from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from project.src.model_registry import ModelRegistry
from project.src.feature_analyzer import FeatureAnalyzer

from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.utils.knn import WeightedKNNClassifier
from torchvision.models import resnet50, ResNet50_Weights

import torchvision.models as models


class BenchmarkRunner:
    #First-stage benchmark runner.
    #- create model from config
    #- optionally load checkpoint
    #- build evaluation dataloader
    #- extract embeddings from validation/test split
    #- run feature analysis
    #- save embeddings/results if requested

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._resolve_device(config)
        self.feature_analyzer = FeatureAnalyzer()

        self.feature_model : Optional[torch.nn.Module] = None

    @staticmethod
    def _resolve_device(config: Dict[str, Any]) -> torch.device:
        # run it on cpu or gpu
        requested_gpus = str(config.get("gpus", "0")).strip()
        if torch.cuda.is_available() and requested_gpus != "":
            return torch.device("cuda")
        return torch.device("cpu")

    def create_model(self):
        #Create model from registry based on config
        model = ModelRegistry.create_model(self.config)
        model = model.to(self.device)
        model.eval()
        return model

    def log_load_report(self, msg, name: str) -> None:
        print(f"\n[LOAD REPORT] {name}")
        print(f"Missing keys: {len(msg.missing_keys)}")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")

        if msg.missing_keys:
            print("First missing keys:")
            for key in msg.missing_keys[:20]:
                print(f" - {key}")

        if msg.unexpected_keys:
            print("First unexpected keys:")
            for key in msg.unexpected_keys[:20]:
                print(f" - {key}")


    def load_checkpoint(self, model) -> None:
        #Load checkpoint into model if checkpoint path is provided.
        #- solo-learn does not support checkpoints no more so we need to fetch them from original repos.
        #- this implementation first tries plain state_dict loading, then tries common nested keys

        checkpoint_path = self.config.get("checkpoint")
        if not checkpoint_path:
            return
        self._load_local_checkpoint(model, Path(checkpoint_path))

    def _load_local_checkpoint(self, model, ckpt_path: Path) -> None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[CHECKPOINT] Loading local checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        else:
            raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

        print("[CHECKPOINT] Original keys sample:")
        for k in list(state_dict.keys())[:20]:
            print(f" - {k}")

        if any(k.startswith("backbone.") for k in state_dict.keys()):
            msg = model.load_state_dict(state_dict, strict=False)
            self.log_load_report(msg, "local solo-learn checkpoint -> full model")
            return

        backbone_state = {}
        for k, v in state_dict.items():
            k_clean = (
                k.replace("module.", "")
                .replace("encoder.", "")
                .replace("resnet.", "")
                .replace("backbone.", "")
            )

            if not k_clean.startswith("fc."):
                backbone_state[k_clean] = v

        msg = model.backbone.load_state_dict(backbone_state, strict=False)
        self.log_load_report(msg, "local checkpoint -> model.backbone")

    def _create_resnet50_feature_model(self) -> torch.nn.Module:
        feature_model = models.resnet50(weights=None)
        feature_model.fc = torch.nn.Identity()
        feature_model = feature_model.to(self.device)
        feature_model.eval()
        return feature_model

    def _load_from_hub(self):
        # fetches weights of fully pretrained models with resnet50 architecture from pytorch hub
        method = self.config["method"].lower()
        print(f"[HUB] Fetching official weights for {method}...")
        # from official repositories
        #DINO
        if method == "dino":
            # https://github.com/facebookresearch/dino
            official_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            self.feature_model = official_model.to(self.device).eval()
            print("[HUB] DINO official feature model loaded directly.")
            return

        elif method == "barlow_twins":
            # https://github.com/facebookresearch/barlowtwins
            official_model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            self.feature_model = official_model.to(self.device).eval()
            print("[HUB] Barlow Twins official feature model loaded directly.")
            return
        elif method == "vicreg":
            # supports on offical github downloading ckpts
            # https://github.com/facebookresearch/vicreg
            official_model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
            self.feature_model = official_model.to(self.device).eval()
            print("[HUB] VICReg official feature model loaded directly.")
            return

        elif method ==  "simclr":
            # methods checkpoints no longer available on repositories smh
            ckpt_path = Path(f"project/models_out/checkpoints/simclr-resnet50-1x.pth")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Download weights for {method} and put em in  {ckpt_path}")

            state_dict = torch.load(ckpt_path, map_location=self.device)

            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            else:
                state_dict = state_dict
            # mapping for solo learn
            print("[SIMCLR] Original keys sample:")
            for k in list(state_dict.keys())[:20]:
                print(f"  - {k}")

            backbone_state = {}

            for k, v in state_dict.items():
                k_clean = k.replace("module.", "")

                # Różne checkpointy SimCLR mają różne prefiksy.
                possible_prefixes = [
                    "encoder.",
                    "backbone.",
                    "resnet.",
                    "net.",
                ]

                matched = False
                for prefix in possible_prefixes:
                    if k_clean.startswith(prefix):
                        new_key = k_clean.replace(prefix, "")
                        backbone_state[new_key] = v
                        matched = True
                        break

                if not matched:
                    # Jeśli wygląda jak czysty ResNet key, weź go.
                    if (
                            k_clean.startswith("conv1.")
                            or k_clean.startswith("bn1.")
                            or k_clean.startswith("layer")
                            or k_clean.startswith("fc.")
                    ):
                        backbone_state[k_clean] = v

            # Nie ładuj fc, bo używamy backbone jako feature extractor.
            backbone_state = {
                k: v for k, v in backbone_state.items()
                if not k.startswith("fc.")
            }

            feature_model = self._create_resnet50_feature_model()
            msg = feature_model.load_state_dict(backbone_state, strict=False)
            self.log_load_report(msg, "SimCLR checkpoint -> official ResNet50 feature_model")
            self.feature_model = feature_model

        elif method ==  "simsiam":
            # methods checkpoints no longer available on repositories smh
            ckpt_path = Path(f"project/models_out/checkpoints/simsiam-resnet50-1x.pth.tar")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Download weights for {method} and put em in  {ckpt_path}")

            state_dict = torch.load(ckpt_path, map_location=self.device)

            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            else:
                state_dict = state_dict

            # mapping for solo learn
            print("[SIMSIAM] Original keys sample:")
            for k in list(state_dict.keys())[:20]:
                print(f"  - {k}")

            # module.encoder.0.*  -> backbone
            # module.encoder.1.*  -> projector
            # module.predictor.*  -> predictor
            backbone_state = {}
            for k, v in state_dict.items():
                k_clean = k.replace("module.", "")

                if k_clean.startswith("encoder.0."):
                    new_key = k_clean.replace("encoder.0.", "")
                    backbone_state[new_key] = v

            if not backbone_state:
                raise ValueError(
                    "No backbone keys found for SimSiam. Expected keys like 'module.encoder.0.conv1.weight'."
                )

            feature_model = self._create_resnet50_feature_model()
            msg = feature_model.load_state_dict(backbone_state, strict=False)
            self.log_load_report(msg, "SimSiam encoder.0 -> official ResNet50 feature_model")
            self.feature_model = feature_model

    def _log_dataset_info(self, loader: DataLoader, split_name: str) -> None:
        dataset = loader.dataset
        print(f"\n[DATASET] {split_name}")
        print(f"Samples: {len(dataset)}")

        if hasattr(dataset, "classes"):
            print(f"Num classes: {len(dataset.classes)}")
            print(f"First classes: {dataset.classes[:10]}")

        if hasattr(dataset, "class_to_idx"):
            first_items = list(dataset.class_to_idx.items())[:10]
            print(f"First class_to_idx: {first_items}")

    def create_eval_loader(self) -> DataLoader:
        dataset = self.config["dataset"]
        data_dir = self.config.get("data_dir", "./datasets")
        train_dir = self.config.get("train_dir", None)
        val_dir = self.config.get("val_dir", None)
        batch_size = self.config.get("batch_size", 256)
        num_workers = self.config.get("num_workers", 4)

        _, val_loader = prepare_data_classification(
            dataset,
            data_dir,
            train_dir,
            val_dir,
            batch_size,
            num_workers,
        )

        self._log_dataset_info(val_loader, split_name="eval")
        return val_loader

    @torch.no_grad()
    def extract_embeddings(
            self,
            model,
            dataloader: DataLoader,
            use_projector: bool,
            split_name: str = "eval",
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_embeddings = []
        all_labels = []

        # Jeśli załadowaliśmy oficjalny model z torch.hub,
        # używamy go bezpośrednio jako feature extractor.
        # Jeśli nie, używamy modelu solo-learn.
        feature_model = self.feature_model if self.feature_model is not None else model
        feature_model.eval()

        first_batch_logged = False

        for batch in dataloader:
            X, y = batch
            X = X.to(self.device, non_blocking=True)

            out = feature_model(X)

            if isinstance(out, dict):
                if "feats" not in out:
                    raise ValueError(f"Output dict does not contain 'feats'. Keys: {out.keys()}")
                feats = out["feats"]
            else:
                feats = out

            # Projector ma sens tylko dla modelu solo-learn.
            # Oficjalne modele z torch.hub zwykle zwracają już backbone features.
            if use_projector:
                if self.feature_model is not None:
                    print("[WARNING] use_projector=True ignored for official feature_model.")
                else:
                    projector = getattr(feature_model, "projector", None)
                    if projector is not None:
                        feats = projector(feats)
                    else:
                        print("[WARNING] Projector requested but not found in model!")

            if feats.ndim > 2:
                feats = torch.flatten(feats, start_dim=1)

            if not first_batch_logged:
                print(f"\n[FIRST BATCH] {split_name}")
                print(f"Feature shape: {tuple(feats.shape)}")
                print(f"Feature mean: {feats.mean().item():.6f}")
                print(f"Feature std: {feats.std().item():.6f}")
                print(f"Feature min: {feats.min().item():.6f}")
                print(f"Feature max: {feats.max().item():.6f}")
                print(f"Labels shape: {tuple(y.shape)}")
                print(f"Labels sample: {y[:20].tolist()}")
                first_batch_logged = True

            all_embeddings.append(feats.detach().cpu())
            all_labels.append(y.detach().cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        print(f"\n[EMBEDDINGS] {split_name}")
        print(f"Shape: {embeddings.shape}")
        print(f"Mean: {embeddings.mean():.6f}")
        print(f"Std: {embeddings.std():.6f}")
        print(f"Min: {embeddings.min():.6f}")
        print(f"Max: {embeddings.max():.6f}")
        print(f"Unique labels: {len(np.unique(labels))}")
        print(f"First labels: {labels[:20]}")

        return embeddings, labels

    def create_train_loader(self) -> DataLoader:
        dataset = self.config["dataset"]
        data_dir = self.config.get("data_dir", "./datasets")
        train_dir = self.config.get("train_dir", None)
        val_dir = self.config.get("val_dir", None)
        batch_size = self.config.get("batch_size", 256)
        num_workers = self.config.get("num_workers", 4)

        train_loader, _ = prepare_data_classification(
            dataset,
            data_dir,
            train_dir,
            val_dir,
            batch_size,
            num_workers,
        )

        self._log_dataset_info(train_loader, split_name="train")

        return train_loader

    def save_embeddings(self,embeddings: np.ndarray,labels: np.ndarray, split_name: str = "eval" ) -> Dict[str, str]:
        #Save embeddings and labels to configured output directory.

        output_dir = Path(self.config.get("embeddings_dir", "./outputs/embeddings"))
        output_dir.mkdir(parents=True, exist_ok=True)

        run_name = self.config.get("name", "benchmark_run")
        emb_path = output_dir / f"{run_name}_{split_name}_embeddings.npy"
        labels_path = output_dir / f"{run_name}_{split_name}_labels.npy"
        np.save(emb_path, embeddings)
        np.save(labels_path, labels)

        return {
            f"{split_name}_embeddings_path": str(emb_path),
            f"{split_name}_labels_path": str(labels_path),
        }

    def run_feature_analysis(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:

        # Delegate embedding-space analysis to FeatureAnalyzer
        return self.feature_analyzer.analyze(embeddings=embeddings, labels=labels)

    def run_knn_eval(
            self,
            train_embeddings: np.ndarray,
            train_labels: np.ndarray,
            test_embeddings: np.ndarray,
            test_labels: np.ndarray,
    ) -> Dict[str, Any]:
        k = int(self.config.get("knn_k", 20))
        temperature = float(self.config.get("knn_temperature", 0.1))
        distance_fx = self.config.get("knn_distance_fx", "cosine")

        knn = WeightedKNNClassifier(
            k=k,
            T=temperature,
            distance_fx=distance_fx,
        )

        knn(
            train_features=torch.from_numpy(train_embeddings),
            train_targets=torch.from_numpy(train_labels),
            test_features=torch.from_numpy(test_embeddings),
            test_targets=torch.from_numpy(test_labels),
        )

        acc1, acc5 = knn.compute()

        return {
            "k": k,
            "temperature": temperature,
            "distance_fx": distance_fx,
            "acc1": float(acc1),
            "acc5": float(acc5),
        }

    def run_linear_eval(
            self,
            train_embeddings: np.ndarray,
            train_labels: np.ndarray,
            test_embeddings: np.ndarray,
            test_labels: np.ndarray,
    ) -> Dict[str, Any]:
        max_iter = int(self.config.get("linear_max_iter", 1000))
        c_value = float(self.config.get("linear_c", 1.0))

        clf = LogisticRegression(
            max_iter=max_iter,
            C=c_value,
            n_jobs=-1,
        )
        clf.fit(train_embeddings, train_labels)

        predictions = clf.predict(test_embeddings)
        acc = accuracy_score(test_labels, predictions)

        return {
            "accuracy": float(acc),
            "max_iter": max_iter,
            "c": c_value,
        }

    def run(self) -> Dict[str, Any]:
        # first create model
        #optionally load checkpoint
        # then create eval loader
        # extract & save embeddings
        # analyze features

        model = self.create_model()
        if self.config.get("checkpoint") == "pytorch_hub":
            model = None
            self._load_from_hub()
        else:
            model = self.create_model()
            self.load_checkpoint(model)

        use_projector = bool(self.config.get("use_projector", False))

        train_loader = self.create_train_loader()
        eval_loader = self.create_eval_loader()

        train_embeddings, train_labels = self.extract_embeddings(
            model=model,
            dataloader=train_loader,
            use_projector=use_projector,
            split_name="train",
        )

        eval_embeddings, eval_labels = self.extract_embeddings(
            model=model,
            dataloader=eval_loader,
            use_projector=use_projector,
            split_name="eval",
        )

        saved_train_paths = self.save_embeddings(train_embeddings, train_labels, split_name="train")
        saved_eval_paths = self.save_embeddings(eval_embeddings, eval_labels, split_name="eval")

        analysis_result = self.run_feature_analysis(eval_embeddings, eval_labels)

        knn_result = self.run_knn_eval(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=eval_embeddings,
            test_labels=eval_labels,
        )

        linear_result = self.run_linear_eval(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            test_embeddings=eval_embeddings,
            test_labels=eval_labels,
        )

        result = {
            "method": self.config["method"],
            "dataset": self.config["dataset"],
            "num_train_samples": int(len(train_labels)),
            "num_eval_samples": int(len(eval_labels)),
            "embedding_dim": int(eval_embeddings.shape[1]),
            "use_projector": use_projector,
            "checkpoint": self.config.get("checkpoint"),
            **saved_train_paths,
            **saved_eval_paths,
            "analysis": analysis_result,
            "knn_eval": knn_result,
            "linear_eval": linear_result,
        }

        return result

    # TODO:
    # implement a method for iterating over checkpoints of a given method to check how embeddigins change every iteration.

    # state_dict = official_model.state_dict()
    # if "conv1.weight" in state_dict:
    # print("[HUB] Skipping conv1.weight due to size mismatch (CIFAR 3x3 vs ImageNet 7x7)")
    # del state_dict["conv1.weight"]

    # model.backbone.load_state_dict(state_dict, strict=False)