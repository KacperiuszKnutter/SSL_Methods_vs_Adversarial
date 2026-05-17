from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from project.src.model_registry import ModelRegistry
from project.src.feature_analyzer import FeatureAnalyzer

from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.utils.knn import WeightedKNNClassifier
from torchvision.models import resnet50, ResNet50_Weights
from solo.utils.auto_umap import OfflineUMAP
import torchvision.models as models

from project.src.report_builder import BenchmarkReportBuilder

import matplotlib.pyplot as plt
import umap
import seaborn as sns
import os

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
        self.umap = OfflineUMAP()
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

        # isolate cache for each method to avoid collisions
        import os
        os.environ["TORCH_HOME"] = f"./.torch_cache/{method}"

        # from official repositories
        #DINO
        if method == "dino":
            # https://github.com/facebookresearch/dino
            official_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
            official_model.fc = torch.nn.Identity()
            self.feature_model = official_model.to(self.device).eval()
            print("[HUB] DINO official feature model loaded directly.")
            return

        elif method == "barlow_twins":
            # https://github.com/facebookresearch/barlowtwins
            official_model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
            official_model.fc = torch.nn.Identity()
            self.feature_model = official_model.to(self.device).eval()
            print("[HUB] Barlow Twins official feature model loaded directly.")
            return
        elif method == "vicreg":
            # supports on offical github downloading ckpts
            # https://github.com/facebookresearch/vicreg
            official_model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
            official_model.fc = torch.nn.Identity()
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
        data_dir = Path(self.config.get("data_dir", "./datasets"))
        train_dir = self.config.get("train_dir")
        val_dir = self.config.get("val_dir")

        # Bezpieczne sklejanie ścieżek
        train_path = str(data_dir / train_dir) if train_dir else str(data_dir)
        val_path = str(data_dir / val_dir) if val_dir else str(data_dir)

        batch_size = self.config.get("batch_size", 256)
        num_workers = self.config.get("num_workers", 4)

        # Używamy argumentów nazwanych!
        _, val_loader = prepare_data_classification(
            dataset=dataset,
            train_data_path=train_path,
            val_data_path=val_path,
            batch_size=batch_size,
            num_workers=num_workers,
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
        data_dir = Path(self.config.get("data_dir", "./datasets"))
        train_dir = self.config.get("train_dir")
        val_dir = self.config.get("val_dir")

        # Bezpieczne sklejanie ścieżek
        train_path = str(data_dir / train_dir) if train_dir else str(data_dir)
        val_path = str(data_dir / val_dir) if val_dir else str(data_dir)

        batch_size = self.config.get("batch_size", 256)
        num_workers = self.config.get("num_workers", 4)

        # Używamy argumentów nazwanych!
        train_loader, _ = prepare_data_classification(
            dataset=dataset,
            train_data_path=train_path,
            val_data_path=val_path,
            batch_size=batch_size,
            num_workers=num_workers,
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

    def compute_dense_metrics(self,model : torch.nn.Module, dataloader: DataLoader, num_batches: int = 5) -> Dict[str, float]:
        print("[DENSE METRICS] Computing spatial metrics...")
        dense_features_list = []

        def hook(module, input, output):
            # Zapisujemy wyjście [B, 2048, 7, 7] odpinając je od grafu obliczeniowego
            dense_features_list.append(output.detach())

        hook_handle = None
        # Szukamy ostatniej warstwy splotowej (zazwyczaj layer4 w ResNet50)
        if model is None:
            # og take self.feature_model
            for name, module in self.feature_model.named_modules():
                if name == 'layer4' or name == 'network.layer4':
                    hook_handle = module.register_forward_hook(hook)
                    break
            self.feature_model.eval()
        else:
            for name, module in model.named_modules():
                # DODANO 'backbone.layer4' dla kompatybilności z solo-learn
                if name in ['layer4', 'network.layer4', 'backbone.layer4']:
                    hook_handle = module.register_forward_hook(hook)
                    break
            model.eval()

        if hook_handle is None:
            print("[WARNING] Could not attach hook to 'layer4'. Skipping dense metrics.")
            return {}


        total_spatial_redundancy = 0.0
        total_dense_similarity = 0.0
        pair_count = 0
        img_count = 0

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break  # Liczymy tylko dla kilku batchy, by nie tracić czasu

                images = images.to(self.device)
                dense_features_list.clear()

                # Przepuszczamy przez sieć (Hook złapie dense features)
                if model is None:
                    self.feature_model(images)
                else:
                    model(images)

                if not dense_features_list:
                    continue

                features = dense_features_list[0]  # Kształt: [B, 2048, 7, 7]
                B, C, H, W = features.shape
                num_patches = H * W  # Zwykle 49 (7x7)

                # Zmieniamy kształt do [B, 49, 2048]
                features = features.view(B, C, num_patches).transpose(1, 2)

                # L2 Normalizacja kosinusowa dla każdej łatki z osobna
                features = torch.nn.functional.normalize(features, p=2, dim=2)

                # =======================================================
                # METRYKA 1: Patch Diversity / Spatial Redundancy
                # Zróżnicowanie łatek na tym SAMYM obrazku.
                # =======================================================
                sim_matrix_self = torch.bmm(features, features.transpose(1, 2))  # [B, 49, 49]

                # Maskujemy przekątną (bo łatka zawsze pasuje do samej siebie na 100%)
                mask = ~torch.eye(num_patches, dtype=torch.bool, device=self.device)

                # Liczymy średnie podobieństwo różnych łatek do siebie (im więcej, tym gorzej/większe rozmycie)
                off_diag_sims = sim_matrix_self[:, mask]
                batch_spatial_redundancy = off_diag_sims.mean().item()

                total_spatial_redundancy += batch_spatial_redundancy * B
                img_count += B

                # =======================================================
                # METRYKA 2: Dense Similarity Score
                # Kosinusowe dopasowanie łatek między RÓŻNYMI losowymi obrazkami.
                # =======================================================
                # Łączymy obrazki w pary (0 z 1, 2 z 3 itd.)
                for j in range(0, B - 1, 2):
                    feat_A = features[j:j + 1]  # [1, 49, 2048]
                    feat_B = features[j + 1:j + 2]  # [1, 49, 2048]

                    sim_matrix_pair = torch.bmm(feat_A, feat_B.transpose(1, 2))  # [1, 49, 49]

                    # Szukamy "najlepszego przyjaciela" w obrazie B dla każdej łatki z A
                    max_sims, _ = sim_matrix_pair.max(dim=2)

                    total_dense_similarity += max_sims.mean().item()
                    pair_count += 1

        hook_handle.remove()  # Sprzątamy

        result = {}
        if img_count > 0:
            result["spatial_patch_redundancy"] = total_spatial_redundancy / img_count
        if pair_count > 0:
            result["mean_dense_similarity_score"] = total_dense_similarity / pair_count

        return result

    def run(self) -> Dict[str, Any]:
        # first create model
        #optionally load checkpoint
        # then create eval loader
        # extract & save embeddings
        # analyze features

        source = self.config.get("checkpoint_source")
        path = self.config.get("checkpoint")

        if path == "pytorch_hub" or source == "official_repo":
            model = None
            self._load_from_hub()
        else:
            # Create a solo-learn based model on checkpoints downloaded as .pth/.ckpt
            # we need to prevent from solo-learn to changing the kernel filters to 3x3 for smaller datasets
            smaller_sets = ["cifar10", "cifar100","stl10"]
            old_dataset = self.config["dataset"]
            if self.config["dataset"] in smaller_sets:
                # workaround to create proper kernels for now
                self.config["dataset"] = "imagenet100"

            model = self.create_model()
            # save it back as it was
            self.config["dataset"] = old_dataset
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

        num_classes = self.config.get("benchmark", {}).get("num_classes", 100)

        # Opcjonalny przyrostek dla nazwy pliku, jeśli badamy projektor
        proj_suffix = "_with_projector" if use_projector else "_backbone_only"

        if path =="pytorch_hub" or source == "official_repo":

            run_name = self.config.get("name", "benchmark_run")

            figures_dir = Path(self.config.get("figures_dir", "project/models_out/figures")) / run_name

            figures_dir.mkdir(parents=True, exist_ok=True)

            filename_train = f"{run_name}_umap_train.pdf"
            filepath_train = figures_dir / filename_train

            filename_eval = f"{run_name}_umap_eval.pdf"
            filepath_eval = figures_dir / filename_eval

            curr_model = self.feature_model if not model else model
            self.umap.plot(self.device,curr_model, train_loader, filepath_train )
            self.umap.plot(self.device,curr_model, eval_loader, filepath_eval)

        else:
            self.plot_umap_projection(
                features=eval_embeddings,
                labels=eval_labels,
                num_classes=num_classes,
                project_suffix=proj_suffix
            )

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

        eval_loader = self.create_eval_loader()
        dense_metrics = self.compute_dense_metrics(model, eval_loader, num_batches=10)

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
            "dense_metrics": dense_metrics
        }

        return result



    def plot_umap_projection(self, features, labels, num_classes, project_suffix="backbone_only"):
        print("[ReportBuilder] UMAP projection...")

        # Redukcja wymiarowości
        # Zostawiłem metrykę 'cosine', ponieważ dla cech z modeli SSL jest ona
        # znacznie lepsza niż domyślna Euklidesowa używana w surowym solo-learn
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(features)

        # load data into df
        df = pd.DataFrame()
        df["feat_1"] = embedding[:, 0]
        df["feat_2"] = embedding[:, 1]
        df["Y"] = labels

        # figsize
        plt.figure(figsize=(9, 9))

        # pallet depends on num of classes
        if num_classes <= 10:
            palette_name = "tab10"
        else:
            palette_name = "husl"

        # Scatterplot
        ax = sns.scatterplot(
            x="feat_1",
            y="feat_2",
            hue="Y",
            palette=sns.color_palette(palette_name, num_classes),
            data=df,
            legend="full",
            alpha=0.3,  # To daje efekt chmury przenikających się kropek
        )

        # Ukrywanie etykiet i osi (styl solo-learn)
        ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, right=False, bottom=False, top=False)

        # fit legends
        if num_classes > 100:
            anchor = (0.5, 1.8)
        else:
            anchor = (0.5, 1.35)

        plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))

        method_name = self.config["method"]
        dataset_name = self.config["dataset"]

        title = f"UMAP - {method_name} on {dataset_name} ({project_suffix})"
        plt.title(title, fontsize=14, pad=20)

        # apply layout
        plt.tight_layout()

        # save
        run_name = self.config.get("name", "benchmark_run")
        figures_dir = Path(self.config.get("figures_dir", "project/models_out/figures")) / run_name
        figures_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{method_name}_{dataset_name}_{project_suffix}_umap.png"
        filepath = figures_dir / filename

        # do not cut the legend
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[UMAP] UMAP saved as: {filename}")

    # TODO:
    # implement a method for iterating over checkpoints of a given method to check how embeddigins change every iteration.
