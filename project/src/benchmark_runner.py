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

    def load_checkpoint(self, model) -> None:
        #Load checkpoint into model if checkpoint path is provided.
        #- solo-learn checkpoints may store weights under different keys
        #- this implementation first tries plain state_dict loading, then tries common nested keys

        checkpoint_path = self.config.get("checkpoint")
        if not checkpoint_path:
            return

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format at: {ckpt_path}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"[CHECKPOINT] Loaded from: {ckpt_path}")
        if missing:
            print(f"[CHECKPOINT] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[CHECKPOINT] Unexpected keys: {len(unexpected)}")

    def create_eval_loader(self) -> DataLoader:

        #Build validation/test loader using solo-learn classification dataloader utility.

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
        return val_loader



    @torch.no_grad()
    def extract_embeddings(self,model,dataloader: DataLoader,use_projector: bool ,) -> Tuple[np.ndarray, np.ndarray]:
        #Extract embeddings and labels from dataloader.

        #Default behavior:
        #- uses model(X)
        #- expects output dict with "feats"
        #- if use_projector=True and model exposes projector, tries to use it

        all_embeddings = []
        all_labels = []

        for batch in dataloader:
            # solo classification dataloader usually returns X, y
            X, y = batch
            X = X.to(self.device, non_blocking=True)

            out = model(X)

            if not isinstance(out, dict) or "feats" not in out:
                raise ValueError(
                    "Model forward output must be a dict containing key 'feats'."
                )

            feats = out["feats"]

            if use_projector:
                projector = getattr(model, "projector", None)
                if projector is not None:
                    feats = projector(feats)

            all_embeddings.append(feats.detach().cpu())
            all_labels.append(y.detach().cpu())

        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

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
        return train_loader

    def save_embeddings(self,embeddings: np.ndarray,labels: np.ndarray, split_name: str = "eval" ) -> Dict[str, str]:
        #Save embeddings and labels to configured output directory.

        output_dir = Path(self.config.get("embeddings_dir", "./outputs/embeddings"))
        output_dir.mkdir(parents=True, exist_ok=True)

        run_name = self.config.get("name", "benchmark_run")
        emb_path = output_dir / f"{run_name}_{split_name}_embeddings.npy"
        labels_path = output_dir / f"{run_name}_{split_name}labels.npy"

        np.save(emb_path, embeddings)
        np.save(labels_path, labels)

        return {
            "embeddings_path": str(emb_path),
            "labels_path": str(labels_path),
        }

    def run_feature_analysis(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Delegate embedding-space analysis to FeatureAnalyzer.
        """
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
            multi_class="auto",
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
        self.load_checkpoint(model)

        use_projector = bool(self.config.get("use_projector", False))

        train_loader = self.create_train_loader()
        eval_loader = self.create_eval_loader()

        train_embeddings, train_labels = self.extract_embeddings(
            model=model,
            dataloader=train_loader,
            use_projector=use_projector,
        )

        eval_embeddings, eval_labels = self.extract_embeddings(
            model=model,
            dataloader=eval_loader,
            use_projector=use_projector,
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