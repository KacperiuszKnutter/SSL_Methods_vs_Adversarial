from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Type

from omegaconf import OmegaConf

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
SOLO_LEARN_ROOT = PROJECT_ROOT / "solo-learn"

if str(SOLO_LEARN_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLO_LEARN_ROOT))

from solo.methods import BarlowTwins, BYOL, DINO, SimCLR, VICReg
from solo.methods.base import BaseMethod, BaseMomentumMethod


class ModelRegistry:
    """
    Registry for supported SSL methods based on solo-learn.

    Responsibilities:
    - keep mapping from method name to concrete solo-learn class
    - validate if method is supported
    - build OmegaConf config in solo-learn format
    - detect momentum-based methods
    - instantiate the final model
    """

    _REGISTRY: Dict[str, Type] = {
        "barlow_twins": BarlowTwins,
        "simclr": SimCLR,
        "byol": BYOL,
        "vicreg": VICReg,
        "dino": DINO,
    }

    @classmethod
    def list_models(cls) -> List[str]:
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        return model_name in cls._REGISTRY

    @classmethod
    def get_model_class(cls, model_name: str) -> Type:
        if not cls.is_supported(model_name):
            supported = ", ".join(cls.list_models())
            raise ValueError(
                f"Unsupported model '{model_name}'. Supported models: {supported}"
            )
        return cls._REGISTRY[model_name]

    @classmethod
    def is_momentum_method(cls, model_name: str) -> bool:
        method_cls = cls.get_model_class(model_name)
        return issubclass(method_cls, BaseMomentumMethod)

    @classmethod
    def _build_base_cfg_dict(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the shared/base config structure expected by solo-learn BaseMethod.
        """
        return {
            "method": config["method"],
            "backbone": {
                "name": config.get("backbone", "resnet18"),
                "kwargs": config.get("backbone_args", {}),
            },
            "data": {
                "dataset": config.get("dataset", "cifar10"),
                "num_classes": config.get("num_classes", 10),
                "num_large_crops": config.get("num_large_crops", 2),
                "num_small_crops": config.get("num_small_crops", 0),
            },
            "optimizer": {
                "name": config.get("optimizer", "sgd"),
                "batch_size": config.get("batch_size", 256),
                "lr": config.get("lr", 0.01),
                "weight_decay": config.get("weight_decay", 1e-5),
                "classifier_lr": config.get("classifier_lr", 0.5),
                "kwargs": config.get("extra_optimizer_args", {"momentum": 0.9}),
                "exclude_bias_n_norm_wd": config.get(
                    "exclude_bias_n_norm_lars",
                    False,
                ),
            },
            "scheduler": {
                "name": config.get("scheduler", "warmup_cosine"),
                "min_lr": config.get("min_lr", 0.0),
                "warmup_start_lr": config.get("warmup_start_lr", 0.0),
                "warmup_epochs": config.get("warmup_epochs", 10),
                "lr_decay_steps": config.get("lr_decay_steps", None),
                "interval": config.get("scheduler_interval", "step"),
            },
            "knn_eval": {
                "enabled": config.get("online_knn_eval", False),
                "k": config.get("online_knn_k", 20),
                "distance_func": config.get("online_knn_distance_fx", "euclidean"),
            },
            "performance": {
                "disable_channel_last": config.get("disable_channel_last", False),
            },
            "max_epochs": config.get("max_epochs", 100),
            "accumulate_grad_batches": config.get("accumulate_grad_batches", 1),
            "method_kwargs": {},
        }

    @classmethod
    def _extract_method_kwargs(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only method-specific parameters from the flat project config.
        Everything else belongs to shared/base solo-learn config sections.
        """
        reserved_keys = {
            "method",
            "backbone",
            "backbone_args",
            "dataset",
            "num_classes",
            "num_large_crops",
            "num_small_crops",
            "optimizer",
            "batch_size",
            "lr",
            "weight_decay",
            "classifier_lr",
            "extra_optimizer_args",
            "exclude_bias_n_norm_lars",
            "scheduler",
            "scheduler_interval",
            "min_lr",
            "warmup_start_lr",
            "warmup_epochs",
            "lr_decay_steps",
            "max_epochs",
            "accumulate_grad_batches",
            "gpus",
            "checkpoint",
            "random_weights",
            "use_projector",
            "mode",
            "name",
            "report_dir",
            "embeddings_dir",
            "data_dir",
            "train_dir",
            "val_dir",
            "cifar",
            "save_checkpoint",
            "checkpoint_dir",
            "load_config",
            "online_knn_eval",
            "online_knn_k",
            "online_knn_distance_fx",
            "disable_channel_last",
            "save_config",
            "config_name",
            "knn_k",
            "knn_temperature",
            "knn_distance_fx",
            "linear_max_iter",
            "linear_c",
            "base_tau",
            "final_tau",
            "momentum_classifier",
        }

        method_kwargs: Dict[str, Any] = {}
        for key, value in config.items():
            if key not in reserved_keys:
                method_kwargs[key] = value

        return method_kwargs

    @classmethod
    def _add_family_specific_sections(
        cls,
        cfg_dict: Dict[str, Any],
        config: Dict[str, Any],
        method_name: str,
    ) -> Dict[str, Any]:
        """
        Add config sections required by specific method families.
        """
        if cls.is_momentum_method(method_name):
            cfg_dict["momentum"] = {
                "base_tau": config.get("base_tau", 0.99),
                "final_tau": config.get("final_tau", 1.0),
                "classifier": config.get("momentum_classifier", False),
            }

        return cfg_dict

    @classmethod
    def build_solo_cfg(cls, config: Dict[str, Any]):
        """
        Convert flat project config into OmegaConf config expected by solo-learn.
        """
        if "method" not in config:
            raise KeyError("Config must contain key: 'method'")

        method_name = config["method"]
        method_cls = cls.get_model_class(method_name)

        cfg_dict = cls._build_base_cfg_dict(config)
        cfg_dict["method_kwargs"] = cls._extract_method_kwargs(config)
        cfg_dict = cls._add_family_specific_sections(cfg_dict, config, method_name)

        cfg = OmegaConf.create(cfg_dict)

        # Let the concrete solo-learn method inject defaults / assertions.
        cfg = method_cls.add_and_assert_specific_cfg(cfg)
        return cfg

    @classmethod
    def create_model(cls, config: Dict[str, Any]):
        """
        Create a solo-learn model from the flat project config.
        """
        if "method" not in config:
            raise KeyError("Config must contain key: 'method'")

        method_name = config["method"]
        method_cls = cls.get_model_class(method_name)
        cfg = cls.build_solo_cfg(config)
        return method_cls(cfg)

    @classmethod
    def describe(cls) -> Dict[str, str]:
        return {
            "barlow_twins": "Redundancy reduction method based on cross-correlation.",
            "simclr": "Contrastive learning with NT-Xent loss.",
            "byol": "Bootstrap self-supervised learning without negative pairs.",
            "vicreg": "Variance-Invariance-Covariance Regularization.",
            "dino": "Self-distillation with no labels.",
        }