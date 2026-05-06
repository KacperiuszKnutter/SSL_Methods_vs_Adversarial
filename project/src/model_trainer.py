from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from project.src.model_registry import ModelRegistry

from solo.utils.checkpointer import Checkpointer
from solo.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
    prepare_transform,
)
from solo.utils.classification_dataloader import prepare_data as prepare_data_classification


class ModelTrainer:
# class for training models based on solo-learn implementations

    #create SSL model from YAML config
    #create SSL train dataloader
    # create optional validation dataloader for online evaluation
    #configure checkpoint saving
    #configure optional early stopping
    #  train


    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_name = config.get("name", "ssl_train_run")
        self.checkpoint_dir = Path(
            config.get("checkpoint_dir", "project/models_out/checkpoints/mine")
        ) / self.run_name

        self.log_dir = Path(
            config.get("log_dir", "project/models_out/logs")
        )
        # make sure it exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def create_model(self):
        model = ModelRegistry.create_model(self.config)
        return model

    def create_train_loader(self):
        # get function set the provided value if not set it to default
        # in this case set default as cifar10
        dataset = self.config.get("dataset", "cifar10")
        data_dir = self.config.get("data_dir", "./datasets")
        train_dir = self.config.get("train_dir", None)

        batch_size = int(self.config.get("batch_size", 256))
        num_workers = int(self.config.get("num_workers", 4))

        transform = self.build_ssl_transform(dataset)

        num_crops_per_aug = self.config.get("num_crops_per_aug", [2])
        transform = prepare_n_crop_transform(
            transform,
            num_crops_per_aug=num_crops_per_aug,
        )

        train_dataset = prepare_datasets(
            dataset,
            transform,
            data_dir=data_dir,
            train_dir=train_dir,
            no_labels=False,
        )

        train_loader = prepare_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return train_loader

    def create_val_loader(self):
        if not self.config.get("online_eval", True):
            return None

        dataset = self.config.get("dataset", "cifar10")
        data_dir = self.config.get("data_dir", "./datasets")
        train_dir = self.config.get("train_dir", None)
        val_dir = self.config.get("val_dir", None)

        batch_size = int(self.config.get("batch_size", 256))
        num_workers = int(self.config.get("num_workers", 4))

        _, val_loader = prepare_data_classification(
            dataset,
            data_dir,
            train_dir,
            val_dir,
            batch_size,
            num_workers,
        )

        return val_loader

    def create_callbacks(self):
        callbacks = []

        callbacks.append(
            LearningRateMonitor(logging_interval="epoch")
        )

        if self.config.get("save_checkpoint", True):
            args = Namespace(**self.config)

            checkpoint_frequency = int(
                self.config.get("checkpoint_frequency", 10)
            )

            callbacks.append(
                Checkpointer(
                    args=args,
                    logdir=str(self.checkpoint_dir),
                    frequency=checkpoint_frequency,
                )
            )

        early_cfg = self.config.get("early_stopping", {})
        if early_cfg.get("enabled", False):
            callbacks.append(
                EarlyStopping(
                    monitor=early_cfg.get("monitor", "valid_acc1"),
                    mode=early_cfg.get("mode", "max"),
                    patience=int(early_cfg.get("patience", 20)),
                    min_delta=float(early_cfg.get("min_delta", 0.0)),
                )
            )

        return callbacks

    def create_logger(self):
        return CSVLogger(
            save_dir=str(self.log_dir),
            name=self.run_name,
        )


def build_ssl_transform(self, dataset: str):
    augmentations = self.config.get("augmentations", {})

    if not augmentations:
        raise ValueError(
            "Missing 'augmentations' section in YAML config. "
            "Add augmentations to your method config."
        )

    if "view1" in augmentations and "view2" in augmentations:
        transform = [
            prepare_transform(dataset, **augmentations["view1"]),
            prepare_transform(dataset, **augmentations["view2"]),
        ]
        num_crops_per_aug = self.config.get("num_crops_per_aug", [1, 1])
    else:
        transform = [prepare_transform(dataset, **augmentations)]
        num_crops_per_aug = self.config.get("num_crops_per_aug", [2])

    return prepare_n_crop_transform(
        transform,
        num_crops_per_aug=num_crops_per_aug,
    )

    def create_trainer(self, callbacks, logger):
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        devices = self.config.get("devices", None)
        if devices is None:
            gpus = str(self.config.get("gpus", "0"))
            devices = 1 if accelerator == "gpu" and gpus != "" else "auto"

        trainer = Trainer(
            max_epochs=int(self.config.get("max_epochs", 100)),
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            callbacks=callbacks,
            precision=self.config.get("precision", "32-true"),
            accumulate_grad_batches=int(
                self.config.get("accumulate_grad_batches", 1)
            ),
            log_every_n_steps=int(self.config.get("log_every_n_steps", 50)),
            enable_checkpointing=False,
        )

        return trainer

    def train(self) -> Dict[str, Any]:
        model = self.create_model()
        train_loader = self.create_train_loader()
        val_loader = self.create_val_loader()

        callbacks = self.create_callbacks()
        logger = self.create_logger()
        trainer = self.create_trainer(callbacks, logger)

        ckpt_path = self.config.get("resume_from_checkpoint", None)

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )

        return {
            "run_name": self.run_name,
            "method": self.config.get("method"),
            "dataset": self.config.get("dataset"),
            "backbone": self.config.get("backbone"),
            "max_epochs": self.config.get("max_epochs"),
            "checkpoint_dir": str(self.checkpoint_dir),
            "log_dir": str(self.log_dir / self.run_name),
            "best_model_path": getattr(trainer.checkpoint_callback, "best_model_path", None),
        }