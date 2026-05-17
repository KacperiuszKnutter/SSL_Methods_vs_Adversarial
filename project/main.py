import os.path
import os
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import yaml
from typing import Any, Dict, Optional
from project.src.benchmark_runner import BenchmarkRunner
from project.src.model_registry import ModelRegistry

from project.src.report_builder import  BenchmarkReportBuilder
# How to use
# calls for benchmarks for supported methods:
# python -m project.main --b --config project/configs/dino-final-resnet50-cifar10-cifar10.yaml
# python -m project.main --b --config project/configs/bt-resnet50-imagenet1k-to-cifar10.yaml
# python -m project.main --b --config project/configs/simclr-resnet50-imagenet1k-to-cifar10.yaml
# python -m project.main --b --config project/configs/simsiam-resnet50-imagenet1k-to-cifar10.yaml
# python -m project.main --b --config project/configs/vicreg-resnet50-imagenet1k-to-cifar10.yaml

# call for training: python main.py --t --method barlow_twins --dataset cifar10 --epochs 100
# call for benchmark: python main.py --b --method simclr --dataset cars --checkpoint ./ckpt.ckpt
# call for analysis: python main.py --a --method vicreg --dataset places --checkpoint ./ckpt.ckpt
# cal for a full pipeline: python main.py --f --method byol --dataset cifar10 --epochs 50 --no-projector


# -------------------------------------- BASE CONFIGS FOR TESTING METHODS -----------------------------------------------
# path to configs for models,
CONFIG_PATH = Path("project/configs")

# ----------------------------------------- Functions strictly for parsing arguments -----------------------------------
def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="SSL pipeline based on solo-learn")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    # type of desired experiment
    mode_group.add_argument("--b", action="store_true", help="Benchmark mode")
    mode_group.add_argument("--t", action="store_true", help="Training mode")
    #mode_group.add_argument("--a", action="store_true", help="Analysis mode")
    mode_group.add_argument("--f", action="store_true", help="Fine-tuning a model")

    # desired method
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config, e.g. project/configs/byol_cifar10_resnet18.yaml",
    )
    # optional overrides
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--use-projector", action="store_true", default=None)
    parser.add_argument("--no-projector", action="store_true")

    return parser

def resolve_mode(args: Namespace) -> str:
    if args.b:
        return "benchmark"
    if args.t:
        return "train"
    if args.f:
        return "finetune"
    raise ValueError("No mode selected")

# ------------------------------------------ Building config for each method --------------------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        path = CONFIG_PATH / config_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML dictionary: {path}")

    print(f"[CONFIG] Loaded: {path}")
    return config

# ------------------------------------------ YAML config utilities -----------------------------------------------------
def apply_cli_overrides(config: Dict[str, Any], args: Namespace) -> Dict[str, Any]:
    config = dict(config)

    config["mode"] = resolve_mode(args)

    if args.checkpoint is not None:
        config["checkpoint"] = args.checkpoint

    if args.name is not None:
        config["name"] = args.name

    if args.dataset is not None:
        config["dataset"] = args.dataset

    if args.backbone is not None:
        config["backbone"] = args.backbone

    if args.data_dir is not None:
        config["data_dir"] = args.data_dir

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    if args.no_projector:
        config["use_projector"] = False
    elif args.use_projector:
        config["use_projector"] = True

    if "name" not in config or config["name"] is None:
        config["name"] = f"{config['method']}-{config['dataset']}-{config['mode']}"

    if "backbone_args" not in config:
        config["backbone_args"] = {"zero_init_residual": True}

    return config
# ---------------------------- Methods for calling other classes for a specific task: training, fine-tuning, loading checkpoints etc. --------------------
def run_train(config: dict) -> None:
    print(f"[TRAIN] method={config['method']} dataset={config['dataset']}")

    checkpoint_dir = Path(config.get("checkpoint_dir", "./outputs/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TRAIN] Checkpoints will be saved to: {checkpoint_dir}")

    # TODO: training logic (Lightning Trainer)

    if config.get("save_checkpoint"):
        print("[TRAIN] Checkpoint saving ENABLED")
    else:
        print("[TRAIN] Checkpoint saving DISABLED")

def run_benchmark(config: dict) -> None:
    print(f"[BENCHMARK] method={config['method']} dataset={config['dataset']}")
    #load checkpoint, evaluate embeddings, kNN, linear eval
    runner = BenchmarkRunner(config)
    result = runner.run()

    report_builder = BenchmarkReportBuilder(config)
    report_paths = report_builder.build(result)

    #print(result)
    for key, path in report_paths.items():
        print(f"Report saved for --> {key}: {path}")

def fine_tuning(config: dict) -> None:
    print(f"[FULL] method={config['method']} dataset={config['dataset']}")

# launches specific mode base on the entry args
def dispatch_mode(config: dict) -> None:
    mode = config["mode"]
    if mode == "train":
        run_train(config)
    elif mode == "benchmark":
        run_benchmark(config)
    elif mode == "finetune":
        fine_tuning(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def build_mode_config(raw_config: Dict[str, Any], mode: str) -> Dict[str, Any]:
    # flattens given yaml structure
    # YAML structure:
    #   experiment:
    #   model:
    #   checkpoint:
    #   outputs:
    #   benchmark:
    #   pretrain:
    #   finetune:


    experiment_cfg = raw_config.get("experiment", {})
    model_cfg = raw_config.get("model", {})
    checkpoint_cfg = raw_config.get("checkpoint", {})
    outputs_cfg = raw_config.get("outputs", {})

    common: Dict[str, Any] = {}

    # Common model fields: method, backbone, num_classes.
    common.update(model_cfg)

    # Output paths: embeddings_dir, report_dir, figures_dir, checkpoint_dir, log_dir.
    common.update(outputs_cfg)

    # Experiment metadata.
    common["name"] = experiment_cfg.get(
        "name",
        f"{model_cfg.get('method', 'model')}-{mode}",
    )
    common["seed"] = experiment_cfg.get("seed", 42)
    common["description"] = experiment_cfg.get("description", "")

    # Checkpoint metadata.
    if checkpoint_cfg:
        common["checkpoint_source"] = checkpoint_cfg.get("source")
        common["checkpoint"] = checkpoint_cfg.get("path")
        common["pretrain_dataset"] = checkpoint_cfg.get("pretrain_dataset")
        common["checkpoint_architecture"] = checkpoint_cfg.get("architecture")
        common["checkpoint_verified"] = checkpoint_cfg.get("verified", False)
        common["hub_repo"] = checkpoint_cfg.get("hub_repo")
        common["hub_model"] = checkpoint_cfg.get("hub_model")

    if mode == "benchmark":
        benchmark_cfg = dict(raw_config.get("benchmark", {}))
        flat = {**common, **benchmark_cfg}

        # Flatten kNN section.
        knn_cfg = benchmark_cfg.get("knn", {})
        flat["knn_k"] = knn_cfg.get("k", 20)
        flat["knn_temperature"] = knn_cfg.get("temperature", 0.1)
        flat["knn_distance_fx"] = knn_cfg.get("distance_fx", "cosine")

        # Flatten linear eval section.
        linear_cfg = benchmark_cfg.get("linear_eval", {})
        flat["linear_max_iter"] = linear_cfg.get("max_iter", 1000)
        flat["linear_c"] = linear_cfg.get("c", 1.0)
        flat["linear_standardize"] = linear_cfg.get("standardize", True)

        # Benchmark-specific checkpoint can override global checkpoint.
        if benchmark_cfg.get("checkpoint") is not None:
            flat["checkpoint"] = benchmark_cfg.get("checkpoint")

        if benchmark_cfg.get("checkpoint_source") is not None:
            flat["checkpoint_source"] = benchmark_cfg.get("checkpoint_source")

        flat["mode"] = "benchmark"
        return flat

    if mode == "train":
        pretrain_cfg = dict(raw_config.get("pretrain", {}))
        flat = {**common, **pretrain_cfg}

        flat["mode"] = "train"
        return flat

    if mode == "finetune":
        finetune_cfg = dict(raw_config.get("finetune", {}))
        flat = {**common, **finetune_cfg}

        # Fine-tune checkpoint can override global checkpoint.
        if finetune_cfg.get("checkpoint") is not None:
            flat["checkpoint"] = finetune_cfg.get("checkpoint")

        if finetune_cfg.get("checkpoint_source") is not None:
            flat["checkpoint_source"] = finetune_cfg.get("checkpoint_source")

        flat["mode"] = "finetune"
        return flat

    raise ValueError(f"Unsupported mode: {mode}")

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raw_config = load_config(args.config)
    mode = resolve_mode(args)

    config = build_mode_config(raw_config, mode)
    config = apply_cli_overrides(config, args)

    print("Starting the pipeline!")
    print(config)

    dispatch_mode(config)

if __name__ == "__main__":
    main()
