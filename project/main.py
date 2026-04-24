import os.path
import os
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import yaml
from typing import Any, Dict, Optional
from project.src.benchmark_runner import BenchmarkRunner
from project.src.model_registry import ModelRegistry
# How to use
# python -m project.main --b --config project/configs/test/byol_cifar10_resnet18.yaml
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
    mode_group.add_argument("--a", action="store_true", help="Analysis mode")
    mode_group.add_argument("--f", action="store_true", help="Full pipeline mode")

    # desired method
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config, e.g. project/configs/test/byol_cifar10_resnet18.yaml",
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
    if args.a:
        return "analyze"
    if args.f:
        return "full"
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
    print(result)

def run_analysis(config: dict) -> None:
    print(f"[ANALYZE] method={config['method']} dataset={config['dataset']}")
    # full analysis pca, svd, correlation
    # this will use report_builder as well in order to generate reports on the fly

def run_full(config: dict) -> None:
    print(f"[FULL] method={config['method']} dataset={config['dataset']}")
    run_train(config)
    run_benchmark(config)
    run_analysis(config)

# launches specific mode base on the entry args
def dispatch_mode(config: dict) -> None:
    mode = config["mode"]
    if mode == "train":
        run_train(config)
    elif mode == "benchmark":
        run_benchmark(config)
    elif mode == "analyze":
        run_analysis(config)
    elif mode == "full":
        run_full(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    print("Starting the pipeline!")
    print(config)

    # actual experiment
    dispatch_mode(config)


if __name__ == "__main__":
    main()
