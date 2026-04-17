from argparse import ArgumentParser, Namespace
from copy import deepcopy
# How to use
# call for training: python main.py --t --method barlow_twins --dataset cifar10 --epochs 100
# call for benchmark: python main.py --b --method simclr --dataset cars --checkpoint ./ckpt.ckpt
# call for analysis: python main.py --a --method vicreg --dataset places --checkpoint ./ckpt.ckpt
# cal for a full pipeline: python main.py --f --method byol --dataset cifar10 --epochs 50 --no-projector

# copied from solo-learn overview paper
BASE_CONFIG = {
    "backbone": "resnet18",
    "num_classes": 10,
    "cifar": True,
    "zero_init_residual": True,
    "max_epochs": 100,
    "optimizer": "sgd",
    "lars": True,
    "lr": 0.01,
    "gpus": "0",
    "grad_clip_lars": True,
    "weight_decay": 0.00001,
    "classifier_lr": 0.5,
    "exclude_bias_n_norm_lars": True,
    "accumulate_grad_batches": 1,
    "extra_optimizer_args": {"momentum": 0.9},
    "scheduler": "warmup_cosine",
    "min_lr": 0.0,
    "warmup_start_lr": 0.0,
    "warmup_epochs": 10,
    "num_crops_per_aug": [2, 0],
    "num_large_crops": 2,
    "num_small_crops": 0,
    "eta_lars": 0.02,
    "lr_decay_steps": None,
    "dali_device": "gpu",
    "batch_size": 256,
    "num_workers": 4,
    "data_dir": "./datasets",
    "train_dir": None,
    "val_dir": None,
    "dataset": "cifar10",
    "name": "experiment",
}
# TODO: iBot
METHOD_CONFIGS = {
    "barlow_twins": {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 2048,
        "lamb": 5e-3,
        "scale_loss": 0.025,
        "backbone_args": {"cifar": True, "zero_init_residual": True},
    },
    "simclr": {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 128,
        "temperature": 0.2,
        "backbone_args": {"cifar": True, "zero_init_residual": True},
    },
    "byol": {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 256,
        "pred_hidden_dim": 4096,
        "backbone_args": {"cifar": True, "zero_init_residual": True},
    },
    "vicreg": {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 2048,
        "sim_loss_weight": 25.0,
        "var_loss_weight": 25.0,
        "cov_loss_weight": 1.0,
        "backbone_args": {"cifar": True, "zero_init_residual": True},
    },
    "dino": {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 65536,
        "num_prototypes": 65536,
        "backbone_args": {"cifar": True, "zero_init_residual": True},
    },
}

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="SSL pipeline based on solo-learn")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    # type of desired experiment
    mode_group.add_argument("--b", action="store_true", help="Benchmark mode")
    mode_group.add_argument("--t", action="store_true", help="Training mode")
    mode_group.add_argument("--a", action="store_true", help="Analysis mode")
    mode_group.add_argument("--f", action="store_true", help="Full pipeline mode")

    # desired method
    parser.add_argument("--method", type=str, required=True,choices=list(METHOD_CONFIGS.keys()))
    # data set to train, test etc
    parser.add_argument("--dataset", type=str, default="cifar10")
    # backbone architecture
    parser.add_argument("--backbone", type=str, default="resnet18")
    # num of epochs
    parser.add_argument("--epochs", type=int, default=100)
    # amount of images for each iter
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="./datasets")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--name", type=str, default=None)
    # load a certain point in the training
    parser.add_argument("--checkpoint", type=str, default=None)
    #specified for training,
    parser.add_argument("--random-weights", action="store_true")
    parser.add_argument("--use-projector", action="store_true", default=True)
    parser.add_argument("--no-projector", action="store_true")
    parser.add_argument("--report-dir", type=str, default="./outputs/reports")
    parser.add_argument("--embeddings-dir", type=str, default="./outputs/embeddings")

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


def build_config(args: Namespace) -> dict:
    if args.method not in METHOD_CONFIGS:
        raise ValueError(f"Unsupported method: {args.method}")

    config = deepcopy(BASE_CONFIG)
    config.update(deepcopy(METHOD_CONFIGS[args.method]))

    config["method"] = args.method
    config["dataset"] = args.dataset
    config["backbone"] = args.backbone
    config["max_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["num_workers"] = args.num_workers
    config["data_dir"] = args.data_dir
    config["train_dir"] = args.train_dir
    config["val_dir"] = args.val_dir
    config["gpus"] = args.gpus
    config["checkpoint"] = args.checkpoint
    config["random_weights"] = args.random_weights
    config["use_projector"] = False if args.no_projector else True
    config["mode"] = resolve_mode(args)

    if args.name:
        config["name"] = args.name
    else:
        config["name"] = f"{args.method}-{args.dataset}-{config['mode']}"

    config["report_dir"] = args.report_dir
    config["embeddings_dir"] = args.embeddings_dir

    config["cifar"] = "cifar" in args.dataset.lower()
    config["backbone_args"] = {
        "cifar": config["cifar"],
        "zero_init_residual": config["zero_init_residual"],
    }

    return config

def run_train(config: dict) -> None:
    print(f"[TRAIN] method={config['method']} dataset={config['dataset']}")
    # TODO: create model, trainer.fit(...)

def run_benchmark(config: dict) -> None:
    print(f"[BENCHMARK] method={config['method']} dataset={config['dataset']}")
    #load checkpoint, evaluate embeddings, kNN, linear eval

def run_analysis(config: dict) -> None:
    print(f"[ANALYZE] method={config['method']} dataset={config['dataset']}")
    # full analysis pca, svd, correlation
    # this will use report_builder as well in order to generate reports on the fly

def run_full(config: dict) -> None:
    print(f"[FULL] method={config['method']} dataset={config['dataset']}")
    run_train(config)
    run_benchmark(config)
    run_analysis(config)


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

def main():
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)

    print("Starting the pipeline!")
    print(config)

    dispatch_mode(config)


if __name__ == "__main__":
    main()
