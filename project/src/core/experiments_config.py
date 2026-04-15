from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Data Class solely for setting up models parameters for the experiment

@dataclass
class ExperimentConfig:
    model_name: str
    backbone: str
    dataset_name: str
    output_dir: Path = Path("outputs")
    dataset_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    user_projector: bool = True
    random_weights: bool = False
    batch_size: int = 128
    num_workers: int = 4
    device: str = "cuda"