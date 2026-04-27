import torchvision
from pathlib import Path

ROOT = "./datasets"

#auxiliary script for fetching neccesary datasets

def fetch_data_set(dataset_name: str) -> None:
    # Mapowanie nazw na klasy torchvision
    dataset_classes = {
        "cifar10": torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
        "mnist": torchvision.datasets.MNIST,
        "fashionmnist": torchvision.datasets.FashionMNIST,
        "svhn": torchvision.datasets.SVHN,
    }

    if dataset_name in dataset_classes:
        print(f"[DATA] Downloading {dataset_name}...")
        cls = dataset_classes[dataset_name]
        # download both sets train and eval
        cls(root=ROOT, train=True, download=True)
        cls(root=ROOT, train=False, download=True)
        print(f"[SUCCESS] {dataset_name} ready in {ROOT}")
    else:
        print(f"[ERROR] Dataset {dataset_name} not found")


# Run for cifar for now
fetch_data_set("cifar10")