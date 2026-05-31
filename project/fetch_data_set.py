import torchvision
from pathlib import Path
import kagglehub
import os
import shutil
import urllib.request
import tarfile
from torchvision.datasets import STL10
import ssl

ROOT = "./datasets"

#auxiliary script for fetching neccesary datasets

def fetch_data_set(dataset_name: str) -> None:
    # map names for torchvis
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

def fetch_stl10():
    os.makedirs(ROOT, exist_ok=True)

    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    tar_path = os.path.join(ROOT, "stl10_binary.tar.gz")
    extract_path = os.path.join(ROOT, "stl10_binary")

    print(f"[1/3] Pobieranie oficjalnego archiwum STL-10 (2.5 GB) z {url}...")
    # Pobieramy plik (to potrwa chwilę w zależności od łącza)
    urllib.request.urlretrieve(url, tar_path)
    print("[SUCCESS] Pomyślnie pobrano plik .tar.gz!")

    print("[2/3] Wypakowywanie archiwum...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Wypakowuje bezpośrednio do ./datasets, tworząc natywny stl10_binary
        tar.extractall(path=ROOT)
    print("[SUCCESS] Wypakowano pomyślnie!")

    print("[3/3] Sprzątanie...")
    os.remove(tar_path)

    print(f"\n[GOTOWE] Dataset znajduje się w: {os.path.abspath(extract_path)}")
    print("Możesz teraz użyć tej ścieżki w solo-learn!")

def fetch_via_kagglehub(dataset_name: str) -> None:
    dataset_mapping = {
        "imagenet100": "ambityga/imagenet100"
    }

    if dataset_name in dataset_mapping:
        target_path = ROOT.join(dataset_name)

        # see if already exists
        if os.path.exists(target_path):
            print(f"[INFO] {dataset_name} already exists in {target_path}")
            return

        print(f"[DATA] Downloading {dataset_name} to cache...")
        # download to cache
        downloaded_path = kagglehub.dataset_download(dataset_mapping[dataset_name])
        print(f"[SUCCESS] {dataset_name} ready in {target_path}")
    else:
        print(f"[ERROR] Dataset {dataset_name} not found")



# Run for cifar for now
#fetch_data_set("cifar10")

# run for ImageNet-100
#fetch_via_kagglehub("imagenet100")

# download stl10
fetch_stl10()