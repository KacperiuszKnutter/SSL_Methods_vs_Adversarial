# SSL_Methods_vs_Adversarial

This repository contains the codebase for an engineering thesis project investigating the robustness and efficiency of various Self-Supervised Learning (SSL) models. 

The research focuses on a deep analysis of latent feature vectors produced by popular SSL architectures (such as SimCLR, Barlow Twins, DINO, SimSiam, and VICReg). It explores model compression and knowledge distillation techniques to transfer knowledge from large-scale pre-trained models (e.g., ResNet-50 on ImageNet) to smaller, more efficient networks (e.g., ResNet-18 on CIFAR-10). Ultimately, the project aims to evaluate the robustness of these representations against various forms of adversarial attacks.

> **⚠️ Note on Current Project Status:** > At this stage of the research, the adversarial attack modules are **not yet implemented**. The current focus of the repository is solely on feature extraction, space degradation analysis (PCA/SVD), and model compression via Knowledge Distillation.

## Credits & Acknowledgements
This repository is heavily inspired by and based on the **[solo-learn](https://github.com/vturrisi/solo-learn)** library. All credit for the foundational implementations of the self-supervised learning methods goes to the original authors of the `solo-learn` repository.

## Project Structure
The repository is organized to clearly separate configurations, source code, and outputs:

* `configs/` - Contains YAML configuration files for different stages (`benchmark`, `finetune`, `pretrain`).
* `src/` - Core Python modules (`benchmark_runner.py`, `compression_runner.py`, `model_registry.py`, etc.).
* `notebooks/` - Jupyter Notebook templates for generating reports and analyzing feature spaces.
* `models_out/` - Automatically generated directory storing outputs:
  * `checkpoints/` - Downloaded and trained model weights.
  * `embeddings/` - Extracted feature vectors (`.npy`).
  * `figures/` - Generated plots (UMAP, PCA, SVD, loss curves).
  * `reports/` - Text summaries and JSON benchmark results.
* `main.py` - The main entry point for running the pipeline.
* `fetch_data_set.py` - Script for downloading necessary datasets (e.g., CIFAR-10, STL-10).

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/KacperiuszKnutter/SSL_Methods_vs_Adversarial.git](https://github.com/KacperiuszKnutter/SSL_Methods_vs_Adversarial.git)
   cd SSL_Methods_vs_Adversarial
   ```
2. **Create and activate a virtual environment (recommended):** 
```python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```
3. **Install dependencies:**
(Note: ensure your virtual environment is active)
```
Bash
pip install -r reqiurements.txt
```


## Usage
1. Prepare Datasets
Before running any benchmarks, download the required datasets using the provided script ( or download it yourself from trusted source ):
```
Bash
python fetch_data_set.py
```
2. Run the Pipeline
The execution is controlled via YAML configuration files located in the configs/ directory. To run a specific benchmark or compression task, pass the corresponding config to main.py (adjust the path inside main.py or pass it as an argument depending on your setup).

Example execution:
```
Bash
python -m project.main --b --config project/configs/dino-final-resnet50-cifar10-cifar10.yaml
```
(Make sure to point to the correct .yaml file, e.g., configs/benchmark/bt-resnet50-imagenet1k-to-cifar10.yaml, inside the code before running).

Outputs, including UMAP visualisations, PCA degradation plots, and distillation history, will be automatically saved to the models_out/ directory.