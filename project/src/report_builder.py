from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


class BenchmarkReportBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_name = config.get("name", "benchmark_run")

        # make sure the paths exist!
        self.report_dir = Path(config.get("report_dir", "project/models_out/reports")) / self.run_name
        self.figure_dir = Path(config.get("figures_dir", "project/models_out/figures")) / self.run_name

        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    #building all the neccesary details for jupyter notebooks. And visualization
    def build(self, result: Dict[str, Any]) -> Dict[str, str]:
        saved = {}

        saved["json_report"] = str(self.save_json(result))

        pca_path = self.plot_pca_2d(result)
        if pca_path:
            saved["pca_2d_plot"] = str(pca_path)

        saved["pca_variance_plot"] = str(self.plot_pca_cumulative_variance(result))
        saved["svd_plot"] = str(self.plot_singular_values(result))
        saved["metrics_plot"] = str(self.plot_eval_metrics(result))
        saved["summary_txt"] = str(self.save_summary_txt(result))

        return saved

    def save_json(self, result: Dict[str, Any]) -> Path:
        path = self.report_dir / "benchmark_result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return path

    def save_summary_txt(self, result: Dict[str, Any]) -> Path:
        path = self.report_dir / "summary.txt"

        basic = result["analysis"]["basic_statistics"]
        svd = result["analysis"]["svd"]
        pca = result["analysis"]["pca"]
        knn = result.get("knn_eval", {})
        linear = result.get("linear_eval", {})

        text = f"""
Benchmark summary

Method: {result.get("method")}
Dataset: {result.get("dataset")}
Checkpoint: {result.get("checkpoint")}
Use projector: {result.get("use_projector")}

Train samples: {result.get("num_train_samples")}
Eval samples: {result.get("num_eval_samples")}
Embedding dim: {result.get("embedding_dim")}

kNN Acc@1: {knn.get("acc1")}
kNN Acc@5: {knn.get("acc5")}
Linear accuracy: {linear.get("accuracy")}

Active dimensions: {basic.get("active_dimensions")}
Dead dimensions: {basic.get("dead_dimensions")}
Dead dimension ratio: {basic.get("dead_dimension_ratio")}

Effective rank: {svd.get("effective_rank")}
SVD top-1 energy: {svd.get("energy_ratio_top_1")}
SVD top-5 energy: {svd.get("energy_ratio_top_5")}
SVD top-10 energy: {svd.get("energy_ratio_top_10")}

PCA components for 80% variance: {pca.get("components_for_80_variance")}
PCA components for 90% variance: {pca.get("components_for_90_variance")}
PCA components for 95% variance: {pca.get("components_for_95_variance")}
""".strip()

        path.write_text(text, encoding="utf-8")
        return path

    def plot_pca_2d(self, result: Dict[str, Any]) -> Optional[Path]:
        pca = result["analysis"]["pca"]
        projection = pca.get("projection_2d")

        if projection is None:
            return None

        projection = np.asarray(projection)
        path = self.figure_dir / "pca_2d.png"

        plt.figure(figsize=(8, 6))
        plt.scatter(projection[:, 0], projection[:, 1], s=6, alpha=0.7)
        plt.title(f"PCA 2D projection - {self.run_name}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

        return path

    def plot_pca_cumulative_variance(self, result: Dict[str, Any]) -> Path:
        pca = result["analysis"]["pca"]
        cumulative = np.asarray(pca["cumulative_explained_variance"])

        path = self.figure_dir / "pca_cumulative_variance.png"

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(cumulative) + 1), cumulative)
        plt.axhline(0.8, linestyle="--")
        plt.axhline(0.9, linestyle="--")
        plt.axhline(0.95, linestyle="--")
        plt.title(f"PCA cumulative explained variance - {self.run_name}")
        plt.xlabel("Number of PCA components")
        plt.ylabel("Cumulative explained variance")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

        return path

    def plot_singular_values(self, result: Dict[str, Any]) -> Path:
        svd = result["analysis"]["svd"]
        singular_values = np.asarray(svd["singular_values"])

        path = self.figure_dir / "svd_singular_values.png"

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(singular_values) + 1), np.log10(singular_values + 1e-12))
        plt.title(f"SVD singular values log10 - {self.run_name}")
        plt.xlabel("Singular value index")
        plt.ylabel("log10(singular value)")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

        return path

    def plot_eval_metrics(self, result: Dict[str, Any]) -> Path:
        knn = result.get("knn_eval", {})
        linear = result.get("linear_eval", {})

        labels = ["kNN Acc@1", "kNN Acc@5", "Linear Acc"]
        values = [
            float(knn.get("acc1", 0.0)),
            float(knn.get("acc5", 0.0)),
            float(linear.get("accuracy", 0.0)),
        ]

        path = self.figure_dir / "eval_metrics.png"

        plt.figure(figsize=(7, 5))
        plt.bar(labels, values)
        plt.ylim(0.0, 1.0 if max(values) <= 1.0 else 100.0)
        plt.title(f"Evaluation metrics - {self.run_name}")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

        return path