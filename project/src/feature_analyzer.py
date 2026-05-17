from __future__ import annotations

from typing import Any,Dict,Optional
import numpy as np
from sklearn.decomposition import PCA
import math

class FeatureAnalyzer:
    def __init__(self, dead_dimension_thresh: float = 1e-8):
        # dead dimension threshold tells how low the weight needs to be so we can round it to zero
        # and classify as dead
        self.dead_dimension_thresh = dead_dimension_thresh


    def analyze(self, embeddings: np.ndarray, labels: Optional[np.ndarray]) -> Dict[str, Any]:
        # main entry point for analysis of feature-spaces

        # validate the shape and type of embeddings np.ndarray
        embeddings = self.validate_embeddings(embeddings)
        # statistics of embeddings
        basic_statistics = self.compute_basic_stats(embeddings)

        # svd decomposition
        svd_stats = self.perform_svd_decomposition(embeddings)

        # need to fetch effective_rank
        eff_rank = svd_stats["effective_rank"]

        # pca analysis to see on the chart
        pca_stats = self.perform_pca_analysis(embeddings, eff_rank)

        advanced_stats = self.compute_advanced_metrics(embeddings)

        result = {
            "basic_statistics": basic_statistics,
            "pca": pca_stats,
            "svd": svd_stats,
            "advanced_metrics": advanced_stats
        }
        # stats for classes
        if labels is not None:
            result["label_stats"] = self.compute_label_stats(labels)

        return result

    def compute_advanced_metrics(self, embeddings: np.ndarray) -> Dict[str, Any]:

        # calculates metrics of use of the vector
        # Cannot cross the limit of 130.000 probes

        if embeddings.shape[0] > 5000:
            idx = np.random.choice(embeddings.shape[0], 5000, replace=False)
            emb_sample = embeddings[idx]
        else:
            emb_sample = embeddings

        # corr of dimentions (redundancy)
        centered = emb_sample - np.mean(emb_sample, axis=0)
        std = np.std(emb_sample, axis=0) + 1e-8
        normalized = centered / std

        # Pearson's corr matrix
        corr_matrix = (normalized.T @ normalized) / (emb_sample.shape[0] - 1)

        # 100x100 dim heatmap
        corr_matrix_sample = corr_matrix[:100, :100].tolist()

        # zero the eigen
        np.fill_diagonal(corr_matrix, 0.0)

        # sum the squares of other values as a metric of cumulative redundancy
        redundancy = np.sum(corr_matrix ** 2) / (corr_matrix.shape[0] ** 2)

        # --- UNIFORMITY ---
        # L2-Norm
        norms = np.linalg.norm(emb_sample, axis=1, keepdims=True) + 1e-8
        l2_normalized = emb_sample / norms

        # Calculate distancees squared between pares of probes (NxN)
        similarity = l2_normalized @ l2_normalized.T
        sq_distances = 2.0 - 2.0 * similarity

        # Uniformity = log( średnia( exp(-2 * sq_distances) ) )
        # , ignore the diagonal
        exp_dist = np.exp(-2.0 * sq_distances)
        np.fill_diagonal(exp_dist, 0.0)
        n = emb_sample.shape[0]
        uniformity = np.log(np.sum(exp_dist) / (n * (n - 1)))

        return {
            "redundancy": float(redundancy),
            "uniformity": float(uniformity),
            "correlation_matrix_sample": corr_matrix_sample
        }


    def perform_pca_analysis(self, embeddings, effective_rank : float )-> Dict[str, Any]:
        # fetch the number of probes and amounts of features
        n_samples, n_features = embeddings.shape

        # max of PCA components
        max_components = min(n_samples, n_features)

        # first center data, subtract mean of every feature
        centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)

        # pca model
        pca = PCA(n_components=max_components)
        # fit pca to our data and project
        transformed = pca.fit_transform(centered)


        #use of variance explained by every part
        explained_variance_ratio = pca.explained_variance_ratio_
        # cumulated variance explained by parts of dimensions
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # for charts
        display_limit = int(math.ceil(effective_rank / 100.0)) * 100 + 100
        display_limit = max(50, min(display_limit, max_components))

        display_cumulative = cumulative_variance[:display_limit]

        # searchsorted looks for index in arr , calc components needed for cum variance accordingly to the full table
        comp_80 = int(np.searchsorted(cumulative_variance, 0.80) + 1)
        comp_90 = int(np.searchsorted(cumulative_variance, 0.90) + 1)
        comp_95 = int(np.searchsorted(cumulative_variance, 0.95) + 1)

        return {
            "n_components_total": int(max_components),
            "n_components_display": int(display_limit),
            "explained_variance_ratio": explained_variance_ratio[:display_limit].tolist(),
            "cumulative_explained_variance": display_cumulative.tolist(),
            "first_component_ratio": float(explained_variance_ratio[0]),
            "components_for_80_variance": comp_80,
            "components_for_90_variance": comp_90,
            "components_for_95_variance": comp_95,
            "projection_2d": transformed[:, :2].tolist() if transformed.shape[1] >= 2 else None,
        }
    def perform_svd_decomposition(self, embeddings: np.ndarray) -> Dict[str, Any]:
                #SVD on centered embeddings.

        # SVD --> X = U * S * V^T
        # where U, V are orthogonal matrixes, S contains singular vals of matrix (eigenvalues)
        # ^T - transposed
        # eigenvalues contain information regarding how much of the picture data is stored
        # in each dimension, if many of them are 0, it might suggest of model collapse

        # center before decomposition
        centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)

        # only care about singular values
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

        # eigenvalues squared are proportional to the variance in other directions
        squared = singular_values ** 2
        # sum of eigenvals sqrd
        total = np.sum(squared)

        if total <= 0:
            # corner case
            normalized_energy = np.zeros_like(squared)
            effective_rank = 0.0
        else:
            # normalize energy so it sums up to 1.0
            normalized_energy = squared / total
            # entropy of the energy across dimnts
            entropy = -np.sum(normalized_energy * np.log(normalized_energy + 1e-12))
            # the higher effective rank the better use of dimentions
            effective_rank = float(np.exp(entropy))

        # relative treshhold < 1% of highest  singular
        svd_thresh = singular_values[0] * 1e-3
        return {
            "singular_values": singular_values.tolist(),
            "top_10_singular_values": singular_values[:10].tolist(),
            "effective_rank": effective_rank,
            "energy_ratio_top_1": float(normalized_energy[0]) if len(normalized_energy) > 0 else 0.0,
            "energy_ratio_top_5": float(np.sum(normalized_energy[:5])) if len(normalized_energy) >= 5 else float(
                np.sum(normalized_energy)),
            "energy_ratio_top_10": float(np.sum(normalized_energy[:10])) if len(normalized_energy) >= 10 else float(
                np.sum(normalized_energy)),
            "small_singular_values_count": int(np.sum(singular_values < svd_thresh)),
        }

    def validate_embeddings(self, embeddings):
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array of shape (N, D), got shape {embeddings.shape}"
            )

        if embeddings.shape[0] < 2:
            raise ValueError("Need at least 2 samples for feature analysis.")

        return embeddings.astype(np.float64, copy=False)

    def compute_basic_stats(self, embeddings: np.ndarray) -> Dict[str, Any]:
        # variance of every feature
        variances = np.var(embeddings, axis=0)
        # mean of every feature
        means = np.mean(embeddings, axis=0)
        # norm of every feature
        norms = np.linalg.norm(embeddings, axis=1)

        # amount of dead dimentions
        dead_dimensions = int(np.sum(variances < self.dead_dimension_thresh))
        # amount of active dimentions
        active_dimensions = int(embeddings.shape[1] - dead_dimensions)

        return {
            "num_samples": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "mean_feature_value": float(np.mean(means)),
            "mean_feature_variance": float(np.mean(variances)),
            "std_feature_variance": float(np.std(variances)),
            "dead_dimensions": dead_dimensions,
            "active_dimensions": active_dimensions,
            "dead_dimension_ratio": float(dead_dimensions / embeddings.shape[1]),
            "mean_embedding_norm": float(np.mean(norms)),
            "std_embedding_norm": float(np.std(norms)),
        }

    def compute_label_stats(self, labels: np.ndarray) -> Dict[str, Any]:
        unique, counts = np.unique(labels, return_counts=True)
        return {
            "num_classes": int(len(unique)),
            "class_counts": {int(k): int(v) for k, v in zip(unique, counts)},
        }