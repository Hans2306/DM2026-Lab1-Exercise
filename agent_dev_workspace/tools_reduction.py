"""Dimensionality reduction and label encoding tools."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import LabelBinarizer
from langchain_core.tools import tool


def make_tools(session):
    """Build and return reduction tools bound to the given session."""

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduce dimensionality of the DTM (PCA, t-SNE, UMAP) and plot 2D scatter plot.

        Args:
            method: Reduction method ("pca", "tsne", or "umap").
            n_components: Number of components (must be >= 2).
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors parameter for UMAP.
            random_state: Random state seed for reproducibility.

        Returns:
            Dict containing result_id, status, method, coordinates, and explained_variance_ratio (PCA only).
        """
        if n_components < 2:
            return {"status": "error", "message": f"n_components ({n_components}) must be at least 2 for 2D visualization."}

        if session.feature_matrix is None:
            return {"status": "error", "message": "No Document-Term Matrix found. Call build_dtm_tool first."}

        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Call load_dataset_tool first."}

        matrix = session.feature_matrix.toarray()
        method_lower = method.lower()

        if method_lower == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(matrix)
            explained_variance = list(np.round(model.explained_variance_ratio_, 4))
        elif method_lower == "tsne":
            model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = model.fit_transform(matrix)
            explained_variance = None
        elif method_lower == "umap":
            model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = model.fit_transform(matrix)
            explained_variance = None
        else:
            return {"status": "error", "message": f"Unsupported method: {method}. Choose 'pca', 'tsne', or 'umap'."}

        coords_rounded = np.round(coords, 4).tolist()
        session.artifacts[f"reduced_coords_{method_lower}"] = coords

        # Generate Scatter Plot
        categories = session.dataframe["category_name"] if "category_name" in session.dataframe.columns else session.dataframe.get("label", None)
        fig, ax = plt.subplots(figsize=(8, 6))
        if categories is not None:
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=categories, palette="tab10", ax=ax)
        else:
            ax.scatter(coords[:, 0], coords[:, 1])

        ax.set_title(f"Dimensionality Reduction ({method_lower.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dim_red")

        summary = {
            "result_id": result_id,
            "status": "success",
            "method": method_lower,
            "coordinates": coords_rounded,
        }
        if explained_variance is not None:
            summary["explained_variance_ratio"] = explained_variance

        session.store_result(
            "reduce_dimensions_tool",
            {"method": method, "n_components": n_components},
            summary,
        )

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encode categorical target labels in session.dataframe.

        Binarizes category labels from session.dataframe["category_name"] or "label"
        into binary one-hot representation, storing the result in session.artifacts["binarized_labels"].

        Returns:
            Dict containing result_id, status, classes, and binarized_matrix.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Call load_dataset_tool first."}

        col = "category_name" if "category_name" in session.dataframe.columns else "label"
        if col not in session.dataframe.columns:
            return {"status": "error", "message": "No category/label column found in session.dataframe."}

        labels = session.dataframe[col].astype(str).values
        lb = LabelBinarizer()
        binarized = lb.fit_transform(labels)
        classes = list(lb.classes_)

        # Two classes -> 2 columns; One class -> 1 column of ones
        if len(classes) == 2 and binarized.shape[1] == 1:
            binarized = np.hstack([1 - binarized, binarized])
        elif len(classes) == 1:
            binarized = np.ones((len(labels), 1), dtype=int)

        session.artifacts["binarized_labels"] = binarized

        result_id = session.next_result_id("bin_labels")

        summary = {
            "result_id": result_id,
            "status": "success",
            "classes": classes,
            "n_samples": len(labels),
            "binarized_matrix": binarized.tolist(),
        }

        session.store_result(
            "binarize_labels_tool",
            {},
            summary,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
