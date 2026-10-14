"""Exploration tools for the text mining pipeline."""

from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    """Build and return exploration tools bound to the given session."""

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int,
    ) -> Dict[str, Any]:
        """Calculate cosine similarity between two documents in the loaded dataset.

        Uses document term vectors from session.feature_matrix (or count_vectorizer)
        to compute normalized dot-product similarity between doc_index_1 and doc_index_2.

        Args:
            doc_index_1: Integer row index of the first document.
            doc_index_2: Integer row index of the second document.

        Returns:
            Dict containing result_id, doc_index_1, doc_index_2, text_1, text_2,
            and cosine_similarity value.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No Document-Term Matrix found. Call build_dtm_tool first."}

        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Call load_dataset_tool first."}

        n_docs = session.feature_matrix.shape[0]

        if not (0 <= doc_index_1 < n_docs):
            return {"status": "error", "message": f"doc_index_1 ({doc_index_1}) out of bounds [0, {n_docs - 1}]."}

        if not (0 <= doc_index_2 < n_docs):
            return {"status": "error", "message": f"doc_index_2 ({doc_index_2}) out of bounds [0, {n_docs - 1}]."}

        # Extract rows from feature_matrix
        v1 = session.feature_matrix[doc_index_1]
        v2 = session.feature_matrix[doc_index_2]

        norm1 = np.linalg.norm(v1.toarray())
        norm2 = np.linalg.norm(v2.toarray())

        if norm1 == 0 or norm2 == 0:
            sim_value = 0.0
        else:
            sim_value = float(cosine_similarity(v1, v2)[0, 0])

        text_1 = str(session.dataframe["text"].iloc[doc_index_1]) if "text" in session.dataframe.columns else ""
        text_2 = str(session.dataframe["text"].iloc[doc_index_2]) if "text" in session.dataframe.columns else ""

        result_id = session.next_result_id("cosine_sim")

        summary = {
            "result_id": result_id,
            "status": "success",
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "text_1": text_1,
            "text_2": text_2,
            "cosine_similarity": round(sim_value, 4),
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Compute feature correlation matrix and heatmap over top-20 terms by variance.

        Operates on global DTM (session.feature_matrix and session.feature_names).
        Calculates variance across all documents, picks top 20 terms, computes Pearson correlation,
        builds a seaborn heatmap set on session.pending_figure, and stores the summary result.

        Returns:
            Dict containing result_id, status, top_terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix found. Call build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        # Compute variance per term across all documents
        variances = np.var(dense_matrix, axis=0)

        # Pick top 20 terms by variance (or fewer if total terms < 20)
        n_terms = min(20, len(feature_names))
        top_indices = np.argsort(variances)[::-1][:n_terms]

        top_terms = list(feature_names[top_indices])
        top_matrix = dense_matrix[:, top_indices]

        # Compute Pearson correlation matrix across top terms
        corr_matrix = np.corrcoef(top_matrix, rowvar=False)
        # Handle single term edge case
        if n_terms == 1:
            corr_matrix = np.array([[1.0]])

        # Generate Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
        )
        ax.set_title("Top Feature Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feat_corr")

        corr_matrix_rounded = np.round(corr_matrix, 4).tolist()

        summary = {
            "result_id": result_id,
            "status": "success",
            "top_terms": top_terms,
            "correlation_matrix": corr_matrix_rounded,
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {},
            summary,
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
