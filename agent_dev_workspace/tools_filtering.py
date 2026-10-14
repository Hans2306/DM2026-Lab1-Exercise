"""Feature filtering tools for the text mining pipeline."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    """Build and return feature filtering tools bound to the given session."""

    @tool
    def variance_filter_tool(
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Filter vocabulary terms based on feature variance across documents in global DTM.

        Calculates variance for each term across all documents in session.feature_matrix.
        Flags terms with variance > threshold as kept and variance <= threshold as removed.
        Stores a full report DataFrame sorted by variance for visualize_result_tool.

        Args:
            threshold: Minimum variance threshold for keeping a term (default 0.1).

        Returns:
            Dict containing result_id, threshold, total_terms, kept_count,
            removed_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix found. Call build_dtm_tool first."}

        # Compute per-term variance (ddof=0 matching standard population/sample variance convention)
        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0)

        feature_names = session.feature_names

        # Build full report dataframe for visualize_result_tool
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances,
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_mask = variances > threshold
        kept_terms = [name for name, keep in zip(feature_names, kept_mask) if keep]
        removed_terms = [name for name, keep in zip(feature_names, kept_mask) if not keep]

        result_id = session.next_result_id("variance")

        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=report_df)

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
    ) -> Dict[str, Any]:
        """Calculate Pearson correlation between term counts and a target category indicator.

        Computes Pearson r for each term count against a binary vector indicating if
        a document belongs to target_class (one-vs-rest). Stores a full report DataFrame
        sorted by correlation strength (|r| descending) for visualize_result_tool.

        Args:
            target_class: The target category name to measure correlation against.

        Returns:
            Dict containing result_id, target_class, total_terms, and pearson_correlations mapping.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix found. Call build_dtm_tool first."}

        if session.labels is None:
            return {"status": "error", "message": "No document category labels found in session state."}

        # Binary indicator for target_class (one-vs-rest)
        target_binary = (session.labels == target_class).astype(float)

        if np.all(target_binary == 0):
            return {"status": "error", "message": f"Target class '{target_class}' not found in document labels."}

        dense_matrix = session.feature_matrix.toarray()
        n_terms = len(session.feature_names)
        correlations = []

        for i in range(n_terms):
            col = dense_matrix[:, i]
            # Handle zero variance columns safely
            if np.std(col) == 0 or np.std(target_binary) == 0:
                r_val = 0.0
            else:
                r_val, _ = pearsonr(col, target_binary)
                r_val = float(r_val) if not np.isnan(r_val) else 0.0
            correlations.append(r_val)

        # Build full report sorted by correlation strength (|r| descending)
        report_df = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": correlations,
            "abs_r": np.abs(correlations),
        }).sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        term_corr_map = dict(zip(report_df["term"], report_df["pearson_r"].round(4)))

        result_id = session.next_result_id("pearson")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "total_terms": n_terms,
            "pearson_correlations": term_corr_map,
        }

        session.store_result("pearson_filter_tool", {"target_class": target_class}, summary, full_report=report_df)

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
    ) -> Dict[str, Any]:
        """Calculate Spearman rank correlation between term counts and a target category indicator.

        Computes Spearman rank correlation rho for each term count against a binary vector
        indicating if a document belongs to target_class (one-vs-rest). Stores a full report
        DataFrame sorted by correlation strength (|rho| descending) for visualize_result_tool.

        Args:
            target_class: The target category name to measure rank correlation against.

        Returns:
            Dict containing result_id, target_class, total_terms, and spearman_correlations mapping.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix found. Call build_dtm_tool first."}

        if session.labels is None:
            return {"status": "error", "message": "No document category labels found in session state."}

        # Binary indicator for target_class (one-vs-rest)
        target_binary = (session.labels == target_class).astype(float)

        if np.all(target_binary == 0):
            return {"status": "error", "message": f"Target class '{target_class}' not found in document labels."}

        dense_matrix = session.feature_matrix.toarray()
        n_terms = len(session.feature_names)
        correlations = []

        for i in range(n_terms):
            col = dense_matrix[:, i]
            # Handle zero variance columns safely
            if np.std(col) == 0 or np.std(target_binary) == 0:
                rho_val = 0.0
            else:
                rho_val, _ = spearmanr(col, target_binary)
                rho_val = float(rho_val) if not np.isnan(rho_val) else 0.0
            correlations.append(rho_val)

        # Build full report sorted by correlation strength (|rho| descending)
        report_df = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": correlations,
            "abs_r": np.abs(correlations),
        }).sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        term_corr_map = dict(zip(report_df["term"], report_df["spearman_r"].round(4)))

        result_id = session.next_result_id("spearman")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "total_terms": n_terms,
            "spearman_correlations": term_corr_map,
        }

        session.store_result("spearman_filter_tool", {"target_class": target_class}, summary, full_report=report_df)

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
