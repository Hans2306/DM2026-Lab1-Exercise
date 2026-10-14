"""Document-Term Matrix (DTM) construction, term frequency, and visualization tools for the text mining pipeline."""

from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    """Build and return DTM management tools bound to the given session."""

    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
    ) -> Dict[str, Any]:
        """Build a Document-Term Matrix (DTM) from the loaded dataset's text column.

        Extracts unigram or n-gram counts using CountVectorizer and stores the resulting
        sparse matrix, feature names, and vectorizer artifact into session state. Calculates
        matrix sparsity metrics.

        Args:
            max_features: Maximum number of vocabulary features to extract (None for all).
            ngram_range: Lower and upper boundary of n-gram values, e.g. [1, 1] or [1, 2].

        Returns:
            Dict containing summary metrics: result_id, n_documents, n_terms,
            non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Call load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "'text' column missing from loaded dataframe."}

        # Handle empty/missing text entries cleanly so row count stays aligned with session.dataframe
        text_series = session.dataframe["text"].fillna("").astype(str)

        ngram_tuple = tuple(ngram_range) if len(ngram_range) == 2 else (1, 1)

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_tuple,
        )

        try:
            dtm_matrix = vectorizer.fit_transform(text_series)
        except ValueError as e:
            # Handle cases where vocabulary is completely empty
            return {"status": "error", "message": f"Failed to build DTM: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = dtm_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].to_numpy())

        # Compute sparsity metrics matching Master notebook logic
        n_rows, n_cols = dtm_matrix.shape
        non_zero = int(dtm_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "ngram_range": ngram_range}, summary)

        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate overall frequency for each term across all documents in the global DTM.

        Sums column occurrences across the document-term matrix (session.feature_matrix)
        and reports total term frequencies sorted in descending order.

        Args:
            top_n: Optional integer to restrict returned term dictionary to top N terms.

        Returns:
            Dict containing result_id, n_terms, total_term_count, top_terms summary,
            and term_frequencies mapping.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix found. Call build_dtm_tool first."}

        # Calculate column sums (total frequency per term across all documents)
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        term_freqs = dict(zip(session.feature_names, map(int, col_sums)))

        # Sort terms by frequency descending
        sorted_freqs = dict(sorted(term_freqs.items(), key=lambda item: item[1], reverse=True))

        filtered_freqs = dict(list(sorted_freqs.items())[:top_n]) if top_n is not None else sorted_freqs

        result_id = session.next_result_id("term_freq")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_terms": len(session.feature_names),
            "total_term_count": int(sum(col_sums)),
            "term_frequencies": filtered_freqs,
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary)

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Generate a heatmap visualization for a slice of the global Document-Term Matrix.

        Renders a positional slice of raw term counts for up to n_documents and n_terms
        and sets the resulting heatmap plot on session.pending_figure.

        Args:
            n_terms: Number of terms (columns) from DTM to include (default 20).
            n_documents: Number of documents (rows) from DTM to include (default 20).

        Returns:
            Dict containing result_id, n_documents, n_terms, term_labels,
            document_labels, and matrix_slice (nested list of raw counts).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No Document-Term Matrix found. Call build_dtm_tool first."}

        dtm_rows, dtm_cols = session.feature_matrix.shape
        actual_docs = min(n_documents, dtm_rows)
        actual_terms = min(n_terms, dtm_cols)

        # Slice the raw DTM positional corner
        slice_matrix = session.feature_matrix[0:actual_docs, 0:actual_terms].toarray()
        term_labels = session.feature_names[0:actual_terms]
        doc_labels = [f"Doc {i}" for i in range(actual_docs)]

        # Generate Heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.4)))
        sns.heatmap(
            slice_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=term_labels,
            yticklabels=doc_labels,
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"Document-Term Matrix Heatmap ({actual_docs} docs x {actual_terms} terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure for agentic notebook rendering
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")

        matrix_list = slice_matrix.tolist()

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "term_labels": term_labels,
            "document_labels": doc_labels,
            "matrix_slice": matrix_list,
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
