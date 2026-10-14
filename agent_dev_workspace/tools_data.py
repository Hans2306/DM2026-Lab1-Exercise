"""Data exploration and preparation tools for agentic data mining pipeline."""

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    """Factory creating data preparation and exploration tools bound to session."""

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """Lists files and directories in the repository or a given subdirectory.

        Args:
            subdirectory: Relative path of the folder to list (default is repo root).

        Returns:
            Dictionary containing result_id, subdirectory, directories, and files.
        """
        # Resolve path relative to working directory
        target_path = os.path.normpath(subdirectory) if subdirectory else "."

        if not os.path.exists(target_path):
            return {"error": f"Path not found: {subdirectory}"}

        try:
            entries = os.listdir(target_path)
        except Exception as e:
            return {"error": f"Failed to list directory: {str(e)}"}

        dirs = []
        files = []
        for entry in entries:
            # Skip hidden files/directories and cache folders
            if entry.startswith(".") or entry == "__pycache__":
                continue
            full_entry = os.path.join(target_path, entry)
            if os.path.isdir(full_entry):
                dirs.append(entry)
            else:
                files.append(entry)

        dirs.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": dirs,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing or empty text values in the working dataframe.

        Returns:
            Dictionary containing result_id, n_missing, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Working dataframe has no 'text' column."}

        # Check for NaN/None or empty / whitespace-only string
        is_missing = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_indices = df.index[is_missing].tolist()

        result_id = session.next_result_id("missing")
        summary = {
            "result_id": result_id,
            "n_missing": len(missing_indices),
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate documents and optionally drops all copies.

        When drop is True, removes every copy of a duplicated row (keep=False),
        matching the pipeline requirement.

        Args:
            drop: Whether to drop duplicates from session.dataframe. Defaults to False.

        Returns:
            Dictionary containing result_id, n_duplicates, duplicate_indices, and remaining_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Working dataframe has no 'text' column."}

        # Create a hashable version of all columns for duplicate detection,
        # converting list columns (such as 'unigrams') into tuples so pandas can hash them.
        hashable_df = df.copy()
        for col in hashable_df.columns:
            if hashable_df[col].apply(lambda x: isinstance(x, list)).any():
                hashable_df[col] = hashable_df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        # Check duplicated rows across the whole row (including category labels), matching assignment rule
        dup_occurrences = hashable_df.duplicated(keep="first")
        n_duplicates = int(dup_occurrences.sum())

        all_dup_mask = hashable_df.duplicated(keep=False)
        duplicate_indices = df.index[all_dup_mask].tolist()

        if drop:
            # keep=False drops EVERY copy of a duplicated row, matching Master's specification
            session.dataframe = df[~all_dup_mask].reset_index(drop=True)
            remaining_rows = len(session.dataframe)
        else:
            remaining_rows = len(df)

        result_id = session.next_result_id("duplicates")
        summary = {
            "result_id": result_id,
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "remaining_rows": remaining_rows,
            "dropped": drop,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: int,
        random_state: Optional[int] = None,
        replace_session: bool = False,
    ) -> Dict[str, Any]:
        """Subsamples rows from the working dataframe.

        Args:
            n: Number of rows to sample.
            random_state: Random seed for reproducible sampling. Defaults to None.
            replace_session: Whether to replace session.dataframe with the sample. Defaults to False.

        Returns:
            Dictionary containing result_id, n_sampled, sampled_indices, and replaced_session.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if n > len(df):
            return {"error": f"Requested sample size {n} exceeds dataframe size {len(df)}."}

        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        if replace_session:
            session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "n_sampled": len(sampled_indices),
            "sampled_indices": sampled_indices,
            "replace_session": replace_session,
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "random_state": random_state, "replace_session": replace_session},
            summary,
            full_report=sampled_df,
        )
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text into unigrams for each document in the working dataframe.

        Uses nltk.word_tokenize on lowercased text, setting session.dataframe['unigrams']
        and session.artifacts['tokens'].

        Returns:
            Dictionary containing result_id, n_documents, and sample_tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Working dataframe has no 'text' column."}

        def _tokenize(val: Any) -> List[str]:
            if pd.isna(val):
                return []
            s = str(val).strip()
            if not s:
                return []
            return nltk.word_tokenize(s.lower())

        tokens_series = df["text"].apply(_tokenize)
        session.dataframe["unigrams"] = tokens_series
        tokens_list = tokens_series.tolist()
        session.artifacts["tokens"] = tokens_list

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "sample_tokens": tokens_list[0] if tokens_list else [],
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Calculates text length statistics overall and per category, plotting a box plot.

        Computes text length (character count) for each document, calculates
        summary statistics, and creates a box plot attached to session.pending_figure.

        Returns:
            Dictionary containing overall and per-category text length statistics.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Working dataframe has no 'text' column."}

        # Calculate text_length as character count, handling NaNs as 0 length
        text_length = df["text"].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        df_copy = df.copy()
        df_copy["text_length"] = text_length

        # Overall describe stats
        overall_stats = text_length.describe().to_dict()

        # Per category stats
        per_category_stats = {}
        if "category_name" in df_copy.columns:
            for cat, group in df_copy.groupby("category_name"):
                per_category_stats[str(cat)] = group["text_length"].describe().to_dict()

        # Build boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in df_copy.columns:
            sns.boxplot(data=df_copy, x="category_name", y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df_copy, y="text_length", ax=ax)
        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Text Length Distribution by Category")
        fig.tight_layout()

        # Set session.pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id("describe")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "by_category": per_category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Peeks at the top rows of the working dataframe.

        Args:
            n: Number of rows to inspect. Defaults to 5.
            columns: Optional list of columns to include. If None, includes all columns.

        Returns:
            Dictionary containing result_id, total_rows, total_columns, column_names, and rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        target_cols = columns if columns is not None else list(df.columns)
        # Filter to only existing columns
        target_cols = [c for c in target_cols if c in df.columns]

        sub_df = df[target_cols].head(n)
        # Convert NaN values to None for clean JSON serialization
        records = sub_df.where(pd.notnull(sub_df), None).to_dict(orient="records")

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": target_cols,
            "inspected_rows": len(records),
            "rows": records,
        }

        session.store_result(
            "inspect_data_tool",
            {"n": n, "columns": columns},
            summary,
            full_report=sub_df,
        )
        return summary

    return [
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        describe_data_tool,
        inspect_data_tool,
    ]
