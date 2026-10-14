"""Frequent-pattern mining tool for agentic data mining pipeline."""

import os
import tempfile
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def make_tools(session):
    """Factory creating pattern mining tools bound to session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mine frequent word patterns for a specific category after vocabulary filtering.

        Args:
            category_name: The category label to mine.
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support count, required for 'fpgrowth' and 'maxfpgrowth'.
            k: Top-k pattern count, required for 'topk'.

        Returns:
            Dictionary summarizing mined patterns.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        # 1. Filter session.dataframe to category_name
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category: {category_name}"}

        # Build category-local CountVectorizer with english stopwords dropped
        vectorizer = CountVectorizer(stop_words="english")
        dtm_counts = vectorizer.fit_transform(cat_df["text"])
        feature_names = np.array(vectorizer.get_feature_names_out())

        if len(feature_names) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = []
            return summary

        # 2. Apply filtering_method
        dense_counts = dtm_counts.toarray()
        kept_indices: np.ndarray

        if filtering_method == "variance":
            # Variance across documents for each term
            variances = np.var(dense_counts, axis=0)
            low = np.percentile(variances, 5)
            high = np.percentile(variances, 95)
            # Inclusive percentile boundaries matching helpers/text_filtering_helpers.py
            kept_mask = (variances >= low) & (variances <= high)
            kept_indices = np.where(kept_mask)[0]

        elif filtering_method == "term_frequency":
            # Column sums
            frequencies = np.sum(dense_counts, axis=0)
            low = np.percentile(frequencies, 5)
            high = np.percentile(frequencies, 95)
            # Inclusive percentile boundaries matching helpers/text_filtering_helpers.py
            kept_mask = (frequencies >= low) & (frequencies <= high)
            kept_indices = np.where(kept_mask)[0]

        elif filtering_method == "tfidf":
            # TfidfTransformer mean score across docs; keep top 80% (>= 20th percentile)
            tfidf_mat = TfidfTransformer().fit_transform(dtm_counts).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            kept_mask = mean_tfidf >= p20
            kept_indices = np.where(kept_mask)[0]
        else:
            return {"error": f"Unknown filtering method: {filtering_method}"}

        # 3. Handle zero survived terms cleanly
        if len(kept_indices) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = []
            return summary

        filtered_feature_names = feature_names[kept_indices]
        filtered_counts = dense_counts[:, kept_indices]

        # 4. Convert surviving terms to transactional database via PAMI DF2DB converter
        # Items are terms present with count >= 1
        binary_df = pd.DataFrame(
            (filtered_counts >= 1).astype(int),
            columns=filtered_feature_names,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "transactions.txt")
            from PAMI.extras.convert import DF2DB

            converter = DF2DB.DF2DB(binary_df)
            converter.convert2TransactionalDatabase(
                oFile=db_path,
                condition=">=",
                thresholdValue=1,
            )

            # Validate input parameters for algorithm
            if algorithm in ("fpgrowth", "maxfpgrowth") and min_sup is None:
                return {"error": f"min_sup is required for {algorithm}"}
            if algorithm == "topk" and k is None:
                return {"error": "k is required for topk"}

            # Run PAMI mining algorithm
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup, sep="\t")
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(db_path, k=k, sep="\t")
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

            if hasattr(miner, "start"):
                miner.start()
            elif hasattr(miner, "mine"):
                miner.mine()
            elif hasattr(miner, "startMining"):
                miner.startMining()

            patterns_dict = miner.getPatterns()

        patterns_list = []
        if patterns_dict:
            for pattern, support in patterns_dict.items():
                items = pattern.split("\t") if isinstance(pattern, str) else list(pattern)
                patterns_list.append({"pattern": items, "support": int(support)})

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns_list),
            "patterns": patterns_list,
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
            full_report=pd.DataFrame(patterns_list) if patterns_list else None,
        )
        session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = patterns_list

        return summary

    return [mine_patterns_tool]
