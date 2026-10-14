"""Unit tests for mine_patterns_tool in tools_patterns.py against TEST_FIXTURE.md (#17 & #20)."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _setup_session():
    """Helper to set up a clean session loaded with sample_fixture.csv."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_mine_patterns_variance_and_tf_filters():
    """Known answer #17: variance and term_frequency keep zero terms for both categories."""
    session, mine_patterns_tool = _setup_session()

    for cat in ["catA", "catB"]:
        for method in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1,
            })
            assert res.get("n_patterns") == 0
            assert res.get("patterns") == []
            assert res.get("note") == "no terms survived filtering"


def test_mine_patterns_tfidf_fpgrowth():
    """Known answer #17: tfidf filtering with fpgrowth for catA and catB."""
    session, mine_patterns_tool = _setup_session()

    # catA at min_sup=3: keeps alpha, pattern {alpha} with support 3
    res_a3 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3,
    })
    assert res_a3["n_patterns"] == 1
    assert res_a3["patterns"][0]["pattern"] == ["alpha"]
    assert res_a3["patterns"][0]["support"] == 3

    # catA at min_sup=4: 0 patterns
    res_a4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_a4["n_patterns"] == 0
    assert res_a4["patterns"] == []

    # catB at min_sup=4: keeps gamma, pattern {gamma} with support 4
    res_b4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_b4["n_patterns"] == 1
    assert res_b4["patterns"][0]["pattern"] == ["gamma"]
    assert res_b4["patterns"][0]["support"] == 4


def test_mine_patterns_topk_and_maxfpgrowth():
    """Known answer #20: topk and maxfpgrowth behavior on tfidf-filtered data."""
    session, mine_patterns_tool = _setup_session()

    # Top-K (k=1): catA -> {alpha} (3), catB -> {gamma} (4)
    topk_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })
    assert topk_a["n_patterns"] == 1
    assert topk_a["patterns"][0]["pattern"] == ["alpha"]
    assert topk_a["patterns"][0]["support"] == 3

    topk_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })
    assert topk_b["n_patterns"] == 1
    assert topk_b["patterns"][0]["pattern"] == ["gamma"]
    assert topk_b["patterns"][0]["support"] == 4

    # MaxFPGrowth: returns 0 patterns for both categories even at min_sup=1
    for cat in ["catA", "catB"]:
        max_res = mine_patterns_tool.invoke({
            "category_name": cat,
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": 1,
        })
        assert max_res["n_patterns"] == 0
        assert max_res["patterns"] == []
