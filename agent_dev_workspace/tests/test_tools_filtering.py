"""Tests for tools_filtering.py using agent_dev/sample_fixture.csv and known answers 5 & 6."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against Known Answer 5 from TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build Document-Term Matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["status"] == "success"
    assert res["total_terms"] == 5
    assert res["kept_count"] == 4
    assert res["removed_count"] == 1

    # Known Answer 5: threshold=0.15 keeps alpha, beta, delta, gamma and removes always
    assert sorted(res["kept_terms"]) == ["alpha", "beta", "delta", "gamma"]
    assert res["removed_terms"] == ["always"]

    # Verify full_report stored for visualize_result_tool
    stored_result = session.results_store.get(res["result_id"])
    assert stored_result is not None
    full_report = stored_result.get("full_report")
    assert full_report is not None
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "variance"

    # Check specific variance values from Known Answer 5 table
    var_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert var_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert var_dict["always"] == pytest.approx(0.109375, abs=1e-3)
    assert var_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert var_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert var_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)


def test_pearson_filter_tool_known_answer_6():
    """Test pearson_filter_tool against Known Answer 6 from TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build Document-Term Matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load filtering tools
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run pearson_filter_tool with target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB"})

    assert res["status"] == "success"
    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5

    corrs = res["pearson_correlations"]
    assert corrs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # Verify full_report stored for visualize_result_tool
    stored_result = session.results_store.get(res["result_id"])
    assert stored_result is not None
    full_report = stored_result.get("full_report")
    assert full_report is not None
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "pearson_r"


def test_spearman_filter_tool_known_answer_6():
    """Test spearman_filter_tool against Known Answer 6 from TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build Document-Term Matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load filtering tools
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run spearman_filter_tool with target_class="catB"
    res = spearman_filter_tool.invoke({"target_class": "catB"})

    assert res["status"] == "success"
    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5

    corrs = res["spearman_correlations"]
    assert corrs["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Verify full_report stored for visualize_result_tool
    stored_result = session.results_store.get(res["result_id"])
    assert stored_result is not None
    full_report = stored_result.get("full_report")
    assert full_report is not None
    assert full_report.columns[0] == "term"
    assert full_report.columns[1] == "spearman_r"
