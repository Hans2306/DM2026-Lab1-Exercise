"""Tests for tools_dtm.py using agent_dev/sample_fixture.csv and known answers 4, 14, & 21."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """Test build_dtm_tool against Known Answer 4 from TEST_FIXTURE.md."""
    session = SessionState()

    # Load load_dataset_tool from premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    assert load_res["n_documents"] == 8

    # Load build_dtm_tool from student's make_tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Execute build_dtm_tool with defaults
    result = build_dtm_tool.invoke({})

    # Known Answer 4 assertions
    assert result["status"] == "success"
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Check session state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]


def test_term_frequency_tool_known_answer_14():
    """Test term_frequency_tool against Known Answer 14 from TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build DTM first
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})

    # Execute term_frequency_tool
    tf_res = term_frequency_tool.invoke({})

    assert tf_res["status"] == "success"
    assert tf_res["n_terms"] == 5
    assert tf_res["total_term_count"] == 27

    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    assert tf_res["term_frequencies"] == expected_freqs


def test_dtm_heatmap_tool_known_answer_21():
    """Test dtm_heatmap_tool against Known Answer 21 from TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build DTM first
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Execute default heatmap (n_terms=20, n_documents=20)
    heatmap_res = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert heatmap_res["status"] == "success"
    assert heatmap_res["n_documents"] == 8
    assert heatmap_res["n_terms"] == 5
    assert heatmap_res["term_labels"] == ["alpha", "always", "beta", "delta", "gamma"]

    # Expected 8x5 matrix from Known Answer 21
    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    assert heatmap_res["matrix_slice"] == expected_matrix

    # Check pending figure created
    assert session.pending_figure is not None

    # Test custom slice (n_terms=2, n_documents=3)
    slice_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert slice_res["n_documents"] == 3
    assert slice_res["n_terms"] == 2
    assert slice_res["term_labels"] == ["alpha", "always"]
    assert slice_res["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
