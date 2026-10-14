"""Unit tests for tools_data.py against TEST_FIXTURE.md."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_list_files_tool():
    """Known answer #11 in TEST_FIXTURE.md: subdirectory 'newdataset'."""
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res.get("directories") == []
    assert res.get("files") == ["Reddit-stock-sentiment.csv"]


def test_check_missing_tool():
    """Known answer #2 in TEST_FIXTURE.md: exactly 1 row has missing text (row 7)."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})
    assert res.get("n_missing") == 1
    assert 7 in res.get("missing_indices", [])


def test_check_duplicates_tool():
    """Known answer #3 in TEST_FIXTURE.md: exactly 1 duplicate; drop=True leaves 6 rows."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Check without dropping
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check.get("n_duplicates") == 1
    assert 5 in res_check.get("duplicate_indices", [])
    assert 6 in res_check.get("duplicate_indices", [])
    assert res_check.get("remaining_rows") == 8
    assert len(session.dataframe) == 8

    # Drop with keep=False: rows 5 and 6 are both dropped, leaving exactly 6 rows
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop.get("remaining_rows") == 6
    assert len(session.dataframe) == 6


def test_check_duplicates_tool_after_tokenization():
    """Verifies check_duplicates_tool works when session.dataframe contains list columns (unigrams)."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Add unigrams column containing lists
    tokenize_tool.invoke({})
    assert "unigrams" in session.dataframe.columns
    assert isinstance(session.dataframe["unigrams"].iloc[0], list)

    # Check duplicate detection on dataframe with list column
    res_check = check_duplicates_tool.invoke({"drop": False})
    assert res_check.get("n_duplicates") == 1
    assert 5 in res_check.get("duplicate_indices", [])
    assert 6 in res_check.get("duplicate_indices", [])
    assert res_check.get("remaining_rows") == 8
    assert len(session.dataframe) == 8

    # Drop duplicates with keep=False
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop.get("remaining_rows") == 6
    assert len(session.dataframe) == 6
    # Verify unigrams column is retained as lists
    assert isinstance(session.dataframe["unigrams"].iloc[0], list)


def test_sample_data_tool():
    """Known answer #13 in TEST_FIXTURE.md: n=4, random_state=42 gives rows [1, 5, 0, 7]."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res.get("n_sampled") == 4
    assert res.get("sampled_indices") == [1, 5, 0, 7]


def test_tokenize_tool():
    """Known answer #10 in TEST_FIXTURE.md: lowercased nltk.word_tokenize per document."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})
    assert res.get("n_documents") == 8
    assert "unigrams" in session.dataframe.columns
    assert "tokens" in session.artifacts

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]
    assert session.artifacts["tokens"] == expected_tokens


def test_describe_data_tool():
    """Known answer #18 in TEST_FIXTURE.md: text_length describe statistics and pending_figure."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})
    assert "overall" in res
    assert "by_category" in res
    assert session.pending_figure is not None

    overall = res["overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    cat_a = res["by_category"]["catA"]
    assert cat_a["count"] == 4
    assert cat_a["mean"] == pytest.approx(17.25, abs=1e-3)
    assert cat_a["std"] == pytest.approx(12.5, abs=1e-3)
    assert cat_a["min"] == 0
    assert cat_a["25%"] == pytest.approx(12.75, abs=1e-3)
    assert cat_a["50%"] == pytest.approx(20.0, abs=1e-3)
    assert cat_a["75%"] == pytest.approx(24.5, abs=1e-3)
    assert cat_a["max"] == 29

    cat_b = res["by_category"]["catB"]
    assert cat_b["count"] == 4
    assert cat_b["mean"] == pytest.approx(22.5, abs=1e-3)
    assert cat_b["std"] == pytest.approx(5.7446, abs=1e-3)
    assert cat_b["min"] == 18
    assert cat_b["25%"] == pytest.approx(18.0, abs=1e-3)
    assert cat_b["50%"] == pytest.approx(21.0, abs=1e-3)
    assert cat_b["75%"] == pytest.approx(25.5, abs=1e-3)
    assert cat_b["max"] == 30


def test_inspect_data_tool():
    """Unit test for inspect_data_tool verifying row inspection and formatting."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n": 3, "columns": ["text", "category_name"]})
    assert res.get("total_rows") == 8
    assert res.get("inspected_rows") == 3
    assert len(res.get("rows", [])) == 3
    first_row = res["rows"][0]
    assert "text" in first_row
    assert "category_name" in first_row
    assert first_row["category_name"] == "catA"
