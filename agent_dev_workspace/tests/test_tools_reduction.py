import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca_known_answer_7():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    build_dtm_tool.invoke({})
    
    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})
    assert res["status"] == "success"
    assert res["method"] == "pca"
    
    coords = np.array(res["coordinates"])
    expected_pca = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    
    assert np.allclose(coords, expected_pca, atol=1e-3)
    assert res["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne_known_answer_8():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    build_dtm_tool.invoke({})
    
    res = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    assert res["status"] == "success"
    assert res["method"] == "tsne"
    
    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)
    
    expected_tsne = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973],
    ])
    
    assert np.allclose(coords, expected_tsne, atol=1e-1)
    assert session.pending_figure is not None


def test_reduce_dimensions_umap_known_answer_9():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    build_dtm_tool.invoke({})
    
    res = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    assert res["status"] == "success"
    assert res["method"] == "umap"
    
    coords = np.array(res["coordinates"])
    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])
    
    assert np.allclose(coords, expected_umap, atol=1e-1)
    assert session.pending_figure is not None


def test_reduce_dimensions_invalid_components():
    session = SessionState()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    
    res = reduce_dimensions_tool.invoke({"n_components": 1})
    assert res["status"] == "error"
    assert "at least 2" in res["message"]


def test_binarize_labels_tool_binary():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    res = binarize_labels_tool.invoke({})
    assert res["status"] == "success"
    assert res["classes"] == ["catA", "catB"]
    assert res["n_samples"] == 8
    
    bin_matrix = np.array(res["binarized_matrix"])
    assert bin_matrix.shape == (8, 2)
    # Row 0 is catA ([1, 0]), Row 3 is catB ([0, 1])
    assert list(bin_matrix[0]) == [1, 0]
    assert list(bin_matrix[3]) == [0, 1]


def test_binarize_labels_tool_single_category():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["hello", "world"],
        "category_name": ["catA", "catA"]
    })
    
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    res = binarize_labels_tool.invoke({})
    assert res["status"] == "success"
    assert res["classes"] == ["catA"]
    assert res["n_samples"] == 2
    
    bin_matrix = np.array(res["binarized_matrix"])
    assert bin_matrix.shape == (2, 1)
    assert np.array_equal(bin_matrix, np.array([[1], [1]]))
