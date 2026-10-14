import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool_known_answer_16():
    # Setup single session state
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Build student tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    # 1. Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM matrix
    build_dtm_tool.invoke({})
    
    # 3. Test Known Answer 16 pairs
    # Pair 0 vs 1: ~0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["status"] == "success"
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Pair 0 vs 3: ~0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["status"] == "success"
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)
    
    # Pair 3 vs 4: ~0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["status"] == "success"
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    # Pair 0 vs 7 (empty text): exactly 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["status"] == "success"
    assert res_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-4)


def test_feature_correlation_matrix_tool_known_answer_19():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    exploration_tools = make_exploration_tools(session)
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    build_dtm_tool.invoke({})
    
    res = feature_correlation_matrix_tool.invoke({})
    assert res["status"] == "success"
    
    # Check top terms ordered by variance: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_terms"] == expected_terms
    
    # Check correlation matrix shape and specific known correlation values
    corr_matrix = np.array(res["correlation_matrix"])
    assert corr_matrix.shape == (5, 5)
    
    # Diagonal elements are 1.0
    for i in range(5):
        assert corr_matrix[i, i] == pytest.approx(1.0, abs=1e-3)
        
    # Check key pairs: alpha vs beta (indices 0 and 3) -> ~0.8885
    assert corr_matrix[0, 3] == pytest.approx(0.8885, abs=1e-3)
    
    # gamma vs delta (indices 1 and 2) -> ~0.8307
    assert corr_matrix[1, 2] == pytest.approx(0.8307, abs=1e-3)
    
    # Check pending figure is generated
    assert session.pending_figure is not None
