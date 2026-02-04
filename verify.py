"""
Verification script for Self-Evolving Code Agent
Run this script to test all modules: python verify.py
"""
import sys
import os
import time
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from src.storage import SQLiteStorage
from src.vector_store import VectorStore
from src.embedding import EmbeddingEngine, DecisionVectorExtractor
from src.data_recorder import PromptRecorder, TrajectoryTracker, LatentSpaceRecorder
from src.evolution_engine import EvolutionEngine


def test_storage():
    print("Testing SQLiteStorage...")
    storage = SQLiteStorage(db_path=":memory:")
    
    session_id = str(uuid.uuid4())
    storage.create_session(session_id)
    
    prompt_id = storage.save_prompt("Test prompt", "test", session_id, {"test": True})
    assert prompt_id > 0, "Failed to save prompt"
    
    traj_id = storage.save_trajectory(
        session_id=session_id, step_order=1, action="test_action",
        action_input={"key": "value"}, observation="Test observation",
        reward=0.85, execution_time=0.5, token_usage=100, cost=0.0002
    )
    assert traj_id > 0, "Failed to save trajectory"
    
    storage.save_latent_space(traj_id, [0.1]*768, [0.2]*512, [0.3]*256)
    
    prompts = storage.get_prompts()
    assert len(prompts) >= 1, "Failed to retrieve prompts"
    
    trajectories = storage.get_trajectories(session_id)
    assert len(trajectories) >= 1, "Failed to retrieve trajectories"
    
    stats = storage.get_session_stats()
    assert "total_sessions" in stats, "Failed to get session stats"
    
    print("  ✓ Storage module passed")
    return True


def test_vector_store():
    print("Testing VectorStore...")
    vector_store = VectorStore()
    
    vector_store.upsert_prompt_embedding(1, [0.1]*1536, {"text": "test"})
    vector_store.upsert_trajectory_embedding(1, [0.2]*1536, {"action": "test"})
    vector_store.upsert_latent_embedding(1, [0.3]*768, {"type": "test"})
    
    results = vector_store.search_similar_prompts([0.1]*1536, limit=5)
    assert isinstance(results, list), "Search should return list"
    
    results = vector_store.search_similar_trajectories([0.2]*1536, limit=5)
    assert isinstance(results, list), "Search should return list"
    
    stats = vector_store.get_collection_stats("prompts")
    assert "vectors_count" in stats, "Stats should contain vectors_count"
    
    print("  ✓ VectorStore module passed")
    return True


def test_embedding():
    print("Testing EmbeddingEngine...")
    engine = EmbeddingEngine()
    
    embedding = engine.embed_text("test prompt")
    assert len(embedding) == 1536, f"Embedding should be 1536 dim, got {len(embedding)}"
    
    embeddings = engine.embed_batch(["prompt1", "prompt2"])
    assert len(embeddings) == 2, "Batch embedding should return 2 embeddings"
    assert len(embeddings[0]) == 1536, "Each embedding should be 1536 dim"
    
    trajectory_data = {"thought": "test", "action": "code_gen", "reward": 0.9}
    embedding = engine.embed_trajectory(trajectory_data)
    assert len(embedding) == 1536, "Trajectory embedding should be 1536 dim"
    
    print("  ✓ EmbeddingEngine passed")
    return True


def test_decision_vector():
    print("Testing DecisionVectorExtractor...")
    extractor = DecisionVectorExtractor()
    
    vec = extractor.extract_decision("test_action", 0.85, "reasoning")
    assert len(vec) == 512, f"Decision vector should be 512 dim, got {len(vec)}"
    
    agg = extractor.aggregate_decisions([vec, vec])
    assert len(agg) == 512, "Aggregated vector should be 512 dim"
    
    print("  ✓ DecisionVectorExtractor passed")
    return True


def test_data_recorder():
    print("Testing DataRecorder components...")
    storage = SQLiteStorage(db_path=":memory:")
    
    recorder = PromptRecorder(storage)
    prompt_id = recorder.record("test prompt", "test_type", "session1")
    assert prompt_id > 0, "Prompt recording failed"
    
    tracker = TrajectoryTracker(storage)
    session_id = tracker.start_session("session1", {"test": True})
    
    tracker.start_step(thought="Test step")
    traj_id = tracker.record_step(
        action="test", action_input={"key": "value"},
        observation="result", reward=0.9
    )
    assert traj_id > 0, "Trajectory recording failed"
    
    tracker.end_session()
    
    latent_recorder = LatentSpaceRecorder(storage)
    latent_id = latent_recorder.record(
        traj_id, [0.1]*768, [0.2]*512, [0.3]*256
    )
    assert latent_id > 0, "Latent space recording failed"
    
    print("  ✓ DataRecorder components passed")
    return True


def test_evolution_engine():
    print("Testing EvolutionEngine...")
    storage = SQLiteStorage(db_path=":memory:")
    vector_store = VectorStore()
    embedding_engine = EmbeddingEngine()
    
    engine = EvolutionEngine(storage, vector_store, embedding_engine)
    
    patterns = engine.analyze_trajectory_patterns()
    assert "success_rate" in patterns, "Patterns should contain success_rate"
    
    recommendations = engine.recommend_strategy("code_generation")
    assert "recommended_actions" in recommendations, "Should return recommendations"
    
    score = engine.calculate_evolution_score()
    assert "overall_score" in score, "Score should contain overall_score"
    
    result = engine.evolve("reward > 0.8", {"reward": 0.9, "timestamp": time.time()})
    assert "status" in result, "Evolution should return status"
    
    print("  ✓ EvolutionEngine passed")
    return True


def test_integration():
    print("Testing full integration...")
    storage = SQLiteStorage(db_path=":memory:")
    vector_store = VectorStore()
    embedding_engine = EmbeddingEngine()
    
    session_id = str(uuid.uuid4())
    storage.create_session(session_id)
    
    prompt_id = storage.save_prompt("Write a factorial function", "code_generation", session_id)
    embedding = embedding_engine.embed_text("Write a factorial function")
    vector_store.upsert_prompt_embedding(prompt_id, embedding, {"type": "code_generation"})
    
    traj_id = storage.save_trajectory(
        session_id=session_id, step_order=1, action="code_generation",
        action_input={"prompt": "factorial"}, observation="Generated code",
        reward=0.85, token_usage=500, cost=0.001
    )
    
    similar = vector_store.search_similar_prompts(embedding, limit=5)
    assert isinstance(similar, list), "Similar search should return list"
    
    patterns = storage.get_trajectories(session_id)
    assert len(patterns) >= 1, "Should retrieve saved trajectory"
    
    print("  ✓ Integration test passed")
    return True


def run_all_tests():
    print("\n" + "="*50)
    print("Self-Evolving Code Agent Verification")
    print("="*50 + "\n")
    
    tests = [
        ("Storage Module", test_storage),
        ("VectorStore Module", test_vector_store),
        ("Embedding Module", test_embedding),
        ("DecisionVector Module", test_decision_vector),
        ("DataRecorder Module", test_data_recorder),
        ("EvolutionEngine Module", test_evolution_engine),
        ("Integration Test", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*50 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
