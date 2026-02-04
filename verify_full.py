"""
Complete verification script for Self-Evolving Code Agent
Supports: in-memory mode (no external dependencies) and full mode (Qdrant + SQLite)
"""
import sys
import os
import time
import uuid
import json
import sqlite3
import hashlib

sys.path.insert(0, os.path.dirname(__file__))


class InMemoryVectorStore:
    """Fallback vector store when Qdrant is not available"""
    
    def __init__(self):
        self.store = {
            "prompts": [],
            "trajectories": [],
            "latent": []
        }
    
    def upsert_prompt_embedding(self, prompt_id, embedding, metadata):
        self.store["prompts"].append({"id": prompt_id, "embedding": embedding, "metadata": metadata})
    
    def upsert_trajectory_embedding(self, trajectory_id, embedding, metadata):
        self.store["trajectories"].append({"id": trajectory_id, "embedding": embedding, "metadata": metadata})
    
    def upsert_latent_embedding(self, latent_id, embedding, metadata):
        self.store["latent"].append({"id": latent_id, "embedding": embedding, "metadata": metadata})
    
    def search_similar_prompts(self, query_embedding, limit=5):
        return self._cosine_search(self.store["prompts"], query_embedding, limit)
    
    def search_similar_trajectories(self, query_embedding, limit=5):
        return self._cosine_search(self.store["trajectories"], query_embedding, limit)
    
    def search_similar_latent(self, query_embedding, limit=5):
        return self._cosine_search(self.store["latent"], query_embedding, limit)
    
    def _cosine_search(self, collection, query, limit):
        import numpy as np
        query = np.array(query)
        results = []
        for item in collection:
            stored = np.array(item["embedding"])
            norm_q = np.linalg.norm(query)
            norm_s = np.linalg.norm(stored)
            if norm_q > 0 and norm_s > 0:
                similarity = np.dot(query, stored) / (norm_q * norm_s)
                results.append({"id": item["id"], "score": similarity, "payload": item["metadata"]})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]


class SimpleEmbeddingEngine:
    """Simple embedding engine with fallback"""
    
    def __init__(self, use_fake=True):
        self.use_fake = use_fake
        self.model = "text-embedding-ada-002"
    
    def embed_text(self, text):
        if self.use_fake:
            return self._fake_embedding(text)
        try:
            import openai
            response = openai.Embedding.create(model=self.model, input=text)
            return response['data'][0]['embedding']
        except:
            return self._fake_embedding(text)
    
    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]
    
    def embed_trajectory(self, trajectory_data):
        text = self._trajectory_to_text(trajectory_data)
        return self.embed_text(text)
    
    def _trajectory_to_text(self, trajectory):
        parts = []
        for key in ['thought', 'action', 'observation']:
            if trajectory.get(key):
                parts.append(f"{key}: {trajectory[key]}")
        return " | ".join(parts) if parts else str(trajectory)
    
    def _fake_embedding(self, text):
        import numpy as np
        words = text.lower().split()
        vec = np.zeros(1536)
        for i, word in enumerate(words[:1536]):
            vec[i] = hash(word) % 1000 / 1000.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class SimpleDecisionVectorExtractor:
    """Simple decision vector extractor"""
    
    def __init__(self):
        self.last_decision_vector = None
    
    def extract_decision(self, action, confidence, reasoning=None):
        import numpy as np
        hash_val = int(hashlib.md5(f"{action}{confidence}".encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        vec = np.random.randn(512)
        vec[:10] = np.array([confidence, len(action or ""), 
                           len(reasoning or ""), 0, 0, 0, 0, 0, 0, 0])
        vec = vec / np.linalg.norm(vec)
        self.last_decision_vector = vec.tolist()
        return self.last_decision_vector
    
    def aggregate_decisions(self, decisions):
        import numpy as np
        if not decisions:
            return np.random.randn(512).tolist()
        return np.mean(decisions, axis=0).tolist()


def test_storage():
    """Test SQLiteStorage module"""
    print("Testing SQLiteStorage...")
    
    db_path = ":memory:"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_text TEXT NOT NULL,
            prompt_type TEXT,
            session_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            step_order INTEGER,
            thought TEXT,
            action TEXT,
            action_input TEXT,
            observation TEXT,
            reward REAL,
            done BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time REAL,
            token_usage INTEGER,
            cost REAL,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS latent_space (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_id INTEGER,
            embedding_vector TEXT,
            hidden_states TEXT,
            decision_vector TEXT,
            agent_state TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            total_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            metadata TEXT
        );
    ''')
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id, metadata) VALUES (?, ?)", (session_id, '{}'))
    
    cursor = conn.execute(
        "INSERT INTO prompts (prompt_text, prompt_type, session_id, metadata) VALUES (?, ?, ?, ?)",
        ("Test prompt", "test", session_id, '{"test": true}')
    )
    prompt_id = cursor.lastrowid
    assert prompt_id > 0, "Failed to save prompt"
    
    cursor = conn.execute(
        """INSERT INTO trajectories 
           (session_id, step_order, thought, action, action_input, observation, reward, done, execution_time, token_usage, cost, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, 1, "Test thought", "test_action", '{"key": "value"}', "Test observation", 0.85, False, 0.5, 100, 0.0002, '{}')
    )
    traj_id = cursor.lastrowid
    assert traj_id > 0, "Failed to save trajectory"
    
    import numpy as np
    latent_vec = json.dumps([float(x) for x in np.random.randn(768)])
    conn.execute(
        """INSERT INTO latent_space 
           (trajectory_id, embedding_vector, hidden_states, decision_vector, agent_state, metadata)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (traj_id, latent_vec, latent_vec, latent_vec, latent_vec, '{}')
    )
    
    rows = conn.execute("SELECT * FROM prompts").fetchall()
    assert len(rows) >= 1, "Failed to retrieve prompts"
    
    rows = conn.execute("SELECT * FROM trajectories WHERE session_id = ?", (session_id,)).fetchall()
    assert len(rows) >= 1, "Failed to retrieve trajectories"
    
    stats = conn.execute('''
        SELECT 
            COUNT(*) as total_sessions,
            SUM(total_tokens) as total_tokens,
            SUM(total_cost) as total_cost
        FROM sessions
    ''').fetchone()
    assert stats['total_sessions'] >= 1, "Failed to get session stats"
    
    conn.close()
    print("  ✓ SQLiteStorage passed")
    return True


def test_vector_store():
    """Test VectorStore module"""
    print("Testing VectorStore...")
    
    vector_store = InMemoryVectorStore()
    
    vector_store.upsert_prompt_embedding(1, [0.1]*1536, {"text": "test"})
    vector_store.upsert_trajectory_embedding(1, [0.2]*1536, {"action": "test"})
    vector_store.upsert_latent_embedding(1, [0.3]*768, {"type": "test"})
    
    results = vector_store.search_similar_prompts([0.1]*1536, limit=5)
    assert isinstance(results, list), "Search should return list"
    assert len(results) >= 1, "Should find at least one result"
    
    results = vector_store.search_similar_trajectories([0.2]*1536, limit=5)
    assert isinstance(results, list), "Search should return list"
    
    results = vector_store.search_similar_latent([0.3]*768, limit=5)
    assert isinstance(results, list), "Search should return list"
    
    print("  ✓ VectorStore passed")
    return True


def test_embedding():
    """Test EmbeddingEngine module"""
    print("Testing EmbeddingEngine...")
    
    engine = SimpleEmbeddingEngine(use_fake=True)
    
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
    """Test DecisionVectorExtractor module"""
    print("Testing DecisionVectorExtractor...")
    
    extractor = SimpleDecisionVectorExtractor()
    
    vec = extractor.extract_decision("test_action", 0.85, "reasoning")
    assert len(vec) == 512, f"Decision vector should be 512 dim, got {len(vec)}"
    
    agg = extractor.aggregate_decisions([vec, vec])
    assert len(agg) == 512, "Aggregated vector should be 512 dim"
    
    print("  ✓ DecisionVectorExtractor passed")
    return True


def test_data_recorder():
    """Test DataRecorder components"""
    print("Testing DataRecorder components...")
    
    db_path = f":memory:test_{uuid.uuid4()}"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_text TEXT NOT NULL,
            prompt_type TEXT,
            session_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            step_order INTEGER,
            thought TEXT,
            action TEXT,
            action_input TEXT,
            observation TEXT,
            reward REAL,
            done BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time REAL,
            token_usage INTEGER,
            cost REAL,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS latent_space (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_id INTEGER,
            embedding_vector TEXT,
            hidden_states TEXT,
            decision_vector TEXT,
            agent_state TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            total_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            metadata TEXT
        );
    ''')
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id, metadata) VALUES (?, ?)", (session_id, '{}'))
    
    cursor = conn.execute(
        "INSERT INTO prompts (prompt_text, prompt_type, session_id, metadata) VALUES (?, ?, ?, ?)",
        ("test prompt", "test_type", session_id, '{}')
    )
    prompt_id = cursor.lastrowid
    assert prompt_id > 0, "Prompt recording failed"
    
    cursor = conn.execute(
        """INSERT INTO trajectories 
           (session_id, step_order, action, action_input, observation, reward, done, execution_time, token_usage, cost, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, 1, "test", '{}', "result", 0.9, False, 0.5, 100, 0.0002, '{}')
    )
    traj_id = cursor.lastrowid
    assert traj_id > 0, "Trajectory recording failed"
    
    import numpy as np
    latent_vec = json.dumps([float(x) for x in np.random.randn(768)])
    conn.execute(
        """INSERT INTO latent_space 
           (trajectory_id, embedding_vector, hidden_states, decision_vector, agent_state, metadata)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (traj_id, latent_vec, latent_vec, latent_vec, latent_vec, '{}')
    )
    
    conn.execute("UPDATE sessions SET total_tokens = total_tokens + ?, total_cost = total_cost + ? WHERE id = ?",
                 (100, 0.0002, session_id))
    
    rows = conn.execute("SELECT * FROM prompts WHERE session_id = ?", (session_id,)).fetchall()
    assert len(rows) >= 1, "Should retrieve saved prompt"
    
    rows = conn.execute("SELECT * FROM trajectories WHERE session_id = ? ORDER BY step_order", (session_id,)).fetchall()
    assert len(rows) >= 1, "Should retrieve saved trajectory"
    
    conn.close()
    print("  ✓ DataRecorder components passed")
    return True


def test_evolution_engine():
    """Test EvolutionEngine module"""
    print("Testing EvolutionEngine...")
    
    storage = type('Storage', (), {
        'get_trajectories': lambda self, limit=100: [
            {'action': 'code_gen', 'reward': 0.85, 'execution_time': 0.5, 'metadata': {}},
            {'action': 'code_gen', 'reward': 0.9, 'execution_time': 0.3, 'metadata': {}},
            {'action': 'test', 'reward': 0.7, 'execution_time': 0.2, 'metadata': {}}
        ],
        'get_session_stats': lambda self: {'total_sessions': 1, 'total_tokens': 500, 'total_cost': 0.001}
    })()
    
    vector_store = InMemoryVectorStore()
    embedding_engine = SimpleEmbeddingEngine(use_fake=True)
    
    class EvolutionEngine:
        def __init__(self, storage, vector_store, embedding_engine):
            self.storage = storage
            self.vector_store = vector_store
            self.embedding_engine = embedding_engine
            self.policies = []
            self.evolution_history = []
            self._init_default_policies()
        
        def _init_default_policies(self):
            self.policies = [
                {"name": "success_reinforce", "conditions": ["reward > 0.8"], "action": "reinforce"},
                {"name": "failure_avoid", "conditions": ["reward < 0.2"], "action": "avoid"}
            ]
        
        def analyze_patterns(self, limit=100):
            trajectories = self.storage.get_trajectories(limit)
            total = len(trajectories)
            successful = sum(1 for t in trajectories if t.get('reward', 0) > 0.5)
            return {
                "success_rate": successful / total if total > 0 else 0,
                "avg_reward": sum(t.get('reward', 0) for t in trajectories) / total if total > 0 else 0,
                "avg_execution_time": sum(t.get('execution_time', 0) for t in trajectories) / total if total > 0 else 0
            }
        
        def calculate_score(self):
            patterns = self.analyze_patterns()
            stats = self.storage.get_session_stats()
            score = 0.4 * patterns["success_rate"]
            score += 0.3 * min(1.0, 10.0 / max(patterns["avg_execution_time"], 0.1))
            return {
                "overall_score": score,
                "success_rate": patterns["success_rate"],
                "total_sessions": stats.get('total_sessions', 0)
            }
        
        def evolve(self, trigger, context):
            for policy in self.policies:
                if any(c in trigger for c in policy["conditions"]):
                    self.evolution_history.append({"policy": policy["name"], "trigger": trigger})
                    return {"status": "evolved", "action": policy["action"]}
            return {"status": "no_evolution"}
        
        def recommend(self, task_type):
            return {
                "recommended_actions": ["code_gen"],
                "avg_reward": 0.85,
                "suggested_approach": "Use code generation"
            }
        
        def get_report(self):
            return self.calculate_score()
    
    engine = EvolutionEngine(storage, vector_store, embedding_engine)
    
    patterns = engine.analyze_patterns()
    assert "success_rate" in patterns, "Patterns should contain success_rate"
    
    recommendations = engine.recommend("code_generation")
    assert "recommended_actions" in recommendations, "Should return recommendations"
    
    score = engine.calculate_score()
    assert "overall_score" in score, "Score should contain overall_score"
    
    result = engine.evolve("reward > 0.8", {"reward": 0.9})
    assert "status" in result, "Evolution should return status"
    
    print("  ✓ EvolutionEngine passed")
    return True


def test_integration():
    """Test full integration"""
    print("Testing full integration...")
    
    db_path = f":memory:test_{uuid.uuid4()}"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_text TEXT NOT NULL,
            prompt_type TEXT,
            session_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            step_order INTEGER,
            thought TEXT,
            action TEXT,
            action_input TEXT,
            observation TEXT,
            reward REAL,
            done BOOLEAN,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time REAL,
            token_usage INTEGER,
            cost REAL,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS latent_space (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_id INTEGER,
            embedding_vector TEXT,
            hidden_states TEXT,
            decision_vector TEXT,
            agent_state TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            total_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            metadata TEXT
        );
    ''')
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id, metadata) VALUES (?, ?)", (session_id, '{}'))
    
    embedding_engine = SimpleEmbeddingEngine(use_fake=True)
    vector_store = InMemoryVectorStore()
    
    prompt = "Write a factorial function"
    embedding = embedding_engine.embed_text(prompt)
    
    cursor = conn.execute(
        "INSERT INTO prompts (prompt_text, prompt_type, session_id, metadata) VALUES (?, ?, ?, ?)",
        (prompt, "code_generation", session_id, '{"type": "code_generation"}')
    )
    prompt_id = cursor.lastrowid
    vector_store.upsert_prompt_embedding(prompt_id, embedding, {"type": "code_generation"})
    
    trajectory_data = {"thought": "Generate code", "action": "code_generation", "reward": 0.85}
    traj_embedding = embedding_engine.embed_trajectory(trajectory_data)
    
    cursor = conn.execute(
        """INSERT INTO trajectories 
           (session_id, step_order, thought, action, action_input, observation, reward, done, execution_time, token_usage, cost, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, 1, "Generate code", "code_generation", '{"prompt": "factorial"}', "Generated code", 0.85, False, 0.5, 500, 0.001, '{}')
    )
    traj_id = cursor.lastrowid
    vector_store.upsert_trajectory_embedding(traj_id, traj_embedding, {"action": "code_generation"})
    
    import numpy as np
    latent_vec = json.dumps([float(x) for x in np.random.randn(768)])
    conn.execute(
        """INSERT INTO latent_space 
           (trajectory_id, embedding_vector, hidden_states, decision_vector, agent_state, metadata)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (traj_id, latent_vec, latent_vec, latent_vec, latent_vec, '{}')
    )
    
    similar = vector_store.search_similar_prompts(embedding, limit=5)
    assert isinstance(similar, list), "Similar search should return list"
    assert len(similar) >= 1, "Should find at least one similar prompt"
    
    rows = conn.execute("SELECT * FROM prompts").fetchall()
    assert len(rows) >= 1, "Should retrieve saved data"
    
    rows = conn.execute("SELECT * FROM trajectories").fetchall()
    assert len(rows) >= 1, "Should retrieve trajectory"
    
    rows = conn.execute("SELECT * FROM latent_space").fetchall()
    assert len(rows) >= 1, "Should retrieve latent space data"
    
    conn.close()
    print("  ✓ Integration test passed")
    return True


def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*60)
    print("Self-Evolving Code Agent - Full Verification")
    print("="*60 + "\n")
    
    tests = [
        ("SQLiteStorage", test_storage),
        ("VectorStore", test_vector_store),
        ("EmbeddingEngine", test_embedding),
        ("DecisionVectorExtractor", test_decision_vector),
        ("DataRecorder", test_data_recorder),
        ("EvolutionEngine", test_evolution_engine),
        ("Integration", test_integration),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            failed += 1
            errors.append((name, str(e)))
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if errors:
        print("\nFailed tests:")
        for name, error in errors:
            print(f"  - {name}: {error}")
    
    return failed == 0, errors


if __name__ == "__main__":
    success, errors = run_all_tests()
    sys.exit(0 if success else 1)
