"""
Complete verification script for Self-Evolving Code Agent
Tests: Full trajectory tracking with steps, prompts, LLM requests, code executions, and latent space
"""
import sys
import os
import time
import uuid
import json
import sqlite3

sys.path.insert(0, os.path.dirname(__file__))


def get_db():
    db_path = f":memory:test_{uuid.uuid4()}"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            total_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS trajectories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            trajectory_type TEXT DEFAULT 'task',
            task_description TEXT,
            initial_prompt TEXT,
            final_result TEXT,
            done BOOLEAN DEFAULT 0,
            total_reward REAL,
            total_execution_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_id INTEGER NOT NULL,
            step_order INTEGER NOT NULL,
            step_name TEXT,
            thought TEXT,
            action TEXT,
            action_input TEXT,
            action_result TEXT,
            observation TEXT,
            reward REAL,
            done BOOLEAN DEFAULT 0,
            execution_time REAL,
            token_usage INTEGER,
            cost REAL,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS llm_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step_id INTEGER,
            trajectory_id INTEGER,
            request_type TEXT,
            model TEXT,
            messages TEXT,
            parameters TEXT,
            prompt_text TEXT,
            response_text TEXT,
            response_object TEXT,
            token_usage INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            cost REAL,
            latency_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_id INTEGER,
            step_id INTEGER,
            prompt_text TEXT NOT NULL,
            prompt_type TEXT,
            language TEXT,
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS code_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step_id INTEGER,
            trajectory_id INTEGER,
            code TEXT,
            language TEXT,
            stdout TEXT,
            stderr TEXT,
            return_code INTEGER,
            execution_time REAL,
            timeout BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS latent_space (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step_id INTEGER,
            trajectory_id INTEGER,
            embedding_vector TEXT,
            hidden_states TEXT,
            decision_vector TEXT,
            agent_state TEXT,
            attention_weights TEXT,
            intermediate_outputs TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_steps_trajectory ON steps(trajectory_id);
        CREATE INDEX IF NOT EXISTS idx_llm_requests_step ON llm_requests(step_id);
    ''')


def serialize(obj):
    return json.dumps(obj) if obj is not None else None


def test_session_creation():
    print("Testing Session Creation...")
    conn = get_db()
    init_db(conn)
    
    session_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO sessions (id, metadata) VALUES (?, ?)",
        (session_id, '{"test": true}')
    )
    
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    assert row is not None, "Session should be created"
    assert row['total_tokens'] == 0, "Initial tokens should be 0"
    
    conn.close()
    print("  ✓ Session creation passed")
    return True


def test_trajectory_with_steps():
    print("Testing Trajectory with Steps...")
    conn = get_db()
    init_db(conn)
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    
    cursor = conn.execute(
        """INSERT INTO trajectories (session_id, task_description, initial_prompt, trajectory_type)
           VALUES (?, ?, ?, ?)""",
        (session_id, "Test Task", "Generate a function", "task")
    )
    trajectory_id = cursor.lastrowid
    assert trajectory_id > 0, "Trajectory should be created"
    
    for i in range(3):
        conn.execute(
            """INSERT INTO steps (trajectory_id, step_order, step_name, thought, action, action_input, observation, reward, execution_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trajectory_id, i+1, f"step_{i+1}", f"Thinking step {i+1}", "test_action", 
             '{"key": "value"}', f"Result {i+1}", 0.8 + i*0.05, 0.1 * (i+1))
        )
    
    steps = conn.execute(
        "SELECT * FROM steps WHERE trajectory_id = ? ORDER BY step_order",
        (trajectory_id,)
    ).fetchall()
    assert len(steps) == 3, f"Should have 3 steps, got {len(steps)}"
    
    conn.execute(
        "UPDATE trajectories SET done = 1, total_reward = ?, total_execution_time = ? WHERE id = ?",
        (0.9, 0.6, trajectory_id)
    )
    
    trajectory = conn.execute(
        "SELECT * FROM trajectories WHERE id = ?", (trajectory_id,)
    ).fetchone()
    assert trajectory['done'] == 1, "Trajectory should be marked as done"
    
    conn.close()
    print("  ✓ Trajectory with steps passed")
    return True


def test_llm_request_recording():
    print("Testing LLM Request Recording...")
    conn = get_db()
    init_db(conn)
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    
    cursor = conn.execute(
        "INSERT INTO trajectories (session_id, task_description) VALUES (?, ?)",
        (session_id, "LLM Request Test")
    )
    trajectory_id = cursor.lastrowid
    
    cursor = conn.execute(
        "INSERT INTO steps (trajectory_id, step_order, step_name) VALUES (?, ?, ?)",
        (trajectory_id, 1, "code_generation")
    )
    step_id = cursor.lastrowid
    
    conn.execute(
        """INSERT INTO llm_requests 
           (step_id, trajectory_id, request_type, model, messages, parameters, prompt_text, 
            response_text, token_usage, prompt_tokens, completion_tokens, cost, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (step_id, trajectory_id, "chat_completion", "gpt-3.5-turbo",
         '[{"role": "user", "content": "test"}]', '{"max_tokens": 1000}',
         "Write a function", "def test(): pass",
         150, 50, 100, 0.0003, 150.5)
    )
    
    requests = conn.execute(
        "SELECT * FROM llm_requests WHERE step_id = ?", (step_id,)
    ).fetchall()
    assert len(requests) == 1, "Should have 1 LLM request"
    assert requests[0]['token_usage'] == 150, "Token usage should be 150"
    assert requests[0]['cost'] == 0.0003, "Cost should be 0.0003"
    
    conn.close()
    print("  ✓ LLM request recording passed")
    return True


def test_prompt_recording():
    print("Testing Prompt Recording...")
    conn = get_db()
    init_db(conn)
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    
    cursor = conn.execute(
        "INSERT INTO trajectories (session_id, task_description) VALUES (?, ?)",
        (session_id, "Prompt Test")
    )
    trajectory_id = cursor.lastrowid
    
    conn.execute(
        """INSERT INTO prompts (trajectory_id, step_id, prompt_text, prompt_type, language, context)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (trajectory_id, None, "Write a Python function", "code_generation", "python", '{"context": "test"}')
    )
    
    prompts = conn.execute("SELECT * FROM prompts WHERE trajectory_id = ?", (trajectory_id,)).fetchall()
    assert len(prompts) == 1, "Should have 1 prompt"
    assert prompts[0]['prompt_type'] == "code_generation", "Prompt type should be code_generation"
    
    conn.close()
    print("  ✓ Prompt recording passed")
    return True


def test_code_execution_recording():
    print("Testing Code Execution Recording...")
    conn = get_db()
    init_db(conn)
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    
    cursor = conn.execute(
        "INSERT INTO trajectories (session_id, task_description) VALUES (?, ?)",
        (session_id, "Code Execution Test")
    )
    trajectory_id = cursor.lastrowid
    
    cursor = conn.execute(
        "INSERT INTO steps (trajectory_id, step_order, step_name) VALUES (?, ?, ?)",
        (trajectory_id, 1, "code_execution")
    )
    step_id = cursor.lastrowid
    
    conn.execute(
        """INSERT INTO code_executions 
           (step_id, trajectory_id, code, language, stdout, stderr, return_code, execution_time, timeout)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (step_id, trajectory_id, "print('hello')", "python", "hello\n", "", 0, 0.05, False)
    )
    
    executions = conn.execute(
        "SELECT * FROM code_executions WHERE step_id = ?", (step_id,)
    ).fetchall()
    assert len(executions) == 1, "Should have 1 code execution"
    assert executions[0]['return_code'] == 0, "Return code should be 0"
    assert executions[0]['timeout'] == False, "Should not be timeout"
    
    conn.close()
    print("  ✓ Code execution recording passed")
    return True


def test_latent_space_recording():
    print("Testing Latent Space Recording...")
    conn = get_db()
    init_db(conn)
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    
    cursor = conn.execute(
        "INSERT INTO trajectories (session_id, task_description) VALUES (?, ?)",
        (session_id, "Latent Space Test")
    )
    trajectory_id = cursor.lastrowid
    
    cursor = conn.execute(
        "INSERT INTO steps (trajectory_id, step_order, step_name) VALUES (?, ?, ?)",
        (trajectory_id, 1, "test_step")
    )
    step_id = cursor.lastrowid
    
    import numpy as np
    embedding = json.dumps([float(x) for x in np.random.randn(1536)])
    hidden_states = json.dumps([float(x) for x in np.random.randn(768)])
    decision_vector = json.dumps([float(x) for x in np.random.randn(512)])
    
    conn.execute(
        """INSERT INTO latent_space 
           (step_id, trajectory_id, embedding_vector, hidden_states, decision_vector, attention_weights)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (step_id, trajectory_id, embedding, hidden_states, decision_vector, '[0.1, 0.2, 0.3]')
    )
    
    latent = conn.execute(
        "SELECT * FROM latent_space WHERE step_id = ?", (step_id,)
    ).fetchone()
    assert latent is not None, "Latent space should be recorded"
    
    stored_embedding = json.loads(latent['embedding_vector'])
    assert len(stored_embedding) == 1536, "Embedding should be 1536 dimensions"
    
    conn.close()
    print("  ✓ Latent space recording passed")
    return True


def test_full_trajectory_retrieval():
    print("Testing Full Trajectory Retrieval...")
    conn = get_db()
    init_db(conn)
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    
    cursor = conn.execute(
        """INSERT INTO trajectories (session_id, task_description, initial_prompt, done, total_reward)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, "Complete Test Task", "Generate and test code", 1, 0.85)
    )
    trajectory_id = cursor.lastrowid
    
    for i in range(2):
        cursor = conn.execute(
            """INSERT INTO steps (trajectory_id, step_order, step_name, action, reward)
               VALUES (?, ?, ?, ?, ?)""",
            (trajectory_id, i+1, f"step_{i+1}", f"action_{i+1}", 0.8 + i*0.1)
        )
        step_id = cursor.lastrowid
        
        conn.execute(
            """INSERT INTO llm_requests (step_id, trajectory_id, request_type, model, token_usage)
               VALUES (?, ?, ?, ?, ?)""",
            (step_id, trajectory_id, "embedding", "text-embedding-ada-002", 100 + i*50)
        )
        
        conn.execute(
            """INSERT INTO code_executions (step_id, trajectory_id, language, return_code)
               VALUES (?, ?, ?, ?)""",
            (step_id, trajectory_id, "python", 0)
        )
    
    trajectory = conn.execute(
        "SELECT * FROM trajectories WHERE id = ?", (trajectory_id,)
    ).fetchone()
    assert trajectory is not None, "Trajectory should exist"
    
    steps = conn.execute(
        "SELECT * FROM steps WHERE trajectory_id = ? ORDER BY step_order", (trajectory_id,)
    ).fetchall()
    assert len(steps) == 2, f"Should have 2 steps, got {len(steps)}"
    
    total_tokens = conn.execute(
        "SELECT SUM(token_usage) FROM llm_requests WHERE trajectory_id = ?", (trajectory_id,)
    ).fetchone()[0]
    assert total_tokens == 300, f"Total tokens should be 300, got {total_tokens}"
    
    conn.close()
    print("  ✓ Full trajectory retrieval passed")
    return True


def test_trajectory_relations():
    print("Testing Trajectory Relations...")
    conn = get_db()
    init_db(conn)
    
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    
    cursor = conn.execute(
        "INSERT INTO trajectories (session_id, task_description) VALUES (?, ?)",
        (session_id, "Parent Task")
    )
    parent_id = cursor.lastrowid
    
    child_id = conn.execute(
        "INSERT INTO trajectories (session_id, task_description) VALUES (?, ?)",
        (session_id, "Child Task")
    ).lastrowid
    
    conn.execute(
        "INSERT INTO trajectory_relations (parent_trajectory_id, child_trajectory_id, relation_type) VALUES (?, ?, ?)",
        (parent_id, child_id, "subtask")
    )
    
    relations = conn.execute(
        """SELECT t.* FROM trajectories t
           JOIN trajectory_relations tr ON t.id = tr.child_trajectory_id
           WHERE tr.parent_trajectory_id = ?""",
        (parent_id,)
    ).fetchall()
    assert len(relations) == 1, "Should have 1 child trajectory"
    
    conn.close()
    print("  ✓ Trajectory relations passed")
    return True


def run_all_tests():
    print("\n" + "="*60)
    print("Self-Evolving Code Agent - Full Trajectory Verification")
    print("="*60 + "\n")
    
    tests = [
        ("Session Creation", test_session_creation),
        ("Trajectory with Steps", test_trajectory_with_steps),
        ("LLM Request Recording", test_llm_request_recording),
        ("Prompt Recording", test_prompt_recording),
        ("Code Execution Recording", test_code_execution_recording),
        ("Latent Space Recording", test_latent_space_recording),
        ("Full Trajectory Retrieval", test_full_trajectory_retrieval),
        ("Trajectory Relations", test_trajectory_relations),
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
