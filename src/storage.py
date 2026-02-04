import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import contextmanager


class SQLiteStorage:
    def __init__(self, db_path: str = "code_agent.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_text TEXT NOT NULL,
                    prompt_type TEXT,
                    session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON
                );

                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step_order INTEGER,
                    thought TEXT,
                    action TEXT,
                    action_input JSON,
                    observation TEXT,
                    reward REAL,
                    done BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    execution_time REAL,
                    token_usage INTEGER,
                    cost REAL,
                    metadata JSON
                );

                CREATE TABLE IF NOT EXISTS latent_space (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trajectory_id INTEGER,
                    embedding_vector BLOB,
                    hidden_states BLOB,
                    decision_vector BLOB,
                    agent_state BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    metadata JSON
                );

                CREATE INDEX IF NOT EXISTS idx_trajectories_session ON trajectories(session_id);
                CREATE INDEX IF NOT EXISTS idx_latent_trajectory ON latent_space(trajectory_id);
                CREATE INDEX IF NOT EXISTS idx_prompts_created ON prompts(created_at);
            ''')

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_prompt(self, prompt: str, prompt_type: str = None, 
                   session_id: str = None, metadata: Dict = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO prompts (prompt_text, prompt_type, session_id, metadata)
                   VALUES (?, ?, ?, ?)''',
                (prompt, prompt_type, session_id, json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    def save_trajectory(self, session_id: str, step_order: int, thought: str = None,
                       action: str = None, action_input: Dict = None, observation: str = None,
                       reward: float = None, done: bool = False, execution_time: float = None,
                       token_usage: int = None, cost: float = None, metadata: Dict = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO trajectories 
                   (session_id, step_order, thought, action, action_input, observation, 
                    reward, done, execution_time, token_usage, cost, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (session_id, step_order, thought, action, json.dumps(action_input) if action_input else None,
                 observation, reward, done, execution_time, token_usage, cost, 
                 json.dumps(metadata) if metadata else None)
            )
            return cursor.lastrowid

    def save_latent_space(self, trajectory_id: int, embedding_vector: List[float] = None,
                         hidden_states: List[float] = None, decision_vector: List[float] = None,
                         agent_state: List[float] = None, metadata: Dict = None):
        import numpy as np
        def serialize_array(arr):
            if arr is None:
                return None
            return json.dumps([float(x) for x in np.array(arr).flatten()])
        
        with self._get_connection() as conn:
            conn.execute(
                '''INSERT INTO latent_space 
                   (trajectory_id, embedding_vector, hidden_states, decision_vector, agent_state, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (trajectory_id, 
                 serialize_array(embedding_vector),
                 serialize_array(hidden_states),
                 serialize_array(decision_vector),
                 serialize_array(agent_state),
                 json.dumps(metadata) if metadata else None)
            )

    def create_session(self, session_id: str, metadata: Dict = None) -> bool:
        with self._get_connection() as conn:
            conn.execute(
                '''INSERT OR IGNORE INTO sessions (id, metadata) VALUES (?, ?)''',
                (session_id, json.dumps(metadata) if metadata else None)
            )
            return True

    def update_session_stats(self, session_id: str, tokens: int = 0, cost: float = 0.0):
        with self._get_connection() as conn:
            conn.execute(
                '''UPDATE sessions SET total_tokens = total_tokens + ?, total_cost = total_cost + ?
                   WHERE id = ?''',
                (tokens, cost, session_id)
            )

    def end_session(self, session_id: str):
        with self._get_connection() as conn:
            conn.execute(
                '''UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ?''',
                (session_id,)
            )

    def get_prompts(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                '''SELECT * FROM prompts ORDER BY created_at DESC LIMIT ? OFFSET ?''',
                (limit, offset)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_trajectories(self, session_id: str = None, limit: int = 100) -> List[Dict]:
        with self._get_connection() as conn:
            if session_id:
                rows = conn.execute(
                    '''SELECT * FROM trajectories WHERE session_id = ? ORDER BY step_order''',
                    (session_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    '''SELECT * FROM trajectories ORDER BY created_at DESC LIMIT ?''',
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]

    def get_session_stats(self) -> Dict:
        with self._get_connection() as conn:
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost) as total_cost,
                    COUNT(CASE WHEN ended_at IS NOT NULL THEN 1 END) as completed_sessions
                FROM sessions
            ''').fetchone()
            return dict(stats) if stats else {}

    def search_similar_trajectories(self, embedding: List[float], limit: int = 5) -> List[Dict]:
        import numpy as np
        with self._get_connection() as conn:
            rows = conn.execute(
                '''SELECT t.*, ls.embedding_vector FROM trajectories t
                   LEFT JOIN latent_space ls ON t.id = ls.trajectory_id
                   ORDER BY t.id DESC LIMIT ?''',
                (limit * 2,)
            ).fetchall()
            
            query = np.array(embedding)
            similarities = []
            for row in rows:
                if row['embedding_vector']:
                    stored = np.array(json.loads(row['embedding_vector']))
                    similarity = np.dot(query, stored) / (np.linalg.norm(query) * np.linalg.norm(stored))
                    similarities.append((dict(row), similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in similarities[:limit]]
