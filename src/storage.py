import sqlite3
import json
import os
import uuid
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
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    metadata JSON
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
                    metadata JSON,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trajectory_id INTEGER NOT NULL,
                    step_order INTEGER NOT NULL,
                    step_name TEXT,
                    thought TEXT,
                    action TEXT,
                    action_input JSON,
                    action_result JSON,
                    observation TEXT,
                    reward REAL,
                    done BOOLEAN DEFAULT 0,
                    execution_time REAL,
                    token_usage INTEGER,
                    cost REAL,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
                );

                CREATE TABLE IF NOT EXISTS llm_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step_id INTEGER,
                    trajectory_id INTEGER,
                    request_type TEXT,
                    model TEXT,
                    messages JSON,
                    parameters JSON,
                    prompt_text TEXT,
                    response_text TEXT,
                    response_object JSON,
                    token_usage INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    cost REAL,
                    latency_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    FOREIGN KEY (step_id) REFERENCES steps(id),
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
                );

                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trajectory_id INTEGER,
                    step_id INTEGER,
                    prompt_text TEXT NOT NULL,
                    prompt_type TEXT,
                    language TEXT,
                    context JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id),
                    FOREIGN KEY (step_id) REFERENCES steps(id)
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (step_id) REFERENCES steps(id),
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
                );

                CREATE TABLE IF NOT EXISTS latent_space (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step_id INTEGER,
                    trajectory_id INTEGER,
                    embedding_vector JSON,
                    hidden_states JSON,
                    decision_vector JSON,
                    agent_state JSON,
                    attention_weights JSON,
                    intermediate_outputs JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSON,
                    FOREIGN KEY (step_id) REFERENCES steps(id),
                    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id)
                );

                CREATE TABLE IF NOT EXISTS trajectory_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_trajectory_id INTEGER,
                    child_trajectory_id INTEGER,
                    relation_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_trajectory_id) REFERENCES trajectories(id),
                    FOREIGN KEY (child_trajectory_id) REFERENCES trajectories(id)
                );

                CREATE INDEX IF NOT EXISTS idx_steps_trajectory ON steps(trajectory_id);
                CREATE INDEX IF NOT EXISTS idx_steps_order ON steps(step_order);
                CREATE INDEX IF NOT EXISTS idx_llm_requests_step ON llm_requests(step_id);
                CREATE INDEX IF NOT EXISTS idx_llm_requests_trajectory ON llm_requests(trajectory_id);
                CREATE INDEX IF NOT EXISTS idx_prompts_trajectory ON prompts(trajectory_id);
                CREATE INDEX IF NOT EXISTS idx_code_executions_step ON code_executions(step_id);
                CREATE INDEX IF NOT EXISTS idx_latent_space_step ON latent_space(step_id);
                CREATE INDEX IF NOT EXISTS idx_trajectory_session ON trajectories(session_id);
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

    def _serialize(self, obj) -> Optional[str]:
        if obj is None:
            return None
        return json.dumps(obj, ensure_ascii=False)

    def _deserialize(self, text: str) -> Any:
        if text is None:
            return None
        try:
            return json.loads(text)
        except:
            return text

    def create_session(self, session_id: str = None, metadata: Dict = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                '''INSERT OR IGNORE INTO sessions (id, metadata) VALUES (?, ?)''',
                (session_id, self._serialize(metadata))
            )
        return session_id

    def create_trajectory(self, session_id: str, task_description: str = None, 
                         initial_prompt: str = None, trajectory_type: str = 'task',
                         metadata: Dict = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO trajectories (session_id, task_description, initial_prompt, trajectory_type, metadata)
                   VALUES (?, ?, ?, ?, ?)''',
                (session_id, task_description, initial_prompt, trajectory_type, self._serialize(metadata))
            )
            return cursor.lastrowid

    def save_step(self, trajectory_id: int, step_order: int, step_name: str = None,
                thought: str = None, action: str = None, action_input: Dict = None,
                action_result: Dict = None, observation: str = None, reward: float = None,
                execution_time: float = None, token_usage: int = None, cost: float = None,
                error: str = None, metadata: Dict = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO steps 
                   (trajectory_id, step_order, step_name, thought, action, action_input, action_result, 
                    observation, reward, execution_time, token_usage, cost, error, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (trajectory_id, step_order, step_name, thought, action, 
                 self._serialize(action_input), self._serialize(action_result),
                 observation, reward, execution_time, token_usage, cost, 
                 self._serialize(error), self._serialize(metadata))
            )
            return cursor.lastrowid

    def update_step_done(self, step_id: int, done: bool = True, result: str = None):
        with self._get_connection() as conn:
            conn.execute(
                '''UPDATE steps SET done = ?, observation = ? WHERE id = ?''',
                (done, result, step_id)
            )

    def save_llm_request(self, step_id: int = None, trajectory_id: int = None,
                        request_type: str = None, model: str = None, messages: List[Dict] = None,
                        parameters: Dict = None, prompt_text: str = None, response_text: str = None,
                        response_object: Dict = None, token_usage: int = None, 
                        prompt_tokens: int = None, completion_tokens: int = None,
                        cost: float = None, latency_ms: float = None, metadata: Dict = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO llm_requests 
                   (step_id, trajectory_id, request_type, model, messages, parameters, prompt_text, 
                    response_text, response_object, token_usage, prompt_tokens, completion_tokens, 
                    cost, latency_ms, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (step_id, trajectory_id, request_type, model, self._serialize(messages),
                 self._serialize(parameters), prompt_text, response_text, 
                 self._serialize(response_object), token_usage, prompt_tokens, completion_tokens,
                 cost, latency_ms, self._serialize(metadata))
            )
            return cursor.lastrowid

    def save_prompt(self, trajectory_id: int = None, step_id: int = None,
                   prompt_text: str = None, prompt_type: str = None, language: str = None,
                   context: Dict = None, metadata: Dict = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO prompts (trajectory_id, step_id, prompt_text, prompt_type, language, context, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (trajectory_id, step_id, prompt_text, prompt_type, language, 
                 self._serialize(context), self._serialize(metadata))
            )
            return cursor.lastrowid

    def save_code_execution(self, step_id: int = None, trajectory_id: int = None,
                          code: str = None, language: str = None, stdout: str = None,
                          stderr: str = None, return_code: int = None, 
                          execution_time: float = None, timeout: bool = False) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO code_executions 
                   (step_id, trajectory_id, code, language, stdout, stderr, return_code, execution_time, timeout)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (step_id, trajectory_id, code, language, stdout, stderr, 
                 return_code, execution_time, timeout)
            )
            return cursor.lastrowid

    def save_latent_space(self, step_id: int = None, trajectory_id: int = None,
                         embedding_vector: List[float] = None, hidden_states: List[float] = None,
                         decision_vector: List[float] = None, agent_state: List[float] = None,
                         attention_weights: List[float] = None, intermediate_outputs: Dict = None,
                         metadata: Dict = None) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute(
                '''INSERT INTO latent_space 
                   (step_id, trajectory_id, embedding_vector, hidden_states, decision_vector, 
                    agent_state, attention_weights, intermediate_outputs, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (step_id, trajectory_id, self._serialize(embedding_vector),
                 self._serialize(hidden_states), self._serialize(decision_vector),
                 self._serialize(agent_state), self._serialize(attention_weights),
                 self._serialize(intermediate_outputs), self._serialize(metadata))
            )
            return cursor.lastrowid

    def save_trajectory_relation(self, parent_trajectory_id: int, child_trajectory_id: int,
                               relation_type: str = None):
        with self._get_connection() as conn:
            conn.execute(
                '''INSERT INTO trajectory_relations (parent_trajectory_id, child_trajectory_id, relation_type)
                   VALUES (?, ?, ?)''',
                (parent_trajectory_id, child_trajectory_id, relation_type)
            )

    def update_trajectory(self, trajectory_id: int, final_result: str = None, 
                         done: bool = False, total_reward: float = None,
                         total_execution_time: float = None):
        with self._get_connection() as conn:
            conn.execute(
                '''UPDATE trajectories SET 
                   final_result = ?, done = ?, total_reward = ?, 
                   total_execution_time = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?''',
                (final_result, done, total_reward, total_execution_time, trajectory_id)
            )

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

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                '''SELECT * FROM sessions WHERE id = ?''', (session_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_trajectory(self, trajectory_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                '''SELECT * FROM trajectories WHERE id = ?''', (trajectory_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_full_trajectory(self, trajectory_id: int) -> Dict:
        with self._get_connection() as conn:
            trajectory = conn.execute(
                '''SELECT * FROM trajectories WHERE id = ?''', (trajectory_id,)
            ).fetchone()
            
            if not trajectory:
                return None
            
            result = dict(trajectory)
            
            steps = conn.execute(
                '''SELECT * FROM steps WHERE trajectory_id = ? ORDER BY step_order''',
                (trajectory_id,)
            ).fetchall()
            result['steps'] = [dict(s) for s in steps]
            
            for step in result['steps']:
                step_id = step['id']
                
                llm_requests = conn.execute(
                    '''SELECT * FROM llm_requests WHERE step_id = ?''', (step_id,)
                ).fetchall()
                step['llm_requests'] = [dict(r) for r in llm_requests]
                
                prompts = conn.execute(
                    '''SELECT * FROM prompts WHERE step_id = ?''', (step_id,)
                ).fetchall()
                step['prompts'] = [dict(p) for p in prompts]
                
                code_executions = conn.execute(
                    '''SELECT * FROM code_executions WHERE step_id = ?''', (step_id,)
                ).fetchall()
                step['code_executions'] = [dict(c) for c in code_executions]
                
                latent_space = conn.execute(
                    '''SELECT * FROM latent_space WHERE step_id = ?''', (step_id,)
                ).fetchall()
                step['latent_space'] = [dict(l) for l in latent_space]
            
            prompts = conn.execute(
                '''SELECT * FROM prompts WHERE trajectory_id = ? AND step_id IS NULL''',
                (trajectory_id,)
            ).fetchall()
            result['prompts'] = [dict(p) for p in prompts]
            
            return result

    def get_step(self, step_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                '''SELECT * FROM steps WHERE id = ?''', (step_id,)
            ).fetchone()
            if row:
                step = dict(row)
                
                llm_requests = conn.execute(
                    '''SELECT * FROM llm_requests WHERE step_id = ?''', (step_id,)
                ).fetchall()
                step['llm_requests'] = [dict(r) for r in llm_requests]
                
                code_executions = conn.execute(
                    '''SELECT * FROM code_executions WHERE step_id = ?''', (step_id,)
                ).fetchall()
                step['code_executions'] = [dict(c) for c in code_executions]
                
                latent_space = conn.execute(
                    '''SELECT * FROM latent_space WHERE step_id = ?''', (step_id,)
                ).fetchall()
                step['latent_space'] = [dict(l) for l in latent_space]
                
                return step
            return None

    def get_trajectories(self, session_id: str = None, limit: int = 100) -> List[Dict]:
        with self._get_connection() as conn:
            if session_id:
                rows = conn.execute(
                    '''SELECT * FROM trajectories WHERE session_id = ? ORDER BY created_at DESC LIMIT ?''',
                    (session_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    '''SELECT * FROM trajectories ORDER BY created_at DESC LIMIT ?''',
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]

    def get_steps(self, trajectory_id: int) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                '''SELECT * FROM steps WHERE trajectory_id = ? ORDER BY step_order''',
                (trajectory_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_llm_requests(self, trajectory_id: int = None, step_id: int = None) -> List[Dict]:
        with self._get_connection() as conn:
            if trajectory_id:
                rows = conn.execute(
                    '''SELECT * FROM llm_requests WHERE trajectory_id = ? ORDER BY created_at''',
                    (trajectory_id,)
                ).fetchall()
            elif step_id:
                rows = conn.execute(
                    '''SELECT * FROM llm_requests WHERE step_id = ? ORDER BY created_at''',
                    (step_id,)
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

    def get_trajectory_stats(self) -> Dict:
        with self._get_connection() as conn:
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_trajectories,
                    SUM(CASE WHEN done = 1 THEN 1 ELSE 0 END) as completed_trajectories,
                    AVG(total_execution_time) as avg_execution_time,
                    AVG(total_reward) as avg_reward,
                    SUM(token_usage) as total_tokens,
                    SUM(cost) as total_cost
                FROM trajectories
            ''').fetchone()
            return dict(stats) if stats else {}

    def search_trajectories(self, query: str = None, embedding: List[float] = None, 
                          limit: int = 10) -> List[Dict]:
        with self._get_connection() as conn:
            if query:
                rows = conn.execute(
                    '''SELECT * FROM trajectories 
                       WHERE task_description LIKE ? OR initial_prompt LIKE ? OR final_result LIKE ?
                       ORDER BY created_at DESC LIMIT ?''',
                    (f'%{query}%', f'%{query}%', f'%{query}%', limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    '''SELECT * FROM trajectories ORDER BY created_at DESC LIMIT ?''',
                    (limit,)
                ).fetchall()
            return [dict(row) for row in rows]

    def get_child_trajectories(self, parent_trajectory_id: int) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                '''SELECT t.* FROM trajectories t
                   JOIN trajectory_relations tr ON t.id = tr.child_trajectory_id
                   WHERE tr.parent_trajectory_id = ?
                   ORDER BY tr.created_at''',
                (parent_trajectory_id,)
            ).fetchall()
            return [dict(row) for row in rows]
