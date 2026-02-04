import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class PromptRecord:
    id: int
    prompt_text: str
    prompt_type: str
    session_id: str
    created_at: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class TrajectoryStep:
    step_order: int
    thought: str
    action: str
    action_input: Dict
    observation: str
    reward: float
    done: bool
    timestamp: datetime
    execution_time: float
    token_usage: int
    cost: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class LatentSpaceData:
    trajectory_id: int
    embedding_vector: List[float]
    hidden_states: List[float]
    decision_vector: List[float]
    agent_state: List[float]
    created_at: datetime
    metadata: Dict = field(default_factory=dict)


class PromptRecorder:
    def __init__(self, storage):
        self.storage = storage

    def record(self, prompt: str, prompt_type: str = "general",
              session_id: str = None, metadata: Dict = None) -> int:
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        prompt_id = self.storage.save_prompt(
            prompt=prompt,
            prompt_type=prompt_type,
            session_id=session_id,
            metadata=metadata
        )
        return prompt_id

    def record_batch(self, prompts: List[Dict]) -> List[int]:
        prompt_ids = []
        for p in prompts:
            prompt_ids.append(self.record(
                prompt=p["text"],
                prompt_type=p.get("type", "general"),
                session_id=p.get("session_id"),
                metadata=p.get("metadata")
            ))
        return prompt_ids


class TrajectoryTracker:
    def __init__(self, storage):
        self.storage = storage
        self.current_session_id = None
        self.current_step = 0
        self.step_start_time = None

    def start_session(self, session_id: str = None, metadata: Dict = None) -> str:
        if session_id is None:
            session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        self.current_step = 0
        self.storage.create_session(session_id, metadata)
        return session_id

    def start_step(self, thought: str = None):
        self.current_step += 1
        self.step_start_time = time.time()

    def record_step(self, action: str, action_input: Dict = None, 
                   observation: str = None, reward: float = None,
                   done: bool = False, token_usage: int = None,
                   cost: float = None, thought: str = None,
                   metadata: Dict = None) -> int:
        execution_time = time.time() - self.step_start_time if self.step_start_time else 0
        
        trajectory_id = self.storage.save_trajectory(
            session_id=self.current_session_id,
            step_order=self.current_step,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            reward=reward,
            done=done,
            execution_time=execution_time,
            token_usage=token_usage,
            cost=cost,
            metadata=metadata
        )
        
        if token_usage or cost:
            self.storage.update_session_stats(
                self.current_session_id, 
                tokens=token_usage or 0, 
                cost=cost or 0.0
            )
        
        return trajectory_id

    def end_session(self):
        if self.current_session_id:
            self.storage.end_session(self.current_session_id)
            self.current_session_id = None
            self.current_step = 0

    def get_session_trajectory(self, session_id: str = None) -> List[Dict]:
        sid = session_id or self.current_session_id
        return self.storage.get_trajectories(session_id=sid)


class LatentSpaceRecorder:
    def __init__(self, storage):
        self.storage = storage

    def record(self, trajectory_id: int, 
               embedding_vector: List[float] = None,
               hidden_states: List[float] = None,
               decision_vector: List[float] = None,
               agent_state: List[float] = None,
               metadata: Dict = None) -> int:
        latent_id = self.storage.save_latent_space(
            trajectory_id=trajectory_id,
            embedding_vector=embedding_vector,
            hidden_states=hidden_states,
            decision_vector=decision_vector,
            agent_state=agent_state,
            metadata=metadata
        )
        return latent_id

    def record_from_agent(self, agent, trajectory_id: int, 
                         capture_hidden: bool = False, metadata: Dict = None):
        latent_data = {
            "embedding_vector": getattr(agent, 'last_embedding', None),
            "hidden_states": getattr(agent, 'last_hidden_states', None) if capture_hidden else None,
            "decision_vector": getattr(agent, 'last_decision_vector', None),
            "agent_state": getattr(agent, 'get_state_vector', lambda: None)(),
        }
        
        return self.record(trajectory_id, **latent_data, metadata=metadata)
