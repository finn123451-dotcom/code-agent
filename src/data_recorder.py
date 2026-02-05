import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class StepData:
    step_order: int
    step_name: str
    thought: str
    action: str
    action_input: Dict
    action_result: Dict
    observation: str
    reward: float
    execution_time: float
    token_usage: int
    cost: float
    error: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class LLMRequestData:
    request_type: str
    model: str
    messages: List[Dict]
    parameters: Dict
    prompt_text: str
    response_text: str
    response_object: Dict
    token_usage: int
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency_ms: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class LatentSpaceData:
    embedding_vector: List[float]
    hidden_states: List[float]
    decision_vector: List[float]
    agent_state: List[float]
    attention_weights: List[float]
    intermediate_outputs: Dict
    metadata: Dict = field(default_factory=dict)


class TrajectoryRecorder:
    def __init__(self, storage):
        self.storage = storage
        self.current_trajectory_id = None
        self.current_step_order = 0
        self.step_start_time = None
        self.total_execution_time = 0
        self.current_step_id = None

    def start_trajectory(self, session_id: str, task_description: str = None,
                        initial_prompt: str = None, trajectory_type: str = 'task',
                        metadata: Dict = None) -> int:
        self.current_trajectory_id = self.storage.create_trajectory(
            session_id=session_id,
            task_description=task_description,
            initial_prompt=initial_prompt,
            trajectory_type=trajectory_type,
            metadata=metadata
        )
        self.current_step_order = 0
        self.total_execution_time = 0
        return self.current_trajectory_id

    def end_trajectory(self, final_result: str = None, done: bool = True,
                      total_reward: float = None) -> int:
        if self.current_trajectory_id:
            self.storage.update_trajectory(
                trajectory_id=self.current_trajectory_id,
                final_result=final_result,
                done=done,
                total_reward=total_reward,
                total_execution_time=self.total_execution_time
            )
            trajectory_id = self.current_trajectory_id
            self.current_trajectory_id = None
            self.current_step_order = 0
            return trajectory_id
        return None

    def start_step(self, step_name: str = None, thought: str = None):
        self.current_step_order += 1
        self.step_start_time = time.time()
        self.current_step_name = step_name
        self.current_thought = thought

    def record_step(self, action: str, action_input: Dict = None,
                   action_result: Dict = None, observation: str = None,
                   reward: float = None, token_usage: int = None,
                   cost: float = None, error: str = None,
                   metadata: Dict = None) -> int:
        if not self.current_trajectory_id:
            raise ValueError("No active trajectory. Call start_trajectory() first.")

        execution_time = time.time() - self.step_start_time if self.step_start_time else 0
        self.total_execution_time += execution_time

        self.current_step_id = self.storage.save_step(
            trajectory_id=self.current_trajectory_id,
            step_order=self.current_step_order,
            step_name=getattr(self, 'current_step_name', None),
            thought=getattr(self, 'current_thought', None),
            action=action,
            action_input=action_input,
            action_result=action_result,
            observation=observation,
            reward=reward,
            execution_time=execution_time,
            token_usage=token_usage,
            cost=cost,
            error=error,
            metadata=metadata
        )

        self.step_start_time = None
        return self.current_step_id

    def record_llm_request(self, step_id: int = None, request_type: str = None,
                          model: str = None, messages: List[Dict] = None,
                          parameters: Dict = None, prompt_text: str = None,
                          response_text: str = None, response_object: Dict = None,
                          token_usage: int = None, prompt_tokens: int = None,
                          completion_tokens: int = None, cost: float = None,
                          latency_ms: float = None, metadata: Dict = None) -> int:
        s_id = step_id or self.current_step_id
        t_id = self.current_trajectory_id

        return self.storage.save_llm_request(
            step_id=s_id,
            trajectory_id=t_id,
            request_type=request_type,
            model=model,
            messages=messages,
            parameters=parameters,
            prompt_text=prompt_text,
            response_text=response_text,
            response_object=response_object,
            token_usage=token_usage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            metadata=metadata
        )

    def record_prompt(self, prompt_text: str, prompt_type: str = None,
                     language: str = None, context: Dict = None,
                     step_id: int = None, metadata: Dict = None) -> int:
        return self.storage.save_prompt(
            trajectory_id=self.current_trajectory_id,
            step_id=step_id or self.current_step_id,
            prompt_text=prompt_text,
            prompt_type=prompt_type,
            language=language,
            context=context,
            metadata=metadata
        )

    def record_code_execution(self, code: str, language: str = None,
                            stdout: str = None, stderr: str = None,
                            return_code: int = None, execution_time: float = None,
                            timeout: bool = False, step_id: int = None) -> int:
        return self.storage.save_code_execution(
            step_id=step_id or self.current_step_id,
            trajectory_id=self.current_trajectory_id,
            code=code,
            language=language,
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            execution_time=execution_time,
            timeout=timeout
        )

    def record_latent_space(self, embedding_vector: List[float] = None,
                           hidden_states: List[float] = None,
                           decision_vector: List[float] = None,
                           agent_state: List[float] = None,
                           attention_weights: List[float] = None,
                           intermediate_outputs: Dict = None,
                           step_id: int = None, metadata: Dict = None) -> int:
        return self.storage.save_latent_space(
            step_id=step_id or self.current_step_id,
            trajectory_id=self.current_trajectory_id,
            embedding_vector=embedding_vector,
            hidden_states=hidden_states,
            decision_vector=decision_vector,
            agent_state=agent_state,
            attention_weights=attention_weights,
            intermediate_outputs=intermediate_outputs,
            metadata=metadata
        )

    def get_full_trajectory(self, trajectory_id: int = None) -> Dict:
        t_id = trajectory_id or self.current_trajectory_id
        if t_id:
            return self.storage.get_full_trajectory(t_id)
        return None

    def get_steps(self, trajectory_id: int = None) -> List[Dict]:
        t_id = trajectory_id or self.current_trajectory_id
        if t_id:
            return self.storage.get_steps(t_id)
        return []


class SessionRecorder:
    def __init__(self, storage):
        self.storage = storage

    def create_session(self, metadata: Dict = None) -> str:
        return self.storage.create_session(metadata=metadata)

    def end_session(self, session_id: str):
        self.storage.end_session(session_id)

    def update_stats(self, session_id: str, tokens: int = 0, cost: float = 0.0):
        self.storage.update_session_stats(session_id, tokens, cost)


class PromptRecorder:
    def __init__(self, storage):
        self.storage = storage

    def record(self, prompt_text: str, prompt_type: str = None,
              trajectory_id: int = None, step_id: int = None,
              language: str = None, context: Dict = None,
              metadata: Dict = None) -> int:
        return self.storage.save_prompt(
            trajectory_id=trajectory_id,
            step_id=step_id,
            prompt_text=prompt_text,
            prompt_type=prompt_type,
            language=language,
            context=context,
            metadata=metadata
        )

    def record_batch(self, prompts: List[Dict]) -> List[int]:
        ids = []
        for p in prompts:
            ids.append(self.record(
                prompt_text=p["text"],
                prompt_type=p.get("type"),
                trajectory_id=p.get("trajectory_id"),
                step_id=p.get("step_id"),
                language=p.get("language"),
                context=p.get("context"),
                metadata=p.get("metadata")
            ))
        return ids
