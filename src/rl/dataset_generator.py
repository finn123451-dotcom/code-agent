"""
RL Dataset Generator - Generate training datasets from trajectory data
Supports: DPO, PPO, SFT, and custom formats
"""
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum


class DatasetFormat(Enum):
    DPO = "dpo"           # Direct Preference Optimization
    PPO = "ppo"           # Proximal Policy Optimization
    SFT = "sft"           # Supervised Fine-Tuning
    TRAJECTORY = "trajectory"  # Full trajectory format
    REWARD_MODEL = "reward_model"  # Reward model training


@dataclass
class RLTrajectorySample:
    """Single trajectory sample for RL training"""
    sample_id: str
    trajectory_id: int
    task_description: str
    task_type: str
    steps: List[Dict]
    total_reward: float
    success: bool
    language: str
    code_generated: str
    execution_success: bool
    efficiency_score: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class DPOSample:
    """DPO (Direct Preference Optimization) sample"""
    sample_id: str
    trajectory_id: int
    chosen: Dict  # Preferred trajectory/step
    rejected: Dict  # Rejected trajectory/step
    preference_score: float
    task_description: str
    language: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PPOSample:
    """PPO (Proximal Policy Optimization) sample"""
    sample_id: str
    trajectory_id: int
    state: Dict
    action: str
    action_input: Dict
    reward: float
    value: float
    log_prob: float
    done: bool
    info: Dict
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SFTSample:
    """SFT (Supervised Fine-Tuning) sample"""
    sample_id: str
    trajectory_id: int
    messages: List[Dict]
    task_description: str
    language: str
    code_output: str
    success: bool
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class RewardModelSample:
    """Reward model training sample"""
    sample_id: str
    trajectory_id: int
    chosen_output: str
    rejected_output: str
    reward_score: float
    task_type: str
    context: str
    created_at: datetime = field(default_factory=datetime.now)


class RLDatasetGenerator:
    """Generate RL training datasets from trajectory data"""
    
    def __init__(self, storage=None, llm_api_key: str = None):
        self.storage = storage
        self.llm_api_key = llm_api_key
        self.samples_generated = 0
    
    def set_storage(self, storage):
        """Set the storage backend"""
        self.storage = storage
    
    def generate_dataset(self, trajectories: List[Dict], format: DatasetFormat = DatasetFormat.TRAJECTORY,
                        min_success_rate: float = 0.0, max_samples: int = None) -> List[Dict]:
        """Generate a dataset from trajectories"""
        
        filtered = [t for t in trajectories if self._filter_trajectory(t, min_success_rate)]
        
        if max_samples and len(filtered) > max_samples:
            import random
            filtered = random.sample(filtered, max_samples)
        
        if format == DatasetFormat.DPO:
            return self._generate_dpo_dataset(filtered)
        elif format == DatasetFormat.PPO:
            return self._generate_ppo_dataset(filtered)
        elif format == DatasetFormat.SFT:
            return self._generate_sft_dataset(filtered)
        elif format == DatasetFormat.REWARD_MODEL:
            return self._generate_reward_model_dataset(filtered)
        else:
            return self._generate_trajectory_dataset(filtered)
    
    def _filter_trajectory(self, trajectory: Dict, min_success_rate: float) -> bool:
        """Filter trajectory based on success rate"""
        steps = trajectory.get('steps', [])
        if not steps:
            return False
        
        success_rate = sum(s.get('reward', 0) for s in steps) / len(steps)
        return success_rate >= min_success_rate
    
    def _generate_trajectory_dataset(self, trajectories: List[Dict]) -> List[Dict]:
        """Generate full trajectory dataset"""
        samples = []
        
        for trajectory in trajectories:
            sample = self._trajectory_to_sample(trajectory)
            samples.append(sample)
        
        self.samples_generated = len(samples)
        return samples
    
    def _trajectory_to_sample(self, trajectory: Dict) -> Dict:
        """Convert trajectory to RL sample"""
        steps = trajectory.get('steps', [])
        
        code_generated = self._extract_code_from_steps(steps)
        language = self._extract_language(steps)
        execution_success = self._check_execution_success(steps)
        
        avg_reward = sum(s.get('reward', 0) for s in steps) / len(steps) if steps else 0
        
        efficiency = self._calculate_efficiency(steps)
        
        return {
            "sample_id": str(uuid.uuid4()),
            "trajectory_id": trajectory.get('id'),
            "task_description": trajectory.get('task_description', ''),
            "task_type": trajectory.get('trajectory_type', 'unknown'),
            "steps": steps,
            "total_reward": trajectory.get('total_reward', avg_reward),
            "success": avg_reward > 0.5,
            "language": language,
            "code_generated": code_generated,
            "execution_success": execution_success,
            "efficiency_score": efficiency,
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "total_steps": len(steps),
                "total_execution_time": trajectory.get('total_execution_time', 0),
                "complexity": self._calculate_complexity(steps)
            }
        }
    
    def _generate_dpo_dataset(self, trajectories: List[Dict]) -> List[Dict]:
        """Generate DPO dataset with preference pairs"""
        samples = []
        
        for trajectory in trajectories:
            steps = trajectory.get('steps', [])
            
            successful_steps = [s for s in steps if s.get('reward', 0) > 0.7]
            failed_steps = [s for s in steps if s.get('reward', 0) < 0.5]
            
            if successful_steps and failed_steps:
                for chosen in successful_steps[:3]:
                    for rejected in failed_steps[:2]:
                        preference = chosen.get('reward', 0) - rejected.get('reward', 0)
                        
                        sample = {
                            "sample_id": str(uuid.uuid4()),
                            "trajectory_id": trajectory.get('id'),
                            "chosen": {
                                "action": chosen.get('action'),
                                "action_input": chosen.get('action_input'),
                                "thought": chosen.get('thought'),
                                "reward": chosen.get('reward')
                            },
                            "rejected": {
                                "action": rejected.get('action'),
                                "action_input": rejected.get('action_input'),
                                "thought": rejected.get('thought'),
                                "reward": rejected.get('reward')
                            },
                            "preference_score": preference,
                            "task_description": trajectory.get('task_description', ''),
                            "language": self._extract_language(steps),
                            "created_at": datetime.now().isoformat()
                        }
                        samples.append(sample)
            elif successful_steps:
                sample = {
                    "sample_id": str(uuid.uuid4()),
                    "trajectory_id": trajectory.get('id'),
                    "chosen": {
                        "action": successful_steps[0].get('action'),
                        "action_input": successful_steps[0].get('action_input'),
                        "thought": successful_steps[0].get('thought'),
                        "reward": successful_steps[0].get('reward')
                    },
                    "rejected": None,
                    "preference_score": 0.5,
                    "task_description": trajectory.get('task_description', ''),
                    "language": self._extract_language(steps),
                    "created_at": datetime.now().isoformat()
                }
                samples.append(sample)
        
        self.samples_generated = len(samples)
        return samples
    
    def _generate_ppo_dataset(self, trajectories: List[Dict]) -> List[Dict]:
        """Generate PPO dataset with state-action-reward tuples"""
        samples = []
        
        for trajectory in trajectories:
            steps = trajectory.get('steps', [])
            
            for i, step in enumerate(steps):
                state = self._extract_state(step, steps[:i])
                next_state = self._extract_state(step, steps[:i+1]) if i < len(steps) - 1 else None
                
                value = step.get('reward', 0)
                log_prob = -abs(value - 0.5) * 2
                
                sample = {
                    "sample_id": str(uuid.uuid4()),
                    "trajectory_id": trajectory.get('id'),
                    "step_id": step.get('id'),
                    "state": state,
                    "action": step.get('action'),
                    "action_input": step.get('action_input'),
                    "reward": step.get('reward', 0),
                    "value": value,
                    "log_prob": log_prob,
                    "done": step.get('done', False) or i == len(steps) - 1,
                    "info": {
                        "thought": step.get('thought'),
                        "observation": step.get('observation'),
                        "execution_time": step.get('execution_time'),
                        "next_state": next_state
                    },
                    "created_at": datetime.now().isoformat()
                }
                samples.append(sample)
        
        self.samples_generated = len(samples)
        return samples
    
    def _generate_sft_dataset(self, trajectories: List[Dict]) -> List[Dict]:
        """Generate SFT dataset with conversation format"""
        samples = []
        
        for trajectory in trajectories:
            steps = trajectory.get('steps', [])
            
            messages = self._build_conversation(steps)
            
            code_output = self._extract_code_from_steps(steps)
            avg_reward = sum(s.get('reward', 0) for s in steps) / len(steps) if steps else 0
            
            sample = {
                "sample_id": str(uuid.uuid4()),
                "trajectory_id": trajectory.get('id'),
                "messages": messages,
                "task_description": trajectory.get('task_description', ''),
                "language": self._extract_language(steps),
                "code_output": code_output,
                "success": avg_reward > 0.5,
                "created_at": datetime.now().isoformat(),
                "metadata": {
                    "total_steps": len(steps),
                    "total_reward": trajectory.get('total_reward', avg_reward)
                }
            }
            samples.append(sample)
        
        self.samples_generated = len(samples)
        return samples
    
    def _generate_reward_model_dataset(self, trajectories: List[Dict]) -> List[Dict]:
        """Generate reward model training dataset"""
        samples = []
        
        for trajectory in trajectories:
            steps = trajectory.get('steps', [])
            
            successful_steps = [s for s in steps if s.get('reward', 0) > 0.7]
            failed_steps = [s for s in steps if s.get('reward', 0) < 0.5]
            
            if successful_steps and failed_steps:
                chosen = successful_steps[0]
                rejected = failed_steps[0]
                
                context = self._build_context(steps)
                
                sample = {
                    "sample_id": str(uuid.uuid4()),
                    "trajectory_id": trajectory.get('id'),
                    "chosen_output": chosen.get('observation', ''),
                    "rejected_output": rejected.get('observation', ''),
                    "reward_score": chosen.get('reward', 0) - rejected.get('reward', 0),
                    "task_type": trajectory.get('trajectory_type', 'unknown'),
                    "context": context,
                    "created_at": datetime.now().isoformat()
                }
                samples.append(sample)
        
        self.samples_generated = len(samples)
        return samples
    
    def _extract_state(self, step: Dict, previous_steps: List[Dict]) -> Dict:
        """Extract state representation from step"""
        return {
            "step_order": step.get('step_order'),
            "action": step.get('action'),
            "action_input": step.get('action_input'),
            "cumulative_reward": sum(s.get('reward', 0) for s in previous_steps) + step.get('reward', 0),
            "previous_actions": [s.get('action') for s in previous_steps[-5:]]
        }
    
    def _extract_code_from_steps(self, steps: List[Dict]) -> str:
        """Extract code from steps"""
        codes = []
        for step in steps:
            action_input = step.get('action_input', {})
            if isinstance(action_input, dict):
                code = action_input.get('code', '')
                if code:
                    codes.append(code)
        return '\n'.join(codes)
    
    def _extract_language(self, steps: List[Dict]) -> str:
        """Extract programming language from steps"""
        for step in steps:
            action_input = step.get('action_input', {})
            if isinstance(action_input, dict):
                lang = action_input.get('language')
                if lang:
                    return lang
        return 'unknown'
    
    def _check_execution_success(self, steps: List[Dict]) -> bool:
        """Check if code execution was successful"""
        for step in steps:
            if 'execution' in step.get('action', '').lower():
                reward = step.get('reward', 0)
                return reward > 0.5
        return False
    
    def _calculate_efficiency(self, steps: List[Dict]) -> float:
        """Calculate efficiency score"""
        if not steps:
            return 0.0
        
        total_time = sum(s.get('execution_time', 0) for s in steps)
        avg_reward = sum(s.get('reward', 0) for s in steps) / len(steps)
        
        time_score = max(0, 1 - total_time / 60) if total_time > 0 else 1.0
        
        return round(avg_reward * 0.7 + time_score * 0.3, 3)
    
    def _calculate_complexity(self, steps: List[Dict]) -> float:
        """Calculate complexity score"""
        if not steps:
            return 0.0
        
        total_steps = len(steps)
        avg_reward = sum(s.get('reward', 0) for s in steps) / len(steps)
        
        complexity = min(1.0, total_steps / 10 * 0.3 + (1 - avg_reward) * 0.7)
        
        return round(complexity, 3)
    
    def _build_conversation(self, steps: List[Dict]) -> List[Dict]:
        """Build conversation messages from steps"""
        messages = []
        
        for step in steps:
            thought = step.get('thought', '')
            action = step.get('action', '')
            action_input = step.get('action_input', {})
            
            messages.append({
                "role": "assistant",
                "content": thought,
                "tool_calls": [{
                    "name": action,
                    "arguments": action_input
                }] if action else []
            })
            
            observation = step.get('observation', '')
            if observation:
                messages.append({
                    "role": "user",
                    "content": observation
                })
        
        return messages
    
    def _build_context(self, steps: List[Dict]) -> str:
        """Build context string for reward model"""
        actions = [s.get('action', '') for s in steps]
        return f"Actions taken: {' -> '.join(actions[:5])}"
    
    def export_dataset(self, samples: List[Dict], format: str = "jsonl", output_path: str = None) -> str:
        """Export dataset to file"""
        if format == "jsonl":
            lines = [json.dumps(s) for s in samples]
            content = '\n'.join(lines)
        elif format == "json":
            content = json.dumps(samples, indent=2, default=str)
        else:
            content = json.dumps(samples, indent=2, default=str)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        
        return content
    
    def get_dataset_stats(self, samples: List[Dict], format: DatasetFormat) -> Dict:
        """Get statistics about the generated dataset"""
        return {
            "format": format.value,
            "total_samples": len(samples),
            "samples_generated": self.samples_generated,
            "avg_reward": self._avg_key(samples, 'total_reward') if samples else 0,
            "success_rate": self._avg_key(samples, 'success') if samples else 0,
            "languages": list(set(self._get_key(samples, 'language'))) if samples else []
        }
    
    def _avg_key(self, samples: List[Dict], key: str) -> float:
        """Calculate average of a key"""
        values = [s.get(key, 0) for s in samples if s.get(key) is not None]
        return sum(values) / len(values) if values else 0
    
    def _get_key(self, samples: List[Dict], key: str) -> List:
        """Get all values of a key"""
        return [s.get(key) for s in samples if s.get(key)]


class TrajectorySampler:
    """Sample trajectories for training/evaluation"""
    
    def __init__(self, storage=None):
        self.storage = storage
    
    def set_storage(self, storage):
        """Set the storage backend"""
        self.storage = storage
    
    def sample_by_reward(self, trajectories: List[Dict], n: int, 
                       high_reward_only: bool = True) -> List[Dict]:
        """Sample trajectories by reward"""
        if high_reward_only:
            sorted_trajs = sorted(trajectories, 
                                 key=lambda t: t.get('total_reward', 0), 
                                 reverse=True)
            return sorted_trajs[:n]
        return trajectories[:n]
    
    def sample_by_task_type(self, trajectories: List[Dict], task_type: str, 
                          n: int = None) -> List[Dict]:
        """Sample trajectories by task type"""
        filtered = [t for t in trajectories 
                   if t.get('trajectory_type') == task_type]
        return filtered[:n] if n else filtered
    
    def sample_balanced(self, trajectories: List[Dict], n: int) -> List[Dict]:
        """Sample balanced trajectory set"""
        success_trajs = [t for t in trajectories if t.get('total_reward', 0) > 0.7]
        fail_trajs = [t for t in trajectories if t.get('total_reward', 0) <= 0.7]
        
        import random
        n_success = min(len(success_trajs), n // 2)
        n_fail = min(len(fail_trajs), n - n_success)
        
        sampled = random.sample(success_trajs, n_success) if success_trajs else []
        sampled.extend(random.sample(fail_trajs, n_fail) if fail_trajs else [])
        
        return sampled
