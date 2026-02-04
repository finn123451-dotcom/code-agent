from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class EvolutionPolicy:
    name: str
    trigger_conditions: List[str]
    action: str
    parameters: Dict = None


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
            EvolutionPolicy(
                name="successful_strategy_reinforcement",
                trigger_conditions=["reward > 0.8", "action_type == code_generation"],
                action="reinforce_prompt_template",
                parameters={"weight": 1.5}
            ),
            EvolutionPolicy(
                name="failed_strategy_avoidance",
                trigger_conditions=["reward < 0.2", "done == True"],
                action="create_negative_example",
                parameters={"weight": -1.0}
            ),
            EvolutionPolicy(
                name="high_cost_optimization",
                trigger_conditions=["cost > 10.0", "token_usage > 5000"],
                action="optimize_prompt",
                parameters={"target_tokens": 2000}
            ),
            EvolutionPolicy(
                name="efficiency_improvement",
                trigger_conditions=["execution_time > 30.0"],
                action="cache_common_patterns",
                parameters={"threshold": 0.8}
            )
        ]

    def analyze_trajectory_patterns(self, limit: int = 100) -> Dict:
        trajectories = self.storage.get_trajectories(limit=limit)
        patterns = {
            "success_rate": 0,
            "avg_reward": 0,
            "avg_execution_time": 0,
            "common_actions": {},
            "reward_distribution": {}
        }
        
        if not trajectories:
            return patterns
        
        total = len(trajectories)
        successful = sum(1 for t in trajectories if t.get('reward', 0) > 0.5)
        patterns["success_rate"] = successful / total if total > 0 else 0
        patterns["avg_reward"] = sum(t.get('reward', 0) for t in trajectories) / total
        patterns["avg_execution_time"] = sum(t.get('execution_time', 0) for t in trajectories) / total
        
        for t in trajectories:
            action = t.get('action', 'unknown')
            patterns["common_actions"][action] = patterns["common_actions"].get(action, 0) + 1
        
        return patterns

    def find_similar_successful_trajectories(self, current_trajectory: Dict, 
                                             limit: int = 5) -> List[Dict]:
        embedding = self.embedding_engine.embed_trajectory(current_trajectory)
        return self.vector_store.search_similar_trajectories(embedding, limit=limit)

    def generate_improved_prompt(self, original_prompt: str, 
                                 similar_successes: List[Dict]) -> str:
        if not similar_successes:
            return original_prompt
        
        improvements = []
        for success in similar_successes[:3]:
            payload = success.get('payload', {})
            if payload.get('thought'):
                improvements.append(payload['thought'])
        
        if improvements:
            enhanced_prompt = f"{original_prompt}\n\nConsider these successful approaches:\n"
            for i, imp in enumerate(improvements, 1):
                enhanced_prompt += f"{i}. {imp[:200]}\n"
            return enhanced_prompt
        
        return original_prompt

    def calculate_evolution_score(self) -> Dict:
        patterns = self.analyze_trajectory_patterns(limit=100)
        stats = self.storage.get_session_stats()
        
        score = 0.0
        weights = {
            "success_rate": 0.4,
            "efficiency": 0.3,
            "cost_effectiveness": 0.3
        }
        
        score += weights["success_rate"] * patterns["success_rate"]
        score += weights["efficiency"] * min(1.0, 10.0 / max(patterns["avg_execution_time"], 0.1))
        
        if stats.get('total_tokens', 0) > 0:
            cost_per_token = stats.get('total_cost', 0) / stats['total_tokens']
            score += weights["cost_effectiveness"] * max(0, 1 - cost_per_token * 100)
        
        return {
            "overall_score": score,
            "success_rate": patterns["success_rate"],
            "avg_execution_time": patterns["avg_execution_time"],
            "total_sessions": stats.get('total_sessions', 0),
            "total_tokens": stats.get('total_tokens', 0),
            "total_cost": stats.get('total_cost', 0)
        }

    def evolve(self, trigger: str, context: Dict) -> Dict:
        for policy in self.policies:
            if trigger in policy.trigger_conditions:
                evolution_result = self._apply_policy(policy, context)
                if evolution_result:
                    self.evolution_history.append({
                        "policy": policy.name,
                        "trigger": trigger,
                        "result": evolution_result,
                        "timestamp": context.get("timestamp")
                    })
                    return evolution_result
        return {"status": "no_evolution", "reason": "no matching policy"}

    def _apply_policy(self, policy: EvolutionPolicy, context: Dict) -> Dict:
        if policy.action == "reinforce_prompt_template":
            return {"action": "prompt_reinforced", "weight": policy.parameters.get("weight", 1.0)}
        elif policy.action == "create_negative_example":
            return {"action": "negative_example_created", "weight": policy.parameters.get("weight", -1.0)}
        elif policy.action == "optimize_prompt":
            return {"action": "prompt_optimized", "target_tokens": policy.parameters.get("target_tokens")}
        elif policy.action == "cache_common_patterns":
            return {"action": "patterns_cached", "threshold": policy.parameters.get("threshold", 0.8)}
        return None

    def get_best_trajectory(self, limit: int = 10) -> List[Dict]:
        trajectories = self.storage.get_trajectories(limit=100)
        sorted_trajs = sorted(trajectories, key=lambda x: x.get('reward', 0), reverse=True)
        return sorted_trajs[:limit]

    def recommend_strategy(self, task_type: str) -> Dict:
        best_trajectories = self.get_best_trajectory(10)
        
        relevant = [t for t in best_trajectories 
                   if t.get('metadata', {}).get('task_type') == task_type]
        
        if not relevant:
            relevant = best_trajectories[:5]
        
        return {
            "recommended_actions": [t.get('action') for t in relevant[:3]],
            "avg_reward": sum(t.get('reward', 0) for t in relevant) / len(relevant) if relevant else 0,
            "suggested_approach": relevant[0].get('thought') if relevant else None
        }

    def add_custom_policy(self, policy: EvolutionPolicy):
        self.policies.append(policy)

    def get_evolution_report(self) -> Dict:
        return {
            "current_score": self.calculate_evolution_score(),
            "active_policies": len(self.policies),
            "evolution_history_length": len(self.evolution_history),
            "recent_evolutions": self.evolution_history[-10:] if self.evolution_history else [],
            "patterns": self.analyze_trajectory_patterns()
        }
