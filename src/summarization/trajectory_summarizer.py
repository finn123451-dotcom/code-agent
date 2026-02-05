"""
Trajectory Summarizer - Extract summaries from trajectory data
"""
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TrajectorySummary:
    """Summary of a complete trajectory"""
    trajectory_id: int
    task_description: str
    task_type: str
    total_steps: int
    total_duration: float
    success_rate: float
    key_actions: List[str]
    patterns_detected: List[str]
    techniques_used: List[str]
    outcome_summary: str
    code_patterns: List[str]
    languages_used: List[str]
    efficiency_score: float
    complexity_score: float
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class StepSummary:
    """Summary of a single step"""
    step_id: int
    step_order: int
    step_name: str
    action_type: str
    primary_goal: str
    tools_used: List[str]
    success: bool
    execution_time: float
    key_insight: str
    code_snippets: List[str]
    prompt_techniques: List[str]


class TrajectorySummarizer:
    """Extract summaries from trajectory data"""
    
    def __init__(self, llm_api_key: str = None):
        self.llm_api_key = llm_api_key
    
    def summarize_trajectory(self, trajectory: Dict) -> TrajectorySummary:
        """Generate a comprehensive summary of a trajectory"""
        
        steps = trajectory.get('steps', [])
        task_type = trajectory.get('trajectory_type', 'unknown')
        task_desc = trajectory.get('task_description', '')
        
        total_duration = trajectory.get('total_execution_time', 0)
        total_steps = len(steps)
        successful_steps = sum(1 for s in steps if s.get('reward', 0) > 0.5)
        success_rate = successful_steps / total_steps if total_steps > 0 else 0
        
        key_actions = list(set(s.get('action', '') for s in steps if s.get('action')))
        
        patterns_detected = self._detect_patterns(steps)
        techniques_used = self._extract_techniques(steps)
        
        outcome_summary = self._generate_outcome_summary(trajectory, success_rate)
        code_patterns = self._extract_code_patterns(steps)
        languages_used = self._extract_languages(steps)
        
        efficiency_score = self._calculate_efficiency(steps, success_rate)
        complexity_score = self._calculate_complexity(steps)
        
        return TrajectorySummary(
            trajectory_id=trajectory.get('id', 0),
            task_description=task_desc,
            task_type=task_type,
            total_steps=total_steps,
            total_duration=total_duration,
            success_rate=success_rate,
            key_actions=key_actions,
            patterns_detected=patterns_detected,
            techniques_used=techniques_used,
            outcome_summary=outcome_summary,
            code_patterns=code_patterns,
            languages_used=languages_used,
            efficiency_score=efficiency_score,
            complexity_score=complexity_score
        )
    
    def summarize_batch(self, trajectories: List[Dict]) -> List[TrajectorySummary]:
        """Summarize multiple trajectories"""
        return [self.summarize_trajectory(t) for t in trajectories]
    
    def _detect_patterns(self, steps: List[Dict]) -> List[str]:
        """Detect common patterns in steps"""
        patterns = []
        actions = [s.get('action', '') for s in steps]
        
        if any('code_generation' in a for a in actions):
            patterns.append("code_generation")
        if any('code_execution' in a for a in actions):
            patterns.append("code_execution")
        if any('code_analysis' in a for a in actions):
            patterns.append("code_analysis")
        if any('iteration' in str(s.get('metadata', {})).lower() for s in steps):
            patterns.append("iterative_refinement")
        if any('error' in str(s.get('observation', '')).lower() for s in steps):
            patterns.append("error_handling")
        
        unique_patterns = list(set(patterns))
        return unique_patterns if unique_patterns else ["general_task"]
    
    def _extract_techniques(self, steps: List[Dict]) -> List[str]:
        """Extract techniques used in the trajectory"""
        techniques = []
        
        for step in steps:
            thought = step.get('thought', '')
            action = step.get('action', '')
            action_input = step.get('action_input', {})
            
            if 'generate' in action.lower() or 'write' in action.lower():
                techniques.append("code_generation")
            if 'analyze' in action.lower() or 'review' in action.lower():
                techniques.append("code_analysis")
            if 'test' in action.lower() or 'verify' in action.lower():
                techniques.append("testing")
            if 'refactor' in action.lower() or 'improve' in action.lower():
                techniques.append("refactoring")
            if 'explain' in action.lower() or 'describe' in action.lower():
                techniques.append("explanation")
        
        return list(set(techniques))
    
    def _generate_outcome_summary(self, trajectory: Dict, success_rate: float) -> str:
        """Generate a text summary of the outcome"""
        steps = trajectory.get('steps', [])
        total_steps = len(steps)
        successful_steps = sum(1 for s in steps if s.get('reward', 0) > 0.5)
        
        if success_rate >= 0.9:
            outcome = "highly successful"
        elif success_rate >= 0.7:
            outcome = "mostly successful"
        elif success_rate >= 0.5:
            outcome = "partially successful"
        else:
            outcome = "needs improvement"
        
        task_type = trajectory.get('trajectory_type', 'task')
        
        return f"A {task_type} with {total_steps} steps, {successful_steps} successful. Overall outcome: {outcome}."
    
    def _extract_code_patterns(self, steps: List[Dict]) -> List[str]:
        """Extract code patterns from steps"""
        patterns = []
        
        for step in steps:
            action_result = step.get('action_result', {})
            observation = step.get('observation', '')
            
            if isinstance(action_result, dict):
                code_length = action_result.get('code_length', 0)
                if code_length > 0:
                    patterns.append(f"generated_{code_length}_chars")
            
            if 'function' in observation.lower():
                patterns.append("function_definition")
            if 'class' in observation.lower():
                patterns.append("class_definition")
            if 'test' in observation.lower():
                patterns.append("test_code")
            if 'api' in observation.lower() or 'endpoint' in observation.lower():
                patterns.append("api_implementation")
        
        return list(set(patterns))
    
    def _extract_languages(self, steps: List[Dict]) -> List[str]:
        """Extract programming languages used"""
        languages = set()
        
        for step in steps:
            action_input = step.get('action_input', {})
            if isinstance(action_input, dict):
                if action_input.get('language'):
                    languages.add(action_input.get('language'))
        
        return list(languages)
    
    def _calculate_efficiency(self, steps: List[Dict], success_rate: float) -> float:
        """Calculate efficiency score (0-1)"""
        if not steps:
            return 0.0
        
        avg_time = sum(s.get('execution_time', 0) for s in steps) / len(steps)
        
        time_score = max(0, 1 - avg_time / 30) if avg_time > 0 else 1.0
        
        efficiency = (success_rate * 0.7 + time_score * 0.3)
        
        return round(efficiency, 3)
    
    def _calculate_complexity(self, steps: List[Dict]) -> float:
        """Calculate complexity score (0-1)"""
        if not steps:
            return 0.0
        
        total_steps = len(steps)
        total_reward = sum(s.get('reward', 0) for s in steps)
        
        complexity = min(1.0, (total_steps * 0.3 + total_reward * 0.7) / 10)
        
        return round(complexity, 3)
    
    def generate_narrative(self, summary: TrajectorySummary) -> str:
        """Generate a narrative description of the trajectory"""
        narrative = f"""## Trajectory Summary

**Task**: {summary.task_description}
**Type**: {summary.task_type}
**Duration**: {summary.total_duration:.2f} seconds
**Steps**: {summary.total_steps}
**Success Rate**: {summary.success_rate:.1%}

### Key Actions
{', '.join(summary.key_actions)}

### Patterns Detected
{', '.join(summary.patterns_detected)}

### Techniques Used
{', '.join(summary.techniques_used)}

### Languages
{', '.join(summary.languages_used)}

### Code Patterns
{', '.join(summary.code_patterns)}

### Efficiency Score
{summary.efficiency_score}/1.0

### Complexity Score  
{summary.complexity_score}/1.0

### Outcome
{summary.outcome_summary}
"""
        return narrative
    
    def extract_key_learnings(self, trajectories: List[Dict]) -> List[Dict]:
        """Extract key learnings from multiple trajectories"""
        learnings = []
        
        for trajectory in trajectories:
            summary = self.summarize_trajectory(trajectory)
            
            learning = {
                "trajectory_id": summary.trajectory_id,
                "task_type": summary.task_type,
                "success_rate": summary.success_rate,
                "effective_patterns": summary.patterns_detected,
                "techniques_used": summary.techniques_used,
                "efficiency_score": summary.efficiency_score,
                "recommendations": self._generate_recommendations(summary)
            }
            
            learnings.append(learning)
        
        return learnings
    
    def _generate_recommendations(self, summary: TrajectorySummary) -> List[str]:
        """Generate recommendations based on summary"""
        recommendations = []
        
        if summary.efficiency_score < 0.5:
            recommendations.append("Consider optimizing execution time")
        if summary.success_rate < 0.7:
            recommendations.append("Review failed steps for improvement")
        if summary.complexity_score > 0.8:
            recommendations.append("Consider breaking down into smaller steps")
        if len(summary.techniques_used) < 2:
            recommendations.append("Explore more varied techniques")
        
        if not recommendations:
            recommendations.append("Current approach is effective")
        
        return recommendations


class StepSummarizer:
    """Summarize individual steps"""
    
    def summarize_step(self, step: Dict) -> StepSummary:
        """Generate summary of a single step"""
        
        return StepSummary(
            step_id=step.get('id', 0),
            step_order=step.get('step_order', 0),
            step_name=step.get('step_name', ''),
            action_type=step.get('action', ''),
            primary_goal=step.get('thought', '')[:100] if step.get('thought') else '',
            tools_used=self._extract_tools(step),
            success=step.get('reward', 0) > 0.5,
            execution_time=step.get('execution_time', 0),
            key_insight=self._extract_insight(step),
            code_snippets=self._extract_code(step),
            prompt_techniques=self._extract_prompt_techniques(step)
        )
    
    def _extract_tools(self, step: Dict) -> List[str]:
        """Extract tools used in step"""
        tools = []
        action = step.get('action', '')
        action_input = step.get('action_input', {})
        
        if 'code' in action.lower() and 'generation' in action.lower():
            tools.append("code_generation")
        if 'code' in action.lower() and 'execution' in action.lower():
            tools.append("code_execution")
        if 'analysis' in action.lower():
            tools.append("code_analysis")
        if 'file' in action.lower():
            tools.append("file_operation")
        if isinstance(action_input, dict):
            if action_input.get('language'):
                tools.append(f"{action_input.get('language')}_compiler")
        
        return list(set(tools))
    
    def _extract_insight(self, step: Dict) -> str:
        """Extract key insight from step"""
        observation = step.get('observation', '')
        reward = step.get('reward', 0)
        
        if reward > 0.8:
            return f"Successfully completed: {observation[:100]}"
        elif reward > 0.5:
            return f"Partially completed: {observation[:100]}"
        else:
            return f"Needs improvement: {observation[:100]}"
    
    def _extract_code(self, step: Dict) -> List[str]:
        """Extract code snippets from step"""
        snippets = []
        action_result = step.get('action_result', {})
        
        if isinstance(action_result, dict):
            code_length = action_result.get('code_length', 0)
            if code_length > 0:
                snippets.append(f"[Generated {code_length} characters of code]")
        
        return snippets
    
    def _extract_prompt_techniques(self, step: Dict) -> List[str]:
        """Extract prompt engineering techniques used"""
        techniques = []
        thought = step.get('thought', '')
        
        if 'think' in thought.lower() or 'consider' in thought.lower():
            techniques.append("reasoning_before_action")
        if 'step' in thought.lower() or 'first' in thought.lower():
            techniques.append("step_by_step")
        if 'check' in thought.lower() or 'verify' in thought.lower():
            techniques.append("verification")
        
        return techniques
