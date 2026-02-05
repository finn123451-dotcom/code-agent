"""
Test for summarization and RL modules
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def test_trajectory_summarizer():
    """Test trajectory summarizer"""
    from src.summarization import TrajectorySummarizer, StepSummarizer
    
    sample_trajectory = {
        "id": 1,
        "task_description": "Generate a factorial function",
        "trajectory_type": "task",
        "initial_prompt": "Write a Python factorial function",
        "total_execution_time": 2.5,
        "steps": [
            {
                "id": 1,
                "step_order": 1,
                "step_name": "code_generation",
                "thought": "Generate Python code for factorial",
                "action": "code_generation",
                "action_input": {"prompt": "Write factorial", "language": "python"},
                "observation": "Generated 150 characters of code",
                "reward": 0.9,
                "execution_time": 1.0
            },
            {
                "id": 2,
                "step_order": 2,
                "step_name": "code_execution",
                "thought": "Execute the generated code",
                "action": "code_execution",
                "action_input": {"language": "python"},
                "observation": "Code executed successfully",
                "reward": 1.0,
                "execution_time": 1.5
            }
        ]
    }
    
    summarizer = TrajectorySummarizer()
    summary = summarizer.summarize_trajectory(sample_trajectory)
    
    assert summary.trajectory_id == 1
    assert summary.task_type == "task"
    assert summary.total_steps == 2
    assert len(summary.key_actions) >= 2
    assert len(summary.patterns_detected) > 0
    
    print("  ✓ TrajectorySummarizer test passed")
    
    step_summarizer = StepSummarizer()
    step_summary = step_summarizer.summarize_step(sample_trajectory["steps"][0])
    
    assert step_summary.step_name == "code_generation"
    assert step_summary.success == True
    
    print("  ✓ StepSummarizer test passed")
    
    return True


def test_rl_dataset_generator():
    """Test RL dataset generator"""
    from src.rl import RLDatasetGenerator, TrajectorySampler, DatasetFormat
    
    sample_trajectories = [
        {
            "id": 1,
            "task_description": "Generate factorial",
            "trajectory_type": "task",
            "initial_prompt": "Write factorial",
            "total_execution_time": 2.0,
            "total_reward": 0.85,
            "steps": [
                {
                    "id": 1,
                    "step_order": 1,
                    "action": "code_generation",
                    "action_input": {"language": "python", "code": "def fact(n): return 1"},
                    "reward": 0.9,
                    "execution_time": 1.0,
                    "thought": "Generate code",
                    "observation": "Generated code"
                },
                {
                    "id": 2,
                    "step_order": 2,
                    "action": "code_execution",
                    "action_input": {"language": "python"},
                    "reward": 0.8,
                    "execution_time": 1.0,
                    "thought": "Execute code",
                    "observation": "Executed successfully"
                }
            ]
        },
        {
            "id": 2,
            "task_description": "Generate fibonacci",
            "trajectory_type": "task",
            "initial_prompt": "Write fibonacci",
            "total_execution_time": 3.0,
            "total_reward": 0.75,
            "steps": [
                {
                    "id": 3,
                    "step_order": 1,
                    "action": "code_generation",
                    "action_input": {"language": "python", "code": "def fib(n): return n"},
                    "reward": 0.7,
                    "execution_time": 1.5,
                    "thought": "Generate fibonacci",
                    "observation": "Generated fibonacci"
                },
                {
                    "id": 4,
                    "step_order": 2,
                    "action": "code_execution",
                    "action_input": {"language": "python"},
                    "reward": 0.8,
                    "execution_time": 1.5,
                    "thought": "Execute fibonacci",
                    "observation": "Execution successful"
                }
            ]
        }
    ]
    
    generator = RLDatasetGenerator()
    
    trajectory_dataset = generator.generate_dataset(
        sample_trajectories, 
        format=DatasetFormat.TRAJECTORY
    )
    assert len(trajectory_dataset) == 2
    print("  ✓ Trajectory format test passed")
    
    sft_dataset = generator.generate_dataset(
        sample_trajectories,
        format=DatasetFormat.SFT
    )
    assert len(sft_dataset) == 2
    assert len(sft_dataset[0]["messages"]) > 0
    print("  ✓ SFT format test passed")
    
    ppo_dataset = generator.generate_dataset(
        sample_trajectories,
        format=DatasetFormat.PPO
    )
    assert len(ppo_dataset) == 4
    assert ppo_dataset[0]["reward"] is not None
    print("  ✓ PPO format test passed")
    
    sampler = TrajectorySampler()
    sampled = sampler.sample_balanced(sample_trajectories, n=2)
    assert len(sampled) == 2
    print("  ✓ TrajectorySampler test passed")
    
    return True


def test_dataset_export():
    """Test dataset export functionality"""
    from src.rl import RLDatasetGenerator, DatasetFormat
    
    sample_trajectories = [
        {
            "id": 1,
            "task_description": "Test task",
            "trajectory_type": "task",
            "initial_prompt": "Test",
            "total_execution_time": 1.0,
            "total_reward": 0.8,
            "steps": [
                {
                    "id": 1,
                    "step_order": 1,
                    "action": "test_action",
                    "action_input": {},
                    "reward": 0.8,
                    "execution_time": 0.5,
                    "thought": "Test thought",
                    "observation": "Test observation"
                }
            ]
        }
    ]
    
    generator = RLDatasetGenerator()
    samples = generator.generate_dataset(sample_trajectories, format=DatasetFormat.TRAJECTORY)
    
    json_content = generator.export_dataset(samples, format="json")
    assert "trajectory_id" in json_content
    print("  ✓ Dataset export test passed")
    
    return True


def run_all_tests():
    print("\n" + "="*60)
    print("Self-Evolving Code Agent - Summarization & RL Tests")
    print("="*60 + "\n")
    
    tests = [
        ("TrajectorySummarizer", test_trajectory_summarizer),
        ("RLDatasetGenerator", test_rl_dataset_generator),
        ("DatasetExport", test_dataset_export),
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
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
