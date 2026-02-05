"""
Test for trajectory clustering module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def test_trajectory_clustering():
    """Test trajectory clustering"""
    from src.summarization import TrajectoryClusterer, StrategyRecommender
    
    sample_trajectories = [
        {
            "id": 1,
            "task_description": "Generate factorial function",
            "trajectory_type": "task",
            "total_reward": 0.9,
            "total_execution_time": 2.0,
            "steps": [
                {
                    "id": 1,
                    "step_order": 1,
                    "action": "code_generation",
                    "action_input": {"language": "python"},
                    "reward": 0.9,
                    "execution_time": 1.0
                },
                {
                    "id": 2,
                    "step_order": 2,
                    "action": "code_execution",
                    "action_input": {"language": "python"},
                    "reward": 1.0,
                    "execution_time": 1.0
                }
            ]
        },
        {
            "id": 2,
            "task_description": "Generate fibonacci function",
            "trajectory_type": "task",
            "total_reward": 0.85,
            "total_execution_time": 3.0,
            "steps": [
                {
                    "id": 3,
                    "step_order": 1,
                    "action": "code_generation",
                    "action_input": {"language": "python"},
                    "reward": 0.85,
                    "execution_time": 1.5
                },
                {
                    "id": 4,
                    "step_order": 2,
                    "action": "code_execution",
                    "action_input": {"language": "python"},
                    "reward": 0.9,
                    "execution_time": 1.5
                }
            ]
        },
        {
            "id": 3,
            "task_description": "Generate complex algorithm",
            "trajectory_type": "task",
            "total_reward": 0.4,
            "total_execution_time": 15.0,
            "steps": [
                {
                    "id": 5,
                    "step_order": 1,
                    "action": "code_generation",
                    "action_input": {"language": "python"},
                    "reward": 0.4,
                    "execution_time": 8.0
                },
                {
                    "id": 6,
                    "step_order": 2,
                    "action": "code_execution",
                    "action_input": {"language": "python"},
                    "reward": 0.3,
                    "execution_time": 7.0
                }
            ]
        }
    ]
    
    clusterer = TrajectoryClusterer(n_clusters=2, embedding_dim=20)
    clusters = clusterer.fit(sample_trajectories, n_clusters=2)
    
    assert len(clusters) > 0, "Should create at least one cluster"
    print(f"  ✓ Created {len(clusters)} clusters")
    
    for cluster in clusters:
        assert cluster.cluster_id is not None
        assert len(cluster.trajectory_ids) > 0
        print(f"    Cluster {cluster.cluster_id}: {len(cluster.trajectory_ids)} trajectories, avg_reward: {cluster.avg_reward:.2f}")
    
    print("  ✓ Cluster statistics computed")
    
    best_cluster = clusterer.get_best_cluster()
    assert best_cluster is not None
    assert best_cluster.avg_reward >= 0.8
    print(f"  ✓ Best cluster: {best_cluster.cluster_id} with avg_reward: {best_cluster.avg_reward:.2f}")
    
    efficient_cluster = clusterer.get_most_efficient_cluster()
    assert efficient_cluster is not None
    print(f"  ✓ Most efficient cluster: {efficient_cluster.cluster_id}")
    
    analysis = clusterer.analyze_cluster_evolution()
    assert "best_for_exploitation" in analysis
    print("  ✓ Cluster evolution analysis completed")
    
    return True


def test_strategy_recommender():
    """Test strategy recommender"""
    from src.summarization import TrajectoryClusterer, StrategyRecommender
    
    clusterer = TrajectoryClusterer(n_clusters=3, embedding_dim=20)
    
    trajectories = [
        {
            "id": 1,
            "task_description": "Task 1",
            "trajectory_type": "task",
            "total_reward": 0.9,
            "total_execution_time": 2.0,
            "steps": [
                {"id": 1, "step_order": 1, "action": "code_generation", "action_input": {}, "reward": 0.9},
                {"id": 2, "step_order": 2, "action": "code_execution", "action_input": {}, "reward": 1.0}
            ]
        },
        {
            "id": 2,
            "task_description": "Task 2",
            "trajectory_type": "task",
            "total_reward": 0.3,
            "total_execution_time": 10.0,
            "steps": [
                {"id": 3, "step_order": 1, "action": "code_generation", "action_input": {}, "reward": 0.3},
                {"id": 4, "step_order": 2, "action": "debug", "action_input": {}, "reward": 0.2}
            ]
        }
    ]
    
    clusterer.fit(trajectories)
    
    recommender = StrategyRecommender(clusterer)
    
    recommender.learn_from_trajectory(trajectories[0], outcome="success")
    recommender.learn_from_trajectory(trajectories[1], outcome="failure")
    
    recommendation = recommender.recommend(task_type="task")
    
    assert "recommended_cluster" in recommendation
    assert "suggested_actions" in recommendation
    print(f"  ✓ Recommendation: cluster {recommendation['recommended_cluster']}")
    print(f"    Suggested actions: {recommendation['suggested_actions']}")
    
    exploration = recommender.get_exploration_suggestion()
    assert "mode" in exploration
    print(f"  ✓ Exploration suggestion: {exploration['mode']}")
    
    return True


def test_cluster_prediction():
    """Test cluster prediction for new trajectory"""
    from src.summarization import TrajectoryClusterer
    
    trajectories = [
        {
            "id": 1,
            "task_description": "Task 1",
            "trajectory_type": "task",
            "total_reward": 0.9,
            "total_execution_time": 2.0,
            "steps": [
                {"id": 1, "step_order": 1, "action": "code_generation", "action_input": {"language": "python"}, "reward": 0.9}
            ]
        }
    ]
    
    clusterer = TrajectoryClusterer(n_clusters=2, embedding_dim=20)
    clusterer.fit(trajectories)
    
    new_traj = {
        "id": 99,
        "task_description": "New task",
        "trajectory_type": "task",
        "total_reward": 0.8,
        "total_execution_time": 3.0,
        "steps": [
            {"id": 100, "step_order": 1, "action": "code_generation", "action_input": {"language": "python"}, "reward": 0.8}
        ]
    }
    
    cluster_id = clusterer.predict(new_traj)
    assert cluster_id is not None
    print(f"  ✓ Predicted cluster: {cluster_id}")
    
    return True


def run_all_tests():
    print("\n" + "="*60)
    print("Trajectory Clustering Tests")
    print("="*60 + "\n")
    
    tests = [
        ("Trajectory Clustering", test_trajectory_clustering),
        ("Strategy Recommender", test_strategy_recommender),
        ("Cluster Prediction", test_cluster_prediction),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"  ✓ {name}")
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
