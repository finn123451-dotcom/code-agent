"""
Trajectory Clustering Module
Clusters trajectories to discover patterns and enable strategy recommendation
"""
import json
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib


@dataclass
class Cluster:
    """Cluster of similar trajectories"""
    cluster_id: str
    centroid: List[float]
    trajectory_ids: List[int]
    avg_reward: float
    avg_execution_time: float
    success_rate: float
    common_actions: List[str]
    common_patterns: List[str]
    languages: List[str]
    task_types: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ClusterStats:
    """Statistics for a cluster"""
    cluster_id: str
    trajectory_count: int
    reward_distribution: Dict[str, int]
    action_frequency: Dict[str, int]
    pattern_frequency: Dict[str, int]
    language_distribution: Dict[str, int]
    avg_success_rate: float
    avg_efficiency: float
    recommendations: List[str]


class TrajectoryClusterer:
    """
    Cluster trajectories based on embeddings and metadata
    
    Benefits for Self-Evolution:
    - Discover successful patterns automatically
    - Group similar strategies for recommendation
    - Identify anomalies for improvement
    - Guide evolution towards high-reward regions
    """
    
    def __init__(self, n_clusters: int = 5, embedding_dim: int = 1536):
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.clusters: Dict[str, Cluster] = {}
        self.trajectory_to_cluster: Dict[int, str] = {}
        self.centroids: np.ndarray = None
        self.is_fitted = False
        self.feature_extractor = None
    
    def set_feature_extractor(self, extractor):
        """Set feature extractor for trajectory features"""
        self.feature_extractor = extractor
    
    def extract_features(self, trajectory: Dict) -> np.ndarray:
        """Extract features from a trajectory for clustering"""
        features = []
        
        steps = trajectory.get('steps', [])
        
        reward = trajectory.get('total_reward', 0.5)
        features.append(reward)
        
        exec_time = trajectory.get('total_execution_time', 0)
        features.append(min(exec_time / 60.0, 1.0))
        
        n_steps = len(steps)
        features.append(min(n_steps / 20.0, 1.0))
        
        success_rate = sum(s.get('reward', 0) for s in steps) / n_steps if n_steps > 0 else 0.5
        features.append(success_rate)
        
        actions = list(set(s.get('action', '') for s in steps if s.get('action')))
        action_count = len(actions)
        features.append(min(action_count / 10.0, 1.0))
        
        task_type = trajectory.get('trajectory_type', 'unknown')
        type_hash = hash(task_type) % 100 / 100.0
        features.append(type_hash)
        
        language = 'unknown'
        for step in steps:
            action_input = step.get('action_input', {})
            if isinstance(action_input, dict) and action_input.get('language'):
                language = action_input.get('language')
                break
        lang_hash = hash(language) % 100 / 100.0
        features.append(lang_hash)
        
        latent_space = trajectory.get('latent_space', [])
        if latent_space:
            embedding = latent_space[0].get('embedding_vector', [])
            if len(embedding) >= self.embedding_dim:
                features.extend(embedding[:self.embedding_dim - len(features)])
            else:
                features.extend([0.0] * (self.embedding_dim - len(features)))
        else:
            features.extend([0.0] * (self.embedding_dim - len(features)))
        
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        return np.array(features[:self.embedding_dim])
    
    def fit(self, trajectories: List[Dict], n_clusters: int = None) -> List[Cluster]:
        """Cluster trajectories using K-Means"""
        if n_clusters:
            self.n_clusters = n_clusters
        
        if not trajectories:
            return []
        
        features_list = []
        for traj in trajectories:
            features = self.extract_features(traj)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        n_clusters = min(self.n_clusters, len(trajectories))
        if n_clusters < 1:
            n_clusters = 1
        
        self.centroids = np.random.randn(n_clusters, self.embedding_dim)
        self.centroids = self.centroids / np.linalg.norm(self.centroids, axis=1, keepdims=True)
        
        for _ in range(100):
            distances = self._compute_distances(features_array, self.centroids)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(self.centroids)
            for i in range(n_clusters):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centroids[i] = np.mean(features_array[mask], axis=0)
                    norm = np.linalg.norm(new_centroids[i])
                    if norm > 0:
                        new_centroids[i] = new_centroids[i] / norm
            
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        
        self.clusters = {}
        self.trajectory_to_cluster = {}
        
        for i, traj in enumerate(trajectories):
            traj_id = traj.get('id', i)
            traj_features = features_list[i]
            cluster_id = str(labels[i])
            
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = self._create_cluster(cluster_id, self.centroids[labels[i]])
            
            self.clusters[cluster_id].trajectory_ids.append(traj_id)
            self.trajectory_to_cluster[traj_id] = cluster_id
        
        self._update_cluster_stats(trajectories)
        
        self.is_fitted = True
        
        return list(self.clusters.values())
    
    def _compute_distances(self, features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute cosine distances between features and centroids"""
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        distances = 1.0 - np.dot(features, centroids.T)
        return distances
    
    def _create_cluster(self, cluster_id: str, centroid: np.ndarray) -> Cluster:
        """Create a new cluster"""
        return Cluster(
            cluster_id=cluster_id,
            centroid=centroid.tolist() if isinstance(centroid, np.ndarray) else centroid,
            trajectory_ids=[],
            avg_reward=0.0,
            avg_execution_time=0.0,
            success_rate=0.0,
            common_actions=[],
            common_patterns=[],
            languages=[],
            task_types=[]
        )
    
    def _update_cluster_stats(self, trajectories: List[Dict]):
        """Update cluster statistics"""
        for cluster_id, cluster in self.clusters.items():
            cluster_trajs = [t for t in trajectories if t.get('id') in cluster.trajectory_ids]
            
            if not cluster_trajs:
                continue
            
            rewards = [t.get('total_reward', 0) for t in cluster_trajs]
            cluster.avg_reward = np.mean(rewards) if rewards else 0.0
            
            exec_times = [t.get('total_execution_time', 0) for t in cluster_trajs]
            cluster.avg_execution_time = np.mean(exec_times) if exec_times else 0.0
            
            success_count = sum(1 for t in cluster_trajs if t.get('total_reward', 0) > 0.5)
            cluster.success_rate = success_count / len(cluster_trajs) if cluster_trajs else 0.0
            
            action_counter = defaultdict(int)
            pattern_counter = defaultdict(int)
            lang_counter = defaultdict(int)
            type_counter = defaultdict(int)
            
            for traj in cluster_trajs:
                steps = traj.get('steps', [])
                for step in steps:
                    action = step.get('action', '')
                    if action:
                        action_counter[action] += 1
                
                action_input = steps[0].get('action_input', {}) if steps else {}
                if isinstance(action_input, dict):
                    lang = action_input.get('language', 'unknown')
                    if lang:
                        lang_counter[lang] += 1
            
            cluster.common_actions = sorted(action_counter.keys(), 
                                         key=lambda x: action_counter[x], reverse=True)[:5]
            cluster.common_patterns = sorted(pattern_counter.keys(),
                                          key=lambda x: pattern_counter[x], reverse=True)[:5]
            cluster.languages = list(lang_counter.keys())
            cluster.task_types = list(type_counter.keys())
            
            cluster.updated_at = datetime.now()
    
    def predict(self, trajectory: Dict) -> str:
        """Predict cluster for a trajectory"""
        if not self.is_fitted:
            raise ValueError("Clusterer not fitted. Call fit() first.")
        
        features = self.extract_features(trajectory)
        features = features / np.linalg.norm(features)
        
        distances = 1.0 - np.dot(features, self.centroids.T)
        cluster_id = str(np.argmin(distances))
        
        return cluster_id
    
    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get cluster by ID"""
        return self.clusters.get(cluster_id)
    
    def get_all_clusters(self) -> List[Cluster]:
        """Get all clusters"""
        return list(self.clusters.values())
    
    def get_cluster_for_trajectory(self, trajectory_id: int) -> Optional[Cluster]:
        """Get cluster for a trajectory ID"""
        cluster_id = self.trajectory_to_cluster.get(trajectory_id)
        if cluster_id:
            return self.clusters.get(cluster_id)
        return None
    
    def get_similar_trajectories(self, trajectory: Dict, limit: int = 5) -> List[Tuple[Dict, float]]:
        """Find similar trajectories in the same cluster"""
        cluster_id = self.predict(trajectory)
        cluster = self.clusters.get(cluster_id)
        
        if not cluster:
            return []
        
        traj_features = self.extract_features(trajectory)
        traj_features = traj_features / np.linalg.norm(traj_features)
        
        similarities = []
        return self.clusters[cluster_id].trajectory_ids[:limit]
    
    def get_best_cluster(self) -> Optional[Cluster]:
        """Get cluster with highest average reward"""
        if not self.clusters:
            return None
        
        return max(self.clusters.values(), key=lambda c: c.avg_reward)
    
    def get_most_efficient_cluster(self) -> Optional[Cluster]:
        """Get cluster with best reward/time ratio"""
        if not self.clusters:
            return None
        
        def efficiency(c):
            if c.avg_execution_time == 0:
                return c.avg_reward
            return c.avg_reward / (c.avg_execution_time + 1)
        
        return max(self.clusters.values(), key=efficiency)
    
    def get_cluster_stats(self, cluster_id: str) -> Optional[ClusterStats]:
        """Get detailed statistics for a cluster"""
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return None
        
        return ClusterStats(
            cluster_id=cluster_id,
            trajectory_count=len(cluster.trajectory_ids),
            reward_distribution={"high": 0, "medium": 0, "low": 0},
            action_frequency={},
            pattern_frequency={},
            language_distribution={},
            avg_success_rate=cluster.success_rate,
            avg_efficiency=cluster.avg_reward / (cluster.avg_execution_time + 1),
            recommendations=self._generate_recommendations(cluster)
        )
    
    def _generate_recommendations(self, cluster: Cluster) -> List[str]:
        """Generate recommendations based on cluster characteristics"""
        recommendations = []
        
        if cluster.avg_reward > 0.8:
            recommendations.append("This cluster has high success rate. Good for similar tasks.")
        elif cluster.avg_reward < 0.5:
            recommendations.append("This cluster has low success rate. Consider avoiding strategies from here.")
        
        if cluster.avg_execution_time < 5:
            recommendations.append("Efficient cluster with fast execution times.")
        elif cluster.avg_execution_time > 20:
            recommendations.append("Slower cluster. May need optimization.")
        
        if cluster.common_actions:
            recommendations.append(f"Common actions: {', '.join(cluster.common_actions[:3])}")
        
        if cluster.languages:
            recommendations.append(f"Primary languages: {', '.join(cluster.languages)}")
        
        return recommendations
    
    def analyze_cluster_evolution(self) -> Dict:
        """Analyze evolution potential of each cluster"""
        analysis = {
            "best_for_exploration": None,
            "best_for_exploitation": None,
            "potential_improvement": [],
            "anomalies": []
        }
        
        if not self.clusters:
            return analysis
        
        sorted_by_reward = sorted(self.clusters.values(), key=lambda c: c.avg_reward, reverse=True)
        
        analysis["best_for_exploitation"] = sorted_by_reward[0].cluster_id if sorted_by_reward else None
        
        if len(sorted_by_reward) > 1:
            analysis["best_for_exploration"] = sorted_by_reward[-1].cluster_id
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.success_rate < 0.3:
                analysis["anomalies"].append({
                    "cluster_id": cluster_id,
                    "reason": "Low success rate",
                    "avg_reward": cluster.avg_reward
                })
            
            if cluster.avg_execution_time > 30 and cluster.avg_reward > 0.5:
                analysis["potential_improvement"].append({
                    "cluster_id": cluster_id,
                    "reason": "High reward but slow execution",
                    "suggestion": "Optimize execution time"
                })
        
        return analysis
    
    def merge_clusters(self, cluster_ids: List[str]) -> Optional[Cluster]:
        """Merge multiple clusters into one"""
        if not all(cid in self.clusters for cid in cluster_ids):
            return None
        
        merged_trajs = []
        centroid_sum = np.zeros(self.embedding_dim)
        
        for cid in cluster_ids:
            cluster = self.clusters[cid]
            merged_trajs.extend(cluster.trajectory_ids)
            centroid = np.array(cluster.centroid) if isinstance(cluster.centroid, list) else cluster.centroid
            centroid_sum += centroid
        
        new_cluster_id = str(uuid.uuid4())[:8]
        new_centroid = (centroid_sum / len(cluster_ids)).tolist()
        
        new_cluster = self._create_cluster(new_cluster_id, new_centroid)
        new_cluster.trajectory_ids = merged_trajs
        
        self._update_cluster_stats([t for t in [] for _ in []])
        self.clusters[new_cluster_id] = new_cluster
        
        for cid in cluster_ids:
            del self.clusters[cid]
        
        return new_cluster
    
    def export_clustering(self) -> Dict:
        """Export clustering results"""
        return {
            "n_clusters": len(self.clusters),
            "clusters": {
                cid: {
                    "cluster_id": c.cluster_id,
                    "centroid": c.centroid,
                    "trajectory_count": len(c.trajectory_ids),
                    "avg_reward": c.avg_reward,
                    "success_rate": c.success_rate,
                    "common_actions": c.common_actions,
                    "languages": c.languages
                }
                for cid, c in self.clusters.items()
            },
            "trajectory_to_cluster": self.trajectory_to_cluster
        }
    
    def load_clustering(self, data: Dict):
        """Load clustering results"""
        self.clusters = {}
        self.trajectory_to_cluster = data.get("trajectory_to_cluster", {})
        
        for cid, cdata in data.get("clusters", {}).items():
            cluster = Cluster(
                cluster_id=cdata["cluster_id"],
                centroid=cdata["centroid"],
                trajectory_ids=[],
                avg_reward=cdata.get("avg_reward", 0),
                avg_execution_time=cdata.get("avg_execution_time", 0),
                success_rate=cdata.get("success_rate", 0),
                common_actions=cdata.get("common_actions", []),
                common_patterns=cdata.get("common_patterns", []),
                languages=cdata.get("languages", []),
                task_types=cdata.get("task_types", [])
            )
            self.clusters[cid] = cluster
        
        self.is_fitted = True


class StrategyRecommender:
    """
    Recommend strategies based on clustering and trajectory analysis
    
    Uses clusters to:
    - Find similar successful strategies
    - Avoid previously failed approaches
    - Guide exploration vs exploitation
    """
    
    def __init__(self, clusterer: TrajectoryClusterer = None):
        self.clusterer = clusterer or TrajectoryClusterer()
        self.success_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.failure_patterns: Dict[str, List[Dict]] = defaultdict(list)
    
    def learn_from_trajectory(self, trajectory: Dict, outcome: str = "success"):
        """Learn from trajectory outcome"""
        traj_id = trajectory.get('id')
        cluster_id = self.clusterer.predict(trajectory) if self.clusterer.is_fitted else "unknown"
        
        pattern = {
            "trajectory_id": traj_id,
            "task_type": trajectory.get('trajectory_type'),
            "reward": trajectory.get('total_reward', 0),
            "actions": list(set(s.get('action') for s in trajectory.get('steps', [])))
        }
        
        if outcome == "success":
            self.success_patterns[cluster_id].append(pattern)
        else:
            self.failure_patterns[cluster_id].append(pattern)
    
    def recommend(self, task_type: str = None, context: Dict = None) -> Dict:
        """Generate strategy recommendations"""
        recommendations = {
            "recommended_cluster": None,
            "suggested_actions": [],
            "actions_to_avoid": [],
            "confidence": 0.0,
            "reasoning": []
        }
        
        if not self.clusterer.is_fitted:
            recommendations["reasoning"].append("No clustering data available")
            return recommendations
        
        best_cluster = self.clusterer.get_best_cluster()
        if best_cluster:
            recommendations["recommended_cluster"] = best_cluster.cluster_id
            recommendations["suggested_actions"] = best_cluster.common_actions[:3]
            recommendations["confidence"] = best_cluster.avg_reward
            recommendations["reasoning"].append(
                f"Cluster {best_cluster.cluster_id} has highest avg reward: {best_cluster.avg_reward:.2f}"
            )
        
        for cid in self.success_patterns:
            patterns = self.success_patterns[cid]
            high_reward = [p for p in patterns if p.get('reward', 0) > 0.7]
            if high_reward:
                recommendations["suggested_actions"].extend(
                    action for p in high_reward for action in p.get('actions', [])
                )
        
        for cid in self.failure_patterns:
            patterns = self.failure_patterns[cid]
            low_reward = [p for p in patterns if p.get('reward', 0) < 0.3]
            if low_reward:
                recommendations["actions_to_avoid"].extend(
                    action for p in low_reward for action in p.get('actions', [])
                )
        
        recommendations["suggested_actions"] = list(set(recommendations["suggested_actions"]))[:5]
        recommendations["actions_to_avoid"] = list(set(recommendations["actions_to_avoid"]))[:5]
        
        return recommendations
    
    def get_exploration_suggestion(self) -> Dict:
        """Suggest exploration vs exploitation balance"""
        if not self.clusterer.is_fitted:
            return {"mode": "explore", "reason": "No clustering data"}
        
        analysis = self.clusterer.analyze_cluster_evolution()
        
        if analysis["anomalies"]:
            return {
                "mode": "explore",
                "reason": "Anomalies detected needing investigation",
                "target_clusters": [a["cluster_id"] for a in analysis["anomalies"]]
            }
        
        best = self.clusterer.get_best_cluster()
        if best and best.avg_reward > 0.8:
            return {
                "mode": "exploit",
                "reason": "High-reward cluster found",
                "target_cluster": best.cluster_id
            }
        
        return {
            "mode": "balanced",
            "reason": "Explore to find better strategies"
        }
