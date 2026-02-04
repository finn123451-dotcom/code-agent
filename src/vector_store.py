import numpy as np
from typing import List, Dict, Any, Optional
import json


class VectorStore:
    def __init__(self, url: str = "localhost", port: int = 6333, api_key: str = None):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance, PointStruct
            
            self.client = QdrantClient(url=url, port=port, api_key=api_key)
            self.initialized = True
            self._init_collections()
        except ImportError:
            print("Qdrant not installed. Using in-memory fallback.")
            self.client = None
            self.initialized = False
            self._in_memory_store = {
                "prompts": [],
                "trajectories": [],
                "latent": [],
                "embeddings": {}
            }

    def _init_collections(self):
        from qdrant_client.models import VectorParams, Distance
        
        collections = [
            ("prompts", 1536),
            ("trajectories", 1536),
            ("latent_space", 768),
            ("agent_decisions", 512)
        ]
        
        for name, size in collections:
            try:
                self.client.get_collection(name)
            except:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE)
                )

    def upsert_prompt_embedding(self, prompt_id: int, embedding: List[float], 
                                metadata: Dict = None):
        if self.initialized:
            from qdrant_client.models import PointStruct
            self.client.upsert(
                collection_name="prompts",
                points=[PointStruct(id=prompt_id, vector=embedding, payload=metadata or {})]
            )
        else:
            self._in_memory_store["prompts"].append({
                "id": prompt_id,
                "embedding": embedding,
                "metadata": metadata
            })

    def upsert_trajectory_embedding(self, trajectory_id: int, embedding: List[float],
                                    metadata: Dict = None):
        if self.initialized:
            from qdrant_client.models import PointStruct
            self.client.upsert(
                collection_name="trajectories",
                points=[PointStruct(id=trajectory_id, vector=embedding, payload=metadata or {})]
            )
        else:
            self._in_memory_store["trajectories"].append({
                "id": trajectory_id,
                "embedding": embedding,
                "metadata": metadata
            })

    def upsert_latent_embedding(self, latent_id: int, embedding: List[float],
                               metadata: Dict = None):
        if self.initialized:
            from qdrant_client.models import PointStruct
            self.client.upsert(
                collection_name="latent_space",
                points=[PointStruct(id=latent_id, vector=embedding, payload=metadata or {})]
            )
        else:
            self._in_memory_store["latent"].append({
                "id": latent_id,
                "embedding": embedding,
                "metadata": metadata
            })

    def search_similar_prompts(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        if self.initialized:
            results = self.client.search(
                collection_name="prompts",
                query_vector=query_embedding,
                limit=limit
            )
            return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
        else:
            import numpy as np
            query = np.array(query_embedding)
            similarities = []
            for item in self._in_memory_store["prompts"]:
                stored = np.array(item["embedding"])
                similarity = np.dot(query, stored) / (np.linalg.norm(query) * np.linalg.norm(stored))
                similarities.append({"id": item["id"], "score": similarity, "payload": item["metadata"]})
            return sorted(similarities, key=lambda x: x["score"], reverse=True)[:limit]

    def search_similar_trajectories(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        if self.initialized:
            results = self.client.search(
                collection_name="trajectories",
                query_vector=query_embedding,
                limit=limit
            )
            return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
        else:
            import numpy as np
            query = np.array(query_embedding)
            similarities = []
            for item in self._in_memory_store["trajectories"]:
                stored = np.array(item["embedding"])
                similarity = np.dot(query, stored) / (np.linalg.norm(query) * np.linalg.norm(stored))
                similarities.append({"id": item["id"], "score": similarity, "payload": item["metadata"]})
            return sorted(similarities, key=lambda x: x["score"], reverse=True)[:limit]

    def search_similar_latent(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        if self.initialized:
            results = self.client.search(
                collection_name="latent_space",
                query_vector=query_embedding,
                limit=limit
            )
            return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
        else:
            import numpy as np
            query = np.array(query_embedding)
            similarities = []
            for item in self._in_memory_store["latent"]:
                stored = np.array(item["embedding"])
                similarity = np.dot(query, stored) / (np.linalg.norm(query) * np.linalg.norm(stored))
                similarities.append({"id": item["id"], "score": similarity, "payload": item["metadata"]})
            return sorted(similarities, key=lambda x: x["score"], reverse=True)[:limit]

    def get_collection_stats(self, collection_name: str) -> Dict:
        if self.initialized:
            info = self.client.get_collection(collection_name)
            return {"vectors_count": info.vectors_count, "config": info.config}
        else:
            return {"vectors_count": len(self._in_memory_store.get(collection_name, []))}
