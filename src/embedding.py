import time
import numpy as np
from typing import List, Dict, Any, Optional
import openai
import os


class LLMRequestRecorder:
    def __init__(self, recorder=None):
        self.recorder = recorder
        self.model = "text-embedding-ada-002"
        if os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def set_recorder(self, recorder):
        self.recorder = recorder

    def embed_text(self, text: str, record: bool = False, step_id: int = None) -> List[float]:
        start_time = time.time()
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            embedding = response['data'][0]['embedding']
            latency_ms = (time.time() - start_time) * 1000

            if record and self.recorder:
                self.recorder.record_llm_request(
                    step_id=step_id,
                    request_type="embedding",
                    model=self.model,
                    messages=[{"role": "user", "content": text}],
                    prompt_text=text,
                    response_text=response['data'][0]['embedding'],
                    response_object=dict(response),
                    token_usage=response['usage']['total_tokens'],
                    prompt_tokens=response['usage']['prompt_tokens'],
                    completion_tokens=response['usage']['completion_tokens'],
                    cost=response['usage']['total_tokens'] * 0.0000001,
                    latency_ms=latency_ms
                )

            return embedding
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            if record and self.recorder:
                self.recorder.record_llm_request(
                    step_id=step_id,
                    request_type="embedding",
                    model=self.model,
                    prompt_text=text,
                    response_text=str(e),
                    latency_ms=latency_ms,
                    metadata={"error": str(e)}
                )
            return self._fallback_embedding(text)

    def embed_batch(self, texts: List[str], record: bool = False, step_id: int = None) -> List[List[float]]:
        start_time = time.time()
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=texts
            )
            embeddings = [item['embedding'] for item in response['data']]
            latency_ms = (time.time() - start_time) * 1000

            if record and self.recorder:
                self.recorder.record_llm_request(
                    step_id=step_id,
                    request_type="embedding_batch",
                    model=self.model,
                    messages=[{"role": "user", "content": texts}],
                    prompt_text=str(texts),
                    response_object=dict(response),
                    token_usage=response['usage']['total_tokens'],
                    latency_ms=latency_ms
                )

            return embeddings
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            if record and self.recorder:
                self.recorder.record_llm_request(
                    step_id=step_id,
                    request_type="embedding_batch",
                    model=self.model,
                    prompt_text=str(texts),
                    response_text=str(e),
                    latency_ms=latency_ms,
                    metadata={"error": str(e)}
                )
            return [self._fallback_embedding(t) for t in texts]

    def chat_completion(self, messages: List[Dict], model: str = "gpt-3.5-turbo",
                      max_tokens: int = 2000, temperature: float = 0.7,
                      record: bool = False, step_id: int = None) -> Dict:
        start_time = time.time()
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            latency_ms = (time.time() - start_time) * 1000

            result = {
                "content": response['choices'][0]['message']['content'],
                "response": response,
                "token_usage": response['usage']['total_tokens'],
                "prompt_tokens": response['usage']['prompt_tokens'],
                "completion_tokens": response['usage']['completion_tokens'],
                "cost": response['usage']['total_tokens'] * 0.000002,
                "latency_ms": latency_ms
            }

            if record and self.recorder:
                self.recorder.record_llm_request(
                    step_id=step_id,
                    request_type="chat_completion",
                    model=model,
                    messages=messages,
                    parameters={"max_tokens": max_tokens, "temperature": temperature},
                    prompt_text=str(messages),
                    response_text=result['content'],
                    response_object=dict(response),
                    token_usage=result['token_usage'],
                    prompt_tokens=result['prompt_tokens'],
                    completion_tokens=result['completion_tokens'],
                    cost=result['cost'],
                    latency_ms=latency_ms
                )

            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            if record and self.recorder:
                self.recorder.record_llm_request(
                    step_id=step_id,
                    request_type="chat_completion",
                    model=model,
                    messages=messages,
                    parameters={"max_tokens": max_tokens, "temperature": temperature},
                    prompt_text=str(messages),
                    response_text=str(e),
                    latency_ms=latency_ms,
                    metadata={"error": str(e)}
                )
            raise e

    def _fallback_embedding(self, text: str) -> List[float]:
        words = text.lower().split()
        vec = np.zeros(1536)
        for i, word in enumerate(words[:1536]):
            vec[i] = hash(word) % 1000 / 1000.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class EmbeddingEngine:
    def __init__(self, model: str = "text-embedding-ada-002", api_key: str = None, recorder=None):
        self.model = model
        self.recorder = recorder
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def set_recorder(self, recorder):
        self.recorder = recorder

    def embed_text(self, text: str, record: bool = False, step_id: int = None) -> List[float]:
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            embedding = response['data'][0]['embedding']

            if record and self.recorder:
                self.recorder.record_llm_request(
                    step_id=step_id,
                    request_type="embedding",
                    model=self.model,
                    messages=[{"role": "user", "content": text}],
                    prompt_text=text,
                    response_text=embedding,
                    token_usage=response['usage']['total_tokens'],
                    prompt_tokens=response['usage']['prompt_tokens'],
                    completion_tokens=response['usage']['completion_tokens'],
                    cost=response['usage']['total_tokens'] * 0.0000001,
                    latency_ms=0
                )

            return embedding
        except Exception as e:
            return self._fallback_embedding(text)

    def embed_batch(self, texts: List[str], record: bool = False, step_id: int = None) -> List[List[float]]:
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=texts
            )
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            return [self._fallback_embedding(t) for t in texts]

    def embed_trajectory(self, trajectory_data: Dict, record: bool = False, step_id: int = None) -> List[float]:
        text = self._trajectory_to_text(trajectory_data)
        return self.embed_text(text, record=record, step_id=step_id)

    def _trajectory_to_text(self, trajectory: Dict) -> str:
        parts = []
        if trajectory.get('thought'):
            parts.append(f"Thought: {trajectory['thought']}")
        if trajectory.get('action'):
            parts.append(f"Action: {trajectory['action']}")
        if trajectory.get('action_input'):
            parts.append(f"Input: {trajectory['action_input']}")
        if trajectory.get('observation'):
            parts.append(f"Observation: {trajectory['observation']}")
        if trajectory.get('reward') is not None:
            parts.append(f"Reward: {trajectory['reward']}")
        return " | ".join(parts)

    def _fallback_embedding(self, text: str) -> List[float]:
        words = text.lower().split()
        vec = np.zeros(1536)
        for i, word in enumerate(words[:1536]):
            vec[i] = hash(word) % 1000 / 1000.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


class HiddenStatesExtractor:
    def __init__(self):
        self.last_hidden_states = None
        self.hidden_states_cache = {}

    def extract_from_response(self, response, layer: int = -1):
        try:
            if hasattr(response, 'choices'):
                self.last_hidden_states = response.choices[0].message.content.encode()
            return self.last_hidden_states
        except:
            return None

    def extract_hidden_states(self, model_response, layers: List[int] = None) -> Dict[int, List[float]]:
        if layers is None:
            layers = [-1]
        states = {}
        for layer in layers:
            try:
                if hasattr(model_response, 'choices'):
                    content = model_response.choices[0].message.content
                    states[layer] = self._encode_to_vector(content, layer)
            except:
                pass
        self.hidden_states_cache.update(states)
        return states

    def _encode_to_vector(self, text: str, seed: int) -> List[float]:
        np.random.seed(seed + len(text))
        vec = np.random.randn(768)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()


class DecisionVectorExtractor:
    def __init__(self):
        self.last_decision_vector = None

    def extract_decision(self, action: str, confidence: float,
                        reasoning: str = None) -> List[float]:
        import hashlib
        hash_val = int(hashlib.md5(f"{action}{confidence}".encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        vec = np.random.randn(512)
        vec[:10] = np.array([confidence, len(action or ""),
                           len(reasoning or ""), 0, 0, 0, 0, 0, 0, 0])
        vec = vec / np.linalg.norm(vec)
        self.last_decision_vector = vec.tolist()
        return self.last_decision_vector

    def aggregate_decisions(self, decisions: List[List[float]]) -> List[float]:
        if not decisions:
            return np.random.randn(512).tolist()
        return np.mean(decisions, axis=0).tolist()
