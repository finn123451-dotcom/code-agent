import uuid
import time
from typing import Dict, List, Any, Optional
import openai
import os


class SelfEvolvingCodeAgent:
    def __init__(self, api_key: str = None, qdrant_url: str = "localhost", 
                 qdrant_port: int = 6333, db_path: str = "code_agent.db"):
        from .storage import SQLiteStorage
        from .vector_store import VectorStore
        from .data_recorder import PromptRecorder, TrajectoryTracker, LatentSpaceRecorder
        from .embedding import EmbeddingEngine, HiddenStatesExtractor, DecisionVectorExtractor
        from .evolution_engine import EvolutionEngine
        
        self.storage = SQLiteStorage(db_path)
        self.vector_store = VectorStore(url=qdrant_url, port=qdrant_port, api_key=api_key)
        self.embedding_engine = EmbeddingEngine(api_key=api_key)
        
        self.prompt_recorder = PromptRecorder(self.storage)
        self.trajectory_tracker = TrajectoryTracker(self.storage)
        self.latent_recorder = LatentSpaceRecorder(self.storage)
        
        self.hidden_states_extractor = HiddenStatesExtractor()
        self.decision_extractor = DecisionVectorExtractor()
        
        self.evolution_engine = EvolutionEngine(self.storage, self.vector_store, self.embedding_engine)
        
        self.session_id = None
        self.current_step = 0
        self.last_embedding = None
        self.last_hidden_states = None
        self.last_decision_vector = None
        
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def start_session(self, session_id: str = None, metadata: Dict = None) -> str:
        self.session_id = self.trajectory_tracker.start_session(session_id, metadata)
        return self.session_id

    def end_session(self):
        self.trajectory_tracker.end_session()
        self.session_id = None
        self.current_step = 0

    def generate(self, prompt: str, language: str = "python", 
               context: List[Dict] = None) -> Dict:
        start_time = time.time()
        self.trajectory_tracker.start_step(thought=f"Generating {language} code for: {prompt[:100]}")
        
        self.prompt_recorder.record(
            prompt=prompt,
            prompt_type="code_generation",
            session_id=self.session_id,
            metadata={"language": language}
        )
        
        context_messages = []
        if context:
            for c in context:
                context_messages.append({"role": c.get("role", "user"), "content": c.get("content")})
        
        context_messages.insert(0, {
            "role": "system", 
            "content": f"You are an expert {language} programmer. Write efficient, clean code."
        })
        context_messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=context_messages,
            max_tokens=2000
        )
        
        generated_code = response['choices'][0]['message']['content']
        token_usage = response['usage']['total_tokens']
        cost = token_usage * 0.000002
        
        embedding = self.embedding_engine.embed_text(prompt)
        self.last_embedding = embedding
        
        decision_vec = self.decision_extractor.extract_decision(
            action="code_generation",
            confidence=0.85,
            reasoning=f"Generated {len(generated_code)} chars in {language}"
        )
        self.last_decision_vector = decision_vec
        
        trajectory_id = self.trajectory_tracker.record_step(
            action="code_generation",
            action_input={"prompt": prompt, "language": language},
            observation=f"Generated {len(generated_code)} characters of code",
            reward=0.8,
            token_usage=token_usage,
            cost=cost,
            thought=f"Successfully generated code using GPT-3.5"
        )
        
        self.latent_recorder.record(
            trajectory_id=trajectory_id,
            embedding_vector=embedding,
            decision_vector=decision_vec,
            metadata={"language": language, "code_length": len(generated_code)}
        )
        
        self.vector_store.upsert_prompt_embedding(trajectory_id, embedding, {
            "prompt": prompt, "language": language
        })
        
        self.evolution_engine.evolve("reward > 0.8", {
            "reward": 0.8, "action_type": "code_generation", "timestamp": time.time()
        })
        
        return {
            "code": generated_code,
            "token_usage": token_usage,
            "cost": cost,
            "execution_time": time.time() - start_time
        }

    def analyze_code(self, code: str) -> Dict:
        from .code_analyzer import CodeAnalyzer
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(code)
        
        self.trajectory_tracker.start_step(thought="Analyzing code structure")
        trajectory_id = self.trajectory_tracker.record_step(
            action="code_analysis",
            action_input={"code_length": len(code)},
            observation=f"Analysis completed: {result.get('status')}",
            reward=0.9,
            thought="Code analysis successful"
        )
        
        return result

    def execute(self, code: str, language: str = "python", 
               timeout: int = 30) -> Dict:
        from .code_executor import CodeExecutor
        executor = CodeExecutor()
        
        self.trajectory_tracker.start_step(thought=f"Executing {language} code")
        result = executor.execute_code(code, language, timeout)
        
        reward = 1.0 if result['status'] == 'success' else 0.0
        
        trajectory_id = self.trajectory_tracker.record_step(
            action="code_execution",
            action_input={"language": language, "timeout": timeout},
            observation=result.get('stdout', '')[:500],
            reward=reward,
            execution_time=result.get('execution_time') if isinstance(result.get('execution_time'), (int, float)) else None
        )
        
        return result

    def search_similar_prompts(self, query: str, limit: int = 5) -> List[Dict]:
        embedding = self.embedding_engine.embed_text(query)
        return self.vector_store.search_similar_prompts(embedding, limit=limit)

    def search_similar_trajectories(self, query: Dict, limit: int = 5) -> List[Dict]:
        embedding = self.embedding_engine.embed_trajectory(query)
        return self.vector_store.search_similar_trajectories(embedding, limit=limit)

    def get_evolution_report(self) -> Dict:
        return self.evolution_engine.get_evolution_report()

    def get_recommendation(self, task_type: str) -> Dict:
        return self.evolution_engine.recommend_strategy(task_type)

    def complete_task(self, task: str) -> Dict:
        self.trajectory_tracker.start_step(thought=f"Processing task: {task[:100]}")
        
        trajectory_id = self.trajectory_tracker.record_step(
            action="task_received",
            action_input={"task": task},
            observation="Task received and queued",
            reward=0.5,
            thought="Initial task understanding"
        )
        
        return {"status": "task_queued", "task": task, "session_id": self.session_id}

    def get_session_stats(self) -> Dict:
        stats = self.storage.get_session_stats()
        evolution_score = self.evolution_engine.calculate_evolution_score()
        return {**stats, **evolution_score}
