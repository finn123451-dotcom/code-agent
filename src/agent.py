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
        from .data_recorder import TrajectoryRecorder, SessionRecorder, PromptRecorder
        from .embedding import EmbeddingEngine, HiddenStatesExtractor, DecisionVectorExtractor, LLMRequestRecorder
        from .evolution_engine import EvolutionEngine
        
        self.storage = SQLiteStorage(db_path)
        self.vector_store = VectorStore(url=qdrant_url, port=qdrant_port, api_key=api_key)
        
        self.trajectory_recorder = TrajectoryRecorder(self.storage)
        self.session_recorder = SessionRecorder(self.storage)
        self.prompt_recorder = PromptRecorder(self.storage)
        
        self.embedding_engine = EmbeddingEngine(api_key=api_key)
        self.hidden_states_extractor = HiddenStatesExtractor()
        self.decision_extractor = DecisionVectorExtractor()
        self.llm_recorder = LLMRequestRecorder()
        
        self.evolution_engine = EvolutionEngine(self.storage, self.vector_store, self.embedding_engine)
        
        self.session_id = None
        self.current_trajectory_id = None
        
        if api_key:
            openai.api_key = api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def start_session(self, session_id: str = None, metadata: Dict = None) -> str:
        self.session_id = self.session_recorder.create_session(metadata=metadata)
        return self.session_id

    def end_session(self):
        if self.session_id:
            self.session_recorder.end_session(self.session_id)
            self.session_id = None

    def start_trajectory(self, task_description: str = None, 
                        initial_prompt: str = None) -> int:
        if not self.session_id:
            self.session_id = self.session_recorder.create_session()
        
        self.current_trajectory_id = self.trajectory_recorder.start_trajectory(
            session_id=self.session_id,
            task_description=task_description,
            initial_prompt=initial_prompt,
            trajectory_type="task"
        )
        return self.current_trajectory_id

    def end_trajectory(self, final_result: str = None, done: bool = True) -> int:
        if self.current_trajectory_id:
            result = self.trajectory_recorder.end_trajectory(
                final_result=final_result,
                done=done
            )
            self.current_trajectory_id = None
            return result
        return None

    def generate(self, prompt: str, language: str = "python", 
                context: List[Dict] = None) -> Dict:
        start_time = time.time()
        
        if not self.current_trajectory_id:
            self.start_trajectory(task_description=f"Generate {language} code", initial_prompt=prompt)
        
        self.trajectory_recorder.start_step(
            step_name="code_generation",
            thought=f"Generating {language} code for: {prompt[:100]}"
        )
        
        self.trajectory_recorder.record_prompt(
            prompt_text=prompt,
            prompt_type="code_generation",
            language=language,
            context={"messages": context} if context else None
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
        
        llm_start = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=context_messages,
            max_tokens=2000
        )
        latency_ms = (time.time() - llm_start) * 1000
        
        generated_code = response['choices'][0]['message']['content']
        token_usage = response['usage']['total_tokens']
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        cost = token_usage * 0.000002
        
        self.trajectory_recorder.record_llm_request(
            request_type="chat_completion",
            model="gpt-3.5-turbo",
            messages=context_messages,
            parameters={"max_tokens": 2000},
            prompt_text=prompt,
            response_text=generated_code,
            response_object=dict(response),
            token_usage=token_usage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency_ms=latency_ms
        )
        
        embedding = self.embedding_engine.embed_text(prompt)
        
        decision_vec = self.decision_extractor.extract_decision(
            action="code_generation",
            confidence=0.85,
            reasoning=f"Generated {len(generated_code)} chars in {language}"
        )
        
        step_id = self.trajectory_recorder.record_step(
            action="code_generation",
            action_input={"prompt": prompt, "language": language},
            action_result={"code_length": len(generated_code)},
            observation=f"Generated {len(generated_code)} characters of code",
            reward=0.8,
            token_usage=token_usage,
            cost=cost,
            thought=f"Successfully generated code using GPT-3.5"
        )
        
        self.trajectory_recorder.record_latent_space(
            step_id=step_id,
            embedding_vector=embedding,
            decision_vector=decision_vec,
            metadata={"language": language, "code_length": len(generated_code)}
        )
        
        self.vector_store.upsert_prompt_embedding(
            self.current_trajectory_id, embedding, {
            "prompt": prompt, "language": language
        })
        
        self.evolution_engine.evolve("reward > 0.8", {
            "reward": 0.8, "action_type": "code_generation", "timestamp": time.time()
        })
        
        return {
            "code": generated_code,
            "token_usage": token_usage,
            "cost": cost,
            "execution_time": time.time() - start_time,
            "trajectory_id": self.current_trajectory_id,
            "step_id": step_id
        }

    def analyze_code(self, code: str) -> Dict:
        from .code_analyzer import CodeAnalyzer
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(code)
        
        self.trajectory_recorder.start_step(
            step_name="code_analysis",
            thought="Analyzing code structure"
        )
        
        step_id = self.trajectory_recorder.record_step(
            action="code_analysis",
            action_input={"code_length": len(code)},
            action_result=result,
            observation=f"Analysis completed: {result.get('status')}",
            reward=0.9,
            thought="Code analysis successful"
        )
        
        self.trajectory_recorder.record_latent_space(
            step_id=step_id,
            embedding_vector=self.embedding_engine.embed_text(code[:1000]),
            metadata={"analysis_result": result.get('status')}
        )
        
        return result

    def execute(self, code: str, language: str = "python", 
               timeout: int = 30) -> Dict:
        from .code_executor import CodeExecutor
        executor = CodeExecutor()
        
        self.trajectory_recorder.start_step(
            step_name="code_execution",
            thought=f"Executing {language} code"
        )
        
        exec_start = time.time()
        result = executor.execute_code(code, language, timeout)
        execution_time = time.time() - exec_start
        
        reward = 1.0 if result['status'] == 'success' else 0.0
        
        step_id = self.trajectory_recorder.record_step(
            action="code_execution",
            action_input={"language": language, "timeout": timeout},
            action_result={"status": result['status'], "stdout_length": len(result.get('stdout', ''))},
            observation=result.get('stdout', '')[:500],
            reward=reward,
            execution_time=execution_time,
            error=result.get('stderr') if result['status'] == 'error' else None
        )
        
        self.trajectory_recorder.record_code_execution(
            step_id=step_id,
            code=code,
            language=language,
            stdout=result.get('stdout', ''),
            stderr=result.get('stderr', ''),
            return_code=result.get('return_code'),
            execution_time=execution_time,
            timeout=result['status'] == 'timeout'
        )
        
        return result

    def complete_task(self, task: str) -> Dict:
        if not self.current_trajectory_id:
            self.start_trajectory(task_description=task)
        
        self.trajectory_recorder.start_step(
            step_name="task_received",
            thought=f"Processing task: {task[:100]}"
        )
        
        step_id = self.trajectory_recorder.record_step(
            action="task_received",
            action_input={"task": task},
            action_result={"status": "queued"},
            observation="Task received and queued",
            reward=0.5,
            thought="Initial task understanding"
        )
        
        return {
            "status": "task_queued", 
            "task": task, 
            "session_id": self.session_id,
            "trajectory_id": self.current_trajectory_id,
            "step_id": step_id
        }

    def get_full_trajectory(self, trajectory_id: int = None) -> Dict:
        t_id = trajectory_id or self.current_trajectory_id
        if t_id:
            return self.trajectory_recorder.get_full_trajectory(t_id)
        return None

    def get_trajectory_steps(self, trajectory_id: int = None) -> List[Dict]:
        t_id = trajectory_id or self.current_trajectory_id
        if t_id:
            return self.trajectory_recorder.get_steps(t_id)
        return []

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

    def get_session_stats(self) -> Dict:
        stats = self.storage.get_session_stats()
        trajectory_stats = self.storage.get_trajectory_stats()
        evolution_score = self.evolution_engine.calculate_evolution_score()
        return {**stats, **trajectory_stats, **evolution_score}
