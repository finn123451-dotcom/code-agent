from .code_generator import CodeGenerator
from .code_analyzer import CodeAnalyzer
from .code_executor import CodeExecutor
from .file_operator import FileOperator


class CodeAgent:
    def __init__(self, api_key: str = None):
        self.code_generator = CodeGenerator(api_key)
        self.code_analyzer = CodeAnalyzer()
        self.code_executor = CodeExecutor()
        self.file_operator = FileOperator()

    def generate(self, prompt: str, language: str = "python") -> str:
        return self.code_generator.generate_code(prompt, language)

    def generate_with_context(self, prompt: str, context: str, language: str = "python") -> str:
        return self.code_generator.generate_with_context(prompt, context, language)

    def analyze(self, code: str) -> dict:
        return self.code_analyzer.analyze_code(code)

    def execute(self, code: str, language: str = "python", timeout: int = 30) -> dict:
        return self.code_executor.execute_code(code, language, timeout)

    def read(self, filepath: str) -> dict:
        return self.file_operator.read_file(filepath)

    def write(self, filepath: str, content: str) -> dict:
        return self.file_operator.write_file(filepath, content)

    def append(self, filepath: str, content: str) -> dict:
        return self.file_operator.append_file(filepath, content)

    def list_dir(self, path: str = ".") -> dict:
        return self.file_operator.list_directory(path)

    def create_dir(self, path: str) -> dict:
        return self.file_operator.create_directory(path)

    def delete(self, filepath: str) -> dict:
        return self.file_operator.delete_file(filepath)

    def search(self, directory: str, pattern: str) -> dict:
        return self.file_operator.search_files(directory, pattern)

    def complete_task(self, task: str) -> dict:
        results = {"task": task, "steps": []}
        results["steps"].append({"action": "task_received", "result": task})
        return results
