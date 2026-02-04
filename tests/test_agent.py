import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agent import CodeAgent
from src.code_analyzer import CodeAnalyzer
from src.code_executor import CodeExecutor
from src.file_operator import FileOperator


def test_code_analyzer():
    analyzer = CodeAnalyzer()
    code = """
def hello():
    print("Hello")
    
class Test:
    def method(self):
        pass
"""
    result = analyzer.analyze_code(code)
    assert result["status"] == "success"
    assert len(result["functions"]) >= 1
    assert len(result["classes"]) >= 1


def test_code_executor():
    executor = CodeExecutor()
    result = executor.execute_code("print('test')", "python")
    assert result["status"] == "success"
    assert "test" in result["stdout"]


def test_file_operator():
    operator = FileOperator()
    test_file = "test_write.txt"
    
    result = operator.write_file(test_file, "Hello, World!")
    assert result["status"] == "success"
    
    result = operator.read_file(test_file)
    assert result["status"] == "success"
    assert result["content"] == "Hello, World!"
    
    operator.delete_file(test_file)


def test_code_agent_init():
    agent = CodeAgent()
    assert hasattr(agent, 'code_generator')
    assert hasattr(agent, 'code_analyzer')
    assert hasattr(agent, 'code_executor')
    assert hasattr(agent, 'file_operator')


if __name__ == "__main__":
    test_code_analyzer()
    test_code_executor()
    test_file_operator()
    test_code_agent_init()
    print("All tests passed!")
