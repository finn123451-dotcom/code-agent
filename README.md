# Code Agent

A Python-based code agent that can generate, analyze, execute code, and perform file operations.

## Features

- **Code Generation**: Generate code using OpenAI API
- **Code Analysis**: Analyze Python code structure and complexity
- **Code Execution**: Execute Python, JavaScript, and bash scripts
- **File Operations**: Read, write, search, and manage files

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.agent import CodeAgent

agent = CodeAgent(api_key="your-openai-api-key")

# Generate code
code = agent.generate("Write a function to calculate factorial", "python")

# Analyze code
analysis = agent.analyze(code)

# Execute code
result = agent.execute("print('Hello, World!')", "python")

# File operations
agent.write("test.py", code)
content = agent.read("test.py")
```

## License

MIT
