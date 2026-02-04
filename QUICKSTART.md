# Self-Evolving Code Agent - Quick Start Guide

## Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional, for Qdrant)

## Installation

### Option 1: With Docker (Recommended for Qdrant)

```bash
# Start Qdrant
docker-compose up -d qdrant

# Install Python dependencies
pip install -r requirements.txt
```

### Option 2: In-Memory Mode (No external dependencies)

```bash
pip install -r requirements.txt
```

## Running Verification

### Full Test Suite
```bash
python verify_full.py
```

### Quick Test
```bash
pytest tests/ -v
```

## Quick Usage

```python
from src.agent import SelfEvolvingCodeAgent

# Initialize (uses in-memory mode if Qdrant unavailable)
agent = SelfEvolvingCodeAgent(
    api_key="your-openai-key",
    qdrant_url="localhost",  # Optional
    qdrant_port=6333           # Optional
)

# Start a session
agent.start_session(metadata={"task": "code_generation"})

# Generate code
result = agent.generate("Write a factorial function", "python")
print(result["code"])

# Analyze code
analysis = agent.analyze_code(result["code"])

# Execute code
exec_result = agent.execute(result["code"], "python")

# Get evolution report
report = agent.get_evolution_report()
print(f"Evolution Score: {report['overall_score']}")

# End session
agent.end_session()
```

## Project Structure

```
code-agent/
├── src/
│   ├── __init__.py
│   ├── agent.py              # SelfEvolvingCodeAgent
│   ├── storage.py            # SQLite storage
│   ├── vector_store.py       # Qdrant vector database
│   ├── data_recorder.py      # Prompt, Trajectory, LatentSpace recording
│   ├── embedding.py          # Embedding generation
│   ├── evolution_engine.py    # Self-evolution logic
│   ├── code_generator.py     # Code generation
│   ├── code_analyzer.py      # Code analysis
│   ├── code_executor.py      # Code execution
│   └── file_operator.py      # File operations
├── tests/
│   ├── __init__.py
│   └── test_agent.py
├── verify_full.py           # Full verification script
├── verify.py                # Basic verification
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Features Implemented

✓ Prompt recording and vectorization
✓ Detailed trajectory tracking (timestamps, tokens, cost, execution time)
✓ Latent space capture (embeddings, hidden states, decision vectors)
✓ Qdrant vector storage for similarity search
✓ SQLite persistent storage
✓ Self-evolution engine with pattern analysis
✓ Strategy recommendations based on historical success

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for embeddings and code generation
- `QDRANT_URL`: Qdrant server URL (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)
