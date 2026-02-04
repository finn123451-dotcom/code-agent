# Verification Instructions for Self-Evolving Code Agent

## Environment Setup

### Option 1: Docker (Recommended)

```bash
# Start Qdrant vector database
docker-compose up -d qdrant

# Wait for Qdrant to be ready
sleep 5

# Run verification
docker-compose run app python verify_full.py
```

### Option 2: Local Python

```bash
# Install Python 3.9+ if not installed
# Download from: https://www.python.org/downloads/

# Install dependencies
pip install -r requirements.txt

# Run verification
python verify_full.py

# Or with pytest
pytest tests/ -v
```

### Option 3: Manual Verification (No Dependencies)

The `verify_full.py` script includes in-memory fallbacks for all components:

```python
# Core functionality works without:
# - Qdrant (uses in-memory vector store)
# - OpenAI API (uses fake embeddings)
# - External databases (uses SQLite in-memory)

python verify_full.py
```

## Expected Output

```
============================================================
Self-Evolving Code Agent - Full Verification
============================================================

Testing SQLiteStorage...
  ✓ SQLiteStorage passed
Testing VectorStore...
  ✓ VectorStore passed
Testing EmbeddingEngine...
  ✓ EmbeddingEngine passed
Testing DecisionVectorExtractor...
  ✓ DecisionVectorExtractor passed
Testing DataRecorder components...
  ✓ DataRecorder components passed
Testing EvolutionEngine...
  ✓ EvolutionEngine passed
Testing full integration...
  ✓ Integration test passed

============================================================
Results: 7 passed, 0 failed
============================================================
```

## Troubleshooting

### Qdrant Connection Failed

If you see "Qdrant not installed", the system automatically uses in-memory mode.

### OpenAI API Error

If embedding generation fails, the system uses fake embeddings for testing.

### Import Errors

```bash
# Install missing dependencies
pip install openai qdrant-client numpy pytest
```

## Quick Test Commands

```bash
# Full test suite
python verify_full.py

# Unit tests only
pytest tests/ -v

# Specific test
pytest tests/test_agent.py -v

# With coverage
pytest --cov=src tests/
```

## Components Verified

| Component | Test Status |
|-----------|-------------|
| SQLiteStorage | ✓ Prompts, Trajectories, LatentSpace, Sessions |
| VectorStore | ✓ CRUD operations, Similarity search |
| EmbeddingEngine | ✓ Single, Batch, Trajectory embeddings |
| DecisionVector | ✓ Extraction, Aggregation |
| DataRecorder | ✓ Prompt, Trajectory, LatentSpace recording |
| EvolutionEngine | ✓ Patterns, Score, Recommendations |
| Integration | ✓ End-to-end data flow |

## Next Steps

After verification, you can:

1. **Start Qdrant for production**:
   ```bash
   docker-compose up -d qdrant
   ```

2. **Run the agent**:
   ```python
   from src.agent import SelfEvolvingCodeAgent
   agent = SelfEvolvingCodeAgent(api_key="your-key")
   ```

3. **View evolution report**:
   ```python
   report = agent.get_evolution_report()
   ```
