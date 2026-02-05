# Trajectory Summarization & RL Dataset Generation

## New Modules

### src/summarization/
- `TrajectorySummarizer` - Extract summaries from trajectory data
- `StepSummarizer` - Summarize individual steps

### src/rl/
- `RLDatasetGenerator` - Generate RL training datasets
- `TrajectorySampler` - Sample trajectories for training
- `DatasetFormat` - Supported formats: DPO, PPO, SFT, TRAJECTORY, REWARD_MODEL

## Usage

```python
from src.summarization import TrajectorySummarizer
from src.rl import RLDatasetGenerator, DatasetFormat

# Summarize a trajectory
summarizer = TrajectorySummarizer()
summary = summarizer.summarize_trajectory(trajectory)

# Generate RL dataset
generator = RLDatasetGenerator()
dataset = generator.generate_dataset(trajectories, format=DatasetFormat.SFT)

# Export dataset
generator.export_dataset(dataset, format="jsonl", output_path="dataset.jsonl")
```
