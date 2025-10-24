# Easy RL for LLMs

## Installation

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

Add your HuggingFace token to `.env`:

```
HF_TOKEN=hf_...
```

## Training

### REINFORCE

Train Llama-3.2-1B-Instruct to maximise lexical richness using REINFORCE algorithm:

```bash
uv run --env-file .env -m easy_rl.train_reinforce
```

REINFORCE with baseline:

```bash
uv run --env-file .env -m easy_rl.train_reinforce --baseline=mean
```

There are three options for baseline:
- `mean`: mean reward of batch
- `ema`: exponential moving average of rewards
- `scst`: reward of model's greedy decode for same prompt

#### REINFORCE Leave-One-Out (RLOO)

RLOO extends SCST from 1 additional generation per prompt to k.
The baseline for each completion is the mean rewards of the other k-1 completions.
All k generations then contribute to one joint policy update.

To run RLOO:

```bash
uv run --env-file .env -m easy_rl.train_rloo
```