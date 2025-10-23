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

```bash
uv run --env-file .env -m easy_rl.train_reinforce
```