# NanoRL

NanoRL contains minimal implementations of reinforcement learning algorithms applied to LLMs.

The implementations focus on policy-gradient methods for finetuning LLMs on a simple toy task: 
maximising lexical richness in generated text (i.e. discouraging word repetition).

Each algorithm is implemented in a single self-contained file. 

Currently implemented algorithms:
- REINFORCE
- REINFORCE Leave-One-Out (RLOO)

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
uv run --env-file .env -m nano_rl.reinforce
```

REINFORCE with baseline:

```bash
uv run --env-file .env -m nano_rl.reinforce --baseline=mean
```

There are three options for baseline:
- `mean`: mean reward of batch
- `ema`: exponential moving average of rewards
- `scst`: reward of model's greedy decode for same prompt

#### REINFORCE Leave-One-Out (RLOO)

RLOO [1] extends SCST from 1 additional generation per prompt to k.
The baseline for each completion is the mean rewards of the other k-1 completions.
All k generations then contribute to one joint policy update.

To run RLOO:

```bash
uv run --env-file .env -m nano_rl.rloo
```

### References

[1] "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" https://arxiv.org/pdf/2402.14740