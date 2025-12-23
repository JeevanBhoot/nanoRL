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

Train Llama-3.2-1B-Instruct to maximise lexical richness using the REINFORCE algorithm:

```bash
uv run --env-file .env -m nano_rl.reinforce
```

A baseline reduces variance in gradient estimates without introducing bias. To run with a baseline:

```bash
uv run --env-file .env -m nano_rl.reinforce --baseline=mean
```

Baseline options:
- `mean`: mean reward of batch
- `ema`: exponential moving average of rewards
- `scst`: reward of model's greedy decode for same prompt

If using `ema` baseline, the `--ema-beta` argument controls the smoothing factor (default 0.9):

```bash
uv run --env-file .env -m nano_rl.reinforce --baseline=ema --ema-beta 0.5
```

Higher values weigh historical rewards more whilst lower values adapt faster to recent batches.

To penalise short responses, increase the value of `--brevity-penalty-scale` (default 0.0):

```bash
uv run --env-file .env -m nano_rl.reinforce --brevity-penalty-scale 0.5
```

For a full list of arguments, see `nano_rl/reinforce.py`.

#### REINFORCE Leave-One-Out (RLOO)

RLOO [1] extends SCST from 1 additional generation per prompt to k.
The baseline for each completion is the mean reward of the other k-1 completions.
All k generations then contribute to one joint policy update.

To run RLOO:

```bash
uv run --env-file .env -m nano_rl.rloo
```

By default, `k=4`. To vary this, set the `--group-size` argument e.g.

```bash
uv run --env-file .env -m nano_rl.rloo --group-size 8
```

Note that increasing `k` will increase GPU memory consumption.

For a full list of arguments, see `nano_rl/rloo.py`.

### References

[1] "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" https://arxiv.org/pdf/2402.14740