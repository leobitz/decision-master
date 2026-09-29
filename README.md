# decision-master

Fast inference for the Qwen3-based DecisionMaster model: given a query (and optional context), it returns a probability distribution over a variable-size list of candidate answers.

## Install

```bash
pip install -e .            # or: pip install -e ".[dev]" to run the tests
```

## Usage

```python
from decision_master import DecisionMaster

# Defaults to the Hub repo `leobitz/decision-master` at revision `base`.
dm = DecisionMaster.from_pretrained()

pred = dm.decide(
    query="Which of these is a fruit?",
    candidates=["apple", "table", "car"],
    context="A grocery list.",       # optional
)
print(pred.best, pred.probabilities)

# Many decisions at once (returned in input order):
preds = dm.predict([
    {"query": "...", "context": "...", "candidates": ["a", "b"]},
    {"query": "...", "candidates": ["x", "y", "z"]},
], batch_size=32)
```

Other sources: `DecisionMaster.from_pretrained("leobitz/decision-master", revision="base")` or a local directory (e.g. a training checkpoint such as `checkpoints/best_val`; `model.safetensors` or `model.pt` are both accepted).

CLI:

```bash
decision-master decide --query "Which is a fruit?" --candidate apple --candidate table
decision-master decide --input decisions.jsonl > predictions.jsonl
```

## Publishing a model

Convert a training checkpoint into a self-contained repo (safetensors, embedded Qwen3 config, tokenizer), then upload it to the `base` revision:

```bash
decision-master export --model checkpoints/best_val --output hf_repo
hf upload leobitz/decision-master hf_repo --revision base
```

## Why it is fast

- Each decision is packed as a single sequence `[context | cand 0 | cand 1 | ...]` with a tree attention mask and explicit position ids. The shared context is computed once and one ordinary batched transformer pass scores all candidates (verified equal to running `context + candidate` separately in `tests/`).
- PyTorch SDPA attention, bf16/fp16 on GPU, `inference_mode`, weights loaded directly onto the device without random init.
- Examples are sorted by length and batched under a token budget (`max_batch_tokens`) to minimise padding; the CPU packs the next batch while the GPU is busy, with a single host sync at the end.

## Tests

```bash
pytest
```
