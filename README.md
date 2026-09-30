# decision-master

Given a query (and optional context), DecisionMaster returns a probability distribution over a variable-size list of candidate answers. It is a Qwen3-based model built for fast inference.

## Install

```bash
pip install -e .            # add ".[dev]" to run the tests
```

## Quick start

```python
from decision_master import DecisionMaster

# The tag "base" loads leobitz/decision-master-base from the Hugging Face Hub.
model = DecisionMaster("base")

pred = model.decide(
    query="What is the best category?",
    context="Customer says they were billed twice and wants a refund.",  # optional
    candidates=["billing", "shipping", "technical", "other"],
)
print(pred.best, pred.probabilities[pred.best_index])
print(pred.ranked())   # [(candidate, probability), ...] sorted high to low
```

The first argument is a model tag (`"base"` -> `leobitz/decision-master-base`, default `"base"`), a full Hub repo id, or a local checkpoint directory (`model.safetensors` or `model.pt` plus `config.json` and tokenizer files). `DecisionMaster.from_pretrained(...)` is an equivalent alias.

```python
DecisionMaster()                                            # same as "base"
DecisionMaster("leobitz/decision-master-base", revision="main")
DecisionMaster("checkpoints/best_val")
```

Optional arguments: `device="cuda"`, `dtype=torch.bfloat16` (default on GPU; float32 on CPU), `cache_dir`, `token`.

## Batch prediction

```python
preds = model.predict([
    {"query": "Is the sentiment positive or negative?", "context": "The food was cold.", "candidates": ["positive", "negative"]},
    {"query": "What is the capital of France?", "candidates": ["Berlin", "Paris", "Madrid"]},
], batch_size=32)
```

Results come back in input order. Decisions can have different candidate counts. Lower `max_batch_tokens` (default 16384) if you run out of GPU memory.

## JEV-style questions

`decide_jev` answers several typed questions against one shared `state`:

```python
result = model.decide_jev({
    "state": "A customer was charged twice and wants the duplicate charge refunded immediately.",
    "questions": {
        "refund_requested": {
            "type": "noul",
            "instructions": "Is the customer explicitly asking for a refund?",
            "criteria": {"true": "The customer wants money returned.", "false": "They are not asking for a refund."},
        },
        "owner_team": {
            "type": "choice",
            "instructions": "Which team should own this case?",
            "criteria": {"billing": "Charges and refunds.", "technical": "Bugs and outages."},
        },
        "priority": {
            "type": "score",
            "instructions": "How urgent is this case?",
            "criteria": ["Low", "Medium", "High"],   # ordered low to high
        },
    },
})
result["answers"]["owner_team"]["choice"]
```

- `state` can be a string, a JSON object, or a list of texts.
- `criteria` is either `{label: description}` or a list of labels.
- `choice` and `score` answers contain `type`, `choice`, `probabilities` and `confidence` (the top probability).
- `score` also has `score`, the probability-weighted 0-based level position (for `Low`/`Medium`/`High`, 0 to 2).
- `noul` answers are `{"type": "noul", "noul": p}` where `p` is the probability the proposition is true. `criteria` is optional and the `true`/`false` candidates are added internally (given `true`/`false` descriptions are reused).

## Command line

```bash
decision-master decide --query "Which is a fruit?" --candidate apple --candidate table
decision-master decide --input decisions.jsonl > predictions.jsonl   # one {"query","context","candidates"} per line
```

## Publishing a checkpoint

Convert a training checkpoint into a self-contained Hub repo (safetensors, embedded Qwen3 config, tokenizer), then upload it:

```bash
decision-master export --model checkpoints/best_val --output hf_repo
hf upload leobitz/decision-master-base hf_repo
```

## Why it is fast

- Each decision is packed as one sequence, `[context | cand 0 | cand 1 | ...]`, with a tree attention mask. The shared context is computed once and a single batched pass scores every candidate. Tests check this equals running `context + candidate` separately.
- PyTorch SDPA attention, bf16/fp16 on GPU, `inference_mode`, and weights loaded straight onto the device.
- Examples are sorted by length and batched under a token budget to limit padding. The CPU packs the next batch while the GPU works.

## Tests

```bash
pytest
```
