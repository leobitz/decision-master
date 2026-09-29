import random

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from decision_master.batching import Encoded, pack_batch, plan_batches
from decision_master.config import DecisionMasterConfig
from decision_master.modeling import DecisionMasterModel
from decision_master.schema import Decision, Prediction

VOCAB = 50


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    backbone = Qwen3Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
    )
    backbone._attn_implementation = "sdpa"
    config = DecisionMasterConfig(set_layers=1, set_heads=4, set_ff_mult=2, dropout=0.0)
    m = DecisionMasterModel(config, backbone).eval()
    m.qwen.rotary_emb = Qwen3RotaryEmbedding(backbone)
    return m


def _random_items(rng, n_items):
    items = []
    for _ in range(n_items):
        prefix = [rng.randrange(VOCAB) for _ in range(rng.randint(3, 20))]
        cands = [[rng.randrange(VOCAB) for _ in range(rng.randint(1, 6))] for _ in range(rng.randint(2, 5))]
        items.append(Encoded(prefix=prefix, candidates=cands))
    return items


@torch.no_grad()
def test_tree_batch_matches_independent_forward(model):
    items = _random_items(random.Random(1), 6)
    batch = pack_batch(items, pad_id=0)
    got = model.candidate_states(batch)

    for b, item in enumerate(items):
        for j, cand in enumerate(item.candidates):
            ids = torch.tensor([item.prefix + cand])
            want = model.qwen(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
            torch.testing.assert_close(got[b, j], want, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_probabilities_are_normalised_over_real_candidates(model):
    items = _random_items(random.Random(2), 4)
    probs = model(pack_batch(items, pad_id=0))
    for row, item in zip(probs, items):
        n = len(item.candidates)
        assert row[:n].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert torch.all(row[n:] == 0)


@torch.no_grad()
def test_result_independent_of_batch_composition(model):
    items = _random_items(random.Random(3), 5)
    together = model(pack_batch(items, pad_id=0))
    for i, item in enumerate(items):
        alone = model(pack_batch([item], pad_id=0))[0]
        n = len(item.candidates)
        torch.testing.assert_close(together[i, :n], alone[:n], atol=1e-5, rtol=1e-4)


def test_plan_batches_covers_all_and_respects_limits():
    items = _random_items(random.Random(4), 40)
    batches = plan_batches(items, batch_size=8, max_batch_tokens=400)
    assert sorted(i for b in batches for i in b) == list(range(40))
    assert all(len(b) <= 8 for b in batches)


def test_decision_validation_and_prediction():
    with pytest.raises(ValueError):
        Decision(query="q", candidates=["only one"]).validate()
    p = Prediction(["a", "b", "c"], [0.2, 0.5, 0.3])
    assert p.best == "b" and p.best_index == 1
    assert [c for c, _ in p.ranked()] == ["b", "c", "a"]
