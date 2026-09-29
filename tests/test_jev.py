import pytest

from decision_master import DecisionMaster, Prediction

PAYLOAD = {
    "state": "Customer was charged twice and wants a refund.",
    "questions": {
        "refund_requested": {
            "type": "noul",
            "instructions": "Is the customer asking for a refund?",
            "criteria": {"true": "Wants money back.", "false": "Does not."},
        },
        "department": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments.", "sales": "Pricing."},
        },
        "priority": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "Medium", "High"]},
    },
}


class _Stub(DecisionMaster):
    def __init__(self):
        self.seen = None

    def predict(self, decisions, batch_size=32, max_batch_tokens=16384):
        self.seen = list(decisions)
        return [Prediction(list(d.candidates), [0.2] + [0.8 / (len(d.candidates) - 1)] * (len(d.candidates) - 1)) for d in self.seen]


def test_decide_jev_answers():
    dm = _Stub()
    answers = dm.decide_jev(PAYLOAD)["answers"]

    assert [d.context for d in dm.seen] == [PAYLOAD["state"]] * 3
    assert dm.seen[0].candidates == ["true: Wants money back.", "false: Does not."]

    noul = answers["refund_requested"]
    assert noul["type"] == "noul" and noul["choice"] == "false" and noul["choice_index"] == 1
    assert noul["probabilities"] == {"true": 0.2, "false": 0.8}

    assert answers["department"] == {
        "type": "choice",
        "choice": "sales",
        "probabilities": {"billing": 0.2, "sales": 0.8},
        "confidence": 0.8,
    }

    score = answers["priority"]
    assert score["choice"] == "Medium" and score["confidence"] == pytest.approx(0.4)
    assert score["score"] == pytest.approx(0.4 * 1 + 0.4 * 2)


def test_decide_jev_renders_object_and_array_state():
    q = {"q": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}}
    dm = _Stub()

    dm.decide_jev({"state": {"message": "hi", "retries": 3}, "questions": q})
    assert dm.seen[0].context == '{"message": "hi", "retries": 3}'

    dm.decide_jev({"state": ["first note", "second note"], "questions": q})
    assert dm.seen[0].context == "first note\nsecond note"


def test_decide_jev_rejects_bad_payload():
    with pytest.raises(ValueError):
        _Stub().decide_jev({"state": "x", "questions": {}})
    with pytest.raises(TypeError):
        _Stub().decide_jev({"state": "x", "questions": {"q": {"criteria": "abc"}}})
