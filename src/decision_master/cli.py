from __future__ import annotations

import argparse
import json
import sys

import torch

from .hub import DEFAULT_MODEL_ID, DEFAULT_REVISION
from .predictor import DecisionMaster

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hub repo id or local checkpoint directory.")
    p.add_argument("--revision", default=DEFAULT_REVISION, help="Hub branch/tag (ignored for local dirs).")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=sorted(_DTYPES), default=None)


def _load(args) -> DecisionMaster:
    return DecisionMaster.from_pretrained(
        args.model, args.revision, device=args.device, dtype=_DTYPES.get(args.dtype)
    )


def _cmd_decide(args) -> None:
    dm = _load(args)
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        if not args.query or not args.candidate:
            sys.exit("Provide --input, or --query with at least two --candidate values.")
        rows = [{"query": args.query, "context": args.context, "candidates": args.candidate}]

    for pred in dm.predict(rows, batch_size=args.batch_size):
        print(
            json.dumps(
                {"best": pred.best, "best_index": pred.best_index, "probabilities": pred.probabilities},
                ensure_ascii=False,
            )
        )


def _cmd_export(args) -> None:
    _load(args).save_pretrained(args.output)
    print(f"Saved to {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="decision-master")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("decide", help="Score candidates for one query or a JSONL file.")
    _add_model_args(d)
    d.add_argument("--query")
    d.add_argument("--context", default="")
    d.add_argument("--candidate", action="append", help="Repeat for each candidate.")
    d.add_argument("--input", help="JSONL with query/context/candidates per line.")
    d.add_argument("--batch-size", type=int, default=32)
    d.set_defaults(func=_cmd_decide)

    e = sub.add_parser("export", help="Convert a training checkpoint dir into a self-contained Hub-ready repo.")
    _add_model_args(e)
    e.add_argument("--output", required=True)
    e.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
