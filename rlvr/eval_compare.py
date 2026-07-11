"""SFT vs RLVR comparison on the pristine held-out eval split.

Thin wrapper over pipeline/eval_harness.py: it registers the two served adapters
(sft, rlvr) as an OpenAI/Ollama-compatible model registry and runs the SAME eval
used everywhere else on data/sft_hardcoded_words/eval.jsonl (never trained on). eval_harness
prints valid_rate, within_spec_rate, pass@k, coverage, crossings, filler_fraction,
invalid_entry/crossing_frac for both models side by side.

Prereq: serve each adapter behind an OpenAI/Ollama endpoint first, e.g.
  - Ollama : merge adapter -> GGUF -> `ollama create qwen3-crossword-sft -f Modelfile`
             (repeat for the GRPO adapter), both on http://localhost:11434
  - vLLM   : `python -m vllm.entrypoints.openai.api_server --model <merged> --port 8000`
             (one server per adapter, or --lora-modules), OpenAI API at /v1

Then:
  python rlvr/eval_compare.py --provider ollama --sft qwen3-crossword-sft --rlvr qwen3-crossword-grpo
  python rlvr/eval_compare.py --provider openai --base-url http://localhost:8000/v1 \
         --sft sft-merged --rlvr grpo-merged --samples 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.eval_harness import main as eval_main

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_EVAL = os.path.join(_ROOT, "data", "sft_hardcoded_words", "eval.jsonl")
_REGISTRY_OUT = os.path.join(_ROOT, "rlvr", "_models_sft_vs_rlvr.json")

# NOTE: this delegates to pipeline/eval_harness.py, which scores by INJECTING a palette
# word_source (the old function-call convention). The RLVR reward instead judges the
# model's OWN crossword (word_source=None -> embedded _WORDS, validated vs words_alpha).
# So these numbers measure "fills a given palette", not "create your own". For an
# apples-to-apples RLVR eval, mirror rlvr/reward.reward_from_text over generated outputs.


def _entry(provider: str, base_url: str, model: str) -> dict:
    if provider == "ollama":
        return {"provider": "ollama", "base_url": base_url, "model": model,
                "think": False, "num_ctx": 8192, "concurrency": 2}
    return {"provider": "openai", "base_url": base_url, "model": model, "concurrency": 8}


def main():
    ap = argparse.ArgumentParser(description="SFT vs RLVR eval on held-out specs")
    ap.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--base-url", default=None,
                    help="endpoint base (default: ollama=http://localhost:11434, openai=http://localhost:8000/v1)")
    ap.add_argument("--sft", default="qwen3-crossword-sft", help="served model id/tag for the SFT adapter")
    ap.add_argument("--rlvr", default="qwen3-crossword-grpo", help="served model id/tag for the GRPO adapter")
    ap.add_argument("--eval-file", default=_DEFAULT_EVAL)
    ap.add_argument("--samples", type=int, default=3, help="completions per spec (pass@k)")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-specs", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_url = args.base_url or ("http://localhost:11434" if args.provider == "ollama"
                                 else "http://localhost:8000/v1")
    registry = {
        "sft": _entry(args.provider, base_url, args.sft),
        "rlvr": _entry(args.provider, base_url, args.rlvr),
    }
    with open(_REGISTRY_OUT, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    print(f"registry -> {_REGISTRY_OUT}\n{json.dumps(registry, indent=2)}\n")

    argv = [
        "--eval-file", args.eval_file,
        "--models", "sft", "rlvr",
        "--models-config", _REGISTRY_OUT,
        "--samples", str(args.samples),
        "--temperature", str(args.temperature),
        "--timeout", str(args.timeout),
    ]
    if args.max_specs is not None:
        argv += ["--max-specs", str(args.max_specs)]
    if args.out:
        argv += ["--out", args.out]
    eval_main(argv)


if __name__ == "__main__":
    main()
