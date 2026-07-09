"""Multi-input fuzz-verify: the quality gate for accepting a generator program.

Runs one candidate against several fresh (spec, word_source) draws and scores
each. A program is accepted only if it produces a valid crossword on EVERY draw
(and clears an optional minimum combined_score). This is what rejects programs
that hardcode words or overfit a single lucky seed — the failure mode a
single-input check would miss.
"""

from __future__ import annotations

from harness.sandbox import run_candidate, run_candidate_inprocess
from harness.scorer import Spec, score


def _draw_topic(spec: Spec) -> str:
    return spec.topic_words[0] if spec.topic_words else "general"


def fuzz_verify(code: str, draws, dictionary=None, timeout_s: float = 5.0, mem_mb: int = 1536,
                accept_min_score: float = 0.0, scores=None, vocab_set=None,
                quality_penalty: bool = False, in_process: bool = False) -> dict:
    """Verify `code` across `draws` = list of (Spec, word_source) pairs.

    Returns {accepted, n_valid, n, mean_score, min_score, results}. `accepted`
    requires all draws valid AND min combined_score >= accept_min_score.
    `scores` (optional {WORD: 0-100}) is forwarded to the scorer for fill_quality.
    `vocab_set` (optional set of vocab n crossword-worthy words) enables the
    filler_fraction / vocab_fraction metrics.
    """
    results = []
    for i, (spec, word_source) in enumerate(draws):
        # word_source may be a flat list OR the theme+fill dict. The generator gets
        # it as-is; the scorer gets the flat theme+fill union for validity.
        is_dict = isinstance(word_source, dict)
        gen_ws = word_source if is_dict else list(word_source)
        inp = {"topic": _draw_topic(spec), "word_source": gen_ws, "size": spec.size, "seed": i}
        if in_process:
            run = run_candidate_inprocess(code, inp, timeout_s=timeout_s)
        else:
            run = run_candidate(code, inp, timeout_s=timeout_s, mem_mb=mem_mb)
        if run["status"] != "ok":
            results.append({
                "status": run["status"], "valid": 0, "combined_score": 0.0,
                "runtime_s": run["runtime_s"], "reasons": [run["status"]],
            })
            continue
        flat = (word_source.get("theme", []) + word_source.get("fill", [])) if is_dict else word_source
        sc = score(run["result"], spec, flat, dictionary=dictionary,
                   runtime_s=run["runtime_s"], scores=scores, vocab_set=vocab_set,
                   quality_penalty=quality_penalty)
        results.append({"status": "ok", "runtime_s": run["runtime_s"], **sc})

    scores = [r["combined_score"] for r in results]
    n_valid = sum(1 for r in results if r.get("valid") == 1)
    min_score = min(scores) if scores else 0.0
    accepted = bool(results) and n_valid == len(results) and min_score >= accept_min_score
    return {
        "accepted": accepted,
        "n_valid": n_valid,
        "n": len(results),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min_score": min_score,
        "results": results,
    }
