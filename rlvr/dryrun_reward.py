"""Phase-A local proof for the self-contained (hardcoded-words) reward (no GPU).

Exercises the full path (extract_code -> sandbox -> scorer -> composite) on the SLM's
actual output format and asserts:
  - GOOD (real hardcoded-words programs from data/sft_hardcoded_words) score high,
  - DEGENERATE (empty grid, runtime error, junk words) score low / hit the run floor,
  - JUNK (non-parseable text) hits floor_no_code == 0,
  - MEMORIZED: a program that returns a FIXED grid regardless of word_source is caught
    by the two-run distinctness check and penalized below its un-penalized score.

Run from the repo root:  python rlvr/dryrun_reward.py [--n 4]
Best on Linux/WSL (sandbox mem cap is Linux-only); Windows works for the wiring proof.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlvr.reward import (RewardConfig, canonical_eff, compute_reward, get_palette,
                         reward_from_text, _run, _score_layout)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN = os.path.join(_ROOT, "data", "sft_hardcoded_words", "train.jsonl")
TEST_SIZE = 7  # fast; 11/15 use slow template fills


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def good_samples(n, size=TEST_SIZE):
    out = []
    for r in _rows(_TRAIN):
        eff = (r.get("meta") or {}).get("effective_spec") or {}
        if eff.get("size") != size:
            continue
        out.append((r["messages"][2]["content"], size, (r.get("meta") or {}).get("spec_id")))
        if len(out) >= n:
            break
    return out


_EMPTY = ("def generate_crossword(topic='vocabulary', word_source=None, size=7):\n"
          "    return {'rows': size, 'cols': size, 'cells': [], 'across': [], 'down': []}\n")
_ERROR = ("def generate_crossword(topic='vocabulary', word_source=None, size=7):\n"
          "    raise ValueError('boom')\n")
_JUNK_WORDS = ("def generate_crossword(topic='vocabulary', word_source=None, size=7):\n"
               "    w = 'ZQXJVK'[:size]\n"
               "    cells = [{'r': 0, 'c': c, 'letter': w[c]} for c in range(len(w))]\n"
               "    across = [{'number': 1, 'row': 0, 'col': 0, 'answer': w, 'len': len(w)}]\n"
               "    return {'rows': size, 'cols': size, 'cells': cells, 'across': across, 'down': []}\n")
DEGENERATE = [(_EMPTY, 7, "empty"), (_ERROR, 7, "error"), (_JUNK_WORDS, 7, "junk_words")]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _score_bucket(label, samples, palette, cfg):
    print(f"== {label} ==")
    rs = []
    for text, size, sid in samples:
        r, bd = reward_from_text(text, size, palette, cfg)
        rs.append(r)
        extra = bd.get("reason") or (f"g={bd.get('r_graded')} b={bd.get('r_binary')} "
                                     f"vocab={bd.get('vocab_fraction')} distinct={bd.get('distinct')}")
        print(f"  {sid!s:<12} size{size}  reward={r:.3f}  {extra}")
    print(f"  mean={_mean(rs):.3f}\n")
    return rs


def memorization_test(palette, cfg):
    """Freeze a real good program's output into a literal-returner and confirm it's
    caught (memorized=True) and penalized below its own un-penalized reward."""
    print("== MEMORIZATION (two-run distinctness) ==")
    good = good_samples(1)
    if not good:
        print("  (no good sample; skipped)\n"); return True
    from pipeline.eval_harness import extract_code
    code = extract_code(good[0][0])
    run = _run(code, TEST_SIZE, [], cfg)
    if run.get("status") != "ok":
        print(f"  (good sample didn't run: {run.get('status')}; skipped)\n"); return True
    layout = run["result"]
    literal = (f"def generate_crossword(topic='vocabulary', word_source=None, size={TEST_SIZE}):\n"
               f"    return {layout!r}\n")
    r_lit, bd = reward_from_text(literal, TEST_SIZE, palette, cfg)
    eff = canonical_eff(TEST_SIZE, cfg)
    raw = compute_reward(_score_layout(layout, eff, [], run.get("runtime_s"), palette), eff, cfg)[0]
    penalized = bd.get("memorized") is True and r_lit < raw - 1e-9
    print(f"  literal-returner: reward={r_lit:.3f} (un-penalized would be {raw:.3f}), "
          f"memorized={bd.get('memorized')}")
    print(f"  [{'PASS' if penalized else 'FAIL'}] memorized grid detected AND penalized\n")
    return penalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    cfg = RewardConfig()
    palette = get_palette()
    print(f"palette: {len(palette['targets'])} targets, {len(palette['vocab_set'])} vocab; "
          f"test size={TEST_SIZE}\n")

    good = _score_bucket("GOOD (hardcoded-words solutions)", good_samples(args.n), palette, cfg)
    degen = _score_bucket("DEGENERATE (synthetic)", DEGENERATE, palette, cfg)
    junk = _score_bucket("JUNK (non-parseable)",
                         [("no code here", 7, "junk1"), ("plan only", 9, "junk2")], palette, cfg)
    memo_ok = memorization_test(palette, cfg)

    g, d, j = _mean(good), _mean(degen), _mean(junk)
    error_r = reward_from_text(_ERROR, 7, palette, cfg)[0]
    print(f"SUMMARY  good={g:.3f}  degenerate={d:.3f}  junk={j:.3f}")
    checks = [
        ("good > degenerate", g > d),
        ("good >= 0.30", g >= 0.30),
        ("degenerate <= 0.50", d <= 0.50),
        ("junk == floor_no_code (0.0)", abs(j - cfg.floor_no_code) < 1e-9),
        ("error hits no-run floor", abs(error_r - cfg.floor_no_run) < 1e-9),
        ("memorization caught + penalized", memo_ok),
    ]
    ok = True
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
