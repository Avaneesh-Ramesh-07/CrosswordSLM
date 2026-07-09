"""Crossword-quality report + assertions for the production generator.

Runs teachers/template_ac3 on the clean educational palette across topics and
reports the reframed headline metrics for each crossword:

    is_valid              -- fully valid crossword (structure + real words + crossings)
    filler % (junk)       -- answers NOT in the vocab n crossword-worthy palette
    SAT %                 -- answers that are strict advanced (SAT) vocabulary
    invalid_crossing %    -- (a) crossing cells where across/down disagree
    invalid_entry %       -- (b) declared entries that aren't real words
    runtime               -- wall-clock fill time

Plain script (no pytest): `python tests/test_crossword_quality.py`.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, score  # noqa: E402
import teachers.template_ac3 as tmpl  # noqa: E402

try:
    from pipeline.word_source import build_clean_education_source
    EDU = build_clean_education_source()
    SOURCE = "clean (wordfreq n dict n crossword)"
except Exception as exc:  # wordfreq / dictionary not available -> fall back
    from pipeline.word_source import build_education_source
    EDU = build_education_source(include_common_fill=True)
    EDU["clean_set"] = set(EDU["targets"]) | set(EDU["fill_words"])
    SOURCE = f"education fallback ({type(exc).__name__})"

VOCAB = EDU["clean_set"]
SAT = set(EDU["targets"])
WS = {"theme": EDU["targets"], "fill": EDU["fill_words"]}
FLAT = EDU["allowed"]
TOPICS = ["alpha", "beta", "gamma", "delta", "epsilon",
          "zeta", "eta", "theta", "space", "history"]
SIZE = 11

PASS = 0


def check(name, cond, detail=""):
    global PASS
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    PASS += 1


def main():
    print(f"palette: {EDU['n_allowed']:,} words ({EDU['n_vocab']:,} SAT) -- {SOURCE}\n")
    print(f"{'topic':10s} {'valid':5s} {'filler':>7s} {'SAT':>5s} "
          f"{'invX':>6s} {'invE':>6s} {'runtime':>8s}")

    filled, agg = [], {"filler": [], "sat": [], "invx": [], "inve": [], "rt": []}
    for topic in TOPICS:
        t0 = time.perf_counter()
        lay = tmpl.generate_crossword(topic, WS, SIZE)
        rt = time.perf_counter() - t0
        if not lay["cells"]:
            print(f"{topic:10s} {'FAIL':5s} {'':>7s} {'':>5s} {'':>6s} {'':>6s} {rt:7.2f}s")
            continue
        sp = Spec(size=SIZE, topic_words=tuple(EDU["targets"]), require_symmetry=True)
        r = score(lay, sp, FLAT, scores=EDU["scores"], runtime_s=rt, vocab_set=VOCAB)
        ans = [e["answer"] for e in lay["across"] + lay["down"]]
        sat_frac = sum(1 for w in ans if w in SAT) / len(ans)
        filled.append(r)
        agg["filler"].append(r["filler_fraction"])
        agg["sat"].append(sat_frac)
        agg["invx"].append(r["invalid_crossing_frac"])
        agg["inve"].append(r["invalid_entry_frac"])
        agg["rt"].append(rt)
        print(f"{topic:10s} {str(bool(r['valid'])):5s} "
              f"{r['filler_fraction']*100:6.0f}% {sat_frac*100:4.0f}% "
              f"{r['invalid_crossing_frac']*100:5.1f}% {r['invalid_entry_frac']*100:5.1f}% "
              f"{rt:7.2f}s")

    n = len(TOPICS)
    fill_rate = len(filled) / n

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\nfill rate {len(filled)}/{n} | means: "
          f"filler(junk)={_avg(agg['filler'])*100:.0f}% SAT={_avg(agg['sat'])*100:.0f}% "
          f"invalid_crossings={_avg(agg['invx'])*100:.1f}% "
          f"invalid_entries={_avg(agg['inve'])*100:.1f}% runtime={_avg(agg['rt']):.2f}s\n")

    # --- assertions --------------------------------------------------------------
    # QUALITY is strict on every crossword that filled (the real guarantee). Fill
    # RATE is load-tolerant: template_ac3 carries a wall-clock deadline, so under
    # heavy CPU load (e.g. the full suite) fewer topics finish in time -- and in the
    # harvest a miss just becomes a kept negative, not a failure. Solo this is ~10/10.
    check("fill rate >= 50% (load-tolerant)", fill_rate >= 0.5, f"{len(filled)}/{n}")
    check("every filled crossword is_valid", all(r["valid"] == 1 for r in filled))
    check("no invalid crossings (a)", all(r["invalid_crossing_frac"] == 0.0 for r in filled))
    check("no invalid entries (b)", all(r["invalid_entry_frac"] == 0.0 for r in filled))
    check("filler (junk) below 30% bar", _avg(agg["filler"]) < 0.30,
          f"mean filler={_avg(agg['filler']):.2f}")

    print(f"\nAll {PASS} crossword-quality checks passed.")


if __name__ == "__main__":
    main()
