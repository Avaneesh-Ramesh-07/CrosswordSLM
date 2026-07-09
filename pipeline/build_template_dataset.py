"""Build the fixed-template SFT section (size 11/15) and merge it, size-routed.

Emits many distinct-but-valid fixed-template generator programs (engine x template
selection x template subset), verifies each actually fills within budget with the
clean palette, then writes standard chat-JSONL SFT records:
  system  = the shared contract (build_dataset.SYSTEM)
  user    = the minimal size-based prompt (render_user_prompt) -> size routing
  assistant = a verified fixed-template program
Records carry kind="fixed_template" so they're distinguishable, but use the SAME
prompt convention as 7/9 -> the model learns big size => select-a-template + fill,
small size => construct-from-scratch. Output dir matches the rest (train/dev/eval
+ meta.effective_spec) so dataset_stats and the trainer union it in.

    python pipeline/build_template_dataset.py --size 15 --library data/templates_15.json --out runs/templates15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.scorer import Spec, score
from pipeline.build_dataset import SYSTEM, render_user_prompt
from pipeline.emit_template_generator import emit
from pipeline.word_source import build_clean_education_source

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# size -> (time_budget_s, density_target) mirrors spec_generator.TIME_BUDGET_S
_BUDGET = {11: 12.0, 13: 20.0, 15: 30.0}

ENGINES = {
    "ac3_lcv": (os.path.join(_ROOT, "generations", "gen3", "ac3_lcv.py"), None),
    "mrv_fc_theme": (os.path.join(_ROOT, "generations", "gen2", "mrv_fc_theme.py"), None),
}
SELECTIONS = ["shuffle", "compact_first", "sparse_first", "fixed"]


def _subsets(templates):
    """Named template subsets -> textual + behavioral variety across programs."""
    n = len(templates)
    out = {"full": templates}
    if n >= 8:
        out["firsthalf"] = templates[: n // 2]
        out["secondhalf"] = templates[n // 2:]
        out["even"] = templates[::2]
        out["odd"] = templates[1::2]
    return out


def _phash(code):
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:16]


def _verify(code, spec, ws, scores, clean_set, words, topics=("vocabulary", "words")):
    """Exec our own emitted program and score it on a few topics. Returns
    (ok, metrics) where ok requires ALL topics valid, filler<=0.30, in budget."""
    g = {}
    try:
        exec(compile(code, "<emit>", "exec"), g)
    except Exception as e:  # noqa: BLE001
        return False, {"error": str(e)}
    covs, fils, rts, valids = [], [], [], 0
    for topic in topics:
        t = time.perf_counter()
        try:
            lay = g["generate_crossword"](topic, ws, spec.size)
        except Exception as e:  # noqa: BLE001
            return False, {"error": f"run: {e}"}
        dt = time.perf_counter() - t
        m = score(lay, spec, words, scores=scores, runtime_s=dt, vocab_set=clean_set)
        valids += m["valid"]
        covs.append(m["coverage"]); fils.append(m["filler_fraction"] or 0.0); rts.append(dt)
    ok = (valids == len(topics) and max(fils) <= 0.30 and max(rts) <= spec.time_budget_s)
    return ok, {"coverage": round(sum(covs) / len(covs), 3),
                "filler_fraction": round(max(fils), 3),
                "runtime_s": round(max(rts), 2), "valid_rate": valids / len(topics)}


def build_variants(templates, size, spec, ws, scores, clean_set, words, log):
    """Emit + verify every (engine x selection x subset) config; keep valid, distinct."""
    subsets = _subsets(templates)
    kept, seen = [], set()
    total = 0
    # keep the program's wall deadline under the spec time budget (with margin) so
    # emitted programs never blow the runtime<=budget check (esp. 11x11 @ 12s).
    budget = spec.time_budget_s
    gen_total = round(0.65 * budget, 1)
    gen_sub = round(max(4.0, 0.25 * budget), 1)
    for ename, (epath, fe) in ENGINES.items():
        # mrv_fc fills 15x15 slowly (~13s) and less reliably -> a few configs only,
        # just for algorithm diversity; ac3_lcv (fast, reliable) carries the bulk.
        sels = SELECTIONS if ename == "ac3_lcv" else ["shuffle", "compact_first"]
        sub_names = None if ename == "ac3_lcv" else {"full"}
        for sel in sels:
            for sname, subs in subsets.items():
                if len(subs) < 4 or (sub_names and sname not in sub_names):
                    continue
                total += 1
                note = f"engine={ename} selection={sel} subset={sname}({len(subs)})"
                code = emit(subs, engine_path=epath, selection=sel, fill_extra=fe,
                            total=gen_total, sub=gen_sub, note=note)
                ph = _phash(code)
                if ph in seen:
                    continue
                ok, m = _verify(code, spec, ws, scores, clean_set, words)
                tag = "keep" if ok else "drop"
                log(f"  [{tag}] {note}  valid={m.get('valid_rate')} "
                    f"cov={m.get('coverage')} filler={m.get('filler_fraction')} "
                    f"rt={m.get('runtime_s')}s" + (f" ERR {m.get('error')}" if m.get('error') else ""))
                if ok:
                    seen.add(ph)
                    kept.append({"code": code, "program_hash": ph, "engine": ename,
                                 "selection": sel, "subset": sname, "metrics": m})
    log(f"  variants: kept {len(kept)}/{total} verified-valid, distinct")
    return kept


def _split(spec_id):
    r = int(hashlib.sha1(spec_id.encode()).hexdigest(), 16) % 10
    return "train" if r < 8 else ("dev" if r == 8 else "eval")


def build_records(kept, size, n_records):
    """Pair verified variants with size-routed specs -> chat SFT records."""
    tb = _BUDGET.get(size, float(size * 2))
    rows = []
    for i in range(n_records):
        sid = f"t{size}_{i:04d}"
        var = kept[i % len(kept)]
        eff = {"spec_id": sid, "size": size, "require_symmetry": True, "min_word_len": 3,
               "time_budget_s": tb, "density_target": 0.80, "topic": "vocabulary",
               "split": _split(sid), "approach": "fixed_template"}
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": render_user_prompt(eff)},
                {"role": "assistant", "content": f"```python\n{var['code'].strip()}\n```"},
            ],
            "meta": {"spec_id": sid, "kind": "fixed_template", "engine": var["engine"],
                     "selection": var["selection"], "subset": var["subset"],
                     "program_hash": var["program_hash"], "effective_spec": eff,
                     "split": eff["split"]},
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--library", required=True, help="templates_<size>.json")
    ap.add_argument("--out", required=True, help="run dir; writes <out>/dataset/{train,dev,eval}.jsonl")
    ap.add_argument("--n-records", type=int, default=None, help="default = distinct variants x 6")
    args = ap.parse_args(argv)

    lib = json.load(open(args.library, encoding="utf-8"))
    templates = lib["templates"]
    print(f"library: {len(templates)} fillable {args.size}x{args.size} templates ({args.library})")

    edu = build_clean_education_source(max_len=args.size)
    ws = {"theme": edu["targets"], "fill": edu["fill_words"]}
    words = edu["targets"] + edu["fill_words"]
    spec = Spec(size=args.size, topic_words=tuple(edu["targets"]), require_symmetry=True,
                min_word_len=3, time_budget_s=_BUDGET.get(args.size, args.size * 2))

    def log(m):
        print(m, flush=True)

    kept = build_variants(templates, args.size, spec, ws, edu["scores"], edu["clean_set"], words, log)
    if not kept:
        print("!! no verified variants; aborting")
        return
    n = args.n_records or min(240, len(kept) * 6)
    rows = build_records(kept, args.size, n)

    ddir = os.path.join(args.out, "dataset")
    os.makedirs(ddir, exist_ok=True)
    counts = {"train": 0, "dev": 0, "eval": 0}
    handles = {s: open(os.path.join(ddir, f"{s}.jsonl"), "w", encoding="utf-8") for s in counts}
    for r in rows:
        s = r["meta"]["split"]
        handles[s].write(json.dumps(r) + "\n")
        counts[s] += 1
    for h in handles.values():
        h.close()
    # a transparency dump of the distinct programs kept
    with open(os.path.join(args.out, "variants.jsonl"), "w", encoding="utf-8") as fh:
        for v in kept:
            fh.write(json.dumps({k: v[k] for k in ("program_hash", "engine", "selection",
                                                   "subset", "metrics")}) + "\n")
    print(f"\n{args.size}x{args.size} fixed-template section: {len(kept)} distinct programs, "
          f"{len(rows)} records -> {ddir}")
    print(f"  splits: {counts}")


if __name__ == "__main__":
    main()
