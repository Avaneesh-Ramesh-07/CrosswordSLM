# Independent-agent crossword eval — self-contained prompts

Run these in a **fresh Claude Code agent that does NOT have this repo**. Each agent needs:
1. the **Evaluator** below, saved as `crossword_eval_standalone.py`, and
2. **one** of the two prompts (English or Spanish).

**Purpose:** measure *unaugmented* Claude at this task — no repo, no engineered
engines/templates, no verify-loop, no iteration-to-convergence, one genuine attempt.
The prompts give only the task *rules* (no algorithm hints). The task is genuinely hard
for one-shot code, so a from-scratch generator is expected to be unreliable — especially
at 11×11 and 15×15. The evaluator builds its own clean word palette
(`wordfreq` ∩ a real dictionary) and checks **every entry against the real dictionary** —
palette membership alone is not trusted. Only deps: `wordfreq`, `pyspellchecker`.

When pasting into an agent, paste **the Evaluator + one prompt** together.

---

## Option A — Claude on the web (no code execution)

> **Used in the eval →** this is the prompt actually pasted into claude.ai to get the
> single **web submission** (`submission_en.py`). That one program was scored with
> `pipeline/eval_selfmodel.py` and came back **0% valid at every size, EN + ES**.

claude.ai can't `pip install` or run our scorer, so split the roles: **web Claude writes
the generator in one response** (it cannot execute or iterate — the most unaugmented,
one-shot case), then you paste that program back into the repo and it is scored there.
The program is language-agnostic (it only uses the `word_source` passed in), so **one
program is scored against BOTH the English and Spanish palettes** — no need for two web
prompts.

Paste this into Claude on the web:

```text
Write one self-contained Python function that generates a fixed-grid, American-style
crossword. Output ONLY the code in a single response — no explanation, no iteration.

def generate_crossword(topic: str, word_source, size: int) -> dict

Requirements:
- Python standard library ONLY.
- word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. Use ONLY
  these words; never invent or hardcode answer words.
- CONSTRUCT and FILL the grid, then return:
  {"rows": int, "cols": int,
   "cells": [{"r","c","letter","number"(optional)}],
   "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
- Satisfy ALL: exactly size x size; black squares in 180-degree rotational symmetry;
  every white run (across and down) >= 3 letters; every white cell part of BOTH an
  across and a down entry; all white cells connected; every entry a real word from
  word_source; high white-square density.
- Handle sizes 7, 9, 11, and 15; return within a few seconds.

Output only the Python code.
```

Then copy the program it produces, save it as `submission_en.py` in this repo, and run
(or ask me to run):

```text
python pipeline/eval_selfmodel.py --lang en --submission submission_en.py
python pipeline/eval_selfmodel.py --lang es --submission submission_en.py
```

---

## Option B — a code-capable agent (self-contained, no repo)

> **Used in the eval →** a portable delivery for a fresh code-capable agent (it writes its
> own generator, then runs the bundled evaluator). Offered as an alternative path; the
> actual *scaled* run used the **Opus API fleet (Option C)** below, not this.

## Evaluator — save as `crossword_eval_standalone.py`

```python
#!/usr/bin/env python3
"""Self-contained crossword eval (NO repo needed).  pip install wordfreq pyspellchecker
Builds a clean palette (wordfreq INTERSECT a real dictionary), runs the submitted
generate_crossword(topic, word_source, size) across sizes/seeds, and validates EVERY
entry against the real dictionary. Each run is hard-capped by a per-size time budget."""
import argparse, importlib.util, sys, time, unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

BUDGET = {7: 3.0, 9: 5.0, 11: 12.0, 13: 20.0, 15: 30.0}
TOPICS = {"en": ["vocabulary","words","study","learn","review","practice","school","language"],
          "es": ["vocabulario","palabras","estudio","aprender","repaso","escuela","idioma","examen"]}
STOP = {"MIERDA","PUTA","PUTO","JODER","CONO","POLLA","CABRON","MARICON","EYACULACION",
        "PENE","CULO","TETAS","COJONES","FUCK","SHIT","CUNT","COCK"}

def norm(w):
    w = unicodedata.normalize("NFKD", str(w))
    return "".join(c for c in w if not unicodedata.combining(c)).upper()

def build_palette(lang, max_len, freq_n=60000, min_len=3):
    import wordfreq
    from spellchecker import SpellChecker
    DICT = {norm(w) for w in SpellChecker(language=lang).word_frequency.dictionary}
    seen, ordered = set(), []
    for w in wordfreq.top_n_list(lang, freq_n):
        u = norm(w)
        if u.isalpha() and min_len <= len(u) <= max_len and u not in seen and u in DICT and u not in STOP:
            seen.add(u); ordered.append(u)
    theme = [w for w in ordered if len(w) >= 4]; tset = set(theme)
    return {"theme": theme, "fill": [w for w in ordered if w not in tset],
            "allowed": set(ordered), "DICT": DICT}

def _conn(white):
    if not white: return False
    s = next(iter(white)); seen = {s}; st = [s]
    while st:
        r, c = st.pop()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nb = (r+dr, c+dc)
            if nb in white and nb not in seen: seen.add(nb); st.append(nb)
    return len(seen) == len(white)

def _sym(white, size):
    return all(((r,c) in white) == ((size-1-r,size-1-c) in white)
               for r in range(size) for c in range(size))

def score(layout, size, allowed, theme, DICT, min_len=3):
    R = {"valid":0,"coverage":0.0,"filler":0.0,"dict_frac":0.0,"crossings":0,"n_entries":0}
    if not isinstance(layout, dict) or "across" not in layout or "down" not in layout: return R
    across, down = layout.get("across") or [], layout.get("down") or []
    grid = {}; conflict = oob = False
    def place(word, r, c, dr, dc):
        nonlocal conflict, oob
        for i, ch in enumerate(word):
            rr, cc = r+dr*i, c+dc*i
            if not (0 <= rr < size and 0 <= cc < size): oob = True; return
            if (rr,cc) in grid and grid[(rr,cc)] != ch: conflict = True
            grid[(rr,cc)] = ch
    try:
        for e in across: place(norm(e["answer"]), int(e["row"]), int(e["col"]), 0, 1)
        for e in down:   place(norm(e["answer"]), int(e["row"]), int(e["col"]), 1, 0)
    except Exception: return R
    white = set(grid)
    if not white: return R
    def runs(dr, dc):
        out = []
        for (r, c) in white:
            if (r-dr, c-dc) in white: continue
            w = ""; rr, cc = r, c; L = 0
            while (rr, cc) in white: w += grid[(rr,cc)]; L += 1; rr += dr; cc += dc
            out.append((r, c, w, L))
        return out
    hr, vr = runs(0,1), runs(1,0)
    bad_short = [x for x in hr+vr if x[3] < min_len]
    actual_a = {(r,c,w) for (r,c,w,l) in hr}; actual_d = {(r,c,w) for (r,c,w,l) in vr}
    claimed_a = {(int(e["row"]),int(e["col"]),norm(e["answer"])) for e in across}
    claimed_d = {(int(e["row"]),int(e["col"]),norm(e["answer"])) for e in down}
    answers = [w for (r,c,w,l) in hr+vr if l >= min_len]
    nonword = [w for w in answers if w not in allowed]
    valid = (not conflict and not oob and not bad_short and actual_a == claimed_a
             and actual_d == claimed_d and not nonword and _conn(white) and _sym(white, size)
             and layout.get("rows") == size and layout.get("cols") == size)
    ac = {(r,c+i) for (r,c,w,l) in hr for i in range(l)}
    dn = {(r+i,c) for (r,c,w,l) in vr for i in range(l)}
    R["valid"] = 1 if valid else 0
    R["n_entries"] = len(across)+len(down); R["crossings"] = len(ac & dn)
    if answers:
        R["coverage"] = sum(1 for w in answers if w in theme)/len(answers)
        R["filler"]   = sum(1 for w in answers if w not in allowed)/len(answers)
        R["dict_frac"]= sum(1 for w in answers if w in DICT)/len(answers)
    return R

def run_one(fn, topic, ws, size, timeout):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, topic, ws, size)
        try: return fut.result(timeout=timeout)
        except (FTimeout, Exception): return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en","es"], required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--per-size", type=int, default=8)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")] if a.sizes else ([7,9,11,15] if a.lang=="en" else [7,9,11])
    print("building palette + dictionary (wordfreq + pyspellchecker)...")
    pal = build_palette(a.lang, max(sizes))
    print(f"palette {len(pal['allowed'])} words | dictionary {len(pal['DICT'])} words")
    spec = importlib.util.spec_from_file_location("sub", a.submission)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if not hasattr(mod, "generate_crossword"): sys.exit("submission has no generate_crossword")
    ws = {"theme": pal["theme"], "fill": pal["fill"]}; tset = set(pal["theme"])
    rows = []
    for size in sizes:
        budget = BUDGET.get(size, size*2)
        for t in TOPICS[a.lang][:a.per_size]:
            t0 = time.perf_counter()
            lay = run_one(mod.generate_crossword, t, ws, size, budget*2+3)
            dt = time.perf_counter()-t0
            if lay is None:
                rows.append({"size":size,"valid":0,"within":0,"fully":0,"dict_frac":0,
                             "coverage":0,"crossings":0,"entries":0,"filler":0,"rt":round(dt,2)}); continue
            m = score(lay, size, pal["allowed"], tset, pal["DICT"])
            within = int(m["valid"] and m["filler"] <= 0.30 and dt <= budget)
            rows.append({"size":size,"valid":m["valid"],"within":within,
                         "fully":int(m["valid"] and m["dict_frac"]>=0.999),"dict_frac":m["dict_frac"],
                         "coverage":m["coverage"],"crossings":m["crossings"],"entries":m["n_entries"],
                         "filler":m["filler"],"rt":round(dt,2)})
    def agg(rs):
        n = len(rs) or 1; v = [r for r in rs if r["valid"]]; vn = len(v) or 1
        return (len(rs), sum(r["valid"] for r in rs)/n, sum(r["fully"] for r in rs)/n,
                sum(r["within"] for r in rs)/n, sum(r["dict_frac"] for r in rs)/n,
                sum(r["coverage"] for r in v)/vn, sum(r["crossings"] for r in v)/vn,
                sum(r["entries"] for r in v)/vn, sum(r["filler"] for r in v)/vn, sum(r["rt"] for r in rs)/n)
    hdr = f"{'size':>5}{'n':>4}{'valid%':>8}{'fullyOK%':>10}{'within%':>9}{'dictOK':>8}{'cov':>6}{'cross':>7}{'entries':>8}{'filler%':>9}{'rt':>7}"
    print("\n"+hdr); print("-"*len(hdr))
    for size in sizes:
        n,val,ful,wit,dic,cov,cr,en,fil,rt = agg([r for r in rows if r["size"]==size])
        print(f"{size:>5}{n:>4}{val*100:>7.0f}{ful*100:>9.0f}{wit*100:>8.0f}{dic*100:>7.0f}{cov:>6.2f}{cr:>7.0f}{en:>8.0f}{fil*100:>8.0f}{rt:>7.2f}")
    n,val,ful,wit,dic,cov,cr,en,fil,rt = agg(rows); print("-"*len(hdr))
    print(f"{'ALL':>5}{n:>4}{val*100:>7.0f}{ful*100:>9.0f}{wit*100:>8.0f}{dic*100:>7.0f}{cov:>6.2f}{cr:>7.0f}{en:>8.0f}{fil*100:>8.0f}{rt:>7.2f}")
    print("\nfullyOK% = structurally valid AND every entry a real dictionary word")

if __name__ == "__main__": main()
```

---

## PROMPT 1 — English

```
You are being evaluated as a crossword-generation model. This measures your
UNAUGMENTED, one-shot ability — no external help.

RULES (read carefully):
  - Total budget: at most 10 MINUTES.
  - Write your generator as a SINGLE genuine attempt. You MAY run your own code briefly
    to confirm it executes without crashing, but you MUST NOT change your algorithm in
    response to validity / coverage / filler results.
  - Run the provided evaluator EXACTLY ONCE, at the very end, and report its output as-is.
  - Do NOT hardcode grids or answer words; do NOT look up, copy, or adapt existing
    crossword-generation code or algorithms from anywhere; do NOT tune to the evaluator.
    Emit the program as a model would in one response.

TASK CONTRACT — write ONE self-contained Python program (standard library ONLY) defining:
    generate_crossword(topic: str, word_source, size: int) -> dict
It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
    {"rows": int, "cols": int,
     "cells": [{"r": int, "c": int, "letter": str, "number": int (optional)}],
     "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
It MUST satisfy ALL of:
  - exactly size x size;
  - black squares in 180-degree rotational symmetry;
  - every white run (across and down) >= 3 letters;
  - every white cell 'checked' = part of BOTH an across and a down entry;
  - all white cells form ONE connected region;
  - every entry a real word taken from word_source (never invented/hardcoded);
  - high white-square density (few black squares).
word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words (theme = the
vocabulary to prefer; fill = general words). Use ONLY these words. Your function must
return within: 7x7<=3s, 9x9<=5s, 11x11<=12s, 15x15<=30s (the evaluator hard-kills
longer runs). It must handle sizes 7, 9, 11, and 15.

STEPS:
1. pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org wordfreq pyspellchecker
2. Save the evaluator script you were given as crossword_eval_standalone.py
3. Write submission_en.py per the contract (single pass, <=10 min, stdlib only, only word_source).
4. Run once: python crossword_eval_standalone.py --lang en --submission submission_en.py
5. Paste the printed table verbatim, then one line: at which sizes are you "fullyOK",
   and where do you break down?
```

---

## PROMPT 2 — Spanish

```
You are being evaluated as a crossword-generation model, for SPANISH. This measures
your UNAUGMENTED, one-shot ability — no external help.

RULES (read carefully):
  - Total budget: at most 10 MINUTES.
  - Write your generator as a SINGLE genuine attempt. You MAY run your own code briefly
    to confirm it executes without crashing, but you MUST NOT change your algorithm in
    response to validity / coverage / filler results.
  - Run the provided evaluator EXACTLY ONCE, at the very end, and report its output as-is.
  - Do NOT hardcode grids or answer words; do NOT look up, copy, or adapt existing
    crossword-generation code or algorithms from anywhere; do NOT tune to the evaluator.

TASK CONTRACT — write ONE self-contained Python program (standard library ONLY) defining:
    generate_crossword(topic: str, word_source, size: int) -> dict
It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
    {"rows": int, "cols": int,
     "cells": [{"r": int, "c": int, "letter": str, "number": int (optional)}],
     "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
It MUST satisfy ALL of:
  - exactly size x size;
  - black squares in 180-degree rotational symmetry;
  - every white run (across and down) >= 3 letters;
  - every white cell 'checked' = part of BOTH an across and a down entry;
  - all white cells form ONE connected region;
  - every entry a real word taken from word_source (never invented/hardcoded);
  - high white-square density (few black squares).
word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. The words are
Spanish (already A-Z normalized). Your generator MUST be language-agnostic: use ONLY the
words in word_source and hardcode NO words of any language. Return within:
7x7<=3s, 9x9<=5s, 11x11<=12s (evaluator hard-kills longer runs). Handle sizes 7, 9, 11.

STEPS:
1. pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org wordfreq pyspellchecker
2. Save the evaluator script you were given as crossword_eval_standalone.py
3. Write submission_es.py per the contract (single pass, <=10 min, stdlib only, only word_source).
4. Run once: python crossword_eval_standalone.py --lang es --submission submission_es.py
5. Paste the printed table verbatim, then one line on where you break down. (The Spanish
   palette is thinner than English, so dense large grids are harder to fill.)
```

---

## Option C — Opus API fleet (the actual 100-agent run)

> **Used in the eval →** this is the exact `system` + `user` message sent to **each of the
> N independent `claude-opus-4-8` API calls** in `pipeline/eval_opus_fleet.py`. Every call
> is a **fresh, single-turn request** — no conversation history, no repo, no shared state
> between calls (temperature 1.0 for diversity). Their programs were scored on EN + ES;
> this produced the **100-sample averaged tables** in `GAP_ANALYSIS.md` (unaugmented Opus
> ≈ 5–7% valid, 0% at 11×11/15×15).

**System message (all calls):**

```text
You are an expert Python programmer. When asked for code, output only code.
```

**User message (the clean-room contract — identical to Option A's, rules only, no
algorithm hints):**

```text
Write one self-contained Python function that generates a fixed-grid, American-style
crossword. Output ONLY the code in a single response.

def generate_crossword(topic: str, word_source, size: int) -> dict

Requirements:
- Python standard library ONLY.
- word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. Use ONLY
  these words; never invent or hardcode answer words.
- CONSTRUCT and FILL the grid, then return:
  {"rows": int, "cols": int,
   "cells": [{"r","c","letter","number"(optional)}],
   "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
- Satisfy ALL: exactly size x size; black squares in 180-degree rotational symmetry;
  every white run (across and down) >= 3 letters; every white cell part of BOTH an
  across and a down entry; all white cells form one connected region; every entry a real
  word from word_source; high white-square density.
- Handle sizes 7, 9, 11, and 15.

Output only the Python code.
```

**Two runs, differing by one line** (both otherwise identical):

| run | last requirement line | scoring timeout | result |
|---|---|---|---|
| `opus_fleet_100.json` (n=100) | `- Handle sizes 7, 9, 11, and 15; return within a few seconds.` | per-size budget (3–30 s) | EN ~5% / ES ~7% valid |
| `opus_fleet_notimelimit.json` (n=50) | `- Handle sizes 7, 9, 11, and 15.` (time clause removed) | generous 60 s | EN ~6% / ES ~6% valid; **11×11 & 15×15 still 0%** → time was **not** the bottleneck |

**Request params:** `model=claude-opus-4-8`, `temperature=1.0`, `max_tokens=8000`, via the
`ANTHROPIC_BASE_URL` gateway. Programs are scored in the subprocess sandbox
(`run_candidate`, hard-killed at the timeout) and every entry checked against a real
dictionary.

### Reading the output

| column | meaning |
|---|---|
| `valid%` | structurally valid (size, 180° symmetry, min-run ≥3, all cells checked, connected, entries in palette) |
| **`fullyOK%`** | valid **AND** every entry is a real dictionary word — the honest quality bar |
| `within%` | valid + filler ≤ 30% + within time budget |
| `dictOK` | mean fraction of entries that are real dictionary words |
| `cov` | fraction of entries that are theme vocabulary |
| `cross` / `entries` | interlocking (checked) cells / total across+down entries |
| `filler%` | fraction of entries outside the clean palette |

A from-scratch baseline typically scores well at 7/9 and collapses at 11/15 (random
construction can't produce a fillable dense grid in the time budget) — that gap is what
a template-based approach is designed to close.
