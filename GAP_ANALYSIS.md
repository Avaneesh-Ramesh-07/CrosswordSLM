# Gap Analysis: what you'd have to build yourself to use a raw LLM

## Thesis

Reach for a frontier model like Claude to produce this artifact — a generator that
reliably emits a *valid, dense, vocabulary-optimized* fixed-grid crossword program —
and you quickly discover it is **not turnkey**. Before Claude can return something
even *passable*, **the user has to define and supply the entire apparatus around the
model**: an eval harness to know whether the output is correct, the existing
algorithmic approaches to scaffold from, a word source and vocabulary definition, and
a precise task contract. And after all of that is built, the output is **still
inconsistent** — during this project's own development, **~25% of first-run scripts
were failures** (9 of 12 valid on the first try).

The gap this project closes is therefore not "LLMs can't write crosswords." It is:
**the raw model shifts an enormous amount of undifferentiated work onto the user, and
still doesn't deliver reliably.** The fine-tuned SLM's job is to put that whole
apparatus *into the weights* so the user doesn't have to assemble it — and to remove
the 25% first-run failure by emitting a verified-quality generator on the first try.

**And Claude is the *best* case.** Everything in this document is measured against
Claude, which was the strongest frontier model we tried on the task. The others were
worse — and worse on an *easier* problem (see "Not a Claude-specific problem" below).

## What the user must define *before* Claude can emit something passable

None of the following ships with the model. To get a passable generator out of raw
Claude, the user has to build or specify each of these themselves:

| You must supply | Why the raw model can't do without it |
|---|---|
| **An eval harness / deterministic verifier** | The model has **no ground truth**. It cannot tell you whether its own emitted generator produces a *valid* grid — 180° symmetry, every run ≥ 3, every white cell checked both ways, every entry drawn from the word source, density in range, within the time budget. Without a scorer you wrote yourself, "passable" is unmeasurable, and the model happily returns buggy or invalid code that *looks* right. |
| **Existing approaches (algorithmic scaffolding)** | Asked cold, the model reinvents CSP fill from scratch and gets it wrong (times out, backtracks forever, leaves holes). You have to hand it the canonical methods — AC-3 / MAC arc-consistency, MRV ordering, forward checking, a pattern index, beam search — **and** a pre-verified grid-template library for large grids (random 11×11 construction fills < ~10% of the time). The model is only somewhat reliable when it's *combining given scaffolding*, not originating it. |
| **A word source + vocabulary definition** | The model never picks the words; "vocabulary-optimized" is meaningless until *you* define the target set (here: crossword-scored ∩ frequency ∩ dictionary + SAT). Leave this to the model and it invents words or fills with junk. |
| **A precise task contract (spec)** | The exact function signature, the output layout dict, and the hard rules have to be pinned down, or the model emits something unscoreable and every run is differently shaped. |
| **The loop itself** | One shot is not enough (see below). You end up building search, restarts, and a keep-the-good / label-the-bad policy around the model — i.e. re-deriving the pipeline by hand. |

## Evidence

- **Raw frontier model, unaided:** an hour with Claude yielded a generator whose output
  had **4 / 66 answers as actual vocabulary (~6%)** — one shot, no ground truth, no
  filter. Passable-looking, not actually good.
- **Even with the apparatus supplied, it's inconsistent.** The generators built during
  this project — *with* the harness, seeds, and spec already in hand — were **not**
  one-shot successes. **~25% of first-run scripts failed** (9/12 valid: deadline
  overruns, a scorer dict-vs-set bug, a failure-label substring bug, etc.). Every
  failure was caught only *because* the eval harness existed, and fixed only by
  iteration. "Passable but inconsistent" is the ceiling of the raw model even after the
  user has done all the setup work.

## EVAL — measured, unaugmented Claude (held-out, `pipeline/eval_selfmodel.py`, 2026-07-08)

The anecdote above, turned into numbers. **Setup, chosen to be scrupulously fair to the model:**

- **Clean-room prompt.** The model is given *only* the task rules + output schema — **zero
  algorithm hints** (no CSP / MRV / AC-3 / beam / templates), verified by an automated
  term scan of the exact prompt. It must originate construction *and* fill itself.
- **One shot, no augmentation.** It emits a single program; it cannot execute, test,
  iterate, or see our engines / template library / verify-loop.
- **`word_source` is injected at runtime.** The program is *handed* the full clean palette
  when scored — ~27k English words (SAT theme + educational fill) or ~17k+ Spanish
  (wordfreq ∩ a real dictionary). It is **never starved of words** — it receives the list
  as its argument, exactly as the contract specifies. (Fairness confirmed: a *working*
  generator scores 100% valid at 7/9 through this identical path.)
- **Scored by the harness + a real-dictionary check.** `fullyOK%` = structurally valid
  **and** every entry a real dictionary word — palette membership alone is not trusted
  (raw frequency lists contain acronyms/proper nouns, and stubbed fills produce
  placeholder runs).

### Result — one unaugmented Claude program, 8 seeds/size

English (sizes 7/9/11/15):

| size | valid% | fullyOK% | within% | dictOK |
|---|---|---|---|---|
| 7×7 | 0 | 0 | 0 | 0 |
| 9×9 | 0 | 0 | 0 | 0 |
| 11×11 | 0 | 0 | 0 | 0 |
| 15×15 | 0 | 0 | 0 | 50 |
| **all** | **0** | **0** | **0** | 12 |

Spanish (sizes 7/9/11):

| size | valid% | fullyOK% | within% | dictOK |
|---|---|---|---|---|
| 7×7 | 0 | 0 | 0 | 0 |
| 9×9 | 0 | 0 | 0 | 0 |
| 11×11 | 0 | 0 | 0 | 0 |
| **all** | **0** | **0** | **0** | 0 |

**Zero valid crosswords across all 56 attempts, both languages.**

### The failure is fundamental, not lazy
The program was a *competent-looking* attempt: symmetric black-square generation,
structural validation (min-3, all-checked, connected), MRV-ordered backtracking with
theme-first value ordering, and a greedy fallback. It still scored 0% because:

1. Its pattern search **accepted a zero-black, all-white grid first**, which forces a full
   N×N **word square** (every row *and* column a valid word) — effectively unfillable at
   any real size.
2. The fill then dead-ends almost instantly (`rt ≈ 0`), and the fallback leaves slots as
   `?`-strings → invalid.
3. The English-vs-Spanish gap (`dictOK` 50% at EN-15 vs **0 everywhere in Spanish**) shows
   the few "real" words were incidental English surface knowledge, not genuine fill from
   the provided `word_source`. Swap the language and even that vanishes — proof it never
   actually solved the constraint problem.

**This is the strong form of the thesis:** a one-shot generator that *reads* correct and
produces **zero** valid crosswords. The hard part — satisfying the interlocking fill from
an arbitrary word list — is precisely what unaugmented Claude fails, in any language, and
is exactly what the verified pipeline + template library supplies.

### Fleet — 100 independent Opus generations (same clean-room prompt)

To confirm the single program above wasn't just a bad draw, we ran **100 independent
`claude-opus-4-8` generations** under the *identical* clean-room prompt (temperature 1.0),
**100% of which returned parseable code**, and scored each across sizes on both languages
(n = 100 per size):

English:

| size | valid% | fullyOK% | within% | dictOK |
|---|---|---|---|---|
| 7×7 | 5 | 5 | 5 | 7 |
| 9×9 | 11 | 11 | 11 | 12 |
| 11×11 | 3 | 3 | 3 | 3 |
| 15×15 | 0 | 0 | 0 | 1 |
| **all** | **5** | **5** | **5** | 6 |

Spanish:

| size | valid% | fullyOK% | within% | dictOK |
|---|---|---|---|---|
| 7×7 | 8 | 8 | 8 | 9 |
| 9×9 | 13 | 13 | 13 | 13 |
| 11×11 | 0 | 0 | 0 | 1 |
| **all** | **7** | **7** | **7** | 8 |

Reading it:

- Across **700 scored generations**, unaugmented Opus is valid **~5% (EN) / ~7% (ES)** of
  the time — one shot, no iteration.
- It peaks at **9×9 (~11–13%)** and **collapses to 0% at the large grid** (15×15 EN,
  11×11 ES) — the dense, long-word grids are where it fails outright.
- **`fullyOK% ≈ valid%` and filler is 0% on the valid ones** → when Opus succeeds it fills
  honestly from the provided `word_source`; the low numbers are not a dirty-palette
  artifact. (The single web submission's flat 0% was simply a below-average draw.)

### Time-relaxed re-run (is the low score just a timeout artifact?)

To rule out the time budget, we re-ran the fleet (n=50) with the "return within a few
seconds" clause **removed** from the prompt and the runner timeout raised to **60 s**. It
did **not** rescue the baseline:

| size | EN valid% (tight → 60 s) | ES valid% (tight → 60 s) |
|---|---|---|
| 7×7 | 5 → 12 | 8 → 10 |
| 9×9 | 11 → 10 | 13 → 8 |
| 11×11 | 3 → **0** | 0 → **0** |
| 15×15 | 0 → **0** | — |

Small-grid rates stayed in the same ~8–12% band (differences are within sampling noise,
n=50 vs 100), and **11×11 / 15×15 stayed at 0% even with 60 seconds**. So the large-grid
collapse is a genuine capability limit — unaugmented Opus can't construct *and* fill a
dense big grid **at all**, not merely "not fast enough."

### Contrast — the verified pipeline through the *same* harness

Running the pipeline's own generators back through `eval_selfmodel` (construct engine at
7/9, the fixed-template engines at 11/15), English:

| size | valid% | fullyOK% | dictOK | engine |
|---|---|---|---|---|
| 7×7 | 100 | 100 | 100 | construct |
| 9×9 | 83 | 83 | 83 | construct |
| 11×11 | 100 | 100 | 100 | fixed-template |
| 15×15 | 100 | 100 | 100 | fixed-template |

The gap, measured end-to-end from the identical harness: **~5–7% (unaugmented Opus) vs
~83–100% (pipeline)** overall, and — at the real crossword sizes — **0% vs 100% at 11×11
and 15×15**, the exact sizes where random construction collapses and the pre-verified
template library takes over. (Honesty note: the *construct* engine is not itself flawless
— it missed one 9×9 fill in this small sample; the clean 100% at 11/15 comes from the
template library, and reliable *one-shot* generation across all sizes is what the
fine-tuned SLM is meant to distill.)

### Representative generations (inspectable)

Two actual clean-room Opus programs are saved verbatim (each file's header records its
per-size harness verdict, under the standard per-size time budget):

- [`docs/eval_examples/opus_valid.py`](docs/eval_examples/opus_valid.py) — the *best case*
  found: produces a valid **9×9**, but **times out at 7×7 and 11×11**. Even a "good"
  unaugmented generation is slow and size-fragile.
- [`docs/eval_examples/opus_invalid.py`](docs/eval_examples/opus_invalid.py) — a typical
  failure: invalid at every size (never completes a valid fill in the budget).

## Not a Claude-specific problem — Claude was the strong case

The numbers above are Claude, and Claude was the **best** of the frontier models tried.
The others did worse, and did worse on a *far easier* version of the task:

- **ChatGPT (GPT-5.5)** and **Gemini (Gemini 3.5 Flash)** struggled even on **free-form
  crosswords** — the constraint-relaxed variant with *no* fixed grid, *no* 180° symmetry,
  and *no* full-interlock requirement, where words just have to cross somewhere. That is
  dramatically easier to implement than the dense, fully-checked fixed grid this project
  targets, and they still could not do it reliably.
- So the gap is a property of **frontier LLMs on constraint-satisfaction code
  generation**, not a quirk of one vendor. Claude on the *hard* (fixed-grid) task is the
  optimistic end of the range; the rest of the frontier failing the *easy* (free-form)
  task is the pessimistic end. Both point the same direction: reliable generation has to
  come from the verified pipeline (and, ultimately, the distilled SLM), not the raw model.

## Why the pipeline succeeds where one-shot Claude fails

The reliability is a property of the **system**, not the model — and today that system
is something the *user* has to stand up:

| One-shot Claude | This pipeline (the "teacher") |
|---|---|
| single attempt | model **+ deterministic verifier + search + restarts** |
| no ground truth | every candidate **run through the harness** and scored |
| keeps whatever came out | **keep the good, label + keep the bad** (SOAR negatives) |
| invents from scratch | **reuses canonical CSP algorithms** (CS50/qxw AC-3, pattern index) as scaffolding |

Unreliable generation is *acceptable* when you verify and retain both outcomes:
successes become training solutions; failures become labeled negatives (e.g. the 1002
negatives in the bulk run). But standing up that verify-and-retain loop is exactly the
burden being described — it is work the user must currently do by hand.

## The numbers

| | validity | vocabulary | filler | time |
|---|---|---|---|---|
| Raw Claude, unaided | ~5–7% valid (100-sample eval; **0%** at 11/15) | **~6%** (4/66) | high | ~1 hour |
| Verified pipeline generators | **~83–100%** valid (**100%** at 11/15 via templates) | **~11–24%** strict-SAT (2–4×) | **~0%** crosswordese | sub-second |

Two honest caveats:

1. **The vocabulary ceiling is combinatorial, not a model failing.** In a dense,
   fully-interlocked grid, strict advanced-vocab (SAT) tops out ~20% because rare words
   don't cross-connect. *No* model or solver beats that. This is why "vocabulary" was
   broadened to a **clean educational palette** (frequency ∩ dictionary ∩
   crossword-scored, unioned with SAT) so "optimized for vocabulary" means ~100%
   *real, gettable* words instead of 6% SAT and 94% junk.
2. **Verification covers validity/filler/coverage — not taste.** The harness does not
   judge cluing, theme cleverness, or elegance. Those, if wanted, are a separate signal.

## What this means for the deliverable

- **The value is collapsing the setup burden into the weights.** Instead of the user
  hand-building an eval harness, sourcing canonical algorithms, defining a vocabulary,
  writing a spec, and wrapping a search loop around the model, the tuned SLM **distills
  the output of that whole (Claude + verifier + search + scaffolding) system into
  reliable one-shot generation** — the capability, without the apparatus.
- **The 4/66 hour and the 25% first-run failure are the eval baseline.** The
  base-vs-tuned test: give the base 4B (and one-shot Claude) the same held-out `eval`
  specs, run their emitted generators through the harness, and show they produce
  invalid / ~6%-vocab / timed-out programs at a high failure rate, while the tuned model
  produces valid, clean, in-budget generators at high pass@1. That turns the anecdote
  into the project's headline evidence.

## Prompt for these 100 Claude instances:

System message:


You are an expert Python programmer. When asked for code, output only code.

User message (the full contract):


Write one self-contained Python function that generates a fixed-grid, American-style crossword. Output ONLY the code in a single response.

def generate_crossword(topic: str, word_source, size: int) -> dict

Requirements:
- Python standard library ONLY.
- word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. Use ONLY these words; never invent or hardcode answer words.
- CONSTRUCT and FILL the grid, then return:
  {"rows": int, "cols": int,
   "cells": [{"r","c","letter","number"(optional)}],
   "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
- Satisfy ALL: exactly size x size; black squares in 180-degree rotational symmetry; every white run (across and down) >= 3 letters; every white cell part of BOTH an across and a down entry; all white cells form one connected region; every entry a real word from word_source; high white-square density.
- Handle sizes 7, 9, 11, and 15.

Output only the Python code.
