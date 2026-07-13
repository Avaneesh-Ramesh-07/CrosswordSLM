### Baseline - Frontier Model Evals

We have run **three evaluations of unaugmented `claude-opus-4-8`** on this task.

**Metric key** (the columns in every per-size table below):

- **`size`**: grid dimension. Every crossword is square (`size × size`)
- **`valid%`**: the **functional** validity rate (the scorer's `valid` flag). based on following critera
    1. exactly `size × size`
    2. every across/down run ≥ 3 letters
    3. every white cell part of **both** an across and a down entry
    4. every run a real word from the provided `word_source`
    5. no crossing-letter conflicts
    6. all white cells forming one connected region. Note: black-square
  density is *reported* (`fill_density`) and used as a soft target, but is **not** a validity gate — a sparse grid can still be `valid` if its runs interlock and are all real words. This is the headline number.
- **`within%`**: `valid` **and** number of filler words ≤ 30% **and** no invalid crossing/entry connections **and** the generated program *runs* inside its per-size time budget. Per-size budgets: 7×7 = 3s, 9×9 = 5s, 11×11 = 12s.
- **`dictOK%`**: mean fraction of placed entries (≥ 3 letters) that are real dictionary words, averaged across all agents
- **`fill (n)`**: **fill density** the mean fraction of the grids that are fillable (white) squares. A real crossword is ~0.8,

## EVAL 1
- 50 `claude-opus-4-8` agents (temperature 1.0)
- Prompted to create a function to create crosswords based on given size & word list parameter (see prompt at end)
- Each emitted program is given up to **60 s** to execute — so the model is never penalized for a slow-but-correct fill
- Word list parameter is the vocabulary set found in `data/wordlists/WORD_LIST_FULLY_PURIFIED.txt`, 24,542 vocabulary words (11.1% SAT words)
- Scored across sizes 7/9/11

| size  | valid%  | within% | fill     |
| ----- | ------- | ------- | -------- |
| 7×7   | 6       | 4       | 0.82     |
| 9×9   | 10      | 6       | 0.84     |
| 11×11 | 0       | 0       | 0.35     |
| all   | **5.3** | **3.3** | **0.76** |
 
Result file: `runs/eval/fleet_fully_purified_v2.json`.

- Takeaways: Even with some coding harness and guidance, frontier models lack the ability to create valid crosswords reliably.

## EVAL 2 - Proof of Concept
- 75 `claude-opus-4-8` agents (temperature 1.0)
- 25 each prompted to create crosswords at sizes 7/9/11, respectively
- Bare-bones (no code requirement) "create me a size X size non free-form crossword to teach vocabulary" prompt with variations (based on data/sft/eval.jsonl)
- Evaluation include running all 75 emmitted programs as-is

- Results:
    - 55% couldn't produce a crossword at all or couldn't produce a fully-filled crossword
        - 28 programs crashed
        - 6 run but report no solution
        - 6 print no grid
        - 1 hangs
    - 45% prints a filled but invalid grid
        - 23 analyzable but invalid grids
            - Average 40% black squares
            - For crosswords that are ≤ 35% black squares, all have non-word crossings (`EOQUYCITY`, `BEVXGE`, `CNCL`, misspelled `PIUDENT`)
            - Crosswords that are > 35% black squares are very sparse (average for these is 50% black squares)
        - 11 print poorly-formatted and unintelligible grids

**In total, 0% were valid crosswords, due to capability failure**

- Takeaways: Frontier models default to code when asked to create a crossword. Frontier models create unreliable crosswords with standard user inputs. 

## EVAL 3
- 50 `claude-opus-4-8` agents (temperature 1.0)
- Each prompted to produce crossword-generation code with size specifications (7/9/11) (see prompt at end); code was required to fit harness
- Evaluation include running all 75 emmitted progras as-is
- Word list parameter is the vocabulary set found in `data/wordlists/WORD_LIST_FULLY_PURIFIED.txt`

| size    | valid% | within% | fill  |
| ------- | ------ | ------- | ----- |
| 7×7     | 0      | 0       | —     |
| 9×9     | 0      | 0       | —     |
| 11×11   | 0      | 0       | —     |
| **all** | **0**  | **0**   | **—** |

**0 / 150 valid** — 100% parse rate (every agent emitted runnable code), so the failure is not code quality, it's **termination**: at the default 60 s execution budget.

**Does more execution time rescue them? No.** Re-running all 150 saved programs at a **5× budget (300 s)** yields an identical result: **0 / 150 valid**:

| size  | timeout (ran the full 300 s) | empty / crash (< 0.5 s) | valid |
| ----- | ---------------------------- | ----------------------- | ----- |
| 7×7   | 50                           | 0                       | 0     |
| 9×9   | 47                           | 3                       | 0     |
| 11×11 | 43                           | 7                       | 0     |

- Takeaway: Frontier models default to unusuably slow and unreliable fill algorithms.

- Result files: `runs/eval/eval3.json` (60 s harness run); `runs/eval/_eval3_slow.log` (300 s
  re-score). Emitted programs saved verbatim under `runs/eval/fleet_progs_eval3/`.

  
## Not a Claude-specific problem

Claude Opus 4.8 was the best frontier model to approach the task. GPT-5.5 and Gemini 3.5 Flash struggled on free-form crossword, the easier variant where words don't have to fully interlock.

### Baseline - Best Tuned Qwen3-4B Eval (Non-Hardcoded, T2) 

Feeding the **identical size-specific Eval 3 prompt** to the fine-tuned SLM scored through the **same harness on the same purified palette**:

| size    | valid% | within% | fill     |
| ------- | ------ | ------- | -------- |
| 7×7     | 32     | 32      | 0.84     |
| 9×9     | 8      | 4       | 0.83     |
| 11×11   | 0      | 0       | 0.84     |
| **all** | **13** | **12**  | **0.84** |

- Takeaways
    - Speed: On the identical prompt, the tuned model's generators return a valid grid in **under 60 s of execution**, whereas unaugmented Opus produces **0 valid even given 5 minutes** (EVAL 3)
    - 11x11 needs major improvements --> possibly more training samples

- Result: `finetuned-models/non_hardcoded/t2/…/tuned_nonhardcoded_*.json`; programs under `…/tuned_nonhardcoded_progs/`.

### Comparison - Frontier Model vs Fine-tuned
The SLM more reliably generates valid crossword-generation programs than the frontier models. These models use proper vocabulary words, generate within a reasonable time limit, and maintain the desired fill percentage.

**Example successful crosswords** (`#` = black square; scored on the purified palette):

**Fine-tuned SLM (T2)** — 9×9, valid:
```text
# A R K # # E V E
A W A I T # P A R
M A D D E N I N G
P S I # M A D # #
S H A R P N E S S
# # T O T # R H O
T R I B E S M E N
W O O # D E A L S
O W N # # E L F #
```

**Claude — EVAL 1** (`docs/eval_examples/opus_valid.py`) — 9×9, valid but **slow: 14.3 s** to fill:
```text
L A P S # P A P A
A L L I G A T O R
S T O R E R O O M
T O W E L # P L Y
# # # D A D # # #
R O T # T U N E D
E R A D I C A T E
V A L E N T I N E
S L E W # S L A P
```

**Claude — EVAL 2 and EVAL 3: no successful crossword exists.** Claude scored **0/75** (EVAL 2) and **0/150** (EVAL 3) valid, so there is no valid grid to display — the absence *is* the result. The single valid Claude crossword in any eval is the EVAL 1 one above, and even it takes **14.3 s** (exceeds the 5 s per-size budget for 9×9, so it counts toward `valid%` but fails `within%`). The tuned SLM's valid 7×7 grids finish inside the tight 3 s budget (`within% = valid% = 32%`).

**Example failed crossword**:


**Claude (EVAL 2)** — 11×11, invalid: the *across* words are mostly real (CANDID, GIANT, PROBE, FRAUD, IRATE, NAIVE, SWEET) but the *down* crossings are gibberish (`COPO`, `IDEAHOR`, `TACIAS`):
```text
C A N D I D # B L U R
O # O # D # G I A N T
P R O B E # R A S H #
O # K # A B A S H # S
# S I G H # E # P O #
T O E # O U T # W I T
A B # F R A U D # E #
C E D E # R # E B B #
I R A T E # V A L E #
A # D R # N A I V E #
S W E E T # E Y E D #
```

### Baseline - Training Samples

Evaluating the **gen-5** training-sample generators (`data/sft_non_hardcoded_enhanced` — the reliable positives T2 distills), run through the same harness on the purified palette, 3 runs each:

| size  | valid% | dictOK% | programs |
| ----- | ------ | ------- | -------- |
| 7×7   | 91     | 91      | 18       |
| 9×9   | 100    | 100     | 10       |
| 11×11 | 94     | 94      | 11       |

Unlike the old two-engine sample (100 / 83 / 100), this covers **all 39 distinct 7/9/11 generators** in the gen-5 set — the actual training data the SLM learns from is **~94% valid overall**. Engines are algorithmic fusions (MRV + forward-checking + pattern-index, with random restarts) at 7/9 and template-bearing fusions at 11. (15×15 positives — 6 programs — also score 89%, but are excluded from T2's training.)

### Representative generations (inspectable)

Two actual clean-room Opus programs are saved verbatim (each file's header records its
per-size harness verdict, under the standard per-size time budget):

- [`docs/eval_examples/opus_valid.py`](docs/eval_examples/opus_valid.py) — the *best case*: found: produces a valid **9×9**, but **times out at 7×7 and 11×11**. Even a "good" unaugmented generation is slow and size-fragile.
- [`docs/eval_examples/opus_invalid.py`](docs/eval_examples/opus_invalid.py) — a typical failure: invalid at every size (never completes a valid fill in the budget).

## EVAL 1 Prompt: General Crossword Creator
System message:

You are an expert Python programmer. When asked for code, output only code.

User message:

Write one self-contained Python function that generates a fixed-grid, American-style crossword. Output ONLY the code in a single response.

def generate_crossword(topic: str, word_source, size: int) -> dict

Requirements:
- Python standard library ONLY.
- word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. Use ONLY these words; never invent or hardcode answer words.
- CONSTRUCT and FILL the grid, then return:
  {"rows": int, "cols": int,
   "cells": [{"r","c","letter","number"(optional)}],
   "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
- Satisfy ALL: exactly size x size; every white run (across and down) >= 3 letters; every white cell part of BOTH an across and a down entry; all white cells form one connected region; every entry a real word from word_source; high white-square density.
- Handle sizes 7, 9, 11, and 15.

Output only the Python code.

## EVAL 2 Prompt: Bare Deployment
System message:


You are an expert Python programmer.

User message (one of five equivalent phrasings; `{N}` = requested size):


I need a {N}x{N} fixed-grid crossword for practicing vocabulary (non-free-form).

Other phrasings sampled (same meaning):
- Generate a {N}x{N} fixed-grid crossword to teach vocabulary (not free-form).
- Make a {N}x{N} non-free-form vocabulary crossword.
- Create a {N}x{N} fixed-grid (non-free-form) crossword about vocabulary.
- Build me a {N}x{N} vocabulary crossword on a fixed grid, not free-form.

## EVAL 3 Prompt: Size-Specific Crossword Creator
System message:

You are an expert Python programmer. When asked for code, output only code.

User message (the full contract; `{N}` = the requested size):

Write Python code to generate a {N}x{N}, fixed-grid, American-style crossword. Output ONLY the code in a single response. The main function in your code MUST BE: "def generate_crossword(topic: str, word_source, size: int) -> dict". This is the only one that works with our testing harness

Requirements:
- Python standard library ONLY.
- word_source is a dict {"theme": [...], "fill": [...]} of UPPERCASE words. Use ONLY these words; never invent or hardcode answer words.
- CONSTRUCT and FILL the grid, then return:
  {"rows": int, "cols": int,
   "cells": [{"r","c","letter","number"(optional)}],
   "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
- Satisfy ALL: exactly {N}x{N}; every white run (across and down) >= 3 letters; every white cell part of BOTH an across and a down entry; all white cells form one connected region; every entry a real word from word_source; high white-square density.

Output only the Python code.