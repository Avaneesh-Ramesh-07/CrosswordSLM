# RLVR + GRPO

Refines the SFT crossword SLM with **RL from verifiable rewards**. The SLM (trained on
`../data/sft_hardcoded_words/`) emits a **self-contained** `generate_crossword(topic="vocabulary",
word_source=None, size=N)` whose body starts `word_source = word_source or _WORDS` — called with
no word_source it uses its **own embedded word list** and returns a layout dict. The reward runs
that program and scores the crossword it makes; GRPO uses group-relative advantages over many
sampled programs.

## Files

| file | role |
|---|---|
| `reward.py` | verifiable reward. Runs the program **twice** — once with its own `_WORDS` (graded), once with an injected palette (memorization check) — and requires the two grids to differ. Validates self-chosen words against `data/wordlists/words_alpha.txt`; educational quality = `vocab_fraction` vs the palette. Composite = binary gates + graded shaping. |
| `prompts.py` | GRPO prompt set: the ~5-per-size unique prompts (sizes 7/9/11/15) from `data/sft_hardcoded_words/train.jsonl`, each with a flat `size` column. |
| `dryrun_reward.py` | Phase-A local proof (no GPU): GOOD ≫ DEGENERATE ≫ non-parseable, + the memorization test. |
| `make_colab_grpo.py` → `colab_grpo.ipynb` | GRPO trainer notebook. Continues the SFT LoRA as the trainable policy; TRL `GRPOTrainer` + vLLM. |
| `eval_compare.py` | SFT vs RLVR via `pipeline/eval_harness.main`. **Caveat:** eval_harness injects a palette word_source, so it measures "fill a given palette", not "create your own" — align it with the reward before trusting the numbers. |

## Design (locked)

- **Init policy:** continue the SFT adapter (`qwen3-4b-crossword-qlora`).
- **Output:** self-contained script — `generate_crossword()` callable with no args, uses its own `_WORDS`.
- **Reward:** binary gates (valid / no invalid crossings / ≥70% vocab / black-squares within target / crossings>0) + graded shaping. "Absence-of-bad" credits gated on the grid having entries/crossings, so an empty grid scores ~0.
- **Anti-memorization:** two-run distinctness — a program returning one fixed grid regardless of word_source is caught and penalized (×0.30).
- **Symmetry curriculum:** `require_symmetry=False` to start; anneal to `True` later.
- **Sizes:** 7/9/11/15 (11/15 are template fills; slow — `SIZES` in the notebook defaults to 7/9).

## Workflow

- **Phase A (local, no GPU):** `pip install wordfreq && python rlvr/dryrun_reward.py`.
- **Phase B (Colab GPU, L4/A100):** run `colab_grpo.ipynb` (`SMOKE=True` first), then evaluate.

## Prereqs to run on Colab

- Push `rlvr/` to GitHub (the notebook clones the repo; it also needs `data/sft_hardcoded_words/`
  and `data/wordlists/words_alpha.txt`, which are committed).
- Put the 268 MB SFT adapter on Drive at `MyDrive/slm_ckpt/qwen3-4b-crossword-qlora`.
