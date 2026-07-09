# RLVR + GRPO

Refines the SFT crossword SLM with **RL from verifiable rewards**: sample
`generate_crossword` programs, run each through the project's sandbox + deterministic
scorer, and reward the composite of the verification criteria. GRPO fits because the
verifier is cheap, so group-relative advantages come from many rollouts per prompt.

## Files

| file | role |
|---|---|
| `reward.py` | verifiable reward. `make_reward_fn(cfg)` -> TRL-compatible reward. Reuses `pipeline/eval_harness.build_palette`+`extract_code` and `pipeline/oe_evaluator.evaluate_code` (sandboxed, `in_process=False`). Composite = binary gates (valid / no invalid crossings / >=X% vocab / black-squares within target / crossings>0) + graded shaping. |
| `prompts.py` | GRPO prompt set: the ~10 unique construct-from-scratch prompts (sizes 7/9) from `dataset/train.jsonl`, each with a flat `size` column for the reward. |
| `dryrun_reward.py` | Phase-A local proof (no GPU): asserts GOOD >> DEGENERATE >> non-parseable. |
| `make_colab_grpo.py` -> `colab_grpo.ipynb` | GRPO trainer notebook (builder pattern; edit the .py, regenerate). Continues the SFT LoRA as the trainable policy; TRL `GRPOTrainer` + vLLM colocation. |
| `eval_compare.py` | SFT vs RLVR on held-out `dataset/eval.jsonl` via `pipeline/eval_harness.main` (no edits to it). |
| `dataset/` | verbatim snapshot of `../data/sft/` (2026-07-09). |

## Design (locked)

- **Init policy:** continue the SFT adapter (`qwen3-4b-crossword-qlora`).
- **Prompts:** minimal, SFT-matching (bare "make an NxN vocabulary crossword"); reward
  uses a **canonical per-size Spec** (reward-time constraints aren't in the prompt).
- **Reward:** binary gates + graded shaping. "Absence of bad" credits (no filler / no
  invalid crossings / fast runtime) are gated on the grid having entries/crossings, so
  an empty grid scores ~0 (verified by the dryrun).
- **Symmetry curriculum:** `require_symmetry=False` to start (else `valid` is too sparse);
  anneal to `True` once valid-rate rises.

## Workflow

- **Phase A (local, no GPU):** `python rlvr/dryrun_reward.py` — proves the reward wiring
  and ordering. Needs `wordfreq` (`pip install wordfreq`) for the palette.
- **Phase B (Colab GPU, L4/A100):** run `colab_grpo.ipynb` (`SMOKE=True` first), then
  `rlvr/eval_compare.py` for the SFT-vs-RLVR numbers.

## Ground rules

- Source of truth is `../data/sft/`; `dataset/` is an experiment snapshot. The verifier,
  sandbox, palette, and prompt source are **reused unmodified** — nothing outside `rlvr/`
  is changed.
