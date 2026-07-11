"""Generate rlvr/colab_grpo.ipynb — GRPO (RLVR) refinement of the SFT adapter.

Continues the SFT QLoRA policy with TRL's GRPOTrainer, using the verifiable reward
(rlvr/reward.py) that runs each sampled program through the existing sandbox+scorer.
Follows the train/make_colab_train.py builder pattern (edit this .py, regenerate the
.ipynb). Its env is pinned SEPARATELY from the SFT notebook: SFT pins trl==0.19.1
for the response-only collator, which predates a mature GRPOTrainer + vLLM colocation.

    python rlvr/make_colab_grpo.py
"""

import json
import os

MD, CO = "markdown", "code"
REPO_URL = "https://github.com/Avaneesh-Ramesh-07/CrosswordSLM.git"

cells = [
 (MD, "# GRPO (RLVR): refine the crossword SLM with a verifiable reward\n\n"
      "Starts from the **SFT adapter** (`qwen3-4b-crossword-qlora`) and optimizes it with "
      "GRPO. Each sampled `generate_crossword` program is run through the project's sandbox + "
      "deterministic scorer (`rlvr/reward.py`), which returns a composite of the verification "
      "criteria (valid? no invalid crossings? >=X% vocab? black squares within target? "
      "crossings>0?) plus graded shaping. Prompts are the same bare size-only requests the SFT "
      "model was trained on (`rlvr/prompts.py`). **Run top to bottom.**"),

 (MD, "## 1. Install (GRPO-capable env — SEPARATE from the SFT pins)\n"
      "GRPO + vLLM colocation matured AFTER `trl==0.19.1` (the SFT pin), so we install a recent "
      "`trl`/`vllm` here. If a future release breaks the API, pin the last known-good set. "
      "`wordfreq` is required and **version-sensitive** — it determines the educational palette "
      "the reward scores against (pin it so the palette is reproducible)."),
 (CO, "!pip install -q -U trl vllm peft transformers accelerate datasets bitsandbytes\n"
      "!pip install -q wordfreq"),

 (MD, "## 1b. Preflight: GPU + GRPO API check\n"
      "GRPO runs the policy AND a vLLM generation engine (+ gradients) together, so it needs "
      "more VRAM than SFT. **T4 (16 GB) is too small** — use **L4 (24 GB)** or **A100 (40 GB)**. "
      "This also confirms `GRPOConfig` exposes the fields we set below (catches API drift)."),
 (CO, r'''import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch, inspect
assert torch.cuda.is_available(), "No GPU -- Runtime > Change runtime type > L4 or A100."
gb = 1024 ** 3
total_gb = torch.cuda.mem_get_info()[1] / gb
name = torch.cuda.get_device_properties(0).name
print(f"GPU: {name} | {total_gb:.1f} GB | bf16: {torch.cuda.is_bf16_supported()}")
if total_gb < 22:
    print("=" * 60)
    print(f"WARNING: {total_gb:.0f} GB looks like a T4 -- GRPO+vLLM will likely OOM.")
    print("Use Runtime > Change runtime type > L4 (24GB) or A100 (40GB).")
    print("=" * 60)
import trl
from trl import GRPOConfig, GRPOTrainer
fields = set(inspect.signature(GRPOConfig.__init__).parameters)
need = ["num_generations", "beta", "use_vllm", "max_completion_length"]
print("trl", trl.__version__, "| GRPOConfig has:", {k: (k in fields) for k in need})'''),

 (MD, "## 2. Get the repo (code + data + wordlists) and mount Drive\n"
      "The reward imports `harness/`, `pipeline/`, `rlvr/` and needs `data/wordlists/` for the "
      "palette. The SFT adapter is read from Drive (`MyDrive/slm_ckpt/…`) where the training "
      "notebook saved it."),
 (CO, f'REPO_URL = "{REPO_URL}"\n' + r'''import os, sys
if not os.path.exists("/content/slm"):
    !git clone -q $REPO_URL /content/slm
os.chdir("/content/slm")
sys.path.insert(0, "/content/slm")   # so `import rlvr.reward` / harness / pipeline resolve
for _p in ("rlvr", "harness", "pipeline", "data/sft_hardcoded_words", "data/wordlists/words_alpha.txt"):
    assert os.path.exists(_p), f"missing {_p} -- push rlvr/ + data to the repo"

from google.colab import drive
drive.mount("/content/drive")
# the hardcoded-words SFT LoRA adapter (~268MB). GRPO CONTINUES this adapter (is_trainable=True).
# (You also have a ...-hardcoded-merged fp16 model on Drive; not needed -- the adapter + HF base is lighter.)
SFT_ADAPTER = "/content/drive/MyDrive/slm_ckpt/qwen3-4b-crossword-qlora-hardcoded"
assert os.path.exists(os.path.join(SFT_ADAPTER, "adapter_config.json")), \
    f"LoRA adapter not found at {SFT_ADAPTER} (expected an adapter dir, not the merged model)"
print("repo:", os.getcwd(), "| adapter:", SFT_ADAPTER)'''),

 (MD, "## 3. Config\n"
      "GRPO LR is far below SFT (2e-4): RL nudges an already-good policy. `num_generations` is "
      "the group size (advantages are computed within each prompt's group). The reward runs a "
      "sandbox subprocess per completion, so it — not generation — is usually the per-step cost; "
      "keep `num_generations` modest. `SMOKE=True` runs a tiny loop to validate the pipeline "
      "before committing to a full run."),
 (CO, r'''SMOKE = True   # flip to False for the full run

BASE_MODEL   = "Qwen/Qwen3-4B-Instruct-2507"   # fallback base; cell 4 reads the real one from the adapter config
GRPO_OUT     = "qwen3-4b-crossword-grpo"        # output: the continued (GRPO-refined) LoRA adapter dir
LOAD_4BIT    = True                             # 4-bit QLoRA policy (set False on A100 for cleaner vLLM sync)
USE_VLLM     = True                             # colocated fast rollouts; fallback below if it errors
SIZES        = (7,) if SMOKE else (7, 9)        # 11/15 use slow template fills (~35s x2 per rollout)

from rlvr.reward import RewardConfig
# require_symmetry=False to start (symmetry makes `valid` sparse); anneal to True later.
# The reward runs each program TWICE (own _WORDS + injected palette) for the memorization
# check, sandboxed; per-size timeouts live in rlvr/reward.py.
reward_cfg = RewardConfig(require_symmetry=False, max_workers=8)

num_generations = 4 if SMOKE else 8
grpo_kwargs = dict(
    output_dir="grpo_results",
    learning_rate=1e-6,
    per_device_train_batch_size=num_generations,   # >= num_generations and divisible by it
    gradient_accumulation_steps=1 if SMOKE else 4,
    num_generations=num_generations,
    max_prompt_length=256,          # prompts are short (bare size request)
    max_completion_length=2048,     # a full generator program (raise to 4096 if truncated)
    temperature=1.0,                # exploration for diverse programs
    beta=0.04,                      # KL to the SFT reference
    num_train_epochs=1,
    logging_steps=1,
    save_strategy="no" if SMOKE else "steps",
    save_steps=50,
    report_to="none" if SMOKE else "tensorboard",
    gradient_checkpointing=True,
    use_vllm=USE_VLLM,
)
print("SMOKE:", SMOKE, "| num_generations:", num_generations)'''),

 (MD, "## 4. Load the base in 4-bit + continue the SFT LoRA as the trainable policy\n"
      "The base id is read from the adapter's own `adapter_config.json` (so it can't drift), loaded "
      "4-bit (nf4), then your SFT adapter is attached with `is_trainable=True` — GRPO refines the "
      "*exact* weights SFT learned. The tokenizer ships inside the adapter dir."),
 (CO, r'''import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bf16_ok = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

base_id = json.load(open(os.path.join(SFT_ADAPTER, "adapter_config.json"))).get(
    "base_model_name_or_path") or BASE_MODEL
print("base model:", base_id)

tokenizer = AutoTokenizer.from_pretrained(SFT_ADAPTER, trust_remote_code=True)  # adapter dir ships the tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # generation-time padding

quant = None
if LOAD_4BIT:
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id, quantization_config=quant, device_map={"": 0},
    torch_dtype=compute_dtype, trust_remote_code=True)

# continue the SFT adapter as the trainable policy (resume its weights, don't reinit)
policy = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)
policy.print_trainable_parameters()'''),

 (MD, "## 5. Reward + prompt dataset\n"
      "`build_grpo_dataset` reads the hardcoded-words corpus (`data/sft_hardcoded_words/`) and "
      "yields the ~5-per-size unique prompts for `SIZES`, each with a flat `size` column TRL "
      "forwards to the reward. Low prompt diversity is inherent (it IS the deployment "
      "distribution) — `n_repeats` just adds optimizer steps; real variety comes from "
      "`num_generations`/`temperature`. The reward warms a palette + the 370k-word dictionary "
      "and runs untrusted model code sandboxed."),
 (CO, r'''from rlvr.reward import make_reward_fn, get_palette, get_dictionary
from rlvr.prompts import build_grpo_dataset

get_palette(); get_dictionary()   # warm caches once (needs wordfreq + data/wordlists/)
reward_fn = make_reward_fn(reward_cfg)

n_repeats = 2 if SMOKE else 20
train_ds = build_grpo_dataset(sizes=SIZES, n_repeats=n_repeats)   # default path: data/sft_hardcoded_words
if SMOKE:
    train_ds = train_ds.select(range(min(4, len(train_ds))))
print(train_ds, "\n example:", train_ds[0]["prompt"][-1]["content"], "| size", train_ds[0]["size"])'''),

 (MD, "## 6. Train with GRPO\n"
      "Known sharp edge: **LoRA + vLLM colocation weight-sync** can error on some trl/vllm "
      "combos. If cell errors mention vLLM/weight sync, set `USE_VLLM=False` in cell 3 and rerun "
      "(HF generation — slower but robust), or on an A100 set `LOAD_4BIT=False`."),
 (CO, r'''from trl import GRPOConfig, GRPOTrainer

if SMOKE:
    grpo_kwargs["max_steps"] = 5

args = GRPOConfig(bf16=bf16_ok, fp16=not bf16_ok, **grpo_kwargs)
trainer = GRPOTrainer(
    model=policy,
    reward_funcs=[reward_fn],
    args=args,
    train_dataset=train_ds,
    processing_class=tokenizer,
)
trainer.train()
print("reward log tail:", [round(h.get("reward", 0), 3) for h in trainer.state.log_history if "reward" in h][-10:])'''),

 (MD, "## 7. Save the GRPO adapter to Drive"),
 (CO, r'''trainer.model.save_pretrained(GRPO_OUT)
tokenizer.save_pretrained(GRPO_OUT)
!mkdir -p /content/drive/MyDrive/slm_ckpt
!cp -r {GRPO_OUT} /content/drive/MyDrive/slm_ckpt/ 2>/dev/null; echo "saved {GRPO_OUT} to Drive"'''),

 (MD, "## Next\n"
      "Compare SFT vs RLVR on the pristine held-out `data/sft_hardcoded_words/eval.jsonl` with "
      "`rlvr/eval_compare.py` (reuses `pipeline/eval_harness.py`): valid_rate, within_spec_rate, "
      "coverage, crossings, vocab/filler_fraction, invalid_crossing/entry_frac. If SMOKE looked "
      "healthy (reward varies across steps, KL stable, adapter saved), set `SMOKE=False` and "
      "rerun for the full run."),
]


def build():
    nb_cells = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == CO:
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_grpo.ipynb")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    return out


if __name__ == "__main__":
    path = build()
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    print(f"wrote {path}")
    print(f"cells: {len(nb['cells'])} "
          f"({sum(c['cell_type']=='code' for c in nb['cells'])} code, "
          f"{sum(c['cell_type']=='markdown' for c in nb['cells'])} md)")
