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

 (MD, "## 1. Install (GRPO env — SEPARATE from the SFT pins; NO vLLM)\n"
      "We install a recent `trl` for `GRPOTrainer` but **deliberately remove vLLM**: Colab ships "
      "vLLM built for CUDA 13 while its torch is CUDA 12, so importing it crashes TRL with "
      "`libcudart.so.13: cannot open shared object file`. With vLLM absent, `is_vllm_available()` "
      "is False and GRPO uses HF generation (slower but robust — fine on an A100). `wordfreq` is "
      "required and **version-sensitive** (it determines the reward's palette; pin it)."),
 (CO, "# Stop the dependency whack-a-mole: do NOT -U Colab's mutually-compatible ML stack\n"
      "# (torch/transformers/peft/accelerate/datasets/trl). Only remove the two preinstalled\n"
      "# packages that break GRPO here, and add wordfreq. The GRPOConfig arg-filter in the\n"
      "# train cell absorbs any trl API drift.\n"
      "!pip uninstall -y -q vllm torchao      # vllm(cu13) crashes TRL import; torchao 0.10 breaks recent peft\n"
      "!pip install -q trl wordfreq           # trl only if missing (keeps Colab's); wordfreq required\n"
      "import torch, transformers, peft, trl, accelerate, datasets\n"
      "print(f'torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} '\n"
      "      f'| trl {trl.__version__} | accelerate {accelerate.__version__} | datasets {datasets.__version__}')"),

 (MD, "## 1b. Preflight: GPU + GRPO API check\n"
      "GRPO holds the policy + gradients + HF generation, so give it headroom — **L4 (24 GB)** or "
      "**A100** (T4 is tight). This also confirms `from trl import GRPOTrainer` works (vLLM gone) "
      "and `GRPOConfig` exposes the fields we set below (catches API drift)."),
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
for _p in ("rlvr", "harness", "pipeline", "data/sft_hardcoded_words",
           "data/wordlists/words_alpha.txt", "data/wordlists/WORD_LIST_FULLY_PURIFIED.txt"):
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
LOAD_4BIT    = False                            # bf16 (fits L4/A100, much faster gen); True only on a <=16GB T4
USE_VLLM     = False                            # HF generation. To enable: install cu12 vllm first
                                                #   (!pip install 'vllm==0.23.*') THEN set True, else GRPOTrainer errors
SIZES        = (7,) if SMOKE else (7, 9, 11)    # 15 excluded: ~21s fills + palette lacks len>11 words

from rlvr.reward import RewardConfig
# require_symmetry=False to start (symmetry makes `valid` sparse); anneal to True later.
# The reward runs each program TWICE (own _WORDS + injected palette) for the memorization
# check, sandboxed; per-size timeouts live in rlvr/reward.py.
reward_cfg = RewardConfig(require_symmetry=False, max_workers=8)

num_generations = 2 if SMOKE else 8   # smoke: GRPO minimum group; cost = steps x num_generations x ~3.5k tokens
grpo_kwargs = dict(
    output_dir="grpo_results",
    learning_rate=1e-6,
    per_device_train_batch_size=num_generations,   # >= num_generations and divisible by it
    gradient_accumulation_steps=1 if SMOKE else 4,
    num_generations=num_generations,
    max_prompt_length=256,          # prompts are short (bare size request)
    max_completion_length=4096,     # programs are ~3k tokens (embed the _WORDS list) -> 2048 truncates
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
 (CO, r'''from rlvr.reward import make_reward_fn, get_palette, get_dictionary, get_vocab_set
from rlvr.prompts import build_grpo_dataset

get_palette(); get_dictionary(); get_vocab_set()   # warm caches (wordfreq + data/wordlists/)
reward_fn = make_reward_fn(reward_cfg)

n_repeats = 2 if SMOKE else 3   # HF-gen full run: 15 prompts x 3 x num_generations 8 = ~360 gens (~2-4h). Raise for more training.
train_ds = build_grpo_dataset(sizes=SIZES, n_repeats=n_repeats)   # default path: data/sft_hardcoded_words
if SMOKE:
    train_ds = train_ds.select(range(min(4, len(train_ds))))
print(train_ds, "\n example:", train_ds[0]["prompt"][-1]["content"], "| size", train_ds[0]["size"])'''),

 (MD, "## 6. Train with GRPO\n"
      "Generation is HF (`use_vllm=False`) — no vLLM colocation to break. If you later want faster "
      "rollouts, pin a TRL-supported cu12 vLLM (`vllm==0.23.*`) so `libcudart.so.12` matches, then "
      "set `USE_VLLM=True`. Reward runs each program twice in a sandbox — the per-step bottleneck."),
 (CO, r'''from trl import GRPOConfig, GRPOTrainer
import inspect

if SMOKE:
    grpo_kwargs["max_steps"] = 2   # ~num_generations x 2 generations total -> minutes, just proves the loop

# drop any kwargs this trl version's GRPOConfig doesn't accept (API drifts across releases)
_ok = set(inspect.signature(GRPOConfig.__init__).parameters)
_drop = [k for k in grpo_kwargs if k not in _ok]
if _drop:
    print("note: GRPOConfig doesn't accept these in this trl version, dropping:", _drop)
grpo_kwargs = {k: v for k, v in grpo_kwargs.items() if k in _ok}

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
