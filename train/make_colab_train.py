"""Generate train/colab_train_qlora.ipynb — QLoRA fine-tune of Qwen3-4B on our SFT data.

Adapted from the standard Llama-2 QLoRA SFT notebook (training_example.ipynb), but:
  * base model = Qwen3-4B-Instruct (not Llama-2), modern transformers/trl/peft
  * dataset = our chat JSONL (data/sft/{train,dev,eval}.jsonl), size-upsampled
  * RESPONSE-ONLY loss (mask system+user; train only on the assistant program)
  * dev split used for in-training validation; eval split left untouched
  * save LoRA + merged fp16 model; points to colab_eval.ipynb for base-vs-tuned eval

    python train/make_colab_train.py
"""

import json
import os

MD, CO = "markdown", "code"

REPO_URL = "https://github.com/Avaneesh-Ramesh-07/CrosswordSLM.git"

cells = [
 (MD, "# QLoRA fine-tune: Qwen3-4B → crossword-generator SLM\n\n"
      "Distills the (Claude + verifier + scaffolding) pipeline into one-shot generation. "
      "Trains on `data/sft/train.jsonl` (chat: fixed system contract → minimal size-routed "
      "user prompt → verified assistant program), **response-only loss**, dev for validation, "
      "`eval` held out for the base-vs-tuned test (see `colab_eval.ipynb`)."),

 (MD, "## 1. Install (pinned, Qwen3-capable snapshot)\n"
      "Versions are **pinned**, not `>=`, on purpose: the current `trl` (1.x) **removed** "
      "`DataCollatorForCompletionOnlyLM` and **renamed** `SFTConfig(max_seq_length=)` -> "
      "`max_length=`, which would break cells 6-7 with `-U`. `trl==0.19.1` is the last "
      "release that supports **both** Qwen3 (needs `transformers>=4.51`) and the "
      "response-only collator this notebook relies on."),
 (CO, "!pip install -q -U 'transformers==4.53.*' 'trl==0.19.1' 'peft==0.16.*' "
      "'bitsandbytes==0.46.*' 'accelerate==1.8.*' 'datasets==3.6.*'"),

 (MD, "> **Expected pip warning -- safe to ignore.** You'll likely see a resolver "
      "complaint that Colab's pre-installed `gradio` wants `huggingface-hub>=1.2` but "
      "`huggingface-hub 0.36.x` is installed. That 0.3x version is **required** by "
      "`transformers 4.53`, and `gradio` is **not used** anywhere in this notebook, so the "
      "conflict is cosmetic -- the install still succeeds. **Do not upgrade "
      "`huggingface-hub`** (it would break `transformers`)."),

 (MD, "## 1b. Preflight: confirm the GPU **before** training\n"
      "Colab Pro only helps if you actually got a bigger card — Pro can still hand you a "
      "~16 GB T4, which will OOM this config (especially the fp16 merge in cell 8). This "
      "prints the GPU name + free VRAM and warns if you're under ~20 GB. Want **L4 (24 GB)** "
      "or **A100 (40 GB)**: Runtime -> Change runtime type."),
 (CO, 'import os\n'
      '# Set BEFORE torch initializes CUDA: lets the allocator grow segments instead of\n'
      '# fragmenting (the "reserved but unallocated" memory in OOM tracebacks).\n'
      'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")\n'
      'import torch\n'
      'assert torch.cuda.is_available(), "No GPU attached -- Runtime > Change runtime type > GPU (L4 or A100)."\n'
      'free_b, total_b = torch.cuda.mem_get_info()\n'
      'gb = 1024 ** 3\n'
      'name = torch.cuda.get_device_properties(0).name\n'
      'total_gb, free_gb = total_b / gb, free_b / gb\n'
      'print(f"GPU: {name}")\n'
      'print(f"VRAM: {total_gb:.1f} GB total | {free_gb:.1f} GB free")\n'
      'print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")\n\n'
      '# 4-bit Qwen3-4B QLoRA (seq-len 4096) + the fp16 merge (~8 GB) wants >= ~20 GB.\n'
      'if total_gb < 20:\n'
      '    print()\n'
      '    print("=" * 64)\n'
      '    print(f"WARNING: only {total_gb:.0f} GB VRAM -- this looks like a T4.")\n'
      '    print("This config will likely OOM (especially the fp16 merge in cell 8).")\n'
      '    print("Fix: Runtime > Change runtime type > L4 (24GB) or A100 (40GB),")\n'
      '    print("     then Runtime > Restart session and rerun from cell 1.")\n'
      '    print("=" * 64)\n'
      'else:\n'
      '    print(f"OK -- {total_gb:.0f} GB fits 4-bit Qwen3-4B QLoRA (can raise seq-len toward 8192).")'),

 (MD, "## 2. Get the data\n"
      "**Do ONE of these** (the notebook will not invent data):\n"
      "- **Clone your repo:** set `REPO_URL` below to your GitHub repo. The committed "
      "`data/sft/{train,dev}.jsonl` splits are used as-is.\n"
      "- **Upload:** put `train.jsonl`/`dev.jsonl` in a Colab folder and set `DATA_DIR` "
      "to it (leave `REPO_URL` as the placeholder).\n\n"
      "The raw per-run outputs (`runs/`) are gitignored, so a fresh clone has none; the "
      "committed splits are already merged + upsampled, so we use them directly and only "
      "rebuild when `runs/` is present -- we never clobber the committed data with empties."),
 (CO, f'REPO_URL = "{REPO_URL}"   # <-- REQUIRED unless you set DATA_DIR to an upload\n'
      'DATA_DIR = None            # <-- set to an uploaded folder to skip the clone\n'
      'import os\n\n'
      'if DATA_DIR is None:\n'
      '    assert "<REPO>" not in REPO_URL, (\n'
      '        "Set REPO_URL to your repo, OR set DATA_DIR to a folder containing "\n'
      '        "train.jsonl/dev.jsonl that you uploaded via the Files panel."\n'
      '    )\n'
      '    if not os.path.exists("slm"):\n'
      '        !git clone -q $REPO_URL slm\n'
      '    DATA_DIR = "slm/data/sft"\n'
      '    # committed splits are already merged+upsampled; only rebuild if the\n'
      '    # (gitignored) raw per-run outputs are present -- never clobber with empties.\n'
      '    if os.path.exists("slm/runs"):\n'
      '        !cd slm && python pipeline/merge_dataset.py --upsample 11=3,15=3\n\n'
      'for _f in ("train.jsonl", "dev.jsonl"):\n'
      '    assert os.path.exists(f"{DATA_DIR}/{_f}"), f"missing {_f} in {DATA_DIR}"\n'
      'print("data dir:", DATA_DIR, os.listdir(DATA_DIR))'),

 (MD, "## 3. Config\n"
      "Hyperparameters. **Batch size + gradient checkpointing are auto-tuned to the GPU** "
      "detected in cell 1b: the *effective* batch stays ~16 (the right convergence target for "
      "~2.1k rows) while the *per-device* batch scales with VRAM, and checkpointing is turned "
      "off when there's memory to spare (~28% faster). Wall-clock is set by rows×epochs×seq, "
      "not by batch size — bigger per-device batch just improves GPU utilization."),
 (CO, '# Qwen3-4B instruct. Confirm the exact HF id (alts: "Qwen/Qwen3-4B",\n'
      '# "Qwen/Qwen3-4B-Instruct"). Start from Instruct for fast SFT.\n'
      'model_name  = "Qwen/Qwen3-4B-Instruct-2507"  # base model to fine-tune FROM\n'
      'adapter_dir = "qwen3-4b-crossword-qlora"      # OUTPUT: trained LoRA adapter dir (merged -> adapter_dir + "-merged")\n'
      'output_dir  = "results"                       # trainer checkpoints + logs\n\n'
      '# QLoRA / LoRA\n'
      'lora_r, lora_alpha, lora_dropout = 32, 64, 0.05\n'
      '# Qwen attention + MLP projections\n'
      'target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]\n\n'
      '# programs are long (a full generator); give the sequence room\n'
      'max_seq_length = 4096\n\n'
      'num_train_epochs = 1\n\n'
      '# ---- throughput config: auto-tuned to the GPU detected in cell 1b ----\n'
      '# Wall-clock is set by rows x epochs x seq-len, NOT batch size. We hold the\n'
      '# EFFECTIVE batch at ~16 (right convergence target for ~2.1k rows) and scale the\n'
      '# PER-DEVICE batch with VRAM (better GPU utilization; no effect on the optimizer).\n'
      '# Gradient checkpointing stays ON on every GPU: at seq-len 4096 a 4B model OOMs\n'
      '# WITHOUT it even on a 40GB A100 (measured) -- batch>=2 activations exceed 40GB, and\n'
      '# group_by_length packs the longest sequences together, spiking peak memory.\n'
      '_vram = globals().get("total_gb", 16.0)   # from cell 1b; fallback = conservative 16GB\n'
      'gradient_checkpointing = True\n'
      'if _vram >= 38:        # A100 40GB -- headroom for a bigger micro-batch (better GEMM)\n'
      '    per_device_train_batch_size, gradient_accumulation_steps = 4, 4\n'
      'elif _vram >= 22:      # L4 24GB\n'
      '    per_device_train_batch_size, gradient_accumulation_steps = 2, 8\n'
      'else:                  # T4 16GB (or unknown) -- memory-bound, keep it minimal\n'
      '    per_device_train_batch_size, gradient_accumulation_steps = 1, 16\n'
      'per_device_eval_batch_size = per_device_train_batch_size\n'
      'eff = per_device_train_batch_size * gradient_accumulation_steps\n'
      'print(f"[auto] VRAM~{_vram:.0f}GB -> per_device_batch={per_device_train_batch_size}, "\n'
      '      f"accum={gradient_accumulation_steps} (effective ~{eff}), "\n'
      '      f"grad_checkpointing={gradient_checkpointing}")\n\n'
      'learning_rate = 2e-4\n'
      'lr_scheduler_type = "cosine"\n'
      'warmup_ratio = 0.03\n'
      'weight_decay = 0.0\n'
      'logging_steps = 10'),

 (MD, "## 4. Load Qwen3-4B in 4-bit (QLoRA) + LoRA adapters"),
 (CO, 'import torch\n'
      'from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n'
      'from peft import LoraConfig, prepare_model_for_kbit_training\n\n'
      '# T4 (Colab free tier) has no bf16 -> fall back to fp16 automatically.\n'
      'bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()\n'
      'compute_dtype = torch.bfloat16 if bf16_ok else torch.float16\n'
      'gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU!)"\n'
      'print(f"GPU: {gpu} | bf16 supported: {bf16_ok} -> compute dtype {compute_dtype}")\n\n'
      'bnb_config = BitsAndBytesConfig(\n'
      '    load_in_4bit=True,\n'
      '    bnb_4bit_quant_type="nf4",\n'
      '    bnb_4bit_compute_dtype=compute_dtype,\n'
      '    bnb_4bit_use_double_quant=True,\n'
      ')\n'
      'tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n'
      'if tokenizer.pad_token is None:\n'
      '    tokenizer.pad_token = tokenizer.eos_token\n'
      'tokenizer.padding_side = "right"\n\n'
      'model = AutoModelForCausalLM.from_pretrained(\n'
      '    model_name, quantization_config=bnb_config, device_map={"": 0},\n'
      '    torch_dtype=compute_dtype, trust_remote_code=True,\n'
      ')\n'
      'model.config.use_cache = False\n'
      'model = prepare_model_for_kbit_training(\n'
      '    model, use_gradient_checkpointing=gradient_checkpointing,   # auto-set in cell 3\n'
      '    # reentrant=True: Qwen3 saves a different tensor count on recompute, which trips\n'
      '    # the non-reentrant checkpointer ("A different number of tensors was saved...").\n'
      '    gradient_checkpointing_kwargs={"use_reentrant": True},\n'
      ')\n\n'
      'peft_config = LoraConfig(\n'
      '    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,\n'
      '    target_modules=target_modules, bias="none", task_type="CAUSAL_LM",\n'
      ')'),

 (MD, "## 5. Load data + render the Qwen chat template\n"
      "Each row is `{messages:[system,user,assistant]}`. We render it with the model's own "
      "chat template into a `text` field; the response-only collator (next cell) then masks "
      "everything up to the assistant turn so loss is computed **only on the program**."),
 (CO, 'import json\n'
      'from datasets import Dataset, DatasetDict\n\n'
      '# Load ONLY the `messages` column. The per-row `meta` is curation-only and has an\n'
      '# INCONSISTENT schema across sources (template rows add meta.engine/selection/subset\n'
      '# + effective_spec.approach; effective_spec.time_budget_s is mixed int/float), so\n'
      '# load_dataset("json", ...) fails inferring one Arrow struct for meta. messages is\n'
      '# uniform (list of {role,content} strings), so we drop meta entirely.\n'
      'def _load_split(path):\n'
      '    rows = [{"messages": json.loads(l)["messages"]}\n'
      '            for l in open(path, encoding="utf-8") if l.strip()]\n'
      '    return Dataset.from_list(rows)\n\n'
      'ds = DatasetDict({\n'
      '    "train": _load_split(f"{DATA_DIR}/train.jsonl"),\n'
      '    "dev":   _load_split(f"{DATA_DIR}/dev.jsonl"),\n'
      '})\n\n'
      'def render(row):\n'
      '    # add_generation_prompt=False -> include the assistant turn as the target\n'
      '    return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False,\n'
      '                                                   add_generation_prompt=False)}\n\n'
      'ds = ds.map(render, remove_columns=[c for c in ds["train"].column_names if c != "text"])\n'
      'print(ds)\n'
      'print("\\n--- one rendered example (head) ---\\n", ds["train"][0]["text"][:600])'),

 (MD, "## 6. Response-only loss\n"
      "Qwen renders the assistant turn after `<|im_start|>assistant\\n`. Masking up to that "
      "marker means gradients flow only through the generated program, not the (fixed) system "
      "contract or user prompt."),
 (CO, 'from trl import DataCollatorForCompletionOnlyLM\n'
      'response_template = "<|im_start|>assistant\\n"   # Qwen chat-template assistant marker\n'
      'collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)\n'
      '# sanity: confirm the marker tokenizes and is found in a sample\n'
      'assert response_template in ds["train"][0]["text"], "assistant marker not found — check template"'),

 (MD, "## 7. Train (dev = in-training validation; eval stays untouched)"),
 (CO, 'from trl import SFTTrainer, SFTConfig\n\n'
      'args = SFTConfig(\n'
      '    output_dir=output_dir,\n'
      '    num_train_epochs=num_train_epochs,\n'
      '    per_device_train_batch_size=per_device_train_batch_size,\n'
      '    per_device_eval_batch_size=per_device_eval_batch_size,\n'
      '    gradient_accumulation_steps=gradient_accumulation_steps,\n'
      '    learning_rate=learning_rate,\n'
      '    lr_scheduler_type=lr_scheduler_type,\n'
      '    warmup_ratio=warmup_ratio,\n'
      '    weight_decay=weight_decay,\n'
      '    logging_steps=logging_steps,\n'
      '    optim="paged_adamw_8bit",   # QLoRA-standard 8-bit optimizer: less optimizer memory -> room for a bigger batch\n'
      '    group_by_length=True,       # bucket similar-length rows so batches are not padded up to 4096 (matters once batch>1)\n'
      '    bf16=bf16_ok,\n'
      '    fp16=not bf16_ok,\n'
      '    gradient_checkpointing=gradient_checkpointing,   # auto-set in cell 3\n'
      '    gradient_checkpointing_kwargs={"use_reentrant": True},   # must match cell 4 (fixes Qwen3 CheckpointError)\n'
      '    max_length=max_seq_length,   # canonical arg; max_seq_length is deprecated/ignored\n'
      '    dataset_text_field="text",\n'
      '    packing=False,   # required for response-only masking\n'
      '    eval_strategy="epoch",\n'
      '    save_strategy="epoch",\n'
      '    load_best_model_at_end=True,\n'
      '    metric_for_best_model="eval_loss",\n'
      '    report_to="tensorboard",\n'
      ')\n\n'
      'trainer = SFTTrainer(\n'
      '    model=model,\n'
      '    args=args,\n'
      '    train_dataset=ds["train"],\n'
      '    eval_dataset=ds["dev"],\n'
      '    processing_class=tokenizer,   # tokenize the `text` field with our configured tokenizer\n'
      '    peft_config=peft_config,\n'
      '    data_collator=collator,\n'
      ')\n'
      'trainer.train()'),

 (MD, "## 8. Save LoRA adapter + merged fp16 model"),
 (CO, 'trainer.model.save_pretrained(adapter_dir)\n'
      'tokenizer.save_pretrained(adapter_dir)\n'
      'print("saved LoRA adapter to", adapter_dir)\n\n'
      '# merge to a standalone fp16 model for inference / GGUF export\n'
      'from peft import PeftModel\n'
      'import torch, gc\n'
      'del model, trainer; gc.collect(); torch.cuda.empty_cache()\n'
      'base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,\n'
      '                                            device_map={"": 0}, trust_remote_code=True)\n'
      'merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()\n'
      'merged.save_pretrained(adapter_dir + "-merged")\n'
      'tokenizer.save_pretrained(adapter_dir + "-merged")\n'
      'print("saved merged model to", adapter_dir + "-merged")'),

 (MD, "## 9. Persist to Drive"),
 (CO, 'from google.colab import drive\n'
      'drive.mount("/content/drive")\n'
      '!mkdir -p /content/drive/MyDrive/slm_ckpt\n'
      '# {adapter_dir} is interpolated by IPython from the Python namespace at runtime\n'
      '!cp -r {adapter_dir} {adapter_dir}-merged /content/drive/MyDrive/slm_ckpt/ 2>/dev/null; echo saved'),

 (MD, "## Next\n"
      "Your trained artifacts are in Drive (`MyDrive/slm_ckpt/`): the LoRA adapter "
      "(`qwen3-4b-crossword-qlora`) and the merged fp16 model (`…-merged`) for inference / "
      "GGUF export.\n\n"
      "Eval is run **separately** — the old `colab_eval.ipynb` is stale, don't use it. The "
      "goal is the base-vs-tuned comparison in `GAP_ANALYSIS.md`: score the tuned model on the "
      "pristine held-out `eval.jsonl` through the sandbox+scorer and compare against "
      "unaugmented Opus (~5–7% valid) — target is high pass@1."),
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
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_train_qlora.ipynb")
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
