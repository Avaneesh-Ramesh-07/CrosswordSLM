"""Pull the CrossWordBench dataset (HINT-lab/CrossWordBench) as compact JSONL.

The HF dataset is gated and image-heavy (400+ MB of grid renders we don't need).
We only want the two TEXT columns per puzzle:
  - puzzle_state    : {"grid": [[cell,...]], "wordlist": [[ANSWER,clue,r,c,dir]], ...}
                      where a black square is the literal "-" cell.
  - reference_answer: [{"direction": "across N", "clue": ..., "answer": ...}]

So instead of downloading the parquet, we page the HuggingFace datasets-server
`/rows` API (returns rows as JSON; images come back as URLs, not blobs) and keep
only {id, difficulty, puzzle_state, reference_answer}. Stdlib-only (urllib), so
it runs on the project's Python 3.14 with no pip installs.

Auth: reads a HuggingFace token from ~/.hf_token (or $HF_TOKEN). The token is
never printed. The dataset gate must already be accepted by that account.

Usage (from WSL, which has network access):
    python3 bench/pull_crosswordbench.py --config english
    python3 bench/pull_crosswordbench.py --config english --split 7x7
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT_DIR = os.path.join(_ROOT, "data", "crosswordbench")

DATASET = "HINT-lab/CrossWordBench"
ROWS_API = "https://datasets-server.huggingface.co/rows"
KEEP = ("id", "difficulty", "puzzle_state", "reference_answer")

# Known (config -> splits). english is the primary target for the eval.
SPLITS = {
    "english": ("7x7", "14x14"),
    "english_simple": ("7x7", "14x14"),
    "commonsenseqa": ("7x7",),
    "chinese": ("7x7",),
}


def _token() -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    path = os.path.expanduser("~/.hf_token")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    raise SystemExit("No token: set $HF_TOKEN or write it to ~/.hf_token")


def _get(url: str, token: str, retries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last = e
            # 4xx (bad token/gate) won't fix on retry; surface immediately.
            if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise SystemExit(f"HTTP {e.code} for {url} — check token scope / gate acceptance")
            time.sleep(1.5 * (attempt + 1))
    raise SystemExit(f"Failed after {retries} tries: {url} ({last})")


def pull_split(config: str, split: str, token: str, page: int = 100) -> list:
    """Page the /rows API for one (config, split); return trimmed row dicts."""
    rows, offset = [], 0
    while True:
        url = (f"{ROWS_API}?dataset={DATASET.replace('/', '%2F')}"
               f"&config={config}&split={split}&offset={offset}&length={page}")
        data = _get(url, token)
        batch = data.get("rows", [])
        if not batch:
            break
        for item in batch:
            row = item.get("row", {})
            rows.append({k: row.get(k) for k in KEEP})
        n_total = data.get("num_rows_total")
        offset += len(batch)
        print(f"  {config}/{split}: {offset}" + (f"/{n_total}" if n_total else ""))
        if n_total is not None and offset >= n_total:
            break
        if len(batch) < page:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="english", choices=sorted(SPLITS))
    ap.add_argument("--split", default=None, help="one split (default: all for config)")
    ap.add_argument("--out-dir", default=_OUT_DIR)
    args = ap.parse_args()

    token = _token()
    os.makedirs(args.out_dir, exist_ok=True)
    splits = (args.split,) if args.split else SPLITS[args.config]

    for split in splits:
        rows = pull_split(args.config, split, token)
        out = os.path.join(args.out_dir, f"{args.config}_{split}.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} puzzles -> {out}")


if __name__ == "__main__":
    main()
