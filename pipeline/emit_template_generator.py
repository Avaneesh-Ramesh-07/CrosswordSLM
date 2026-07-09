"""Emit self-contained fixed-template crossword generators.

A fixed-template generator is a complete `generate_crossword(topic, word_source,
size)` program that (unlike the 7/9 construct-from-scratch programs) carries a
BAKED-IN library of real NYT black-square patterns as a code constant and simply
selects one and fills it. No random construction -> no dead-end gamble at 15x15.

We reuse the PROVEN fill engine verbatim: the emitter parses an engine module
(e.g. generations/gen3/ac3_lcv.py), keeps its helper functions unchanged, drops
the construction helpers, inlines a chosen template subset, and appends a
template-driven generate_crossword. Varying (engine, template subset, selection
strategy) yields distinct-but-valid programs for SFT diversity.
"""

from __future__ import annotations

import ast
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# drop the construction + old entry point; keep EVERY other helper the engine
# defines (so per-engine dependencies like _build_pattern_index come along).
_DROP = {"generate_crossword", "_make_structure", "_structure_ok"}
# these must survive the drop or the emitted program can't fill
_REQUIRED = ["_split_source", "_index_by_length", "_slots_and_crossings",
             "_fill", "_build_layout"]

# how the emitted program orders its inlined templates before trying to fill
SELECTIONS = {
    "shuffle": "    rng.shuffle(order)",
    "compact_first": "    order.sort(key=lambda i: len(_TEMPLATES[i]))",   # denser grids first
    "sparse_first": "    order.sort(key=lambda i: -len(_TEMPLATES[i]))",   # more black first
    "fixed": "    pass  # try templates in catalog order",
}

_GEN = '''def generate_crossword(topic, word_source, size):
    deadline = time.perf_counter() + {total}
    rng = random.Random(hash((topic, size)) & 0xFFFFFFFF)
    theme, fill = _split_source(word_source)
    theme_set = set(theme)
    idx = _index_by_length(theme + fill)
    full = {{(r, c) for r in range(size) for c in range(size)}}
    order = list(range(len(_TEMPLATES)))
{selection}
    for ti in order:
        if time.perf_counter() > deadline:
            break
        black = _TEMPLATES[ti]
        white = full - {{(r, c) for (r, c) in black}}
        slots, cell_to_slots = _slots_and_crossings(white, size)
        a = _fill(slots, cell_to_slots, idx, rng, theme_set, {fill_extra}
                  deadline=min(deadline, time.perf_counter() + {sub}))
        if a and len(a) == len(slots):
            return _build_layout(white, size, slots, a)
    return {{"rows": size, "cols": size, "cells": [], "across": [], "down": []}}
'''


def engine_helpers(engine_path):
    """Return kept helpers verbatim: module-level constants (e.g. _LCV_WINDOW) that
    the helpers reference, followed by the _KEEP functions in order."""
    src = open(engine_path, encoding="utf-8").read()
    tree = ast.parse(src)
    consts = [ast.get_source_segment(src, n) for n in tree.body
              if isinstance(n, (ast.Assign, ast.AnnAssign))]
    funcs = [(n.name, ast.get_source_segment(src, n)) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name not in _DROP]
    names = {n for n, _ in funcs}
    missing = [k for k in _REQUIRED if k not in names]
    if missing:
        raise ValueError(f"{engine_path} missing required helpers: {missing}")
    blocks = []
    if consts:
        blocks.append("\n".join(consts))
    blocks.extend(s for _, s in funcs)   # keep engine's own definition order
    return "\n\n\n".join(blocks)


def _templates_const(templates):
    lines = ["_TEMPLATES = ["]
    for t in templates:
        lines.append("    " + json.dumps(t["black"]) + ",")
    lines.append("]")
    return "\n".join(lines)


def emit(templates, engine_path=None, selection="shuffle", total=18.0, sub=6.0,
         budget=200000, fill_extra=None, note=""):
    """Compose a self-contained fixed-template generator program (string).

    fill_extra: kwargs string injected into the _fill(...) call. Defaults to
    "budget=N," (ac3_lcv / mrv_fc); pass "" for engines without a budget kwarg
    (theme_beam)."""
    engine_path = engine_path or os.path.join(_ROOT, "generations", "gen3", "ac3_lcv.py")
    if fill_extra is None:
        fill_extra = f"budget={budget},"
    header = (f'"""Fixed-template crossword generator (baked-in real NYT 15x15 grids '
              f'+ {os.path.basename(engine_path).replace(".py", "")} fill). {note}\n\n'
              f'{len(templates)} pre-verified-fillable black-square patterns are inlined; '
              f'the grid is SELECTED (not randomly constructed) then filled from '
              f'word_source. Self-contained; never hardcodes answer words."""')
    parts = [
        header,
        "",
        "import random",
        "import time",
        "",
        _templates_const(templates),
        "",
        "",
        engine_helpers(engine_path),
        "",
        "",
        _GEN.format(total=total, sub=sub, fill_extra=fill_extra,
                    selection=SELECTIONS[selection]),
    ]
    return "\n".join(parts).rstrip() + "\n"


if __name__ == "__main__":
    # smoke: emit from the current library and print the header + a compile check
    lib = json.load(open(os.path.join(_ROOT, "data", "templates_15.json"), encoding="utf-8"))
    code = emit(lib["templates"][:20], selection="shuffle", note="smoke")
    compile(code, "<emitted>", "exec")
    print(f"emitted {len(code)} chars, compiles OK; {len(lib['templates'])} templates in library")
    print("\n".join(code.splitlines()[:6]))
