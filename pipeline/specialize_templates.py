"""Specialize a multi-size crossword program to ONE grid size.

Several harvested programs bake in hardcoded black-square templates for more than one size
(module-level `_TEMPLATES_7 / _TEMPLATES_9 / _TEMPLATES_11` lists) and pick one at runtime inside
`_structures(size)`. A training record for a 9x9 prompt should carry ONLY the 9x9 template -- not the
11x11 (or 7x7) data it never uses. `specialize_to_size(code, K)` removes the other-size template
DEFINITIONS and collapses the size dispatch so the program keeps EXACTLY its size-K behavior:

  Style A:  if size==11: tpls=_TEMPLATES_11        -->  tpls = _TEMPLATES_9      (K=9)
            elif size==9: tpls=_TEMPLATES_9              tpls = None             (K=7, no template)
            else:         tpls=None
  Style B:  tpls = {7:_T7, 9:_T9, 11:_T11}.get(size) --> tpls = {9:_TEMPLATES_9}.get(size)

The size-K execution path is byte-identical (we only drop code that is dead when size==K), and the
dispatch edits do not touch `rng`, so fill behavior at K is preserved. Every result is re-parsed and,
by the caller, re-executed at K; if anything is off the ORIGINAL is kept (never a broken target).
"""
from __future__ import annotations

import ast
import re

_SIZES = {5, 7, 9, 11, 13, 15}


def _tmpl_size(name: str):
    """`_TEMPLATES_9` -> 9 (only for the recognized grid sizes); else None."""
    m = re.match(r"_TEMPL\w*?_(\d{1,2})$", name or "")
    return int(m.group(1)) if m and int(m.group(1)) in _SIZES else None


def _size_eq(test) -> int | None:
    """`size == 11` (or n/N) -> 11; else None."""
    if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id in ("size", "n", "N") and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, int) and test.comparators[0].value in _SIZES):
        return test.comparators[0].value
    return None


def _abs(line_starts, lineno, col):
    return line_starts[lineno - 1] + col


def template_sizes(code: str) -> set:
    """Sizes for which this program defines a module-level `_TEMPLATES_<N>` list."""
    out = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            n = _tmpl_size(node.targets[0].id)
            if n is not None:
                out.add(n)
    return out


def specialize_to_size(code: str, K: int):
    """Return (new_code, changed: bool, note: str). Removes every `_TEMPLATES_<N!=K>` definition and
    collapses the `_structures` size dispatch to K. Returns the original unchanged (changed=False) if
    there is nothing multi-size to strip, or if the transform would not parse / leaves a dangling ref."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, False, "parse-fail"

    tsizes = template_sizes(code)
    if not (tsizes - {K}):
        return code, False, "already-size-specific"

    src = code
    line_starts = [0]
    for ln in src.split("\n")[:-1]:
        line_starts.append(line_starts[-1] + len(ln) + 1)
    edits = []  # (abs_start, abs_end, replacement)

    # 1) drop module-level _TEMPLATES_<N> for N != K (delete whole line span incl. trailing newline)
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            n = _tmpl_size(node.targets[0].id)
            if n is not None and n != K:
                a1 = line_starts[node.lineno - 1]
                a2 = (line_starts[node.end_lineno] if node.end_lineno < len(line_starts)
                      else len(src))
                edits.append((a1, a2, ""))

    # 2) collapse the dispatch inside _structures
    fn = next((f for f in ast.walk(tree)
               if isinstance(f, ast.FunctionDef) and f.name == "_structures"), None)
    dispatch_done = False
    if fn is not None:
        # style B: a Dict {int: _TEMPLATES_*} literal
        for node in ast.walk(fn):
            if (isinstance(node, ast.Dict) and node.keys
                    and all(isinstance(k, ast.Constant) and isinstance(k.value, int) and k.value in _SIZES
                            for k in node.keys)
                    and any(isinstance(v, ast.Name) and _tmpl_size(v.id) is not None for v in node.values)):
                keys = [k.value for k in node.keys]
                repl = "{%d: _TEMPLATES_%d}" % (K, K) if K in keys else "{}"
                edits.append((_abs(line_starts, node.lineno, node.col_offset),
                              _abs(line_starts, node.end_lineno, node.end_col_offset), repl))
                dispatch_done = True
        # style A: an if/elif chain testing size==N and assigning `var = _TEMPLATES_N`
        if not dispatch_done:
            for node in ast.walk(fn):
                if not (isinstance(node, ast.If) and _size_eq(node.test) is not None):
                    continue
                # only the TOP of the chain (not an elif nested under another size-eq If)
                if any(isinstance(p, ast.If) and _size_eq(p.test) is not None and node in p.orelse
                       for p in ast.walk(fn)):
                    continue
                # find the assignment target var + the else value, walking the chain
                var, else_val = None, "None"
                cur = node
                while isinstance(cur, ast.If):
                    for st in cur.body:
                        if (isinstance(st, ast.Assign) and len(st.targets) == 1
                                and isinstance(st.targets[0], ast.Name)):
                            v = st.value
                            if isinstance(v, ast.Name) and _tmpl_size(v.id) is not None:
                                var = st.targets[0].id
                    nxt = cur.orelse
                    if len(nxt) == 1 and isinstance(nxt[0], ast.If):
                        cur = nxt[0]
                    else:  # final else block
                        for st in nxt:
                            if (isinstance(st, ast.Assign) and len(st.targets) == 1
                                    and isinstance(st.targets[0], ast.Name)):
                                var = var or st.targets[0].id
                                else_val = ast.get_source_segment(src, st.value) or "None"
                        break
                if var is None:
                    continue
                indent = " " * node.col_offset
                rhs = f"_TEMPLATES_{K}" if K in tsizes else else_val
                repl = f"{indent}{var} = {rhs}"
                a1 = line_starts[node.lineno - 1]
                a2 = (line_starts[node.end_lineno] if node.end_lineno < len(line_starts) else len(src))
                # keep the block's own trailing newline: replace up to (not incl.) next line, add newline
                edits.append((a1, a2, repl + "\n"))
                dispatch_done = True
                break

    if not dispatch_done:
        return code, False, "dispatch-not-recognized"

    # apply edits back-to-front
    out = src
    for a1, a2, repl in sorted(edits, key=lambda e: -e[0]):
        out = out[:a1] + repl + out[a2:]

    # guards: must parse, and no dangling reference to a removed template
    try:
        ast.parse(out)
    except SyntaxError:
        return code, False, "post-edit-parse-fail"
    for m in re.findall(r"_TEMPL\w*?_(\d{1,2})\b", out):
        if int(m) in _SIZES and int(m) != K:
            return code, False, f"residual _TEMPLATES_{m}"
    return out, True, "ok"
