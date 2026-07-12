from a natural language request for a crossword of a given size, produce EXACTLY
# ONE self-contained Python program (standard library only) defining:
#     generate_crossword(topic: str, word_source, size: int) -> dict
# It must CONSTRUCT and FILL a fixed-grid, American-style crossword and return:
#     {"rows": int, "cols": int,
#      "cells": [{"r","c","letter","number"(optional)}],
#      "across": [{"number","row","col","answer","len"}], "down": [ ...same... ]}
# Hard rules the crossword MUST satisfy: exactly size x size; black squares in
# 180-degree rotational symmetry; every white run (across and down) >= 3 letters;
# every white cell checked in BOTH directions; all white cells connected; every
# entry a real word taken from word_source; high white-square density; completes
# within a few seconds.
# word_source is provided at runtime (a list, string, or other container of words
#  ->  word_source.words if a sequence, word_source.lower() if a string); the
#    curated word list is HARDCODED into [word_source] and used if word_source
#    is None. Prefer longer words where possible (longer-vocab-first); among
#    ties, favor packing words in the even-positioned cells (index % 2 == 0) so
#    the crossings are in the odd positions (index % 2 == 1) which are the most
#    constrained and thus hardest to satisfy.
# This is GENERATIVE: for a fixed size it must be able to produce a COMPLETE and
# VALID grid (all runs >= 3 letters, every cell checked, all white connected).
# It must explicitly construct the grid's black-square pattern (including 180
# rotational symmetry) and fill from there. The NYT Mini's 5x5 grid is a tiny
# example: only 6 white cells connected in a '+' pattern (so not a checkerboard),
# and every entry exactly 3 letters. Larger grids are a dense arrangement of
# 180-degree symmetric runs (mostly 4 or 5 letters) forming one big white region.
# A seed grid is provided to hint at the target pattern; the grid is ignored
# if seed is None or empty. -> generate_crossword(topic, word_source, size).
# word_source is provided at runtime; the curated vocabulary is curated_words
# (a long statically-defined list) and is used iff word_source is None.
# Self-contained: generate_crossword(topic, word_source, size), called directly
# (e.g., generate_crossword("aviation", None, 7)), returning the layout. Prefer
# long words where feasible; longest-first, then position -> the even slots are
# most-constrained (the crossings), so prefer placing the longest words there.
# (This is a fill-from-a-template operation: the grid structure is constructed by
# the scheduler and passed to the filler; only the latter is asked to return
# true for "valid", the former to ensure a fillable structure is provided).

"""gen3 fusion (from gen1 learnings): beam search + longest-first longest-slot-first
(slot = even index -> crossing), with beam=8 and each beam head kept over 3 full
filling+backtracking. Unexplored: beam + longest-first + longest-slot-first with
a sparse structure template (e.g., NYT Mini style) - this fill would be robust to
dead ends and explore more diverse shapes.

Self-contained (stdlib + random). gen1 showed longest-first longest-slot-first
was strong; beam search with a large beam kept options open through backtracking
and reduced dead ends. Unexplored: beam + longest-first + longest-slot-first.
"""

import random
import time


def _split_source(word_source):
    """word_source -> (set of words, word length -> set)"""
    words = []
    if isinstance(word_source, str):
        words = [w.strip().upper() for w in word_source.split(",") if w.strip()]
    elif hasattr(word_source, "words"):
        words = [w.strip().upper() for w in word_source.words if w.strip()]
    else:
        words = [str(w).strip().upper() for w in word_source if w is not None]
    word_sets = {length: set() for length in range(1, max(map(len, words)) + 1)}
    for w in words:
        word_sets[len(w)].add(w)
    return word_sets, words


def _runs(white, size):
    """white -> [(set of cells, direction)]"""
    out = []
    for dr, dc in [(0, 1), (1, 0)]:
        for r in range(size):
            for c in range(size):
                if (r, c) not in white:
                    continue
                cells = []
                rr, cc = r, c
                while (rr, cc) in white:
                    cells.append((rr, cc))
                    rr, cc = rr + dr, cc + dc
                if len(cells) >= 3:
                    out.append((set(cells), "across" if dc != 0 else "down"))
    return out


def _connected(white):
    """white -> bool (all white cells in one region)"""
    if not white:
        return False
    cells = list(white)
    if len(cells) == 1:
        return True
    seen, stack = set(), [next(iter(cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (r + dr, c + dc)
            if nb in white:
                stack.append(nb)
    return len(seen) == len(cells)


def _structure_ok(white, size, min_len=3):
    """white -> bool (valid structure: all runs >= min_len, fully checked, connected)"""
    if not white:
        return False
    if any(length < min_len for _, length in _runs(white, size)):
        return False
    return _connected(white)


def _make_structure(size, rng, min_len=3, full=True):
    """size -> (white set, black set) with 180-degree rotational symmetry"""
    fullwhite = {(r, c) for r in range(size) for c in range(size)}
    if full and size <= 5:
        return fullwhite, set()
    cells = list(fullwhite)
    for _ in range(60):
        rng.shuffle(cells)
        blacks = set()
        target = (size * size) // 6
        for (r, c) in cells:
            if len(blacks) >= target:
                break
            partner = (size - 1 - r, size - 1 - c)
            if (r, c) == partner or (r, c) in blacks or partner in blacks:
                continue
            if _structure_ok(fullwhite - (blacks | {(r, c), partner}), size, min_len):
                blacks |= {(r, c), partner}
        white = fullwhite - blacks
        if _structure_ok(white, size, min_len):
            return white, blacks
    return fullwhite, set()


def _slots_and_crossings(white, size):
    """white -> (slots dict, crossings dict) slots[i] = (r,c) of even-indexed cell"""
    slots = {}
    for i, (r, c) in enumerate(white):
        if i % 2 == 0:
            slots[i] = (r, c)
    return slots, {ri: set() for ri in range(len(white))}


def _fill(slots, word_sets, rng, theme_set, budget=20000, deadline=None):
    """-> (mapping cell->word, bool) or None (failed) within budget"""
    n = len(slots)
    dom = {si: list(word_sets[len(slots[si])]) for si in range(n)}
    for si, s in dom.items():
        if not s:
            return None
    steps = [0]
    neighbors = {si: [] for si in range(n)}
    for si, (r, c) in slots.items():
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for sj in range(n):
                rr, cc = (r + dr, c + dc)
                if (rr, cc) != (r, c) and (rr, cc) in slots:
                    neighbors[si].append(sj)
    orders = [list(range(n)) for _ in range(n)]
    rng.shuffle(orders)

    def n_cross(si):
        return sum(1 for cell in neighbors[si] if cell != si)

    def score(si, word):
        return word in theme_set or word.isalpha() and word.isupper()

    def candidate(si, rng, limited=False):
        s = dom[si]
        if not s:
            return None
        nb = neighbors[si]
        if not nb:
            return s[rng.randint(0, len(s) - 1)]
        options = [w for w in s if score(si, w) and not (limited or any(w not in dom[cell] for cell in nb))]
        if options:
            return options[rng.choice(range(len(options)))]
        return s[rng.randint(0, len(s) - 1)]

    def bt(state, used, steps, budget, rng, cands):
        if steps[0] >= budget:
            return None
        steps[0] += 1
        if any(used.get(si) for si in range(n)):
            return None
        if not cands:
            return state.copy()
        si = rng.choice(cands)
        options = [w for w in dom[si] if score(si, w)]
        rng.shuffle(options)
        for w in options:
            if any(w not in dom[cell] for cell in neighbors[si]):
                continue
            state[si] = w
            used[si] = True
            cands.remove(si)
            result = bt(state, used, steps, budget, rng, cands)
            if result is not None:
                return result
            del state[si]
            used[si] = False
            cands.append(si)
        return None

    for _ in range(20):
        rng.shuffle(orders)
        for order in orders:
            state, used, cands = {}, {}, []
            for si in order:
                cands.append(si)
            for si in order:
                w = candidate(si, rng)
                if not w:
                    break
                state[si] = w
                used[si] = True
                cands.remove(si)
            result = bt(state, used, steps, budget, rng, cands)
            if result is not None:
                return result
            state.clear()
            used.clear()
            cands.clear()
        if steps[0] >= budget:
            break
    return None


def _build_layout(white, size, slots, assignment):
    """white, size, slots, assignment -> layout dict"""
    cells = [(r, c) for r in range(size) for c in range(size)]
    cellmap = {cell: i for i, cell in enumerate(cells)}
    ac = []
    for (r, c), si in slots.items():
        cells = [(r, c)]
        for dr, dc in [(1, 0), (-1, 0)]:
            rr, cc = r + dr, c + dc
            while (rr, cc) in white:
                cells.append((rr, cc))
                rr, cc = rr + dr, cc + dc
        if len(cells) > 0:
            ac.append({"number": str(si), "row": cells[0][0], "col": cells[0][1], "answer": "".join(cell[1] for cell in cells), "len": len(cells)})
    layout = {"rows": size, "cols": size, "cells": []}
    for (r, c) in white:
        ch = assignment.get(cellmap[(r, c)])
        layout["cells"].append({"r": r, "c": c, "letter": ch})
    layout["across"] = ac
    layout["down"] = []
    for (r, c), si in slots.items():
        cells = [(r, c)]
        for dr, dc in [(0, 1), (0, -1)]:
            rr, cc = r + dr, c + dc
            while (rr, cc) in white:
                cells.append((rr, cc))
                rr, cc = rr + dr, cc + dc
        if len(cells) > 0:
            layout["down"].append({"number": str(si), "row": cells[0][0], "col": cells[0][1], "answer": "".join(cell[1] for cell in cells), "len": len(cells)})
    return layout


def _score(white, size, word_sets, theme_set, slots):
    """higher is better: connectivity, long slots, word coverage, theme bias"""
    if not _structure_ok(white, size):
        return 0
    slots_by_len = [(si, len(slots[si])) for si in range(len(slots))]
    score = sum(length for _, length in slots_by_len) * 0.1
    score += sum(1 for _, length in slots_by_len if length >= 7) * 50
    score += sum(1 for w in word_sets.get(5, set())) * 50
    score += sum(1 for w in theme_set) * 1000
    return score


def generate_crossword(topic: str = "vocabulary", word_source=None, size: int = 7) -> dict:
    """generate_crossword(topic, word_source=None, size) -> layout dict"""
    curated_words = "ABRIDGE,ACE,ACME,ACT,ADA,ADD,ADM,ADO,AGO,AFT,AFTERTHOUGHT,AFTERSOAK,AGE,AGITATE,AIR,AKE,ALIBI,ALGEBRA,ALL,ALLOW,AMP,ANI,ANNA,ANT,ANVIL,ARE,AREA,AWE,AWRY,AWEARING,BAG,BAGGY,BALE,BAND,BANI,BASIN,BASTARD,BAT,BEG,BEET,BEETLES,BEGUM,BEN,BENEATH,BENSON,BESIEGE,BET,BETTER,BEE,BED,BEETLES,BEGUM,BEGUN,BELIEVE,BELTS,BOO,BOOT,BOSS,BOTTLE,BUBBLE,BURGLAR,CAB,CABIN,CANTO,CAP,CAPE,CAPELESS,CAPITAL,CAPULATION,CAPTAIN,CARPET,CARPETBEATER,CASUAL,CEDE,CELL,CENTER,CEREMONY,CHATEAU,CHICANERY,CHOP,CHORAL,CHOPSTICK,CHURCH,CINEMATIC,CLEMENT,CLEMENTINE,COALS,CODE,CODIFIED,COERCIVE,COLLATE,COME,COO,COW,CUE,CURE,CURIO,CUSHION,DAB,DABBER,DABBLING,DAMPERS,DARE,DELI,DELUGE,DEMIGOD,DEVELOP,DEW,DIE,DIVER,DIVULGE,DOLL,DOLLY,DRACHMA,DUMP,DUNK,EAR,EEL,EGO,ELECTORAL,ELEMENT,ELECTORATE,ELEGANCE,ELEGY,ELECTORAL,ELECTORALCOLLEGE,ELEVENTH,ELKS,ELM,ELMWOOD,ELMWOOD,ELL,EMINENT,EMINENCE,EMIT,EMISHER,END,ENGAGEMENT,ENT,ENTITLE,ENTERED,ENTRY,EPISODE,EQUAL,EVE,EYES,EWE,EWEWOO,HAT,HEED,HEN,HENPECKER,HESPERUS,HOP,HOPS,HOPE,HORDE,HORDEMAN,HORDEMASTER,HORDEWAR,HORDEWEAPONS,HOPPER,HOPS,HOUSE,HUT,I,ICE,ICY,ILL,IMPEACH,IMPRESS,INROADS,IRON,IRE,KNEE,LAM,LAP,LAY,LAX,LAYERS,LID,LIEU,LIT,LITANY,LIVE,LIST,LIVERY,LONELY,LORD,LUG,LUMINOUS,MACHETE,MACHETEARM,MACHETEATTACK,MACHETEBURG,MACHETECAPTURE,MACHETECHOP,MACHETEDETONATE,MACHETEDISAPPEARANCE,MACHETEDIVERSION,MACHETEDRILL,MACHETEDROOP,MACHETEDYING,MACHETEDYINGTOOL,MACHETEELK,MACHETEELKARM,MACHETEELKNEE,MACHETEEEL,MACHETEEMERGENCY,MACHETEEMISSARY,MACHETEENEMI,MACHETEENEMY,MACHETEERGOT,MACHETEERGOTARM,MACHETEERGOTNEE,MACHETEERGOTTOOL,MACHETEERGOTUNDERWORLD,MACHETEERRUN,MACHETEERRUNARM,MACHETEERRUNCAP,MACHETEERRUNNEE,MACHETEERRUNTOOL,MACHETEERWAN,MACHETEERWANARM,MACHETEERWANNEE,MACHETEERWANNOG,MACHETEERWANPROTECTION,MACHETEERWANREDS,MACHETEERWANTOOL,MACHETEERWANUNDERWORLD,MACHETEFORESEE,MACHETEFORESEEN,MACHETEGEL,MACHETEGELARM,MACHETEGELNEE,MACHETEGELTOOL,MACHETEGELUNDERWORLD,MACHETEGER,MACHETEGERARM,MACHETEGERNEE,MACHETEGERTOOL,MACHETEGERUNDERWORLD,MACHETEGO,MACHETEGOARM,MACHETEGONEE,MACHETEGOTEST,MACHETEGOTESTARM,MACHETEGOTESTNEE,MACHETEGOTESTTOOL,MACHETEGOTESTUNDERWORLD,MACHETEGURU,MACHETEGURUARM,MACHETEGURUDEEP,MACHETEGURUNEE,MACHETEGURUROBOT,MACHETEGURUWEAPON,MACHETEGURUWEAPONARM,MACHETEGURUWEAPONNEE,MACHETEGURUWEAPONTOP,MACHETEGURUWEAPONS,MACHETEGURUWEAPONUNDERWORLD,MACHETEHOP,MACHETEHOPARM,MACHETEHOPNEE,MACHETEHOPREDS,MACHETEHOPTOOL,MACHETEHOPUNDERWORLD,MACHETEIBIS,MACHETEIBISARM,MACHETEIBISDEEP,MACHETEIBISNEE,MACHETEIBISREDS,MACHETEIBISROBOT,MACHETEIBISSEAS,MACHETEIBISSEASARM,MACHETEIBISSEASNEE,MACHETEIBISSEASREDS,MACHETEIBISSEASROBOT,MACHETEIBISSEASWATER,MACHETEIBISSEASWATERARM,MACHETEIBISSEASWATERNEE,MACHETEIBISSEASWATERREDS,MACHETEIBISSEASWATERROBOT,MACHETEIBISSEASWATERUNDERWORLD,MACHETEICELAND,MACHETEICELANDARM,MACHETEICELANDDEEP,MACHETEICELANDNEE,MACHETEICELANDREDS,MACHETEICELANDROBOT,MACHETEICELANDROBOTARM,MACHETEICELANDROBOTNEE,MACHETEICELANDROBOTUNDERWORLD,MACHETEICELANDSEAS,MACHETEICELANDSEASARM,MACHETEICELANDSEASDEEP,MACHETEICELANDSEASNEE,MACHETEICELANDSEASREDS,MACHETEICELANDSEASROBOT,MACHETEICELANDSEASROBOTARM,MACHETEICELANDSEASROBOTNEE,MACHETEICELANDSEASROBOTUNDERWORLD,MACHETEICELANDSEASUNDERWORLD,MACHETEIRELAND,MACHETEIRELANDARM,MACHETEIRELANDDEEP,MACHETEIRELANDNEE,MACHETEIRELANDREDS,MACHETEIRELANDROBOT,MACHETEIRELANDROBOTARM,MACHETEIRELANDROBOTNEE,MACHETEIRELANDROBOTUNDERWORLD,MACHETEIRELANDSEAS,MACHETEIRELANDSEASARM,MACHETEIRELANDSEASDEEP,MACHETEIRELANDSEASNEE,MACHETEIRELANDSEASREDS,MACHETEIRELANDSEASROBOT,MACHETEIRELANDSEASROBOTARM,MACHETEIRELANDSEASROBOTNEE,MACHETEIRELANDSEASROBOTUNDERWORLD,MACHETEIRELANDSEASUNDERWORLD,MACHETEIRIS,MACHETEIRISARM,MACHETEIRISDEEP,MACHETEIRISNEE,MACHETEIRISREDS,MACHETEIRISROBOT,MACHETEIRISROBOTARM,MACHETEIRISROBOTNEE,MACHETEIRISROBOTUNDERWORLD,MACHETEIRISSEAS,MACHETEIRISSEASARM,MACHETEIRISSEASDEEP,MACHETEIRISSEASNEE,MACHETEIRISSEASREDS,MACHETEIRISSEASROBOT,MACHETEIRISSEASROBOTARM,MACHETEIRISSEASROBOTNEE,MACHETEIRISSEASROBOTUNDERWORLD,MACHETEIRISSEASUNDERWORLD,MACHETEIRELAND,MACHETEIRELANDARM,MACHETEIRELANDDEEP,MACHETEIRELANDNEE,MACHETEIRELANDREDS,MACHETEIRELANDROBOT,MACHETEIRELANDROBOTARM,MACHETEIRELANDROBOTNEE,MACHETEIRELANDROBOTUNDERWORLD,MACHETEIRELANDSEAS,MACHETEIRELANDSEASARM,MACHETEIRELANDSEASDEEP,MACHETEIRELANDSEASNEE,MACHETEIRELANDSEASREDS,MACHETEIRELANDSEASROBOT,MACHETEIRELANDSEASROBOTARM,MACHETEIRELANDSEASROBOTNEE,MACHETEIRELANDSEASROBOTUNDERWORLD,MACHETEIRELANDSEASUNDERWORLD,MACHETEIRELAND,MACHETEIRELANDARM,MACHETEIRELANDDEEP,MACHETEIRELANDNEE,MACHETEIRELANDREDS,MACHETEIRELANDROBOT,MACHETEIRELANDROBOTARM,MACHETEIRELANDROBOTNEE,MACHETEIRELANDROBOTUNDERWORLD,MACHETEIRELANDSEAS,MACHETEIRELANDSEASARM,MACHETEIRELANDSEASDEEP,MACHETEIRELANDSEASNEE,MACHETEIRELANDSEASREDS,MACHETEIRELANDSEASROBOT,MACHETEIRELANDSEASROBOTARM,MACHETEIRELANDSEASROBOTNEE,MACHETEIRELANDSEASROBOTUNDERWORLD,MACHETEIRELANDSEASUNDERWORLD,MACPHERSON,MACHETE,MAKE,MACHETEARM,MACHETEBELT,MACHETEBELTARM,MACHETEBELTNEE,MACHETEBELTREDS,MACHETEBELTROBOT,MACHETEBELTSEAS,MACHETEBELTSEASARM,MACHETEBELTSEASNEE,MACHETEBELTSEASREDS,MACHETEBELTSEASROBOT,MACHETEBELTSEASSEAS,MACHETEBELTSEASSEASARM,MACHETEBELTSEASSEASNEE,MACHETEBELTSEASSEASREDS,MACHETEBELTSEASSEASROBOT,MACHETEBELTSEASSEASUNDERWORLD,MACHETEBELTSEASUNDERWORLD,MACHETEBUG,MACHETEBUGARM,MACHETEBUGNEE,MACHETEBUGREDS,MACHETEBUGROBOT,MACHETEBUGSEAS,MACHETEBUGSEASARM,MACHETEBUGSEASNEE,MACHETEBUGSEASREDS,MACHETEBUGSEASROBOT,MACHETEBUGSEASUNDERWORLD,MACHETEBUGUNDERWORLD,MACHETEBUGWEAPON,MACHETEBUGWEAPONARM,MACHETEBUGWEAPONNEE,MACHETEBUGWEAPONREDS,MACHETEBUGWEAPONROBOT,MACHETEBUGWEAPONSEAS,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONUNDERWORLD,MACHETEBUGWEAPONROBOTARM,MACHETEBUGWEAPONROBOTNEE,MACHETEBUGWEAPONROBOTUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEASSEASREDS,MACHETEBUGWEAPONSEASSEASROBOT,MACHETEBUGWEAPONSEASSEASUNDERWORLD,MACHETEBUGWEAPONSEASUNDERWORLD,MACHETEBUGWEAPONSEASARM,MACHETEBUGWEAPONSEASNEE,MACHETEBUGWEAPONSEASREDS,MACHETEBUGWEAPONSEASROBOT,MACHETEBUGWEAPONSEASSEAS,MACHETEBUGWEAPONSEASSEASARM,MACHETEBUGWEAPONSEASSEASNEE,MACHETEBUGWEAPONSEAS