"""Known-good fixtures for the scorer.

NYT_1976 is the real New York Times crossword from Thursday, Jan 1, 1976
(constructor Alfio Micci, ed. Will Weng) — a genuine 15x15, 180-degree
symmetric, fully-checked, min-length-3 American grid. It is the primary
"this must score valid" fixture.
"""

# Solved grid: one string per row, '.' = black square.
NYT_1976_ROWS = [
    "AHEM.NANA.CLOVE",
    "DIVA.OWES.LAVAS",
    "AMEN.MANICURIST",
    "MANDRAKE.ABIDE.",
    "...AIDE.RIMA...",
    "SPARES.MANATEES",
    "EOSIN.DOZEN.LAT",
    "ASSN.SORES.DOGE",
    "TIE.ETWAS.REPEL",
    "OTTOMANS.CAMERA",
    "...LUMS.SARA...",
    ".LADLE.FUMANCHU",
    "MANHANDLED.DOUR",
    "GROAT.DADE.ELLA",
    "TANTE.STEN.DEAL",
]


def grid_map(rows=NYT_1976_ROWS):
    """Return {(r, c): letter} for white cells of a '.'-delimited grid."""
    g = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch != ".":
                g[(r, c)] = ch
    return g


def size(rows=NYT_1976_ROWS):
    return len(rows)
