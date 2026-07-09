"""Self-audit: which of the scripts written this session were valid on first run,
and which were 'validated wrong' and then patched (the churn to avoid).

A pass = one script/version I wrote and executed. valid=1 means it did what it was
meant to do on that run; valid=0 means I found it wrong and rewrote it (a negative
example, in SOAR terms). The plot tracks the cumulative first-run validity rate.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (label, valid, failure_mode_if_invalid)
PASSES = [
    ("show_output",       1, ""),
    ("diag_big",          1, ""),
    ("diag_template",     1, ""),
    ("diag_templib",      1, ""),
    ("build_templates",   1, ""),
    ("build_more",        1, ""),
    ("template_ac3 v1",   0, "deadline overrun (gamma 24s):\ncopy before AC-3 deadline check"),
    ("test_template v1",  0, "passed {theme,fill} dict to score():\nallowed={'THEME','FILL'} -> valid=0"),
    ("test_template v2",  1, ""),
    ("template_ac3 v2",   0, "two-pass still only 10/12 valid:\ntheme-first ate budget, no restart"),
    ("template_ac3 v3",   1, ""),
    ("regen_templates",   1, ""),
]

labels = [p[0] for p in PASSES]
valid = [p[1] for p in PASSES]
x = list(range(1, len(PASSES) + 1))
cum_rate = [100.0 * sum(valid[:i]) / i for i in x]

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(x, cum_rate, "-", color="#3b6", lw=2, zorder=1, label="cumulative first-run validity %")
for xi, v, rate in zip(x, valid, cum_rate):
    ax.scatter(xi, rate, s=110, zorder=3,
               color="#2a9d4a" if v else "#d1495b",
               edgecolors="white", linewidths=1.2)

for xi, (lab, v, mode) in zip(x, PASSES):
    if not v:
        ax.annotate(mode, (xi, cum_rate[xi - 1]), textcoords="offset points",
                    xytext=(6, -46), fontsize=7.5, color="#d1495b",
                    ha="left", va="top")

ax.axhline(75, ls=":", color="#999", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
ax.set_ylim(60, 103)
ax.set_ylabel("cumulative first-run validity (%)")
ax.set_title("Script validity over passes  (green = valid on first run, red = validated wrong & rewrote)")
ax.grid(axis="y", alpha=0.25)
n_valid, n = sum(valid), len(valid)
ax.text(0.99, 0.03, f"{n_valid}/{n} valid on first run ({100*n_valid//n}%)",
        transform=ax.transAxes, ha="right", fontsize=9, color="#444")
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "script_validity.png")
fig.savefig(out, dpi=130)
print(f"saved -> {out}  ({n_valid}/{n} valid on first run)")
