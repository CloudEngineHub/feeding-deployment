
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

days, cal = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
out = sys.argv[3]

logged = [d for d in days if d["corrections"] is not None]
xs   = [d["n"] for d in logged]
ask  = [sum(1 for c in d["corrections"] if c["channel"] == "ask") for d in logged]
sett = [sum(1 for c in d["corrections"] if c["channel"] == "settings") for d in logged]
n    = len(days)

plt.rcParams.update({"font.size": 8, "font.family": "serif"})

# Authored at exactly \textwidth so \includegraphics[width=\textwidth] applies no
# scaling; the axes rect is then placed so bar centers land on the table's day
# column centers directly below. See calibrate() for where these come from.
W_IN = cal["textwidth_pt"] / 72.27
# Short bottom margin: this axis carries no tick labels of its own (the table's
# Day/Date rows serve as them), so leaving room for them would just open a gap
# between the two panels and weaken the visual pairing.
fig = plt.figure(figsize=(W_IN, 1.30))
ax = fig.add_axes([cal["axes_left"], 0.09, cal["axes_width"], 0.87])

ax.bar(xs, ask, color="#C0504D", width=cal["bar_width"], label="Pre-meal (ask flow)")
ax.bar(xs, sett, bottom=ask, color="#E8A33D", width=cal["bar_width"],
       label="Mid-meal (settings)")

ax.set_ylabel("Corrections", fontsize=7.5, labelpad=2)
ax.set_xlim(0.5, n + 0.5)
ax.set_ylim(0, max([a + s for a, s in zip(ask, sett)] + [1]) + 0.5)
ax.set_xticks(range(1, n + 1))
ax.set_xticklabels([])          # the table's Day/Date rows below label this axis
ax.tick_params(axis="x", length=2, width=0.5)
ax.tick_params(axis="y", labelsize=6.5, length=2, width=0.5)
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))

# Days that have not run yet: an empty column here must not read as "zero
# corrections", because days 10 and 11 genuinely drew none.
for d in days:
    if d["corrections"] is None:
        ax.axvspan(d["n"] - 0.5, d["n"] + 0.5, color="black", alpha=0.055, lw=0)
first_pending = next((d["n"] for d in days if d["corrections"] is None), None)
if first_pending is not None:
    ax.text((first_pending + n) / 2, ax.get_ylim()[1] * 0.5, "not yet run",
            ha="center", va="center", fontsize=6, color="0.5", style="italic")

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_linewidth(0.5)
ax.grid(axis="y", color="0.9", lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right",
          handlelength=1.1, columnspacing=1.0, borderpad=0.1, handletextpad=0.4)
fig.savefig(out)
print("wrote", out)
