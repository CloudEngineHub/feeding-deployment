
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

days, cal = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
out = sys.argv[3]

logged = [d for d in days if d["ran"]]
xs     = [d["n"] for d in logged]
nav    = [d["nav"] for d in logged]
manip  = [d["manip"] for d in logged]
n      = len(days)

plt.rcParams.update({"font.size": 8, "font.family": "serif"})

W_IN = cal["textwidth_pt"] / 72.27
fig = plt.figure(figsize=(W_IN, 1.30))
ax = fig.add_axes([cal["axes_left"], 0.09, cal["axes_width"], 0.87])

NAVC, MANC = "#2E5E8C", "#B0A8A0"
w = cal["bar_width"]
ax.bar(xs, nav, color=NAVC, width=w, label="Navigation (base)")
ax.bar(xs, manip, bottom=nav, color=MANC, width=w, label="Manipulation (arm)")

ax.set_ylabel("Teleoperation", fontsize=7.5, labelpad=2)
ax.set_xlim(0.5, n + 0.5)
# Headroom for the legend, which sits over the tallest bars otherwise.
ax.set_ylim(0, max([a + b for a, b in zip(nav, manip)] + [1]) + 2.6)
ax.set_xticks(range(1, n + 1))
ax.set_xticklabels([])          # the table's Day/Date rows below label this axis
ax.tick_params(axis="x", length=2, width=0.5)
ax.tick_params(axis="y", labelsize=6.5, length=2, width=0.5)
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))

for d in days:
    if not d["ran"]:
        ax.axvspan(d["n"] - 0.5, d["n"] + 0.5, color="black", alpha=0.055, lw=0)

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
