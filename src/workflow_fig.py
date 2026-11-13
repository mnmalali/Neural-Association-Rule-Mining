"""Draws the workflow diagram (figures/fig_workflow.png)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw(path="figures/fig_workflow.png"):
    fig, ax = plt.subplots(figsize=(10, 4.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 44)

    def box(x, y, w, h, t, c="#dbe9f7", fs=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="#33506f", lw=1.3, zorder=2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, zorder=3)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#33506f"), zorder=1)

    box(0.5, 17, 12, 11, "UCI Adult\n48,842 people\n15 attributes")
    box(15, 17, 14, 11, "Pre-processing\nbin numeric cols\none-hot items\n(55 items)")
    box(32, 17, 11, 11, "Split 70/30\nstratified\ntrain / test")
    box(47, 31, 22, 11, "Neural ARM (proposed)\nDenoising autoencoder\n55-64-16-64-55", "#cfe8cf")
    box(47, 16.5, 22, 11, "Rule extraction\nP(c|A) >= tau, estimated\nsupport and lift", "#cfe8cf")
    box(47, 2, 22, 10.5, "Baselines\nApriori, FP-Growth,\nuntrained-AE ablation", "#f7d9d9")
    box(72, 16.5, 11.5, 11, "Exact\nverification\non the data")
    box(86, 26, 13.5, 14, "Evaluation\naccuracy, stability,\nspuriousness,\nunderstandability", "#fff2cc", 9)
    box(86, 5, 13.5, 14, "Explain and audit\nrule cards,\nocclusion drivers,\nfairness, privacy", "#fff2cc", 9)
    arr(12.5, 22.5, 15, 22.5); arr(29, 22.5, 32, 22.5)
    arr(43, 26, 47, 34); arr(43, 20, 47, 8); arr(58, 31, 58, 27.5); arr(69, 22, 72, 22); arr(69, 8, 78, 16.5)
    arr(83.5, 25, 86, 31); arr(83.5, 19, 86, 13)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    draw()
