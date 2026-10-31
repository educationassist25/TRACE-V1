"""
pubfig.py - publication renderer: every TRACE figure as a matplotlib figure with collision-free labels,
exported as PNG / TIFF / JPEG / SVG / PDF / EPS at any resolution.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .figures import CLASS_COLORS, PATTERN_COLORS  # noqa: E402
from .ipa import z_color  # noqa: E402
from .layout import place_labels, text_size  # noqa: E402

FORMATS = {"PNG": "png", "TIFF": "tiff", "JPEG": "jpg", "SVG": "svg", "PDF": "pdf", "EPS": "eps"}
MIME = {"png": "image/png", "tiff": "image/tiff", "jpg": "image/jpeg", "svg": "image/svg+xml",
        "pdf": "application/pdf", "eps": "application/postscript"}


def style(font_family="DejaVu Sans", font_size=8):
    plt.rcParams.update({"font.family": font_family, "font.size": font_size, "axes.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "legend.frameon": False, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def export(fig, fmt: str = "png", dpi: int = 600, transparent: bool = False) -> bytes:
    b = io.BytesIO()
    kw = {}
    if fmt == "tiff":
        kw["pil_kwargs"] = {"compression": "tiff_lzw"}
    if fmt == "jpg":
        kw["pil_kwargs"] = {"quality": 95}
        transparent = False
    fig.savefig(b, format="jpeg" if fmt == "jpg" else fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.05,
                transparent=transparent, **kw)
    plt.close(fig)
    return b.getvalue()


def _repel(fig, ax, x, y, labels, all_x, all_y, fs, point_r=0.035):
    """Place labels for points (x, y) avoiding all points and each other; draws text + leader lines."""
    if len(labels) == 0:
        return
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    W, H = bb.width, bb.height
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    tx = lambda v: (np.asarray(v, float) - x0) / (x1 - x0) * W
    ty = lambda v: (np.asarray(v, float) - y0) / (y1 - y0) * H
    A = np.c_[tx(x), ty(y)]
    O = np.c_[tx(all_x), ty(all_y)]
    ok = np.isfinite(O).all(1)
    O = O[ok]
    sizes = np.array([text_size(t, fs) for t in labels])
    C, placed, lead = place_labels(A, sizes, O, np.full(len(O), point_r), np.zeros((0, 2)), W, H,
                                   anchor_r=np.full(len(A), point_r), margin=0.0)
    for i, t in enumerate(labels):
        if not placed[i]:
            continue
        cx = x0 + C[i, 0] / W * (x1 - x0)
        cy = y0 + C[i, 1] / H * (y1 - y0)
        if lead[i]:
            ax.plot([x[i], cx], [y[i], cy], color="#7F8C8D", lw=0.5, zorder=4)
        ax.text(cx, cy, t, fontsize=fs, ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))


# --------------------------------------------------------------------------------------------- figures
def concordance(conc, beta, n_label=15, w=4.2, h=4.0, fs=7, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    for cls, col in CLASS_COLORS.items():
        d = conc[conc["class"] == cls]
        ax.scatter(d["z_mRNA"], d["z_protein"], s=5 if cls == "Unchanged" else 14, c=col, lw=0.3,
                   edgecolors="white", alpha=0.45 if cls == "Unchanged" else 0.95, label=f"{cls} ({len(d)})",
                   zorder=2 if cls == "Unchanged" else 3)
    lim = np.nanmax(np.abs(conc[["z_mRNA", "z_protein"]].values)) * 1.08
    ax.plot([-lim, lim], [-beta * lim, beta * lim], "--", c="#333", lw=0.8, zorder=1, label=f"Deming slope {beta:.2f}")
    ax.axhline(0, c="#ccc", lw=0.5, zorder=0)
    ax.axvline(0, c="#ccc", lw=0.5, zorder=0)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="mRNA z-score", ylabel="Protein z-score")
    ax.legend(fontsize=fs - 1, loc="upper left", bbox_to_anchor=(1.01, 1))
    top = conc[conc["class"] != "Unchanged"].nsmallest(n_label, "p_discordance")
    _repel(fig, ax, top["z_mRNA"].values, top["z_protein"].values, list(top["gene"]),
           conc["z_mRNA"].values, conc["z_protein"].values, fs)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def pattern_grid(rx, n_label=12, w=5.2, h=4.2, fs=7, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    q = rx["q_label"].fillna(1).clip(1e-12, 1)
    for pat, col in PATTERN_COLORS.items():
        d = rx[rx["pattern"] == pat]
        if len(d):
            qq = d["q_label"].fillna(1).clip(1e-12, 1)
            ax.scatter(d["E"], d["M"], s=6 + 6 * np.clip(-np.log10(qq), 0, 6), c=col, lw=0.3, edgecolors="white",
                       alpha=0.9, label=f"{pat} ({len(d)})", zorder=3 if pat != "No clear pattern" else 2)
    ax.axhline(0, c="#999", lw=0.5, zorder=1)
    ax.axvline(0, c="#999", lw=0.5, zorder=1)
    ax.set(xlabel="Enzyme evidence E (z)", ylabel="Metabolite balance M")
    ax.legend(fontsize=fs - 1, loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=0.8)
    sig = rx[rx["significant"]].copy()
    sig["lab"] = [str(e).split(",")[0] for e in sig["enzymes"]]
    sig = sig.drop_duplicates("lab").head(n_label)
    labs = list(sig["lab"])
    _repel(fig, ax, sig["E"].values, sig["M"].values, labs, rx["E"].values, rx["M"].values, fs)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def pathway_dotplot(pw, top=25, w=6.4, h=None, fs=7, title=""):
    d = pw.dropna(subset=["z_combined"]).head(top).iloc[::-1]
    cols = [c for c in ("mRNA", "protein", "metabolite") if f"z_{c}" in d.columns] + ["combined"]
    h = h or max(2.2, 0.26 * len(d) + 1.2)
    fig, ax = plt.subplots(figsize=(w, h))
    cmap = plt.get_cmap("RdBu_r")
    for i, (_, r) in enumerate(d.iterrows()):
        for j, c in enumerate(cols):
            z, q = r[f"z_{c}"], r[f"q_{c}"]
            if np.isfinite(z):
                ax.scatter(j, i, s=14 + 16 * min(6, -np.log10(max(q, 1e-12))), c=[cmap((np.clip(z, -5, 5) + 5) / 10)],
                           edgecolors="#444", linewidths=0.4, zorder=3)
        ax.text(len(cols) - 0.35, i, r["class"], fontsize=fs - 0.5, va="center", color="#00695C")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([p if len(p) <= 52 else p[:50] + "…" for p in d["pathway"]], fontsize=fs)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=fs)
    ax.xaxis.tick_top()
    ax.set_xlim(-0.6, len(cols) + 1.9)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.spines[["bottom", "right"]].set_visible(False)
    ax.spines["top"].set_visible(True)
    ax.grid(axis="y", color="#EEE", lw=0.5, zorder=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(-5, 5))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("z", fontsize=fs)
    for qv, lab in ((0.05, "q 0.05"), (1e-3, "q 0.001"), (1e-6, "q 1e-6")):
        ax.scatter([], [], s=14 + 16 * min(6, -np.log10(qv)), c="#BBB", edgecolors="#444", lw=0.4, label=lab)
    ax.legend(fontsize=fs - 1, loc="upper left", bbox_to_anchor=(1.08, 0.35), title="size", title_fontsize=fs - 1)
    if title:
        ax.set_title(title, loc="left", fontweight="bold", pad=18)
    return fig


def canonical(cp, top=25, w=7.6, h=None, fs=7, title="", thr=0.05):
    d = cp.head(top).iloc[::-1]
    h = h or max(2.4, 0.24 * len(d) + 1.4)
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(d))
    ax.barh(y, d["minus_log10_p"], color=[z_color(z) for z in d["z_directional"]], edgecolor="#555", lw=0.4,
            height=0.72, zorder=2)
    ax.axvline(-np.log10(thr), color="#E65100", ls="--", lw=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([p if len(p) <= 66 else p[:64] + "…" for p in d["pathway"]], fontsize=fs)
    ax.set_xlabel("-log10(p)  (right-tailed Fisher's exact)")
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax2 = ax.twiny()
    ax2.plot(d["ratio"], y, "-o", color="#E67E22", ms=2.5, lw=0.8, zorder=4)
    ax2.set_xlabel("Ratio (significant / pathway molecules)", color="#E67E22", fontsize=fs)
    ax2.tick_params(axis="x", colors="#E67E22", labelsize=fs - 1)
    ax2.set_xlim(0, max(0.05, float(d["ratio"].max()) * 1.15))
    ax2.spines["top"].set_visible(True)
    import matplotlib.patches as mp
    hs = [mp.Patch(fc=z_color(3), ec="#555", label="z > 0 (increased)"), mp.Patch(fc=z_color(-3), ec="#555", label="z < 0 (decreased)"),
          mp.Patch(fc="#FFFFFF", ec="#555", label="z = 0"), mp.Patch(fc="#BDBDBD", ec="#555", label="no activity pattern")]
    ax.legend(handles=hs, fontsize=fs - 1, loc="lower right")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return fig


def targets_bar(tg, top=20, w=5.6, h=None, fs=7, title=""):
    d = tg.head(top).iloc[::-1]
    h = h or max(2.4, 0.24 * len(d) + 0.9)
    fig, ax = plt.subplots(figsize=(w, h))
    v = np.nan_to_num(-np.log10(d["rra_score"].astype(float).clip(1e-30, 1)))
    ax.barh(range(len(d)), v, color=["#E65100" if c else "#00695C" for c in d["choke_point"]], height=0.7)
    for i, (val, ev) in enumerate(zip(v, d["evidence"])):
        ax.text(val + v.max() * 0.01, i, ev, fontsize=fs - 1.5, va="center", color="#4A5754")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["gene"], fontsize=fs)
    ax.set_xlim(0, v.max() * 2.1 if len(v) else 1)
    ax.set_xlabel("-log10 rank-aggregation score")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def panel_bars(panel, w=4.6, h=None, fs=7, title=""):
    d = panel.sort_values("selection_frequency")
    h = h or max(2.0, 0.22 * len(d) + 0.8)
    cmap = {"mRNA": "#2C6FB7", "protein": "#00695C", "metabolite": "#E65100"}
    fig, ax = plt.subplots(figsize=(w, h))
    ax.barh(range(len(d)), d["selection_frequency"], color=[cmap.get(l, "#888") for l in d["layer"]], height=0.7)
    ax.axvline(0.7, ls=":", c="#555", lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{a} ({b})" for a, b in zip(d["label"], d["layer"])], fontsize=fs)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Selection frequency (outer folds)")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def roc(sig, w=3.2, h=3.2, fs=7, title=""):
    fpr, tpr = sig["roc"]
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot(fpr, tpr, c="#00695C", lw=1.6, label=f"Integrated AUC {sig['auc_mean']:.2f}\n95% CI {sig['auc_ci'][0]:.2f}-{sig['auc_ci'][1]:.2f}")
    ax.plot([0, 1], [0, 1], ":", c="#999")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=fs)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def perm_hist(sig, w=3.4, h=2.6, fs=7, title=""):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.hist(sig["perm_aucs"], bins=20, color="#B8C2C0", edgecolor="white")
    ax.axvline(sig["auc_mean"], color="#E65100", lw=1.6)
    ax.text(sig["auc_mean"], ax.get_ylim()[1] * 0.95, f" observed {sig['auc_mean']:.2f}\n p = {sig['p_perm']:.3f}",
            fontsize=fs, va="top", ha="right" if sig["auc_mean"] > 0.8 else "left")
    ax.set(xlabel="Cross-validated AUC (permuted labels)", ylabel="Count")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def variance_heatmap(r2, w=None, h=1.8, fs=7, title=""):
    w = w or 0.6 * r2.shape[1] + 1.6
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(r2.values, cmap="BuGn", vmin=0, aspect="auto")
    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            v = r2.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=fs - 0.5,
                    color="white" if v > r2.values.max() * 0.6 else "#1F2A28")
    ax.set_xticks(range(r2.shape[1]))
    ax.set_xticklabels(r2.columns, rotation=40, ha="right", fontsize=fs)
    ax.set_yticks(range(r2.shape[0]))
    ax.set_yticklabels(r2.index, fontsize=fs)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("% variance", fontsize=fs)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def factor_scatter(scores, groups, fx, fy, w=3.6, h=3.2, fs=7, title="", label_samples=False):
    fig, ax = plt.subplots(figsize=(w, h))
    cols = ["#00695C", "#E65100", "#2C6FB7", "#8E44AD"]
    g = groups.reindex(scores.index)
    for c, lv in zip(cols, sorted(g.dropna().unique())):
        s = scores[g == lv]
        ax.scatter(s[fx], s[fy], s=26, c=c, edgecolors="white", lw=0.5, label=lv)
    ax.set(xlabel=fx, ylabel=fy)
    ax.legend(fontsize=fs, loc="upper left", bbox_to_anchor=(1.01, 1))
    if label_samples:
        _repel(fig, ax, scores[fx].values, scores[fy].values, list(scores.index), scores[fx].values, scores[fy].values, fs - 1)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def coverage(layers, w=3.6, h=2.6, fs=7, title=""):
    names, tot, mp = [], [], []
    for k, L in layers.items():
        names.append(L.name)
        tot.append(len(L.feat))
        mp.append(int((L.feat["map_rule"] != "unmapped").sum() if k == "met" else (L.feat["ensg"] != "").sum()))
    fig, ax = plt.subplots(figsize=(w, h))
    x = np.arange(len(names))
    ax.bar(x, tot, color="#C9D1CF", width=0.6, label="features")
    ax.bar(x, mp, color="#00695C", width=0.6, label="mapped to KB")
    for i, (t, m) in enumerate(zip(tot, mp)):
        ax.text(i, t, f"{m}/{t}", ha="center", va="bottom", fontsize=fs - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=fs)
    ax.set_ylabel("Features")
    ax.legend(fontsize=fs - 1)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig


def cascade_alluvial(sel, w=7.5, h=None, fs=7, title=""):
    """Static enzyme -> reaction -> metabolite ribbon diagram (publication version of the Sankey)."""
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path
    genes, rxns, mets, links = [], [], [], []
    for _, r in sel.iterrows():
        rn = r["name"] if len(r["name"]) <= 44 else r["name"][:42] + "…"
        rxns.append(rn)
        for g in [x.strip() for x in str(r["enzymes"]).split(",") if x.strip()][:4]:
            if g not in genes:
                genes.append(g)
            links.append((0, genes.index(g), 1, len(rxns) - 1, r["E"]))
        for m, v in [(x.strip(), r["P"]) for x in str(r["products"]).split(";") if x.strip()][:3] + \
                    [(x.strip() + " (S)", r["S"]) for x in str(r["substrates"]).split(";") if x.strip()][:3]:
            if m not in mets:
                mets.append(m)
            links.append((1, len(rxns) - 1, 2, mets.index(m), v))
    n = max(len(genes), len(rxns), len(mets), 1)
    h = h or max(2.6, 0.24 * n + 0.8)
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")
    cols = [genes, rxns, mets]
    bw = 0.025
    cw = lambda items, f: max([text_size(t, f)[0] for t in items] or [0.3]) / w
    g_lab, r_lab = cw(genes, fs), cw(rxns, fs - 0.5)
    xs = [0.0, bw + g_lab + 0.16, 0.0]
    xs[2] = xs[1] + bw + r_lab + 0.14
    box_h = 0.62 / n

    def ypos(c, i):
        k = len(cols[c])
        return 1 - (i + 0.5) / max(k, 1)

    for c0, i0, c1, i1, v in links:
        y0, y1 = ypos(c0, i0), ypos(c1, i1)
        x0 = xs[c0] + bw + (g_lab + 0.01 if c0 == 0 else r_lab + 0.01)
        x1 = xs[c1]
        col = "#C0392B" if (np.isfinite(v) and v > 0) else ("#2C6FB7" if np.isfinite(v) else "#AAAAAA")
        wv = (0.10 + 0.22 * min(abs(v) if np.isfinite(v) else 0.3, 4) / 4) / n
        xm = (x0 + x1) / 2
        verts = [(x0, y0 + wv), (xm, y0 + wv), (xm, y1 + wv), (x1, y1 + wv), (x1, y1 - wv), (xm, y1 - wv),
                 (xm, y0 - wv), (x0, y0 - wv), (x0, y0 + wv)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), fc=col, ec="none", alpha=0.35, zorder=1))
    for c, items in enumerate(cols):
        for i, t in enumerate(items):
            y = ypos(c, i)
            ax.add_patch(Rectangle((xs[c], y - box_h / 2), bw, box_h, fc="#E3F2EF", ec="#00695C", lw=0.6, zorder=3))
            ax.text(xs[c] + bw + 0.006, y, t, fontsize=fs - (0.5 if c == 1 else 0), va="center", ha="left", zorder=4)
    for x, t in zip(xs, ["Enzymes", "Reactions", "Metabolites"]):
        ax.text(x, 1.04, t, fontsize=fs + 1, fontweight="bold", ha="left")
    ax.set_xlim(-0.01, xs[2] + bw + cw(mets, fs) + 0.02)
    ax.set_ylim(-0.02, 1.08)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    return fig
