"""figures.py - interactive Plotly figures for the TRACE app (download as PNG/SVG from each figure)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

C_UP, C_DOWN, C_GREY = "#C0392B", "#2C6FB7", "#9AA5A2"
TEAL, ORANGE = "#00695C", "#E65100"
FONT = dict(family="Arial, Helvetica, sans-serif", size=13, color="#1F2A28")
PATTERN_COLORS = {
    "Throughput-consistent (up)": "#C0392B", "Throughput-consistent (down)": "#2C6FB7",
    "Candidate bottleneck": "#E65100", "Constrained step": "#8E44AD", "Substrate-limited": "#B9770E",
    "Non-abundance regulation (lower flux)": "#00897B", "Non-abundance regulation (higher flux)": "#26A69A",
    "Compensated": "#5D6D7E", "Product drain": "#D35400", "Metabolite-only change": "#7FB3D5",
    "Enzyme-only change": "#F1948A", "No clear pattern": "#D5DBDB",
}
CLASS_COLORS = {"Coupled": "#00695C", "Opposing": "#8E44AD", "Protein-only (post-transcriptional)": "#E65100",
                "mRNA-only (buffered)": "#2C6FB7", "Unchanged": "#D0D5D4"}


def _layout(fig, title, h=520, **kw):
    kw.setdefault("margin", dict(l=60, r=30, t=60, b=55))
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16)), font=FONT, height=h,
                      template="simple_white", **kw)
    return fig


def concordance(conc: pd.DataFrame, beta: float, label_top: int = 12):
    fig = go.Figure()
    for cls, col in CLASS_COLORS.items():
        d = conc[conc["class"] == cls]
        if d.empty:
            continue
        fig.add_trace(go.Scattergl(x=d["z_mRNA"], y=d["z_protein"], mode="markers", name=f"{cls} ({len(d)})",
                                   marker=dict(color=col, size=6 if cls == "Unchanged" else 8,
                                               opacity=0.45 if cls == "Unchanged" else 0.9,
                                               line=dict(width=0.4, color="white")),
                                   text=d["gene"], hovertemplate="%{text}<br>z mRNA %{x:.2f}<br>z protein %{y:.2f}"))
    lim = float(np.nanmax(np.abs(conc[["z_mRNA", "z_protein"]].values))) * 1.05 if len(conc) else 5
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-beta * lim, beta * lim], mode="lines", name=f"Deming fit (slope {beta:.2f})",
                             line=dict(color="#1F2A28", dash="dash", width=1.2)))
    top = conc[conc["class"] != "Unchanged"].nsmallest(label_top, "p_discordance")
    for _, r in top.iterrows():
        fig.add_annotation(x=r["z_mRNA"], y=r["z_protein"], text=r["gene"], showarrow=False, yshift=10,
                           font=dict(size=11))
    fig.add_hline(y=0, line_color="#BBB", line_width=0.8)
    fig.add_vline(x=0, line_color="#BBB", line_width=0.8)
    fig.update_xaxes(title="mRNA z-score", range=[-lim, lim])
    fig.update_yaxes(title="Protein z-score", range=[-lim, lim])
    return _layout(fig, "Gene-protein concordance", 560, legend=dict(orientation="h", y=-0.18))


def pattern_grid(rx: pd.DataFrame, n_label: int = 10):
    d = rx.copy()
    d["size"] = 6 + 3 * np.clip(-np.log10(d["q_label"].clip(1e-12)), 0, 6)
    fig = go.Figure()
    for pat, col in PATTERN_COLORS.items():
        s = d[d["pattern"] == pat]
        if s.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=s["E"], y=s["M"], mode="markers", name=f"{pat} ({len(s)})",
            marker=dict(color=col, size=s["size"], opacity=0.85, line=dict(width=0.5, color="white")),
            text=s["name"], customdata=np.c_[s["enzymes"], s["q_label"], s["q_topology"]],
            hovertemplate="%{text}<br>Enzymes: %{customdata[0]}<br>E %{x:.2f}  M %{y:.2f}"
                          "<br>q label %{customdata[1]:.2g}, q topology %{customdata[2]:.2g}<extra></extra>"))
    for _, r in d[d["significant"]].head(n_label).iterrows():
        fig.add_annotation(x=r["E"], y=r["M"], text=str(r["enzymes"]).split(",")[0], showarrow=True,
                           arrowhead=0, ax=18, ay=-18, font=dict(size=11))
    fig.add_hline(y=0, line_color="#999", line_width=0.8)
    fig.add_vline(x=0, line_color="#999", line_width=0.8)
    for x, y, t in ((0.98, 0.98, "coherent: enzyme up,<br>products up / substrates down"),
                    (0.02, 0.02, "coherent: enzyme down,<br>substrates accumulate"),
                    (0.98, 0.02, "enzyme up, balance down"), (0.02, 0.98, "enzyme down, balance up")):
        fig.add_annotation(xref="paper", yref="paper", x=x, y=y, text=t, showarrow=False,
                           xanchor="right" if x > 0.5 else "left", yanchor="top" if y > 0.5 else "bottom",
                           font=dict(size=10, color="#6B7775"))
    fig.update_xaxes(title="Enzyme evidence E (z)")
    fig.update_yaxes(title="Metabolite balance M = (products - substrates)/sqrt(2)")
    return _layout(fig, "Reaction pattern map (each dot = one reaction)", 620,
                   legend=dict(font=dict(size=11)))


def pathway_dotplot(pw: pd.DataFrame, top: int = 25):
    d = pw.dropna(subset=["z_combined"]).head(top).iloc[::-1]
    cols = [c for c in ("mRNA", "protein", "metabolite") if f"z_{c}" in d.columns] + ["combined", "coherence"]
    xs, ys, zs, ss, txt = [], [], [], [], []
    for _, r in d.iterrows():
        for c in cols:
            if c == "coherence":
                z = r["coherence"]
                q = r["q_coherence"]
                n = r["n_reactions_observed"]
            elif c == "combined":
                z, q, n = r["z_combined"], r["q_combined"], r.get("layers_assessable", np.nan)
            else:
                z, q, n = r[f"z_{c}"], r[f"q_{c}"], r[f"n_{c}"]
            if not np.isfinite(z):
                continue
            xs.append(c)
            ys.append(r["pathway"][:60])
            zs.append(np.clip(z, -6, 6) if c != "coherence" else np.clip(z * 2, -6, 6))
            ss.append(8 + 4 * min(6, -np.log10(max(q, 1e-12))) if np.isfinite(q) else 6)
            txt.append(f"{r['pathway']}<br>{c}: {z:.2f}<br>q = {q:.2g}<br>n = {n}")
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", text=txt, hovertemplate="%{text}<extra></extra>",
                               marker=dict(size=ss, color=zs, colorscale="RdBu_r", cmin=-5, cmax=5,
                                           line=dict(width=0.6, color="#555"),
                                           colorbar=dict(title="z", thickness=12))))
    cls = d["class"].values
    fig.add_trace(go.Scatter(x=["class"] * len(d), y=[p[:60] for p in d["pathway"]], mode="text",
                             text=cls, textfont=dict(size=11, color=TEAL), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(categoryorder="array", categoryarray=cols + ["class"], side="top")
    fig.update_layout(showlegend=False)
    return _layout(fig, "Pathway activity across layers (size = -log10 q)", max(420, 26 * len(d) + 120),
                   margin=dict(l=330, r=30, t=90, b=30))


def cascade_sankey(rx: pd.DataFrame, title: str = "Cascade"):
    """Gene -> reaction -> metabolite ribbons for a set of reactions (rows of the reaction table)."""
    labels, idx, src, dst, val, col = [], {}, [], [], [], []

    def node(key, lab):
        if key not in idx:
            idx[key] = len(labels)
            labels.append(lab)
        return idx[key]

    rgba = lambda v: "rgba(192,57,43,0.55)" if v > 0 else "rgba(44,111,183,0.55)"
    for _, r in rx.iterrows():
        rn = node("r:" + r["reaction"], (r["name"][:40] + "...") if len(r["name"]) > 43 else r["name"])
        for g in [x.strip() for x in str(r["enzymes"]).split(",") if x.strip()][:6]:
            src.append(node("g:" + g, g))
            dst.append(rn)
            val.append(max(abs(r["E"]), 0.3))
            col.append(rgba(r["E"]))
        for m in [x.strip() for x in str(r["products"]).split(",") if x.strip()][:4]:
            src.append(rn)
            dst.append(node("m:" + m, m))
            val.append(max(abs(r["P"]) if np.isfinite(r["P"]) else 0.3, 0.3))
            col.append(rgba(r["P"]) if np.isfinite(r["P"]) else "rgba(150,150,150,0.4)")
        for m in [x.strip() for x in str(r["substrates"]).split(",") if x.strip()][:4]:
            src.append(rn)
            dst.append(node("m:" + m, m + " (substrate)"))
            val.append(max(abs(r["S"]) if np.isfinite(r["S"]) else 0.3, 0.3))
            col.append(rgba(r["S"]).replace("0.55", "0.3") if np.isfinite(r["S"]) else "rgba(150,150,150,0.3)")
    fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=labels, pad=12, thickness=14,
                                                             color="#E3F2EF", line=dict(color=TEAL, width=0.6)),
                              link=dict(source=src, target=dst, value=val, color=col)))
    return _layout(fig, title + "  (enzyme -> reaction -> metabolite; red = up, blue = down)",
                   max(420, 22 * len(labels) // 2 + 200))


def reaction_map(rx: pd.DataFrame, met_z: dict, title: str, seed: int = 1):
    """Bipartite reaction-metabolite map of one pathway; squares = reactions (colour = coherence),
    circles = metabolites (colour = z), enzyme names under each reaction."""
    import networkx as nx
    G = nx.Graph()
    for _, r in rx.iterrows():
        R = "r:" + r["reaction"]
        G.add_node(R, kind="r")
        for m in [x.strip() for x in str(r["substrates"]).split(",") if x.strip()] + \
                 [x.strip() for x in str(r["products"]).split(",") if x.strip()]:
            G.add_node("m:" + m, kind="m")
            G.add_edge(R, "m:" + m)
    if G.number_of_nodes() == 0:
        return _layout(go.Figure(), title + " (no observable reactions)", 300)
    pos = nx.kamada_kawai_layout(G) if G.number_of_nodes() < 200 else nx.spring_layout(G, seed=seed)
    ex, ey = [], []
    for a, b in G.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#B8C2C0", width=1), hoverinfo="skip",
                               showlegend=False))
    rmap = rx.set_index("reaction")
    rn = [n for n in G if n.startswith("r:")]
    cz = [float(rmap.loc[n[2:], "C_std"]) if np.isfinite(rmap.loc[n[2:], "C_std"]) else 0 for n in rn]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in rn], y=[pos[n][1] for n in rn], mode="markers+text", name="reactions",
        marker=dict(symbol="square", size=14, color=cz, colorscale="PuOr_r", cmin=-4, cmax=4,
                    line=dict(color="#333", width=0.8), colorbar=dict(title="reaction<br>coherence z", x=1.02, y=0.77, len=0.42, thickness=10)),
        text=[str(rmap.loc[n[2:], "enzymes"]).split(",")[0] for n in rn], textposition="bottom center",
        textfont=dict(size=10, color="#333"),
        hovertext=[f"{rmap.loc[n[2:], 'name']}<br>{rmap.loc[n[2:], 'enzymes']}<br>{rmap.loc[n[2:], 'pattern']}"
                   for n in rn], hoverinfo="text"))
    mn = [n for n in G if n.startswith("m:")]
    mz = [met_z.get(n[2:], np.nan) for n in mn]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in mn], y=[pos[n][1] for n in mn], mode="markers+text", name="metabolites",
        marker=dict(size=16, color=[z if np.isfinite(z) else 0 for z in mz], colorscale="RdBu_r", cmin=-4,
                    cmax=4, line=dict(color=["#333" if np.isfinite(z) else "#CCC" for z in mz], width=1),
                    colorbar=dict(title="metabolite z", x=1.02, y=0.23, len=0.42, thickness=10)),
        text=[n[2:] for n in mn], textposition="top center", textfont=dict(size=10),
        hovertext=[f"{n[2:]}<br>z = {z:.2f}" if np.isfinite(z) else f"{n[2:]} (not measured)" for n, z in zip(mn, mz)],
        hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title, 680, showlegend=False, margin=dict(l=20, r=120, t=60, b=20))


def network_plot(nodes: pd.DataFrame, edges: list, title: str, seed: int = 2):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes["node"])
    G.add_edges_from([(a, b) for a, b, _ in edges if a in G and b in G])
    try:
        pos = nx.forceatlas2_layout(G, max_iter=300, gravity=2.0, strong_gravity=True, seed=seed)
    except Exception:
        pos = nx.spring_layout(G, seed=seed, k=0.6 / np.sqrt(max(G.number_of_nodes(), 1)))
    ex, ey = [], []
    for a, b in G.edges():
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#C9D1CF", width=0.8), hoverinfo="skip",
                               showlegend=False))
    sym = {"enzyme": "diamond", "reaction": "square", "metabolite": "circle"}
    for t, s in sym.items():
        d = nodes[nodes["type"] == t]
        if d.empty:
            continue
        z = d["z"].fillna(0).clip(-5, 5)
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in d["node"]], y=[pos[n][1] for n in d["node"]], mode="markers+text" if t != "reaction" else "markers",
            name=t, marker=dict(symbol=s, size=11 if t != "reaction" else 8, color=z, colorscale="RdBu_r", cmin=-4,
                                cmax=4, line=dict(width=0.6, color="#333")),
            text=[str(x)[:18] for x in d["label"]] if t != "reaction" else None, textposition="top center",
            textfont=dict(size=9), hovertext=[f"{t}: {l}<br>z = {v:.2f}" for l, v in zip(d["label"], d["z"].fillna(0))],
            hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title + "  (diamond = enzyme, square = reaction, circle = metabolite)", 700,
                   legend=dict(orientation="h", y=-0.05))


def variance_heatmap(r2: pd.DataFrame):
    fig = go.Figure(go.Heatmap(z=r2.values, x=r2.columns, y=r2.index, colorscale="Teal", zmin=0,
                               text=np.round(r2.values, 1), texttemplate="%{text}",
                               colorbar=dict(title="% variance")))
    return _layout(fig, "Variance explained per factor and layer (MOFA+)", 320)


def factor_scatter(scores: pd.DataFrame, groups: pd.Series, fx: str, fy: str):
    fig = go.Figure()
    for g, col in zip(sorted(groups.dropna().unique()), [TEAL, ORANGE, "#2C6FB7", "#8E44AD"]):
        s = scores[groups.reindex(scores.index) == g]
        fig.add_trace(go.Scatter(x=s[fx], y=s[fy], mode="markers", name=g, text=s.index,
                                 marker=dict(size=10, color=col, line=dict(width=0.6, color="white"))))
    fig.update_xaxes(title=fx)
    fig.update_yaxes(title=fy)
    return _layout(fig, "Sample scores", 460)


def roc_plot(sig: dict):
    fpr, tpr = sig["roc"]
    fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=TEAL, width=3),
                               name=f"Integrated (AUC {sig['auc_mean']:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#999"), name="chance"))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return _layout(fig, "Out-of-fold ROC (nested CV, mean prediction over repeats)", 440,
                   legend=dict(x=0.45, y=0.08))


def perm_hist(sig: dict):
    fig = go.Figure(go.Histogram(x=sig["perm_aucs"], nbinsx=20, marker_color="#B8C2C0", name="permuted labels"))
    fig.add_vline(x=sig["auc_mean"], line_color=ORANGE, line_width=3,
                  annotation_text=f"observed {sig['auc_mean']:.2f} (p = {sig['p_perm']:.3f})")
    fig.update_xaxes(title="Cross-validated AUC")
    return _layout(fig, "Label-permutation null", 340, showlegend=False)


def panel_bars(panel: pd.DataFrame):
    d = panel.sort_values("selection_frequency")
    cmap = {"mRNA": "#2C6FB7", "protein": TEAL, "metabolite": ORANGE}
    fig = go.Figure(go.Bar(x=d["selection_frequency"], y=[f"{a} ({b})" for a, b in zip(d["label"], d["layer"])],
                           orientation="h", marker_color=[cmap.get(l, "#888") for l in d["layer"]]))
    fig.add_vline(x=0.7, line_dash="dot", line_color="#555")
    fig.update_xaxes(title="Selection frequency across outer folds", range=[0, 1.02])
    return _layout(fig, "Stability-selected integrated panel", max(320, 24 * len(d) + 120))


def targets_bar(tg: pd.DataFrame, top: int = 20):
    d = tg.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=-np.log10(d["rra_score"].clip(1e-30)), y=d["gene"], orientation="h",
                           marker_color=[ORANGE if c else TEAL for c in d["choke_point"]],
                           text=d["evidence"], hovertemplate="%{y}: %{text}<extra></extra>"))
    fig.update_xaxes(title="-log10 rank-aggregation score")
    return _layout(fig, "Prioritised targets (orange = choke point)", max(380, 24 * len(d) + 120))


def coverage_bars(layers: dict):
    rows = []
    for k, L in layers.items():
        if k == "met":
            mapped = (L.feat["map_rule"] != "unmapped").sum()
        else:
            mapped = (L.feat["ensg"] != "").sum()
        rows.append((L.name, len(L.feat), mapped))
    d = pd.DataFrame(rows, columns=["layer", "features", "mapped"])
    fig = go.Figure([go.Bar(x=d["layer"], y=d["features"], name="features", marker_color="#C9D1CF"),
                     go.Bar(x=d["layer"], y=d["mapped"], name="mapped to metabolic KB", marker_color=TEAL)])
    fig.update_layout(barmode="overlay")
    return _layout(fig, "Identifier mapping coverage", 360)


def config(fmt: str = "png", scale: int = 4, name: str = "trace_figure") -> dict:
    return {"displaylogo": False, "toImageButtonOptions": {"format": fmt, "scale": scale, "filename": name}}
