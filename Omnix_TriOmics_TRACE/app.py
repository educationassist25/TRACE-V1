"""
Omnix TRACE - Tri-omics Reaction-Anchored Cascade Evidence (standalone Streamlit app, v1).

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from trace_core import __version__
from trace_core import figures as F
from trace_core import report as R
from trace_core.cascade import PATTERN_HELP
from trace_core.data import LAYER_NAMES, build_design, cramers_v, make_layer, read_table
from trace_core.engine import run_trace
from trace_core.kb import load_kb

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_data")
LOGO = os.path.join(HERE, "assets", "omnix_logo_web.png")

st.set_page_config(page_title="Omnix TRACE - Tri-Omics Integration", page_icon=LOGO if os.path.exists(LOGO) else None,
                   layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
h1, h2, h3 {color: #00574D;}
div[data-testid="stMetricValue"] {color: #00574D;}
.trace-note {background:#E3F2EF;border-left:4px solid #00695C;padding:10px 14px;border-radius:6px;margin:6px 0 12px 0;}
.trace-warn {background:#FFF3E9;border-left:4px solid #E65100;padding:10px 14px;border-radius:6px;margin:6px 0 12px 0;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading knowledge base")
def get_kb():
    return load_kb()


def show(fig, name="trace_figure"):
    cfg = F.config(st.session_state.get("fig_fmt", "png"), st.session_state.get("fig_scale", 6), name)
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def show_mpl(fig, max_scale: float = 1.15):
    """Show a matplotlib figure as a crisp vector image (SVG, text converted to outlines so it renders exactly as
    laid out, at any zoom or screen density). It scales down with the window and never beyond 115 % of its
    natural size. Returns the natural size in inches."""
    import base64
    import matplotlib.pyplot as plt
    w_in, h_in = fig.get_size_inches()
    buf = io.BytesIO()
    with plt.rc_context({"svg.fonttype": "path"}):
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<div style="width:100%;text-align:center"><img src="data:image/svg+xml;base64,{b64}" '
                f'style="width:100%;max-width:{int(w_in * 96 * max_scale)}px;height:auto;display:inline-block"/></div>',
                unsafe_allow_html=True)
    return float(w_in), float(h_in)


def table(df, height=380, key=None):
    st.dataframe(df, height=height, hide_index=True, key=key)


def csv_button(df, name, label="Download table (CSV)", key=None):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", key=key)


def _fonts():
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    pref = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans", "FreeSans", "Inter", "Carlito",
            "Times New Roman", "Liberation Serif", "DejaVu Serif", "FreeSerif"]
    return [f for f in pref if f in have] or ["DejaVu Sans"]


def download_panel(key: str, builder, w: float, h: float | None, fs: float = 7.0, name: str = "figure"):
    """Publication export for one figure: any format, 300-1200 DPI, custom size / font, or all formats
    at once. `builder(w, h, fs)` returns a matplotlib figure (rendered only when requested)."""
    from trace_core import pubfig as PF
    with st.expander("⬇ Download publication-quality figure (PNG · TIFF · JPEG · SVG · PDF · EPS)"):
        c = st.columns(3) + st.columns(3)          # two rows of three: readable at half-screen width
        fmt = c[0].selectbox("Format", list(PF.FORMATS) + ["All formats (ZIP)"], key=f"{key}_fmt")
        dpi = c[1].selectbox("DPI", [300, 600, 1200], index=1, key=f"{key}_dpi")
        font = c[2].selectbox("Font family", _fonts(), key=f"{key}_font")
        ww = c[3].number_input("Width (in)", 2.0, 30.0, float(w), 0.5, key=f"{key}_w")
        hh = c[4].number_input("Height (in, 0 = auto)", 0.0, 30.0, float(h or 0), 0.5, key=f"{key}_h")
        ff = c[5].number_input("Font (pt)", 4.0, 20.0, float(fs), 0.5, key=f"{key}_fs")
        t1, t2 = st.columns([1, 1])
        transp = t1.checkbox("Transparent background", key=f"{key}_tr")
        if t2.button("Render figure", key=f"{key}_go"):
            with st.spinner("Rendering"):
                PF.style(font, ff)
                fmts = list(PF.FORMATS.values()) if fmt.startswith("All") else [PF.FORMATS[fmt]]
                out = {}
                for f in fmts:
                    fig = builder(ww, hh or None, ff)
                    out[f"{name}.{f}"] = PF.export(fig, f, dpi, transp)
                if len(out) == 1:
                    (fn, data), = out.items()
                    st.session_state[f"_pub_{key}"] = (fn, data, PF.MIME[fn.rsplit('.', 1)[1]])
                else:
                    st.session_state[f"_pub_{key}"] = (f"{name}_all_formats.zip", R.bundle(out), "application/zip")
        if f"_pub_{key}" in st.session_state:
            fn, data, mime = st.session_state[f"_pub_{key}"]
            st.download_button(f"Download {fn} ({len(data) / 1e6:.1f} MB)", data, fn, mime=mime, key=f"{key}_btn",
                               type="primary")


kb = get_kb()

# ===================================================================================== sidebar
with st.sidebar:
    if os.path.exists(LOGO):
        st.image(LOGO, width=96)
    st.markdown(f"### Omnix TRACE\nTri-Omics Integration Engine · v{__version__}")
    st.caption(f"Knowledge base: {kb.version} · Human-GEM {kb.manifest['sources']['Human-GEM'].get('version', '')}")
    st.markdown("**1 · Data**")
    source = st.radio("Dataset", ["Demo: paired tri-omics", "Demo: partially overlapping", "Upload my data"],
                      label_visibility="collapsed")
    up = {}
    if source == "Upload my data":
        st.caption("Wide tables: one row per feature, one column per sample. The first column is the "
                   "identifier unless you choose another. Exports from the Omnix modules work directly.")
        up["rna"] = st.file_uploader("Transcriptomics (genes)", type=["csv", "tsv", "txt", "xlsx"])
        up["prot"] = st.file_uploader("Proteomics (proteins)", type=["csv", "tsv", "txt", "xlsx"])
        up["met"] = st.file_uploader("Metabolomics / Lipidomics (required for the cascade)",
                                     type=["csv", "tsv", "txt", "xlsx"])
        up["meta"] = st.file_uploader("Sample metadata (Sample, Group, optional Batch)", type=["csv", "tsv", "xlsx"])
        up["map"] = st.file_uploader("Optional: metabolite ID table (feature, HMDB / KEGG / ChEBI)",
                                     type=["csv", "tsv", "xlsx"])

# ---------------------------------------------------------------- load raw tables
raw, meta, inputs = {}, None, {}
if source.startswith("Demo"):
    for k, f in (("rna", "demo_transcriptomics.csv"), ("prot", "demo_proteomics.csv"),
                 ("met", "demo_metabolomics.csv")):
        raw[k] = pd.read_csv(os.path.join(DEMO, f))
        inputs[f] = open(os.path.join(DEMO, f), "rb").read()
    meta = pd.read_csv(os.path.join(DEMO, "demo_metadata.csv"))
    if source == "Demo: partially overlapping":
        drop_p = ["S03", "S08", "S15", "S20"]
        drop_m = ["S05", "S11", "S17", "S23"]
        raw["prot"] = raw["prot"].drop(columns=drop_p)
        raw["met"] = raw["met"].drop(columns=drop_m)
else:
    for k in ("rna", "prot", "met"):
        if up.get(k) is not None:
            raw[k] = read_table(up[k])
            up[k].seek(0)
            inputs[up[k].name] = up[k].read()
    if up.get("meta") is not None:
        meta = read_table(up["meta"])
    if up.get("map") is not None and "met" in raw:
        mp = read_table(up["map"])
        key = mp.columns[0]
        raw["met"] = raw["met"].merge(mp.rename(columns={key: raw["met"].columns[0]}), on=raw["met"].columns[0],
                                      how="left", suffixes=("", "_map"))

with st.sidebar:
    id_cols, ok = {}, True
    sample_col, group_col, batch_col, case, ref = "Sample", "Group", None, None, None
    if source == "Upload my data":
        for k, df in raw.items():
            id_cols[k] = st.selectbox(f"{LAYER_NAMES[k]} identifier column", list(df.columns), key=f"id_{k}")
    if meta is not None:
        st.markdown("**2 · Design**")
        mc = list(meta.columns)
        sample_col = st.selectbox("Sample column", mc, index=mc.index("Sample") if "Sample" in mc else 0)
        group_col = st.selectbox("Group column", mc, index=mc.index("Group") if "Group" in mc else min(1, len(mc) - 1))
        bopts = ["(none)"] + [c for c in mc if c not in (sample_col, group_col)]
        batch_col = st.selectbox("Batch column", bopts, index=bopts.index("Batch") if "Batch" in bopts else 0)
        batch_col = None if batch_col == "(none)" else batch_col
        groups = sorted(meta[group_col].astype(str).unique())
        case = st.selectbox("Case group", groups, index=groups.index("Tumor") if "Tumor" in groups else len(groups) - 1)
        ref = st.selectbox("Reference group", [g for g in groups if g != case])
    else:
        ok = False
    st.markdown("**3 · Options**")
    with st.expander("Processing", expanded=False):
        transform = st.selectbox("Log transform", ["auto", "log2", "none"],
                                 help="auto: log2 if values look like raw intensities")
        max_missing = st.slider("Max missing per group", 0.1, 0.9, 0.5, 0.05)
        bc = {k: st.checkbox(f"Batch-correct {LAYER_NAMES[k]}", value=(k == "met" and batch_col is not None),
                             key=f"bc_{k}", disabled=batch_col is None) for k in ("rna", "prot", "met")}
    with st.expander("Statistics", expanded=False):
        B = st.select_slider("Label permutations", [200, 500, 1000, 2000, 5000], value=1000)
        Bt = st.select_slider("Topology permutations", [200, 500, 1000, 2000], value=1000)
        alpha = st.select_slider("FDR threshold (q)", [0.01, 0.05, 0.1], value=0.05)
        pthr = st.slider("Pattern threshold |z|", 1.0, 3.0, 1.5, 0.1)
        moderated = st.checkbox("Moderated t (empirical Bayes)", True)
        seed = st.number_input("Random seed", value=20261030, step=1)
    with st.expander("Engines", expanded=False):
        do_net = st.checkbox("Network, propagation and targets", True)
        do_fac = st.checkbox("Latent factors (MOFA+)", True)
        n_fac = st.slider("Maximum factors", 3, 15, 8)
        do_sig = st.checkbox("Integrated biomarker signature", True)
        sig_rep = st.slider("Outer CV repeats", 3, 20, 10)
        sig_perm = st.select_slider("Signature permutations", [20, 50, 100, 200], value=50)
        drug = st.file_uploader("Optional: druggable genes (one symbol per line)", type=["txt", "csv"])
    with st.expander("Figures", expanded=False):
        st.session_state["fig_fmt"] = st.selectbox("Download format (camera icon on each figure)", ["png", "svg"])
        st.session_state["fig_scale"] = st.select_slider("PNG scale (x screen size)", [4, 6, 8, 10], value=6)
    run = st.button("Run TRACE", type="primary", disabled=not ok or "met" not in raw or len(raw) < 2,
                    )

# ===================================================================================== header
st.markdown("# Omnix TRACE · Tri-Omics Integration")
st.markdown("Directional integration of **transcriptomics → proteomics → enzyme function → metabolomics**. "
            "Every reaction is scored for whether enzyme and metabolite changes point the same way, calibrated "
            "by label and network-topology permutations with FDR control.")

if not ok or "met" not in raw or len(raw) < 2:
    st.markdown('<div class="trace-warn">Load a metabolomics layer, at least one of transcriptomics or proteomics, '
                'and a sample metadata table to begin. Or choose a demo dataset in the sidebar.</div>',
                unsafe_allow_html=True)
    st.stop()

# ===================================================================================== run
if run:
    t0 = time.time()
    prog = st.progress(0.0, text="Preparing layers")
    layers = {}
    for k, df in raw.items():
        layers[k] = make_layer(k, df, meta, kb, id_col=id_cols.get(k), transform=transform,
                               max_missing=max_missing, batch_correct=bc[k], sample_col=sample_col,
                               group_col=group_col, batch_col=batch_col)
    design = build_design(meta, layers, case, ref, sample_col, group_col, batch_col)
    params = dict(B=int(B), B_topo=int(Bt), alpha=float(alpha), pattern_thr=float(pthr), moderated=moderated,
                  seed=int(seed))
    res = run_trace(layers, design, kb, params,
                    progress=lambda f, m: prog.progress(min(0.6, 0.6 * f), text=f"Engine A · {m}"))
    extras = {"kb_version": kb.version}
    if do_net:
        from trace_core import network as N
        prog.progress(0.65, text="Engine C · network propagation and targets")
        druggable = None
        if drug is not None:
            druggable = {x.strip().upper() for x in drug.read().decode(errors="ignore").replace(",", "\n").split("\n") if x.strip()}
            druggable = {g for g in {s for gs in res.cascade.rx_genes.values() for s in gs} if g.upper() in druggable}
        nodes, edges = N.build_graph(res, kb)
        prop = N.propagate(nodes, edges, B=200, seed=int(seed))
        extras["targets"] = N.rank_targets(res, kb, prop, druggable)
        extras["subnet"] = N.subnetwork(res, prop, edges)
        extras["prop"] = prop
        extras["network_params"] = {"restart": 0.3, "null": "degree-binned seed permutation, B=200"}
    if do_fac:
        from trace_core import factors as FA
        prog.progress(0.75, text="Engine B · MOFA+ latent factors")
        try:
            extras["factors"] = FA.run_factors(layers, design, n_factors=n_fac, seed=int(seed) % 10000)
            extras["factor_enrichment"] = FA.factor_enrichment(extras["factors"], layers, kb)
            extras["factor_params"] = {"max_factors": n_fac, "top_var": 1500, "restarts": 3}
        except Exception as e:  # keep the rest of the results
            extras["factors_error"] = str(e)
    if do_sig:
        from trace_core import signature as SG
        prog.progress(0.85, text="Engine D · nested cross-validated signature")
        extras["signature"] = SG.run_signature(layers, design, n_repeats=sig_rep, n_perm=sig_perm, seed=int(seed) % 10000)
        extras["signature_params"] = {"keepX_grid": [5, 10, 20], "repeats": sig_rep, "permutations": sig_perm}
    prog.progress(1.0, text=f"Done in {time.time() - t0:.0f} s")
    st.session_state["trace"] = {"res": res, "layers": layers, "extras": extras, "inputs": inputs,
                                 "run_id": time.time()}

if "trace" not in st.session_state:
    st.markdown('<div class="trace-note">Choose the design and options in the sidebar, then press '
                '<b>Run TRACE</b>. The demo takes about 20-40 seconds.</div>', unsafe_allow_html=True)
    st.stop()

S = st.session_state["trace"]
res, layers, extras = S["res"], S["layers"], S["extras"]
rx, pw = res.reactions, res.pathways
lname = {"rna": "mRNA", "prot": "protein", "met": "metabolite"}
from trace_core import ipa as IPA  # noqa: E402
from trace_core import netview as NV  # noqa: E402
from trace_core import pubfig as PF  # noqa: E402
from trace_core.layout import LAYOUTS  # noqa: E402

if "canonical" not in S:
    S["canonical"] = IPA.canonical_pathways(res, kb, res.params["alpha"])
cp_tbl = S["canonical"]

TAB_NAMES = ["Overview", "Data & mapping", "Concordance", "Reactions", "Cascade", "Pathways",
             "Pathway maps & network", "Targets", "Factors", "Signature", "Export", "Methods"]
try:  # Streamlit >= 1.50: remember the active tab and render only that tab (much faster)
    tabs = st.tabs(TAB_NAMES, key="main_tabs", on_change="rerun")
    LAZY = True
except TypeError:
    tabs = st.tabs(TAB_NAMES)
    LAZY = False


def on(i: int) -> bool:
    if not LAZY:
        return True
    return st.session_state.get("main_tabs", TAB_NAMES[0]) in (TAB_NAMES[i], None)


# ------------------------------------------------------------------------------------------- network UI
def _preset_cb(prefix):
    name = st.session_state.get(f"{prefix}_preset")
    for k, v in NV.PRESETS.get(name, {}).items():
        st.session_state[f"{prefix}_{k}"] = v


SCOPE_LABEL = {"Pathway": "Pathway map (IPA-style)", "Coherent reactions": "Coherent reactions",
               "Enzyme neighbourhood": "Enzyme / metabolite neighbourhood", "Molecule list": "My molecule list",
               "Top targets": "Top prioritised targets"}


def _img(data):
    try:
        st.image(data, width="stretch")
    except TypeError:  # older Streamlit
        st.image(data, use_container_width=True)


def network_ui(prefix: str):
    """One studio for IPA-style pathway maps and every other network view. Scope controls sit on top (they
    apply immediately); style settings live in a collapsible, tabbed panel whose two-column grids stack on
    narrow screens; the figure always spans the full width and scales with the window."""
    d = NV.NetOptions()
    K = lambda f: f"{prefix}_{f}"
    pchoices = pw.loc[pw["n_reactions_observed"] > 0].sort_values("p_combined")["pathway"].tolist()
    if K("scope") not in st.session_state:
        st.session_state[K("scope")] = "Pathway"
    if st.session_state.get(K("pathway")) not in pchoices:
        st.session_state[K("pathway")] = pchoices[0] if pchoices else ""
    for f, v in vars(d).items():   # initialise widget state once
        if K(f) not in st.session_state and not isinstance(v, (list, tuple)):
            st.session_state[K(f)] = v
    c1, c2 = st.columns([1, 1.6])
    c1.selectbox("Draw", list(SCOPE_LABEL), format_func=SCOPE_LABEL.get, key=K("scope"))
    scope = st.session_state[K("scope")]
    with c2:
        if scope == "Pathway":
            st.selectbox("Pathway (most significant first)", pchoices, key=K("pathway"))
        elif scope in ("Enzyme neighbourhood", "Molecule list"):
            cc = st.columns([3, 1])
            cc[0].text_input("Genes / metabolites (comma-separated)", key=K("seeds_txt"))
            if scope == "Enzyme neighbourhood":
                cc[1].number_input("Hops", 1, 2, key=K("hops"))
        elif scope == "Coherent reactions":
            st.caption("Reactions whose enzyme and metabolite changes agree (label and topology q below the FDR).")
        else:
            st.caption("The highest-ranked targets from Engine C and the reactions around them.")
    st.selectbox("Style preset (sets layout, shapes, colours and edges at once; fine-tune below)",
                 ["(custom)"] + list(NV.PRESETS), key=K("preset"), on_change=_preset_cb, args=(prefix,))
    with st.expander("Layout, colours, labels and edges", expanded=False):
        with st.form(f"{prefix}_form", border=False):
            t = st.tabs(["Content", "Layout", "Nodes & colours", "Labels", "Edges", "Canvas"])
            with t[0]:
                c = st.columns(2)
                c[0].slider("Maximum reactions", 5, 120, key=K("max_reactions"))
                c[1].slider("Isozymes shown per reaction", 1, 8, key=K("isozymes_per_reaction"))
                c[0].selectbox("Representation", ["Molecule view (IPA-style)", "Reaction view (bipartite)"], key=K("representation"))
                c[1].selectbox("Gene value", ["Best available (protein > mRNA)", "Protein", "mRNA"], key=K("value"))
                c[0].checkbox("Show unmeasured molecules", key=K("include_unmeasured"))
                c[1].checkbox("Hide isolated molecules", key=K("hide_isolated"))
                c[0].checkbox("Add protein-protein edges (STRING add-on)", key=K("include_ppi"),
                              disabled="ppi_string" not in kb.extra)
            with t[1]:
                c = st.columns(2)
                c[0].selectbox("Layout", LAYOUTS, key=K("layout"))
                c[1].selectbox("Subcellular bands", ["IPA (5 bands)", "Organelles (10 bands)"], key=K("band_detail"))
                c[0].slider("Spacing", 0.8, 2.5, key=K("spacing"), step=0.1)
                c[1].text_input("Focus molecules (radial layout)", "", key=K("focus_txt"))
                c[0].number_input("Layout seed", 1, 9999, key=K("seed"))
                c[1].checkbox("Auto-enlarge canvas so nothing overlaps", key=K("auto_size"))
                c[1].checkbox("Shade network modules (communities)", key=K("hulls"))
            with t[2]:
                c = st.columns(2)
                c[0].selectbox("Colour by", NV.COLOR_BY, key=K("color_by"))
                c[1].selectbox("Shape scheme", NV.SHAPES, key=K("shape_scheme"))
                c[0].selectbox("Palette", list(NV.PALETTES), key=K("palette"))
                c[1].slider("Colour scale |z| max", 1.0, 8.0, key=K("zmax"), step=0.5)
                c[0].selectbox("Node size by", ["Constant", "|z|", "Degree", "Target score"], key=K("size_mode"))
                c[1].slider("Node size (pt)", 6.0, 36.0, key=K("node_size"), step=1.0)
                c[0].checkbox("Predict unmeasured metabolites (MAP-like)", key=K("prediction"))
                c[1].slider("Prediction threshold |E|", 0.5, 3.0, key=K("pred_thr"), step=0.1)
                c[0].text_input("Highlight (comma-separated)", "", key=K("highlight_txt"))
                c[1].color_picker("Highlight outline", key=K("highlight_color"))
            with t[3]:
                c = st.columns(2)
                c[0].selectbox("Label", ["All", "Measured", "Significant", "Top N by |z|", "None"], key=K("label_mode"))
                c[1].slider("Top N", 5, 100, key=K("label_top_n"))
                c[0].slider("Label size (pt)", 5.0, 16.0, key=K("font_size"), step=0.5)
                c[1].selectbox("Font family", _fonts(), key=K("font_family"))
                c[0].checkbox("Label background", key=K("label_box"))
                c[1].checkbox("Gene names inside boxes (KEGG scheme)", key=K("label_inside"))
                c[0].checkbox("Hide labels that cannot be placed cleanly", key=K("hide_unplaced"))
            with t[4]:
                c = st.columns(2)
                c[0].selectbox("Edge colour", ["IPA relationship", "By relation type", "Single colour"], key=K("edge_color"))
                c[1].color_picker("Single colour", key=K("edge_single"))
                c[0].selectbox("Routing", NV.ROUTES, key=K("edge_route"))
                c[1].selectbox("Width by", ["Constant", "|enzyme evidence|"], key=K("edge_width_by"))
                c[0].slider("Edge width", 0.3, 4.0, key=K("edge_width"), step=0.1)
                c[1].slider("Edge opacity", 0.1, 1.0, key=K("edge_alpha"), step=0.05)
                c[0].checkbox("Arrow heads", key=K("arrows"))
                c[1].checkbox("Line style encodes effect (print)", key=K("effect_linestyle"))
            with t[5]:
                c = st.columns(2)
                c[0].text_input("Title", key=K("title"))
                c[1].selectbox("Theme", ["Light", "Dark"], key=K("theme"))
                c[0].slider("Width (in)", 5.0, 30.0, key=K("width_in"), step=0.5)
                c[1].slider("Height (in)", 4.0, 30.0, key=K("height_in"), step=0.5)
                c[0].checkbox("Legend", key=K("show_legend"))
                c[1].checkbox("Draw compartment bands", key=K("show_bands"))
                c[0].selectbox("Background", ["white", "transparent"], key=K("background"))
            st.form_submit_button("Apply", type="primary")
    vals = {f: st.session_state.get(K(f), v) for f, v in vars(d).items() if not isinstance(v, (list, tuple))}
    split = lambda t_: [x.strip() for x in str(st.session_state.get(K(t_), "")).split(",") if x.strip()]
    o = NV.NetOptions(**vals)
    o.seeds = split("seeds_txt") or ["ASS1"]
    o.highlight = split("highlight_txt")
    o.focus = split("focus_txt")
    o.size_range = d.size_range
    if o.scope == "Pathway" and not o.title:
        o.title = o.pathway
    ck = (S.get("run_id"), repr(o))
    cache = S.setdefault("net_cache", {})
    if ck not in cache:
        with st.spinner("Laying out the network"):
            N, E = NV.build_view(res, kb, o, extras.get("targets"))
            if N.empty:
                cache[ck] = (N, E, None, None)
            else:
                g = NV.geometry(N, E, o)
                import matplotlib.pyplot as plt
                buf = io.BytesIO()
                with plt.rc_context({"svg.fonttype": "path"}):
                    NV.render_mpl(N, E, o, g).savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.05)
                plt.close("all")
                cache[ck] = (N, E, g, buf.getvalue())
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
    N, E, g, png = cache[ck]
    if g is None:
        st.info("Nothing to draw for this selection. Widen the scope or show unmeasured molecules.")
        return
    cl = g.get("collisions", {})
    hard = cl.get("label_label", 0) + cl.get("label_node", 0) + cl.get("edge_node", 0)
    if not cl.get("checked", True):
        status = "labels placed radially outside the layout"
    elif hard == 0:
        status = "no overlapping labels or nodes, no edges through nodes"
    else:
        status = f"{hard} overlap(s) remain: raise Spacing or lower Maximum reactions"
    st.caption(f"{len(N)} molecules, {len(E)} relationships · canvas {g['W']:.1f} × {g['H']:.1f} in · {status}")
    view = st.radio("View", ["Exact preview (identical to the export)", "Interactive (hover and zoom)"],
                    horizontal=True, key=f"{prefix}_view", label_visibility="collapsed")
    if view.startswith("Exact"):
        import base64
        st.markdown(f'<img src="data:image/svg+xml;base64,{base64.b64encode(png).decode()}" '
                    f'style="width:100%;height:auto;display:block"/>', unsafe_allow_html=True)
        st.caption("Vector preview: sharp at any zoom (Ctrl/Cmd + to enlarge). Exports are identical.")
    else:
        import streamlit.components.v1 as components
        fig = NV.render_plotly(N, E, o, g)
        cfg = F.config(st.session_state.get("fig_fmt", "png"), st.session_state.get("fig_scale", 6), f"{prefix}_network")
        html = fig.to_html(include_plotlyjs=True, full_html=False, config=cfg)
        components.html(html, height=min(int(fig.layout.height) + 30, 900), scrolling=True)
        st.caption("Shown at true size so no label overlaps: scroll inside the frame to pan, drag to zoom, "
                   "double-click to reset. The exact preview above scales to your window instead.")

    def build(w, h, fs_):
        import dataclasses
        o2 = dataclasses.replace(o, width_in=w, height_in=h or o.height_in * w / o.width_in, font_size=fs_)
        return NV.render_mpl(N, E, o2)
    download_panel(f"{prefix}_pub", build, o.width_in, o.height_in, o.font_size, f"{prefix}_network")
    c = st.columns(2)
    c[0].download_button("Cytoscape (.cyjs)", NV.to_cyjs(N, E, g), f"{prefix}_network.cyjs", key=f"{prefix}_cy",
                         width="stretch")
    c[1].download_button("GraphML", NV.to_graphml(N, E), f"{prefix}_network.graphml", key=f"{prefix}_gml",
                         width="stretch")
    nt, et = NV.export_tables(N, E)
    c[0].download_button("Nodes (CSV)", nt, f"{prefix}_nodes.csv", key=f"{prefix}_nodes", width="stretch")
    c[1].download_button("Edges (CSV)", et, f"{prefix}_edges.csv", key=f"{prefix}_edges", width="stretch")


# ===================================================================================== Overview
with tabs[0]:
  if on(0):
      st.markdown(f'<div class="trace-note"><b>{res.design.tier_text}.</b> Contrast: <b>{res.design.case}</b> vs '
                  f'<b>{res.design.ref}</b>. Results describe evidence consistent with a mechanism, not proof of '
                  f'causation.</div>', unsafe_allow_html=True)
      c = st.columns(3) + st.columns(3)          # two rows: numbers stay readable at half width
      c[0].metric("Observable reactions", f"{len(rx):,}")
      c[1].metric("Coherent reactions", int(rx["significant"].sum()))
      c[2].metric("Metabolite-driven", int(rx["metabolite_driven"].sum()))
      c[3].metric("Shared pathways", int(pw["class"].str.startswith("Shared").sum()))
      c[4].metric("Omics-specific", int(pw["class"].str.endswith("-specific").sum()))
      c[5].metric("mRNA-protein r", f"{res.coupling.get('pearson_z', np.nan):.2f}" if res.concordance is not None else "n/a")
      st.markdown("### Key findings")
      lines = []
      sig = rx[rx["significant"]].drop_duplicates(subset=["enzymes", "E", "M"])
      for _, r in sig.head(5).iterrows():
          lines.append(f"- **{r['enzymes'].split(',')[0]}** · {r['name']} — *{r['pattern']}* "
                       f"(E = {r['E']:.1f}, M = {r['M']:.1f}, q = {r['q_label']:.1e}). {PATTERN_HELP.get(r['pattern'], '')}")
      for _, r in pw[pw["class"].str.startswith("Shared") | pw["class"].str.endswith("-specific")].head(5).iterrows():
          lines.append(f"- Pathway **{r['pathway']}** — {r['class']} (combined z = {r['z_combined']:.1f}, "
                       f"q = {r['q_combined']:.1e}).")
      if len(cp_tbl):
          act = cp_tbl[cp_tbl["prediction"].isin(["Increased activity", "Decreased activity"])].head(3)
          for _, r in act.iterrows():
              lines.append(f"- Canonical pathway **{r['pathway']}**: {r['prediction'].lower()} "
                           f"(directional z = {r['z_directional']:.2f}, p = {r['p']:.1e}).")
      if extras.get("targets") is not None:
          tg = extras["targets"].head(3)
          lines.append("- Top prioritised targets: " + ", ".join(f"**{g}** ({e or 'ranked'})" for g, e in zip(tg["gene"], tg["evidence"])))
      if res.concordance is not None:
          vc = res.concordance["class"].value_counts()
          lines.append(f"- Gene-protein concordance: {vc.get('Coupled', 0)} coupled, "
                       f"{vc.get('Protein-only (post-transcriptional)', 0)} protein-only, "
                       f"{vc.get('mRNA-only (buffered)', 0)} mRNA-only, {vc.get('Opposing', 0)} opposing genes.")
      st.markdown("\n".join(lines) if lines else "No significant findings at the chosen FDR.")
      show(F.pattern_grid(rx), "reaction_patterns")
      download_panel("ov_pat", lambda w, h, fs: PF.pattern_grid(rx, 12, w, h or 4.2, fs), 5.6, 4.2, 7, "reaction_patterns")
      if len(cp_tbl):
          show(F.canonical_bars(cp_tbl, 15), "canonical_pathways")
          download_panel("ov_can", lambda w, h, fs: PF.canonical(cp_tbl, 15, w, h, fs), 7.6, None, 7, "canonical_pathways")

# ===================================================================================== Data & mapping
with tabs[1]:
  if on(1):
      st.markdown("### Samples and evidence tier")
      ov = res.design.overlap
      st.write(pd.DataFrame({"layer": [LAYER_NAMES[k] for k in ov["per_layer"]],
                             "samples in design": list(ov["per_layer"].values())}).assign(
          **{"samples in all layers": ov["all_layers"]}))
      st.caption(res.design.tier_text)
      if "Batch" in res.design.meta.columns:
          v = cramers_v(res.design.meta["Batch"], res.design.meta["Group"])
          msg = "warning: batch and group are confounded" if v > 0.5 else "batch is not confounded with group"
          st.caption(f"Batch vs group Cramér's V = {v:.2f} ({msg}).")
      show(F.coverage_bars(layers), "mapping_coverage")
      download_panel("cov", lambda w, h, fs: PF.coverage(layers, w, h or 2.6, fs), 3.6, 2.6, 7, "mapping_coverage")
      for k, L in layers.items():
          with st.expander(f"{L.name}: processing log and identifier mapping"):
              for line in L.log:
                  st.write("• " + line)
              f = L.feat.reset_index().rename(columns={"index": "feature"})
              table(f.head(500), 260, key=f"map_{k}")
              un = f[f["map_rule"].isin(["unmapped", "not in metabolic KB"])]
              csv_button(un, f"unmapped_{k}.csv", f"Download {len(un)} unmapped features", key=f"un_{k}")

# ===================================================================================== Concordance
with tabs[2]:
  if on(2):
      if res.concordance is None:
          st.info("Concordance needs both transcriptomics and proteomics.")
      else:
          cp = res.coupling
          c = st.columns(4)
          c[0].metric("Genes in both layers", f"{cp['n_genes']:,}")
          c[1].metric("Deming slope (protein on mRNA)", f"{cp['beta']:.2f}")
          c[2].metric("Correlation of z-scores", f"{cp['pearson_z']:.2f}")
          c[3].metric("Median within-sample rho", f"{cp.get('median_within_rho', np.nan):.2f}" if "median_within_rho" in cp else "n/a (unpaired)")
          nl = st.slider("Genes labelled", 0, 40, 15, key="conc_nl")
          show(F.concordance(res.concordance, cp["beta"], nl), "concordance")
          download_panel("conc", lambda w, h, fs: PF.concordance(res.concordance, cp["beta"], nl, w, h or 4.0, fs),
                         5.4, 4.0, 7, "concordance")
          cls = st.multiselect("Show classes", sorted(res.concordance["class"].unique()),
                               default=[c for c in res.concordance["class"].unique() if c != "Unchanged"])
          d = res.concordance[res.concordance["class"].isin(cls)]
          table(d, 360, key="conc_tbl")
          csv_button(d, "gene_protein_concordance.csv", key="conc_dl")

# ===================================================================================== Reactions
with tabs[3]:
  if on(3):
      st.markdown("Each reaction combines **enzyme evidence E** (protein z through the GPR rule) with the "
                  "**metabolite balance M** (products minus substrates). **Coherence C = E × M.** A reaction is "
                  f"called coherent when both label-permutation and topology q < {res.params['alpha']}.")
      nl = st.slider("Enzymes labelled", 0, 40, 14, key="rx_nl")
      show(F.pattern_grid(rx, nl), "reaction_patterns")
      download_panel("rxpat", lambda w, h, fs: PF.pattern_grid(rx, nl, w, h or 4.2, fs), 6.0, 4.4, 7, "reaction_patterns")
      c1, c2, c3 = st.columns([1.2, 2, 2])
      only = c1.radio("Rows", ["Coherent", "Metabolite-driven", "All"], horizontal=False)
      pats = c2.multiselect("Patterns", list(PATTERN_HELP), default=[])
      q = c3.text_input("Search enzyme, reaction or metabolite")
      d = rx
      if only == "Coherent":
          d = d[d["significant"]]
      elif only == "Metabolite-driven":
          d = d[d["metabolite_driven"]]
      if pats:
          d = d[d["pattern"].isin(pats)]
      if q:
          m = d[["name", "enzymes", "substrates", "products", "subsystem"]].apply(
              lambda s_: s_.astype(str).str.contains(q, case=False)).any(axis=1)
          d = d[m]
      cols = ["name", "enzymes", "subsystem", "pattern", "E", "S", "P", "M", "C", "q_label", "q_topology",
              "q_balance", "enzyme_evidence", "substrates", "products", "ec", "kegg_rxn", "reaction"]
      table(d[cols].round(4), 420, key="rx_tbl")
      csv_button(d, "reactions.csv", key="rx_dl")
      with st.expander("What the patterns mean"):
          for k, v in PATTERN_HELP.items():
              st.markdown(f"- **{k}**: {v}")

# ===================================================================================== Cascade
with tabs[4]:
  if on(4):
      st.markdown("Follow the chain **enzyme → reaction → metabolite**. Ribbon width = |z|, red = up, blue = down; "
                  "pale ribbons and italic names are substrates. Full reaction names are shown, each in its own box.")
      mode = st.radio("Select reactions by", ["Pathway", "Top coherent reactions", "Enzyme"], horizontal=True)
      if mode == "Pathway":
          choices = pw.loc[pw["n_reactions_observed"] > 0, "pathway"].tolist()
          p = st.selectbox("Pathway", choices, key="casc_pw")
          sel = rx[rx["subsystem"].str.split("; ").apply(lambda x: p in x)]
          sel = sel.sort_values("p_label").head(st.slider("Reactions shown", 3, 30, 12, key="casc_n"))
      elif mode == "Top coherent reactions":
          sel = rx.sort_values("p_label").head(st.slider("Reactions shown", 3, 30, 12, key="casc_n2"))
      else:
          genes = sorted({g.strip() for e in rx["enzymes"] for g in e.split(",")})
          g = st.selectbox("Enzyme", genes, index=genes.index("ASS1") if "ASS1" in genes else 0)
          sel = rx[rx["enzymes"].str.split(", ").apply(lambda x: g in x)].head(15)
      if len(sel):
          PF.style("DejaVu Sans", 8)
          cw_, _ = show_mpl(PF.cascade_alluvial(sel, w=7.5, fs=8), max_scale=1.3)
          download_panel("casc", lambda w, h, fs: PF.cascade_alluvial(sel, w, h, fs), round(cw_ * 2) / 2, None, 8, "cascade")
          table(sel[["name", "enzymes", "pattern", "E", "S", "P", "C", "q_label"]].round(4), 260, key="casc_tbl")

# ===================================================================================== Pathways
with tabs[5]:
  if on(5):
      st.markdown("#### Canonical pathways (IPA-style)")
      st.caption("Bars: -log10 p of a right-tailed Fisher's exact test of significant molecules (any layer). Colour: "
                 "directional z-score = (up - down)/sqrt(up + down) over significant molecules; orange = increased, "
                 "blue = decreased, grey = fewer than 4 directional molecules. Line: ratio of significant to all "
                 "pathway molecules. Dashed line: p = 0.05.")
      nc = st.slider("Canonical pathways shown", 10, 60, 25, key="can_n")
      if len(cp_tbl):
          show(F.canonical_bars(cp_tbl, nc), "canonical_pathways")
          download_panel("can", lambda w, h, fs: PF.canonical(cp_tbl, nc, w, h, fs), 7.6, None, 7, "canonical_pathways")
          with st.expander("Canonical pathway table"):
              table(cp_tbl.round(5), 320, key="can_tbl")
              csv_button(cp_tbl, "canonical_pathways.csv", key="can_dl")
      st.markdown("#### Cross-layer pathway activity (TRACE)")
      st.markdown("Per-layer pathway activity (Stouffer z standardised by label permutations), combined across "
                  "layers with heterogeneity (Cochran's Q, I²). **Shared-concordant**: all significant layers agree; "
                  "**shared-discordant**: layers disagree; **omics-specific**: one layer only, with the others "
                  "measured but unchanged.")
      ntop = st.slider("Pathways shown", 10, 60, 25)
      show(F.pathway_dotplot(pw, ntop), "pathways")
      download_panel("pwdot", lambda w, h, fs: PF.pathway_dotplot(pw, ntop, w, h, fs), 7.2, None, 7, "pathway_activity")
      cls = st.multiselect("Classes", sorted(pw["class"].unique()), default=[])
      d = pw[pw["class"].isin(cls)] if cls else pw
      table(d.round(4), 380, key="pw_tbl")
      csv_button(d, "pathways.csv", key="pw_dl")

# ===================================================================================== Pathway maps & network studio
with tabs[6]:
  if on(6):
      st.markdown("**Pathway maps and network studio.** Start from an IPA-style pathway map or any other view. "
                  "**Shape** = molecule type; **red / green** = measured up / down; **orange / blue outline** = "
                  "predicted increase / decrease of unmeasured metabolites; **edge colour** = leads to increase "
                  "(orange), decrease (blue), inconsistent with the data (yellow) or not predicted (grey). Edges are "
                  "routed around nodes and every label is placed clear of other labels and nodes.")
      network_ui("net")

# ===================================================================================== Targets
with tabs[7]:
  if on(7):
      if extras.get("targets") is None:
          st.info("Enable 'Network, propagation and targets' in the sidebar and run again.")
      else:
          tg = extras["targets"]
          st.markdown("Targets are ranked by robust rank aggregation of reaction coherence, mRNA-protein agreement, "
                      "choke-point status and network propagation (plus druggability when a list is uploaded). "
                      "**They are hypotheses for experimental follow-up.**")
          nt = st.slider("Targets shown", 5, 50, 20, key="tg_n")
          PF.style("DejaVu Sans", 8)
          tw_, _ = show_mpl(PF.targets_bar(tg, nt, fs=8))
          download_panel("tg", lambda w, h, fs: PF.targets_bar(tg, nt, w, h, fs), round(tw_ * 2) / 2, None, 8, "targets")
          table(tg.round(4), 320, key="tg_tbl")
          csv_button(tg, "targets.csv", key="tg_dl")
      st.markdown("### Reporter metabolites")
      st.caption("Metabolites around which the surrounding reactions are jointly coherent (Patil & Nielsen 2005).")
      table(res.reporter_mets.head(200).round(5), 300, key="rep_tbl")
      if res.regulators is not None:
          st.markdown("### Transcription-factor regulators")
          table(res.regulators.round(5), 300, key="tf_tbl")
      else:
          st.caption("TF regulator inference becomes available when CollecTRI regulons are added to the knowledge "
                     "base (python -m trace_core.kb_build --skip-core --collectri).")

# ===================================================================================== Factors
with tabs[8]:
  if on(8):
      f = extras.get("factors")
      if not f:
          st.info(extras.get("factors_error", "Enable 'Latent factors (MOFA+)' in the sidebar and run again."))
      else:
          st.caption(f"Method: {f['method']}. Factors explaining ≥ 2% variance in at least one layer are shown.")
          show(F.variance_heatmap(f["r2"]), "factor_variance")
          download_panel("fvar", lambda w, h, fs: PF.variance_heatmap(f["r2"], w, h or 1.8, fs), 0.6 * f["r2"].shape[1] + 1.8,
                         1.8, 7, "factor_variance")
          cols = list(f["scores"].columns)
          if len(cols) >= 2:
              a, b = st.columns(2)
              fx = a.selectbox("x", cols, 0)
              fy = b.selectbox("y", cols, 1)
              lbl = st.checkbox("Label samples", False, key="fac_lbl")
              show(F.factor_scatter(f["scores"], res.design.meta["Group"], fx, fy), "factor_scores")
              download_panel("fsc", lambda w, h, fs: PF.factor_scatter(f["scores"], res.design.meta["Group"], fx, fy, w,
                                                                     h or 3.2, fs, label_samples=lbl), 4.2, 3.4, 7, "factor_scores")
          table(f["summary"].round(3), 220, key="fac_sum")
          st.markdown("**Factor associations with group and covariates**")
          table(f["assoc"].round(4), 240, key="fac_assoc")
          if extras.get("factor_enrichment") is not None and len(extras["factor_enrichment"]):
              st.markdown("**Pathways enriched in factor loadings**")
              table(extras["factor_enrichment"].head(300).round(5), 300, key="fac_enr")

# ===================================================================================== Signature
with tabs[9]:
  if on(9):
      s = extras.get("signature")
      if not s:
          st.info("Enable 'Integrated biomarker signature' in the sidebar and run again.")
      else:
          if s.get("status") != "ok":
              st.markdown(f'<div class="trace-warn">{s["status"]}</div>', unsafe_allow_html=True)
          else:
              c = st.columns(4)
              c[0].metric("AUC (nested CV)", f"{s['auc_mean']:.2f}", f"95% CI {s['auc_ci'][0]:.2f}-{s['auc_ci'][1]:.2f}",
                          delta_color="off")
              c[1].metric("Balanced accuracy", f"{s['balanced_accuracy']:.2f}")
              c[2].metric("Permutation p", f"{s['p_perm']:.3f}")
              c[3].metric("Features per block", s["keepX"])
              st.markdown('<div class="trace-warn">Cross-validated performance on one cohort is optimistic for '
                          'clinical use. Validate the panel in an independent cohort.</div>', unsafe_allow_html=True)
              a, b = st.columns(2)
              with a:
                  show(F.roc_plot(s), "signature_roc")
                  download_panel("roc", lambda w, h, fs: PF.roc(s, w, h or w, fs), 3.4, 3.4, 7, "signature_roc")
              with b:
                  show(F.perm_hist(s), "signature_permutation")
                  download_panel("perm", lambda w, h, fs: PF.perm_hist(s, w, h or 2.6, fs), 3.6, 2.6, 7, "signature_permutation")
              show(F.panel_bars(s["panel"]), "signature_panel")
              download_panel("panel", lambda w, h, fs: PF.panel_bars(s["panel"], w, h, fs), 5.0, None, 7, "signature_panel")
              st.write(pd.DataFrame({"model": ["Integrated (all layers)"] + [f"{k} only" for k in s["single_layer_auc"]],
                                     "AUC": [s["auc_mean"]] + list(s["single_layer_auc"].values())}).round(3))
              csv_button(s["panel"], "signature_panel.csv", key="sig_dl")
          lf = s.get("late_fusion", {})
          if lf.get("status") == "ok":
              st.markdown("**Late fusion (uses every sample available in each layer)**")
              st.write(pd.DataFrame({"model": ["Fused"] + list(lf["auc_per_layer"]),
                                     "AUC (CV)": [lf["auc_fused"]] + list(lf["auc_per_layer"].values())}).round(3))

# ===================================================================================== Export
with tabs[10]:
  if on(10):
      st.markdown("### Download results")
      xls = R.excel(res, extras)
      man = R.manifest(res, S["inputs"], kb, extras)
      meth = R.methods_text(res, kb, extras)
      c = st.columns(3)
      c[0].download_button("Excel workbook (all tables)", xls, "TRACE_results.xlsx", type="primary")
      c[1].download_button("Run manifest (JSON)", json.dumps(man, indent=2, default=str), "TRACE_manifest.json")
      c[2].download_button("Methods text (Markdown)", meth, "TRACE_methods.md")
      st.markdown("### All publication figures at once")
      c = st.columns(3)
      fmts = c[0].multiselect("Formats", list(PF.FORMATS), default=["PNG", "PDF", "SVG"], key="all_fmts")
      dpi = c[1].selectbox("DPI", [300, 600, 1200], index=1, key="all_dpi")
      font = c[2].selectbox("Font family", _fonts(), key="all_font")
      if st.button("Build figure set (ZIP)"):
          with st.spinner("Rendering all figures"):
              PF.style(font, 7)
              builders = {"reaction_patterns": lambda: PF.pattern_grid(rx),
                          "pathway_activity": lambda: PF.pathway_dotplot(pw),
                          "mapping_coverage": lambda: PF.coverage(layers)}
              if len(cp_tbl):
                  builders["canonical_pathways"] = lambda: PF.canonical(cp_tbl)
              if res.concordance is not None:
                  builders["concordance"] = lambda: PF.concordance(res.concordance, res.coupling["beta"])
              if extras.get("targets") is not None:
                  builders["targets"] = lambda: PF.targets_bar(extras["targets"])
              if extras.get("factors"):
                  builders["factor_variance"] = lambda: PF.variance_heatmap(extras["factors"]["r2"])
              sg = extras.get("signature")
              if sg and sg.get("status") == "ok":
                  builders["signature_roc"] = lambda: PF.roc(sg)
                  builders["signature_panel"] = lambda: PF.panel_bars(sg["panel"])
              top_pw = pw.loc[pw["n_reactions_observed"] >= 2, "pathway"].head(3).tolist()
              for k_, p_ in enumerate(top_pw):
                  o_ = NV.NetOptions(scope="Pathway", pathway=p_, max_reactions=30, title=p_)
                  builders[f"pathway_map_{k_ + 1}"] = (lambda o_=o_: NV.render_mpl(*NV.build_view(res, kb, o_, extras.get("targets")), o_))
              out = {}
              for nm, b in builders.items():
                  for fl in fmts:
                      ext = PF.FORMATS[fl]
                      out[f"{nm}.{ext}"] = PF.export(b(), ext, dpi)
              st.session_state["figzip"] = R.bundle(out)
      if "figzip" in st.session_state:
          st.download_button("Download figure set (ZIP)", st.session_state["figzip"], "TRACE_figures.zip", type="primary")
      st.markdown("### Everything in one archive")
      if st.button("Build full bundle"):
          files = {"TRACE_results.xlsx": xls, "TRACE_manifest.json": man, "TRACE_methods.md": meth}
          files.update({f"figures/{k}": v for k, v in R.pub_figures(res, extras, "png", 600).items()})
          if len(cp_tbl):
              files["canonical_pathways.csv"] = cp_tbl.to_csv(index=False)
          if extras.get("subnet"):
              from trace_core.network import to_graphml
              files["TRACE_subnetwork.graphml"] = to_graphml(*extras["subnet"])
          st.session_state["fullzip"] = R.bundle(files)
      if "fullzip" in st.session_state:
          st.download_button("Download full bundle (ZIP)", st.session_state["fullzip"], "TRACE_bundle.zip")

# ===================================================================================== Methods
with tabs[11]:
  if on(11):
      st.markdown(R.methods_text(res, kb, extras))
      st.markdown("**IPA-style displays.** Canonical-pathway bars, molecule shapes, measured (red/green) and predicted "
                  "(orange/blue) colours, relationship-coloured edges and subcellular layouts follow the visual "
                  "conventions of Ingenuity Pathway Analysis. They are computed from TRACE statistics and the "
                  "Human-GEM knowledge base; they do not use IPA's proprietary knowledge base or its activation "
                  "z-score algorithm.")
      with st.expander("Run log and timings"):
          for l in res.log:
              st.write("• " + l)
          st.write({k: f"{v:.1f} s" for k, v in res.timings.items()})
      with st.expander("Knowledge base"):
          st.json(kb.manifest)
