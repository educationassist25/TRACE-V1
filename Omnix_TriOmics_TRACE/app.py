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
    cfg = F.config(st.session_state.get("fig_fmt", "png"), st.session_state.get("fig_scale", 4), name)
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def table(df, height=380, key=None):
    st.dataframe(df, height=height, hide_index=True, key=key)


def csv_button(df, name, label="Download table (CSV)", key=None):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", key=key)


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
        st.session_state["fig_scale"] = st.select_slider("PNG scale", [2, 3, 4, 6], value=4)
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
    st.session_state["trace"] = {"res": res, "layers": layers, "extras": extras, "inputs": inputs}

if "trace" not in st.session_state:
    st.markdown('<div class="trace-note">Choose the design and options in the sidebar, then press '
                '<b>Run TRACE</b>. The demo takes about 20-40 seconds.</div>', unsafe_allow_html=True)
    st.stop()

S = st.session_state["trace"]
res, layers, extras = S["res"], S["layers"], S["extras"]
rx, pw = res.reactions, res.pathways
lname = {"rna": "mRNA", "prot": "protein", "met": "metabolite"}

tabs = st.tabs(["Overview", "Data & mapping", "Concordance", "Reactions", "Cascade", "Pathways", "Pathway maps",
                "Network & targets", "Factors", "Signature", "Export", "Methods"])

# ===================================================================================== Overview
with tabs[0]:
    st.markdown(f'<div class="trace-note"><b>{res.design.tier_text}.</b> Contrast: <b>{res.design.case}</b> vs '
                f'<b>{res.design.ref}</b>. Results describe evidence consistent with a mechanism, not proof of '
                f'causation.</div>', unsafe_allow_html=True)
    c = st.columns(6)
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
    show(F.pathway_dotplot(pw, 15), "pathways_top15")

# ===================================================================================== Data & mapping
with tabs[1]:
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
    if res.concordance is None:
        st.info("Concordance needs both transcriptomics and proteomics.")
    else:
        cp = res.coupling
        c = st.columns(4)
        c[0].metric("Genes in both layers", f"{cp['n_genes']:,}")
        c[1].metric("Deming slope (protein on mRNA)", f"{cp['beta']:.2f}")
        c[2].metric("Correlation of z-scores", f"{cp['pearson_z']:.2f}")
        c[3].metric("Median within-sample rho", f"{cp.get('median_within_rho', np.nan):.2f}" if "median_within_rho" in cp else "n/a (unpaired)")
        show(F.concordance(res.concordance, cp["beta"]), "concordance")
        cls = st.multiselect("Show classes", sorted(res.concordance["class"].unique()),
                             default=[c for c in res.concordance["class"].unique() if c != "Unchanged"])
        d = res.concordance[res.concordance["class"].isin(cls)]
        table(d, 360, key="conc_tbl")
        csv_button(d, "gene_protein_concordance.csv", key="conc_dl")

# ===================================================================================== Reactions
with tabs[3]:
    st.markdown("Each reaction combines **enzyme evidence E** (protein z through the GPR rule) with the "
                "**metabolite balance M** (products minus substrates). **Coherence C = E × M.** A reaction is "
                f"called coherent when both label-permutation and topology q < {res.params['alpha']}.")
    show(F.pattern_grid(rx, 14), "reaction_patterns")
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
            lambda s: s.astype(str).str.contains(q, case=False)).any(axis=1)
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
    st.markdown("Follow the chain **enzyme → reaction → metabolite**. Ribbon width = |z|, red = up, blue = down; "
                "substrate links are lighter.")
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
        show(F.cascade_sankey(sel, "Cascade"), "cascade")
        table(sel[["name", "enzymes", "pattern", "E", "S", "P", "C", "q_label"]].round(4), 260, key="casc_tbl")

# ===================================================================================== Pathways
with tabs[5]:
    st.markdown("Per-layer pathway activity (Stouffer z standardised by label permutations), combined across "
                "layers with heterogeneity (Cochran's Q, I²). **Shared-concordant**: all significant layers agree; "
                "**shared-discordant**: layers disagree; **omics-specific**: one layer only, with the others "
                "measured but unchanged. Coherence = mean reaction coherence in the pathway.")
    ntop = st.slider("Pathways shown", 10, 60, 25)
    show(F.pathway_dotplot(pw, ntop), "pathways")
    cls = st.multiselect("Classes", sorted(pw["class"].unique()), default=[])
    d = pw[pw["class"].isin(cls)] if cls else pw
    table(d.round(4), 380, key="pw_tbl")
    csv_button(d, "pathways.csv", key="pw_dl")

# ===================================================================================== Pathway maps
with tabs[6]:
    choices = pw.loc[pw["n_reactions_observed"] >= 2].sort_values("p_combined")["pathway"].tolist()
    if not choices:
        st.info("No pathway has two or more observable reactions.")
    else:
        p = st.selectbox("Pathway", choices, key="map_pw")
        maxr = st.slider("Maximum reactions on the map", 5, 60, 30, key="map_n")
        sel = rx[rx["subsystem"].str.split("; ").apply(lambda x: p in x)].sort_values("p_label").head(maxr)
        mf = res.features["met"]
        mz = mf.sort_values("mean_log2", ascending=False).drop_duplicates("kb_name").set_index("kb_name")["z"].to_dict()
        show(F.reaction_map(sel, mz, f"{p}: reaction map"), "pathway_map")
        st.caption("Squares = reactions (colour = coherence z, label = first enzyme); circles = metabolites "
                   "(colour = z; grey outline = not measured). Hover for details.")

# ===================================================================================== Network & targets
with tabs[7]:
    if extras.get("targets") is None:
        st.info("Enable 'Network, propagation and targets' in the sidebar and run again.")
    else:
        tg = extras["targets"]
        st.markdown("Targets are ranked by robust rank aggregation of reaction coherence, mRNA-protein agreement, "
                    "choke-point status and network propagation (plus druggability when a list is uploaded). "
                    "**They are hypotheses for experimental follow-up.**")
        show(F.targets_bar(tg), "targets")
        table(tg.round(4), 320, key="tg_tbl")
        csv_button(tg, "targets.csv", key="tg_dl")
        N_, E_ = extras["subnet"]
        show(F.network_plot(N_, E_, "Explanatory subnetwork of coherent reactions"), "subnetwork")
        from trace_core.network import to_graphml
        st.download_button("Download subnetwork (GraphML, opens in Cytoscape)", to_graphml(N_, E_),
                           "trace_subnetwork.graphml")
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
    f = extras.get("factors")
    if not f:
        st.info(extras.get("factors_error", "Enable 'Latent factors (MOFA+)' in the sidebar and run again."))
    else:
        st.caption(f"Method: {f['method']}. Factors explaining ≥ 2% variance in at least one layer are shown.")
        show(F.variance_heatmap(f["r2"]), "factor_variance")
        cols = list(f["scores"].columns)
        if len(cols) >= 2:
            a, b = st.columns(2)
            fx = a.selectbox("x", cols, 0)
            fy = b.selectbox("y", cols, 1)
            show(F.factor_scatter(f["scores"], res.design.meta["Group"], fx, fy), "factor_scores")
        table(f["summary"].round(3), 220, key="fac_sum")
        st.markdown("**Factor associations with group and covariates**")
        table(f["assoc"].round(4), 240, key="fac_assoc")
        if extras.get("factor_enrichment") is not None and len(extras["factor_enrichment"]):
            st.markdown("**Pathways enriched in factor loadings**")
            table(extras["factor_enrichment"].head(300).round(5), 300, key="fac_enr")

# ===================================================================================== Signature
with tabs[9]:
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
            with b:
                show(F.perm_hist(s), "signature_permutation")
            show(F.panel_bars(s["panel"]), "signature_panel")
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
    st.markdown("### Download results")
    xls = R.excel(res, extras)
    man = R.manifest(res, S["inputs"], kb, extras)
    meth = R.methods_text(res, kb, extras)
    c = st.columns(3)
    c[0].download_button("Excel workbook (all tables)", xls, "TRACE_results.xlsx", type="primary")
    c[1].download_button("Run manifest (JSON)", json.dumps(man, indent=2, default=str), "TRACE_manifest.json")
    c[2].download_button("Methods text (Markdown)", meth, "TRACE_methods.md")
    st.markdown("### Publication figures")
    fmt = st.selectbox("Format", ["png", "tiff", "pdf", "svg"], key="pubfmt")
    dpi = st.select_slider("Resolution (DPI)", [300, 600, 1200], value=600)
    if st.button("Build figure set"):
        figs = R.pub_figures(res, extras, fmt=fmt, dpi=dpi)
        st.session_state["figzip"] = R.bundle(figs)
    if "figzip" in st.session_state:
        st.download_button("Download figures (ZIP)", st.session_state["figzip"], f"TRACE_figures_{fmt}.zip")
    st.markdown("### Everything in one archive")
    if st.button("Build full bundle"):
        files = {"TRACE_results.xlsx": xls, "TRACE_manifest.json": man, "TRACE_methods.md": meth}
        files.update({f"figures/{k}": v for k, v in R.pub_figures(res, extras, "png", 600).items()})
        if extras.get("subnet"):
            from trace_core.network import to_graphml
            files["TRACE_subnetwork.graphml"] = to_graphml(*extras["subnet"])
        st.session_state["fullzip"] = R.bundle(files)
    if "fullzip" in st.session_state:
        st.download_button("Download full bundle (ZIP)", st.session_state["fullzip"], "TRACE_bundle.zip")

# ===================================================================================== Methods
with tabs[11]:
    st.markdown(R.methods_text(res, kb, extras))
    with st.expander("Run log and timings"):
        for l in res.log:
            st.write("• " + l)
        st.write({k: f"{v:.1f} s" for k, v in res.timings.items()})
    with st.expander("Knowledge base"):
        st.json(kb.manifest)
