# Omnix TRACE v1.0 — Tri-Omics Integration Engine

**TRACE = Tri-omics Reaction-Anchored Cascade Evidence.** This is a standalone Streamlit app. It integrates transcriptomics, proteomics and metabolomics/lipidomics along the directional chain **gene → protein → enzymatic reaction → metabolite**.

The reaction, not the molecule, is the unit of integration. For every reaction TRACE asks whether the change in its enzyme and the changes in its substrates and products point the same way. Each answer is calibrated by permutation and corrected for multiple testing (FDR).

> Results describe evidence *consistent with* a mechanism. They are not causal proof.

---

## 1. Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Pick **Demo: paired tri-omics** in the sidebar.
2. Press **Run TRACE**. The demo takes about 25 s.

The demo contains planted, known biology:

- glycolysis is up;
- ASS1 is lost, creating a bottleneck;
- fatty-acid oxidation is up at the protein level only;
- the glutaminase step is regulated without an enzyme change;
- cholesterol-synthesis mRNA is up but its protein is buffered.

TRACE recovers all five. The tests check this.

Run the tests:

```bash
python -m pytest -q tests
```

All 13 tests pass. They take about 30 s.

## 2. Input data

| File | Format |
| --- | --- |
| Transcriptomics | One row per gene. First column = gene symbol, Ensembl ID or Entrez ID. Remaining numeric columns = samples. |
| Proteomics | One row per protein. ID column = UniProt accession or gene symbol. An extra `Gene` column helps. |
| Metabolomics / Lipidomics | One row per metabolite. ID = name, or HMDB / KEGG / ChEBI. Optional `HMDB`, `KEGG` and `ChEBI` columns are used first. |
| Metadata | `Sample`, `Group`, optional `Batch` and other covariates (age, sex …). |
| Optional ID table | Feature → HMDB / KEGG / ChEBI, for metabolites that do not map by name. |

**Data you need:**

- **Layers:** metabolomics plus at least one of transcriptomics or proteomics.
- **Scale:** raw intensities are log2-transformed automatically. Already-logged data are detected.
- **Omnix exports:** the normalized matrices exported by the existing Omnix modules work as they are.

## 3. What the app computes

| Tab | Engine | Output |
| --- | --- | --- |
| Overview | — | Evidence tier, key findings, headline figures |
| Data & mapping | Harmonization | Sample overlap and tier, batch-group confounding (Cramér's V), ID-mapping coverage, unmapped features |
| Concordance | A | mRNA vs protein z-scores. Each gene is labelled coupled, opposing, protein-only (post-transcriptional) or mRNA-only (buffered). Uses a Deming slope and within-sample Spearman ρ. |
| Reactions | A | Pattern map (E vs M) and a reaction table with q-values (label, topology, balance) |
| Cascade | A | Sankey diagram of enzyme → reaction → metabolite for a pathway, the top reactions or one enzyme |
| Pathways | A | Per-layer pathway z, combined z, Cochran Q / I², and the class: shared-concordant, shared-discordant or omics-specific |
| Pathway maps | A | Interactive reaction–metabolite map of a pathway |
| Network & targets | C | Ranked targets (rank aggregation), explanatory subnetwork with GraphML export, reporter metabolites, TF regulators |
| Factors | B | MOFA+ variance explained, factor scores, covariate associations, pathway enrichment of the loadings |
| Signature | D | Nested-CV multiblock sPLS-DA with AUC (bootstrap CI), permutation p and a stability-selected panel; late fusion |
| Export | — | Excel (all tables), run manifest (JSON), methods text, figure ZIP (PNG/TIFF/PDF/SVG, 300–1200 DPI), full bundle |

Every interactive figure has a camera icon that downloads a high-resolution PNG or SVG.

### Core scores (Engine A)

```text
E_r = enzyme evidence: protein z through the GPR rule
      AND (complex) = smallest |z|, and only if the subunit signs agree
      OR (isozymes) = abundance-weighted mean
      if no protein is measured: mRNA z x observed mRNA-protein coupling ("mRNA proxy")
S_r, P_r = mean z of measured non-currency substrates and products
M_r = (P_r - S_r) / sqrt(2)          metabolite balance
C_r = E_r x M_r                      coherence
```

**Patterns.** Each reaction gets one plain-language pattern:

- Throughput-consistent (up/down)
- Candidate bottleneck
- Constrained step
- Substrate-limited
- Product drain
- Compensated
- Non-abundance regulation
- Metabolite-only change
- Enzyme-only change

### Statistics

- **Per-layer tests:** moderated t-statistics (limma-style empirical-Bayes variance shrinkage), converted to signed z-scores.
- **Label null:** sample labels are permuted jointly across layers, so cross-omics dependence of matched samples is kept. Defaults: 1,000 permutations; up to 5,000 for the label null and 2,000 for the topology null.
- **Topology null:** measured values are shuffled among nodes of similar network degree.
- **Reaction p-values:** taken from the product-normal distribution. Its location, scale and correlation are estimated from the permutations. Small p-values are therefore resolvable, and the p-values agree with the empirical permutation p.
- **FDR:** Benjamini–Hochberg. A reaction is called **coherent** only when both the label q and the topology q are below 0.05. A reaction is **metabolite-driven** when its metabolite balance is significant but its enzyme is unchanged.
- **Signature:** all feature selection happens inside the cross-validation folds. The early-integration signature is disabled below 10 samples per class.

### Validation built into the package

| Check | Result |
| --- | --- |
| Planted bottleneck (ASS1) | Top reaction, label q < 0.001, typed *Candidate bottleneck* |
| Glycolysis | *Shared-concordant*; HK/GPI/GCK steps *Throughput-consistent (up)* |
| Cholesterol synthesis (mRNA up, protein buffered) | *mRNA-specific* |
| Fatty-acid oxidation (protein only) | *Protein-specific*; genes typed *Protein-only* |
| Glutaminase step (enzyme unchanged) | *Metabolite-driven* |
| Global-null data, 12 simulated datasets | Mean fraction of reactions with p < 0.05 = 0.041 (parametric) vs 0.041 (empirical). No FDR-significant reactions or pathways. |
| Signature on null data | AUC < 0.8 (no leakage) |

## 4. Knowledge base

`kb/` is a frozen, versioned snapshot. The app makes no live database calls. `kb/kb_manifest.json` records the sources and versions.

- **Core:** Human-GEM 2.1.1 (CC BY 4.0). It provides:
  - 12,748 reactions with gene-protein-reaction (GPR) rules and stoichiometry;
  - 4,102 metabolites with KEGG, HMDB, ChEBI, PubChem and LIPID MAPS IDs;
  - 2,870 genes with Ensembl, UniProt and Entrez IDs;
  - 145 subsystems, used as pathways.
- **Add-ons:** build these on a computer with internet access. The cloud build environment could not reach these sites.

```bash
python -m trace_core.kb_build --skip-core --kegg --collectri --string   # adds to ./kb
python -m trace_core.kb_build --download --gem-dir kb_src --out kb      # full rebuild from Human-GEM
```

The add-ons provide:

- `--kegg`: KEGG pathway gene and compound sets, which appear as extra pathways prefixed `KEGG:`.
- `--collectri`: TF regulons, which turn on the TF-regulator table.
- `--string`: protein–protein edges (STRING v12, score ≥ 700), which are added to the network.

**Licensing.** KEGG and HMDB content is used now, as you decided. A commercial licence from KEGG (Pathway Solutions) and permission from HMDB are needed before Tri-Omics is sold. Each source is a separate file in `kb/`, so either can be removed without code changes.

## 5. Project layout

```text
app.py                      Streamlit UI (12 tabs)
trace_core/
  data.py                   loading, log/missing handling, within-layer batch correction, ID mapping, design/tiers
  stats.py                  moderated t, BH, permutation labels, product-normal calibrated p-values
  cascade.py                reaction index (GPR, stoichiometry), E/S/P/M/C scoring, pattern typing
  engine.py                 Engine A run: concordance, cascade + nulls, pathways, reporter metabolites, TF regulators
  network.py                Engine C: graph, random walk with restart + null, choke points, rank aggregation, subnetwork
  factors.py                Engine B: MOFA+ (mofapy2) with stability, associations, loading enrichment
  signature.py              Engine D: nested-CV multiblock sPLS-DA, stability selection, permutation, late fusion
  figures.py                interactive Plotly figures
  report.py                 Excel, manifest, methods text, 600-DPI matplotlib figures, ZIP bundles
  kb.py / kb_build.py       knowledge-base loader and builder
  demo.py                   synthetic tri-omics data with ground truth
kb/                         knowledge-base snapshot (parquet + manifest)
demo_data/                  demo matrices, metadata, ground truth
tests/test_trace.py         13 validation tests
docs/                       methods and Omnix integration notes
```

## 6. Adding TRACE to Omnix later

See `docs/OMNIX_INTEGRATION.md`. In short:

1. Copy `trace_core/`, `kb/` and `demo_data/` into `apps/triomics/`.
2. Wrap `app.py` as a portal page.
3. Add an **Export to Tri-Omics** button to the existing modules. This is the only change to them.

## 7. Known limits of v1

- **Two-group contrasts only.** For multi-group designs, run one contrast at a time.
- **Bulk data mixes compartments.** Compartments are collapsed to base metabolites.
- **Lipids map only partly.** They map at species or class level when the name or ID matches Human-GEM. Class-level lipid rules are planned for v1.1.
- **No live database access.** Drug–target annotation comes from an uploaded gene list. A ChEMBL/DGIdb add-on is planned.
- **GNN embeddings (Strategy C, deep-learning part) are deferred to v1.1.** Network propagation is implemented now.
