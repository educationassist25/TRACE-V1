"""TRACE validation tests: parsing, mapping, recovery of planted biology, null calibration, no leakage."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trace_core.data import build_design, load_demo, make_layer, norm_hmdb  # noqa: E402
from trace_core.demo import make_demo  # noqa: E402
from trace_core.engine import run_trace  # noqa: E402
from trace_core.kb import load_kb  # noqa: E402
from trace_core.kb_build import parse_gpr  # noqa: E402
from trace_core.stats import product_normal_p  # noqa: E402


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def _layers(kb, folder, drop=None):
    dfs, meta = load_demo(folder)
    for k, cols in (drop or {}).items():
        dfs[k] = dfs[k].drop(columns=cols)
    layers = {k: make_layer(k, dfs[k], meta, kb, batch_correct=(k == "met"), batch_col="Batch") for k in dfs}
    return layers, build_design(meta, layers, "Tumor", "Control")


@pytest.fixture(scope="module")
def demo_result(kb):
    layers, des = _layers(kb, os.path.join(ROOT, "demo_data"))
    return run_trace(layers, des, kb, {"B": 400, "B_topo": 400}), layers, des


def test_gpr_parser():
    assert parse_gpr("A and (B or C)") == [["A", "B"], ["A", "C"]]
    assert parse_gpr("A or B") == [["A"], ["B"]]
    assert parse_gpr("") == []


def test_identifier_normalisation():
    assert norm_hmdb("HMDB00122") == "HMDB0000122"
    assert norm_hmdb("hmdb0000122") == "HMDB0000122"


def test_product_normal_matches_simulation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400000)
    y = 0.3 * x + np.sqrt(1 - 0.09) * rng.normal(size=400000)
    emp = (np.abs(x * y) >= 3).mean()
    assert abs(product_normal_p(np.array([3.0]), np.array([0.3]))[0] - emp) < 0.002


def test_tier_and_mapping(demo_result):
    res, layers, des = demo_result
    assert des.tier == 1
    assert (layers["met"].feat["map_rule"] != "unmapped").mean() > 0.95


def test_recovers_bottleneck(demo_result):
    rx = demo_result[0].reactions
    r = rx[rx["name"] == "L-Citrulline:L-aspartate ligase (AMP-forming)"].iloc[0]
    assert r["significant"] and r["pattern"] == "Candidate bottleneck" and r["E"] < 0


def test_recovers_glycolysis_throughput(demo_result):
    rx = demo_result[0].reactions
    g = rx[rx["significant"] & rx["subsystem"].str.contains("Glycolysis")]
    assert (g["pattern"] == "Throughput-consistent (up)").any()


def test_pathway_classes(demo_result):
    pw = demo_result[0].pathways.set_index("pathway")
    assert pw.loc["Glycolysis / Gluconeogenesis", "class"] == "Shared-concordant"
    assert pw.loc["Cholesterol biosynthesis 1 (Bloch pathway)", "class"] == "mRNA-specific"
    fao = pw[pw.index.str.contains("Beta oxidation")]
    assert (fao["class"] == "Protein-specific").any()


def test_concordance_classes(demo_result):
    c = demo_result[0].concordance.set_index("gene")["class"]
    assert c.get("ASS1") == "Coupled"
    assert sum(c.get(g) == "Protein-only (post-transcriptional)" for g in ["CPT1A", "ACADM", "HADHA", "HADHB", "ACAA2"]) >= 3
    assert sum(c.get(g) == "mRNA-only (buffered)" for g in ["HMGCR", "HMGCS1", "SQLE", "FDPS", "LSS"]) >= 3


def test_metabolite_driven_step(demo_result):
    rx = demo_result[0].reactions
    r = rx[rx["name"] == "L-glutamine amidohydrolase"]
    assert r["metabolite_driven"].any()


def test_partial_overlap_runs(kb):
    layers, des = _layers(kb, os.path.join(ROOT, "demo_data"),
                          drop={"prot": ["S03", "S08", "S15", "S20"], "met": ["S05", "S11", "S17", "S23"]})
    assert des.tier == 2
    res = run_trace(layers, des, kb, {"B": 200, "B_topo": 200})
    assert res.reactions["significant"].sum() > 0


def test_null_calibration(kb, tmp_path):
    """Global null: no planted effects -> few nominal and essentially no FDR-significant reactions."""
    make_demo(kb, str(tmp_path), seed=99, effects=False)
    layers, des = _layers(kb, str(tmp_path))
    res = run_trace(layers, des, kb, {"B": 400, "B_topo": 400})
    rx = res.reactions
    assert (rx["p_label"] < 0.05).mean() < 0.10
    assert rx["significant"].sum() <= 3
    assert (res.pathways["q_combined"] < 0.05).sum() <= 2


def test_signature_no_leakage(kb, tmp_path):
    from trace_core.signature import run_signature
    make_demo(kb, str(tmp_path), seed=5, effects=False)
    layers, des = _layers(kb, str(tmp_path))
    s = run_signature(layers, des, n_repeats=3, n_perm=10)
    assert s["status"] == "ok" and s["auc_mean"] < 0.8


def test_exports(demo_result, kb):
    from trace_core import report as R
    res = demo_result[0]
    assert len(R.excel(res, {"kb_version": kb.version})) > 10000
    assert "Human-GEM" in R.methods_text(res, kb, {})
    m = R.manifest(res, {"x": b"1"}, kb, {})
    assert m["design"]["tier"] == 1
