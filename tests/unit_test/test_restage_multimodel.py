"""CPU-only tests for the restage multi-model planner (no sglang imports).

What "verified" means here, in the campaign's provenance language:
- On qwen3-omni-30b the planner must REPRODUCE the measured verdicts of the
  2xH100 line (sgl-3arm 19589494 / sgl-solv5 19591990): the planner-chosen
  no-TP rebalance (15.33 audio-s/s) beats the shipped auto-partition (12.96,
  +18.3%), and TP2 (9.97) ranks last. That is the recorded, content-gated
  A/B -- these tests pin the planner to it (zero regret on the measured grid).
- On every other registry entry the planner must be HONEST, not right: every
  throughput carries PREDICTED/PRIOR provenance, memory arithmetic uses the
  config-derived geometry, and no launch is emitted for models sglang-omni
  cannot serve. New GPU measurements are the only way to upgrade these.

Run: python -m pytest tests/unit_test/test_restage_multimodel.py -q
"""
import pytest

from sglang_omni.restage import models as M
from sglang_omni.restage.plan import (
    SGL_DEFAULT_UNIT, SGL_NOTP_UNIT, SGL_TP2_UNIT,
    candidates_for_model, launcher_lines, pool_requests,
)
from sglang_omni.restage.workload import SGLANG_OMNI_QWEN3_LAW, Workload

LAW = SGLANG_OMNI_QWEN3_LAW


def ranked(name, gpus=4, wl=None, gpu_mem=80.0):
    e = M.get(name)
    wl = wl or e.default_workload
    rows = sorted(candidates_for_model(e, gpus, wl, LAW, gpu_mem), key=lambda r: -r[1])
    assert rows, "no candidates for %s" % name
    return rows


# --- measured-anchor regression (qwen3-omni-30b, 2xH100 line) ---------------

def test_planner_beats_shipped_default_on_measured_anchor():
    # the recorded A/B: shipped auto-partition 12.96 vs planner no-TP 15.33
    assert SGL_NOTP_UNIT[0] > SGL_DEFAULT_UNIT[0]
    gain = SGL_NOTP_UNIT[0] / SGL_DEFAULT_UNIT[0] - 1.0
    assert abs(gain - 0.183) < 0.005, "the +18.3%% measured gain drifted: %f" % gain
    assert "MEASURED" in SGL_NOTP_UNIT[1] and "19589494" in SGL_NOTP_UNIT[1]


def test_tp2_ranks_last_at_the_anchor():
    rows = ranked("qwen3-omni-30b", wl=Workload(6343, 4.5))
    assert rows[-1][0] == "tp2_thinker", [r[0] for r in rows]
    assert SGL_TP2_UNIT[0] < SGL_DEFAULT_UNIT[0]  # TP2 loses even to shipped


def test_anchor_registry_agrees_with_plan_constants():
    a = M.get("qwen3-omni-30b").anchors
    assert a["default_auto"][0] == SGL_DEFAULT_UNIT[0]
    assert a["handtuned_notp"][0] == SGL_NOTP_UNIT[0]
    assert a["tp2"][0] == SGL_TP2_UNIT[0]


# --- honesty invariants on every registry entry ------------------------------

@pytest.mark.parametrize("name", sorted(M.REGISTRY))
def test_every_row_carries_provenance(name):
    for _, agg, prov, layout in ranked(name):
        assert agg > 0
        assert ("MEASURED" in prov) or ("PREDICTED" in prov) or ("PRIOR" in prov), prov
        assert layout


@pytest.mark.parametrize("name", sorted(set(M.REGISTRY) - {"qwen3-omni-30b"}))
def test_unmeasured_models_never_claim_measured_throughput(name):
    for _, _, prov, _ in ranked(name):
        head = prov.split("(")[0]
        assert "PREDICTED" in head or "PRIOR" in head, prov


def test_registry_provenance_tags_present():
    for e in M.REGISTRY.values():
        assert any(t in e.provenance for t in ("MEASURED", "DERIVED", "PRIOR")), e.name


# --- memory arithmetic from real geometry ------------------------------------

def test_glm4voice_pool_matches_config_derivation():
    e = M.get("glm-4-voice-9b")
    # DERIVED: 19,085,115,456 B weights; kv 2*40*2*128*2 = 40,960 B/tok
    assert abs(e.backbone_gib - 19085115456 / 2**30) < 0.01
    assert e.kv_bytes_per_tok == 2 * 40 * 2 * 128 * 2
    q = pool_requests(e, Workload(2048, 4.5), 0.85, 80.0)
    manual = (0.85 * 80 - e.backbone_gib) * 2**30 / (40960 * 2048)
    assert abs(q - manual) < 1e-6


def test_big_backbone_forces_tp_wall():
    rows = ranked("qwen3-tts-1.7b", gpu_mem=3.0)  # 3.4 GiB backbone > 3 GiB GPU
    assert rows[0][0] == "tp2_backbone_mandatory"
    assert len(rows) == 1  # the wall is arithmetic: no other shape offered


def test_midsize_models_get_no_colocation():
    # 9B/8B backbones plausibly saturate a GPU alone; the only measured
    # co-location gain is a 1.7B model, so coloc must not be offered here
    for name in ("glm-4-voice-9b", "step-audio-2-mini"):
        assert not any(r[0].startswith("coloc") for r in ranked(name)), name


def test_small_models_offer_colocation_flagged_upper_bound():
    for name in ("cosyvoice2-0.5b", "csm-1b", "chatterbox"):
        rows = ranked(name)
        coloc = [r for r in rows if r[0].startswith("coloc")]
        assert coloc, name
        assert all("UPPER BOUND" in r[2] for r in coloc), name


def test_unknown_kv_geometry_disables_pool_claims():
    e = M.get("chatterbox")
    assert pool_requests(e, e.default_workload, 0.85, 80.0) is None
    for _, _, prov, _ in ranked("chatterbox"):
        assert "pool holds" not in prov


# --- emission gating ----------------------------------------------------------

def test_launcher_only_for_qwen3_omni_shapes():
    assert launcher_lines("dp3_consolidated_mps", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    assert launcher_lines("nonexistent_shape", "x") == []


def test_mps_launcher_uses_node_local_pipe_dir():
    lines = "\n".join(launcher_lines("dp3_consolidated_mps", "m"))
    assert "CUDA_MPS_PIPE_DIRECTORY=/tmp" in lines  # never Lustre/FSx


def test_two_gpu_measured_line_ranks_and_wins():
    # gpus=2 is the hardware the anchors were measured on; must not crash and
    # must reproduce the A/B verdict order: no-TP 15.33 > shipped 12.96 > TP2
    rows = ranked("qwen3-omni-30b", gpus=2, wl=Workload(6343, 4.5))
    assert [r[0] for r in rows] == ["rebalanced_notp", "shipped_auto", "tp2_thinker"]
    assert rows[0][1] == SGL_NOTP_UNIT[0] and "MEASURED" in rows[0][2]
