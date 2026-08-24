"""Restage planner CLI for sglang-omni.

    python -m sglang_omni.restage.plan --gpus 4 --context-tokens 6343 \
        --audio-seconds 4.5 [--model-path Qwen/Qwen3-Omni-30B-A3B-Instruct]
        [--launch-dir OUT] [--calibrate]

Enumerates candidate residency plans for the GPU count, scores them with the
kappa/d(m) arithmetic, prints a ranked table with per-number provenance
(MEASURED vs PREDICTED), and writes ready-to-run launcher scripts for the
winner. --calibrate prints the two probe commands that turn PREDICTED into
MEASURED for a new workload.
"""
import argparse
import math
import os

from .workload import Workload, SGLANG_OMNI_QWEN3_LAW
from . import models as MODELS

Z99 = 2.326

# sharing discounts MEASURED on sglang-omni (sgl-dm 20057386, h020, 4xH100,
# L=6343, same-node 3 arms, matched per-pipeline c8): ded 17.03 audio-s/s
# (rtf99 0.88) | time-slice x3 agg 29.4 -> 0.58 | MPS x3 agg 43.5 -> 0.85.
# Both inside the pre-registered bands [0.40,0.62] / [0.65,0.85]; vLLM-Omni
# measured 0.51 / 0.76-0.93. Caveat: consolidated pipelines exceed rtf99 1
# at c8 (ts 1.46-1.65, mps 0.94-1.32) -- throughput-ratio d; the QoS-gated
# consolidation win on sglang-omni is not established.
D_TS = (0.58, "MEASURED sglang-omni sgl-dm 20057386 (h020, matched c8 ratio; consolidated rtf99>1 at c8)")
D_MPS = (0.85, "MEASURED sglang-omni sgl-dm 20057386 (h020, matched c8 ratio; consolidated rtf99 0.94-1.32 at c8)")

# measured sglang-omni anchors, per 2-GPU unit (2xH100, L=6343 unique-prefix
# long context; sgl-3arm 19589494 arms A/B, sgl-solv5 19591990 arm C):
SGL_DEFAULT_UNIT = (12.96, "MEASURED sglang-omni 2xH100 sgl-3arm 19589494 (shipped auto-partition, c4)")
SGL_NOTP_UNIT = (15.33, "MEASURED sglang-omni 2xH100 sgl-3arm 19589494 (hand-tuned 0.82/0.40, no TP; c8 rtf99 0.985 -- single run, marginal)")
SGL_TP2_UNIT = (9.97, "MEASURED sglang-omni 2xH100 sgl-solv5 19591990 (TP2 thinker @ fraction 0.62, c4; loses at QoS -- compute-bound)")


def kappa_lower(T, delta, D):
    c = 1.0
    while c < 256:
        if c / T + (c + Z99 * math.sqrt(c)) * delta / D >= 1.0:
            return c
        c += 0.1
    return c


def candidates(gpus, wl, law):
    """Yield (name, agg_throughput, provenance, layout) for common shapes."""
    T1 = law.T_sat(wl.audio_seconds, wl.context_tokens)   # per dedicated engine
    delta = law.delta(wl.context_tokens)
    k1 = kappa_lower(T1, delta, wl.audio_seconds)
    rho = T1 / max(k1, 1.0)
    pred = "PREDICTED(law: %s)" % law.source
    unit = min(k1, 1e9) * rho            # per 2-GPU unit (thinker + tails), from the law
    # If the law is uncalibrated but the workload sits at the measured sglang
    # anchor (L~6343, D~4.5), prefer scaling the MEASURED no-TP unit over a
    # borrowed vllm shape -- the provenance string says which one was used.
    if law.source.startswith("UNCALIBRATED") and \
            abs(wl.context_tokens - 6343) / 6343.0 < 0.1 and abs(wl.audio_seconds - 4.5) / 4.5 < 0.1:
        unit = SGL_NOTP_UNIT[0]
        pred = "PREDICTED scaling of 2-GPU unit(%s)" % SGL_NOTP_UNIT[1]

    if gpus == 2:
        # the directly measured 2xH100 arms; numbers are the measured units at
        # the anchor workload (L~6343, D~4.5) -- off-anchor, recalibrate first
        yield ("rebalanced_notp", SGL_NOTP_UNIT[0], SGL_NOTP_UNIT[1],
               [("thinker", [0]), ("tails", [1])])
        yield ("shipped_auto", SGL_DEFAULT_UNIT[0], SGL_DEFAULT_UNIT[1],
               [("thinker", [0]), ("tails", [1])])
        yield ("tp2_thinker", SGL_TP2_UNIT[0], SGL_TP2_UNIT[1],
               [("thinker TP2", [0, 1]), ("tails", [1])])
        return

    if gpus >= 4:
        # DP x2 dedicated pairs (thinker+tails per pair)
        yield ("dp2_dedicated", 2 * unit, pred,
               [("thinker", [0]), ("tails", [1]), ("thinker", [2]), ("tails", [3])])
        # DP x3 tails consolidated on the last GPU
        for mode, (d, prov) in (("timeslice", D_TS), ("mps", D_MPS)):
            yield ("dp3_consolidated_" + mode, 3 * unit * d,
                   "%s x d[%s]=%.2f (%s)" % (pred, mode, d, prov),
                   [("thinker", [0]), ("thinker", [1]), ("thinker", [2]),
                    ("tails x3 shared", [3])])
        # TP2 thinker + tails: measured LOSER on the 2xH100 line (compute-bound);
        # scaled by 2-GPU units for the 4-GPU shape (PREDICTED scaling).
        # one TP2 replica needs 3 GPUs (thinker x2 + tails); on 4 GPUs the 4th idles.
        yield ("tp2_thinker", SGL_TP2_UNIT[0],
               "PREDICTED 1x unit(%s); 4th GPU idle -- anchor at L=6343 only; recalibrate for other L" % SGL_TP2_UNIT[1],
               [("thinker TP2", [0, 1]), ("tails", [2]), ("idle", [3])])


def pool_requests(entry, wl, frac, gpu_mem_gib):
    """How many L-token requests the KV pool holds at a memory fraction.

    Returns None when the KV geometry is unpublished (PRIOR-only entry):
    the pool cap then never binds in ranking and every number stays PRIOR.
    """
    if entry.kv_bytes_per_tok is None or entry.backbone_gib is None:
        return None
    pool_gib = frac * gpu_mem_gib - entry.backbone_gib
    if pool_gib <= 0:
        return 0.0
    return pool_gib * (1 << 30) / (entry.kv_bytes_per_tok * wl.context_tokens)


def candidates_for_model(entry, gpus, wl, law, gpu_mem_gib=80.0):
    """Yield (name, agg_throughput, provenance, layout) for a registry entry.

    qwen3-omni-30b keeps the measured-anchor path (candidates()); every other
    entry ranks from the uncalibrated law + config-derived memory arithmetic,
    so every throughput is PREDICTED/PRIOR until the two probes are run.
    The d(m) values below are the qwen3-omni tails measurement reused as a
    PRIOR band for other models -- the ranking prints that explicitly.
    """
    if entry.name == "qwen3-omni-30b":
        for row in candidates(gpus, wl, law):
            yield row
        return

    T1 = law.T_sat(wl.audio_seconds, wl.context_tokens)
    delta = law.delta(wl.context_tokens)
    k1 = kappa_lower(T1, delta, wl.audio_seconds)
    pred = "PREDICTED(law: %s; constants: %s)" % (law.source, entry.provenance.split(";")[0])
    d_prior = (("timeslice", D_TS[0], "PRIOR: qwen3-omni tails d(ts)=0.58 reused (cross-model band 0.41-0.58)"),
               ("mps", D_MPS[0], "PRIOR: qwen3-omni tails d(mps)=0.85 reused (cross-model band 0.64-0.93)"))

    if entry.backbone_gib is not None and entry.backbone_gib > gpu_mem_gib:
        # arithmetic wall: one backbone does not fit one GPU -> TP mandatory
        yield ("tp2_backbone_mandatory", T1, pred + " -- TP mandatory (weights %.1f GiB > %.0f GiB GPU)"
               % (entry.backbone_gib, gpu_mem_gib), [("backbone TP2", [0, 1]), ("tails", [2])])
        return

    # per-engine footprint: weights + runtime margin + kappa L-token KV pool
    kv_need_gib = (k1 * wl.context_tokens * entry.kv_bytes_per_tok / (1 << 30)
                   if entry.kv_bytes_per_tok is not None else None)
    total_w = ((entry.backbone_gib or 0.0) + (entry.tails_gib or 0.0)) or None
    footprint = (total_w + 2.0 + (kv_need_gib or 0.0)) if total_w is not None else None

    # dedicated: one engine (backbone+tails) per GPU (pair only if it can't fit)
    per_engine_gpus = 2 if (footprint is not None and footprint > 0.9 * gpu_mem_gib) else 1
    n_ded = gpus // per_engine_gpus
    q = pool_requests(entry, wl, 0.85, gpu_mem_gib)
    eff = min(k1, q) if q is not None else k1
    unit = eff * (T1 / max(k1, 1.0))
    cap_note = "" if q is None else " pool holds %.0f reqs @ L=%d (kappa %d %s)" % (
        q, wl.context_tokens, k1, "binds" if k1 <= q else "> pool -> POOL BINDS")
    yield ("dedicated_x%d" % n_ded, n_ded * unit, pred + cap_note,
           [("engine", [i * per_engine_gpus]) for i in range(n_ded)])

    # "small" = plausibly cannot saturate the GPU alone; the only MEASURED
    # co-location gain is the vLLM-Omni TTS-1.7B line, so gate tightly.
    small = footprint is not None and footprint <= 0.15 * gpu_mem_gib
    if small:
        # co-locate n engines on ONE GPU under a sharing mode; n from the full
        # per-engine footprint (weights + margin + kappa-sized KV pool).
        # UPPER BOUND: co-location only gains when a single engine cannot
        # saturate the GPU (MEASURED only on vLLM-Omni TTS x2/x3: 26.1 -> 41.7);
        # for a compute-saturated engine the true d approaches 1/n.
        n_fit = max(1, int((gpu_mem_gib * 0.9) // footprint))
        for n in sorted({2, 3, min(n_fit, 4)}):
            if n > n_fit or n < 2:
                continue
            for mode, d, dprov in d_prior:
                yield ("coloc%d_%s_per_gpu" % (n, mode), gpus * n * unit * d,
                       "%s x %d/GPU x d[%s]=%.2f UPPER BOUND (%s; true d -> 1/n if one engine already saturates)"
                       % (pred, n, mode, d, dprov),
                       [("%d engines shared" % n, [g]) for g in range(gpus)])
    elif total_w is not None:
        # backbone per GPU, tails consolidated on the last GPU
        n_bb = gpus - 1
        for mode, d, dprov in d_prior:
            yield ("dp%d_consolidated_%s" % (n_bb, mode), n_bb * unit * d,
                   "%s x d[%s]=%.2f (%s)" % (pred, mode, d, dprov),
                   [("backbone", [i]) for i in range(n_bb)] + [("tails x%d shared" % n_bb, [gpus - 1])])


def launcher_lines(name, model_path, ports=(8040, 8041, 8042)):
    base = ("python examples/run_qwen3_omni_speech_server.py "
            "--model-path %s" % model_path)
    env = ("export SGLOMNI_THINKER_ARGS='{\"max_running_requests\": 64, "
           "\"cuda_graph_max_bs\": 64}'")
    if name.startswith("dp3_consolidated"):
        lines = [env]
        if name.endswith("mps"):
            lines += ["export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_$$  # node-local!",
                      "export CUDA_MPS_LOG_DIRECTORY=/tmp/mpslog_$$",
                      "mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY",
                      "nvidia-cuda-mps-control -d",
                      "# verify client connections in $CUDA_MPS_LOG_DIRECTORY/control.log"]
        for i, p in enumerate(ports):
            lines.append("%s --port %d --gpu-thinker %d --gpu-talker 3 "
                         "--gpu-code-predictor 3 --gpu-code2wav 3 "
                         "--thinker-mem-fraction-static 0.80 "
                         "--talker-mem-fraction-static 0.10 &" % (base, p, i))
        return lines
    if name == "dp2_dedicated":
        return [env,
                base + " --port 8040 --gpu-thinker 0 --gpu-talker 1 "
                       "--gpu-code-predictor 1 --gpu-code2wav 1 "
                       "--thinker-mem-fraction-static 0.82 --talker-mem-fraction-static 0.40 &",
                base + " --port 8041 --gpu-thinker 2 --gpu-talker 3 "
                       "--gpu-code-predictor 3 --gpu-code2wav 3 "
                       "--thinker-mem-fraction-static 0.82 --talker-mem-fraction-static 0.40 &"]
    if name == "tp2_thinker":
        return [env,
                base + " --port 8040 --thinker-tp-size 2 --gpu-thinker-tp 0,1 "
                       "--gpu-talker 2 --gpu-code-predictor 2 --gpu-code2wav 2 "
                       "--thinker-mem-fraction-static 0.62 --talker-mem-fraction-static 0.15 &"]
    return []


CALIBRATION = """# Calibration probes for this workload (turn PREDICTED into MEASURED):
# 1. single-request probe -> delta, weights, overheads (read server log + ttft)
#    run one server (any layout), send 3 requests at concurrency 1
# 2. saturation probe -> T_sat: closed-loop c-sweep at 3 points around the
#    predicted kappa (%(k)d): c=%(c1)d,%(c2)d,%(c3)d, 60+ samples each,
#    UNIQUE system prefixes (RadixAttention dedupes shared prefixes and hides
#    the capacity regime). Then refit ServiceLaw in workload.py.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-omni-30b",
                    help="registry key (see --list-models)")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-mem-gib", type=float, default=80.0)
    ap.add_argument("--context-tokens", type=int, default=None)
    ap.add_argument("--audio-seconds", type=float, default=None)
    ap.add_argument("--slo-rtf", type=float, default=1.0)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--launch-dir", default=None)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for name, e in MODELS.REGISTRY.items():
            tag = "servable" if e.servable else "planner-only"
            anch = "MEASURED anchors" if e.anchors else "no measurement"
            print("%-18s %-12s %-17s %s" % (name, tag, anch, e.hf_id))
        return

    entry = MODELS.get(args.model)
    wl = Workload(
        args.context_tokens if args.context_tokens is not None else entry.default_workload.context_tokens,
        args.audio_seconds if args.audio_seconds is not None else entry.default_workload.audio_seconds,
        args.slo_rtf, name=entry.default_workload.name)
    model_path = args.model_path or entry.hf_id
    law = SGLANG_OMNI_QWEN3_LAW
    rows = sorted(candidates_for_model(entry, args.gpus, wl, law, args.gpu_mem_gib),
                  key=lambda r: -r[1])

    print("Stack: sglang-omni | model: %s (%s, %s)"
          % (entry.name, entry.hf_id, "servable" if entry.servable
             else "PLANNER-ONLY: no sglang-omni implementation yet"))
    print("Constants: %s" % entry.provenance)
    print("Workload: L=%d tok, D=%.1f s, rtf_p99<=%.1f | law: %s"
          % (wl.context_tokens, wl.audio_seconds, wl.slo_rtf, law.source))
    print("%-26s %10s  %s" % ("plan", "audio-s/s", "provenance"))
    for name, agg, prov, _ in rows:
        print("%-26s %10.1f  %s" % (name, agg, prov))
    best = rows[0]
    print("\nCHOSEN: %s (%.1f audio-s/s, sglang-omni, %s)" % (best[0], best[1], "PREDICTED" if ("PREDICTED" in best[2] or "PRIOR" in best[2]) else "MEASURED"))
    print("  provenance: %s" % best[2])

    if not entry.servable:
        print("\nNo launch emitted: %s has no sglang-omni implementation. The plan"
              "\nabove is the residency shape to target when porting it; after the"
              "\nport, run the two calibration probes before trusting any ranking."
              % entry.name)
        lines = []
    else:
        lines = launcher_lines(best[0], model_path)
        print("\nLaunch:")
        for ln in lines:
            print("  " + ln)
    if lines and args.launch_dir:
        os.makedirs(args.launch_dir, exist_ok=True)
        path = os.path.join(args.launch_dir, "launch_%s.sh" % best[0])
        with open(path, "w") as f:
            f.write("#!/bin/bash\nset -x\n" + "\n".join(lines) + "\nwait\n")
        print("\nwrote " + path)

    if args.calibrate or "PREDICTED" in best[2]:
        k = int(kappa_lower(law.T_sat(wl.audio_seconds, wl.context_tokens),
                            law.delta(wl.context_tokens), wl.audio_seconds))
        print("\n" + CALIBRATION % dict(k=k, c1=max(2, k - 4), c2=k, c3=k + 4))


if __name__ == "__main__":
    main()
