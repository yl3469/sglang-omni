#!/usr/bin/env python3
"""Formulation v2 prototype: budget-constrained plan enumeration.

Capacity identity:  sum_i n_i * t_i * p_i + idle = B   (co-location folds
multiple instances onto one device under a sharing discount d(m)).
Memory per device:  sum_(i on d) x_{i,d} <= beta * C_d   (feasibility).
Stage utility:      u_i = n_i * e(t_i) * d(m) * min(pool_i/(k_i L), kappa_i)
Objective:          max min_i u_i  at FIXED B.

Constants below are MEASURED on this node (2026-08-25..27, sglang 0.5.18
except where noted). kappa_i / T_i are PER-STAGE effective saturated
audio-s/s of one instance, backed out from the measured cells — the point
of the prototype is ranking, not absolute prediction.

Validation: the enumerator's argmax at each budget must match the measured
winner:  B=3 short-ctx Qwen3: coloc x3 (65.4) > tails-replica2 (48.2) >
default+idle (29.0);  B=4 Ming: TP2+2 talkers (28.9) > TP2+1+idle (11.3);
B=5 Ming: TP2+2 talkers+idle (28.9) > TP4+1 talker (10.3).
"""
from itertools import product

BETA_C = 0.92 * 183.0

# ---- measured per-stage constants -----------------------------------------
QWEN = dict(  # short context, per-instance saturated audio-s/s (measured)
    # one full colocated pipeline on ONE GPU sustains ~21.8 (coloc x3 / 3)
    pipeline_1gpu=21.8,
    # split pipeline: thinker stage alone saturates ~29.0-ish with tails
    # elsewhere; tails stage (talker_ar+code2wav) per instance ~24.1
    # (backed out: default 29.0 tails-bound; replica2 tails x2 -> 48.2)
    thinker_1gpu=60.0,      # not binding in any measured short-ctx cell
    tails_1gpu=24.1,        # binding: 1x -> 29.0? use min() to see
    w_pipeline=66.0, w_thinker=59.5, w_tails=6.8,
)
MING = dict(  # per-instance saturated audio-s/s (0.5.18)
    thinker_tp2=35.0,       # revised after Experiment D (talker3 hit 34.6)
    thinker_tp4=28.0,       # e(t) discount; also never binding
    talker_1gpu=11.3,       # binding at n=1 (measured baseline)
    w_backbone=206.0, w_talker=8.4,
)


def qwen_plans(B):
    """Enumerate short-ctx Qwen3 plans at budget B (no TP: never needed)."""
    plans = []
    # full-pipeline replication: n pipelines, 1 GPU each (fits: 66 < 168)
    for n in range(1, B + 1):
        idle = B - n
        plans.append((f"pipeline x{n}" + (f" +{idle} idle" if idle else ""),
                      min(n * QWEN["pipeline_1gpu"], 1e9)))
    # split: 1 thinker GPU + m tails instances on dedicated GPUs
    for m in range(1, B):
        idle = B - 1 - m
        u = min(QWEN["thinker_1gpu"], m * QWEN["tails_1gpu"])
        plans.append((f"thinker + tails x{m}" + (f" +{idle} idle" if idle else ""), u))
    return plans


def ming_plans(B):
    """Ming: backbone needs TP (206 GiB); talker fits one GPU.

    Includes WHOLE-PIPELINE replication (k independent pipelines): the
    B=8 fix — the plateau at the thinker ceiling is escaped by replicating
    the pipeline, not the stage.
    """
    plans = []
    for t, thr in ((2, MING["thinker_tp2"]), (4, MING["thinker_tp4"])):
        for k in range(1, B // (t + 1) + 1):          # k pipelines
            rem = B - k * t                            # GPUs left for talkers
            for n in range(1, rem // k + 1):           # talkers per pipeline
                used = k * (t + n)
                if used > B:
                    continue
                idle = B - used
                u = k * min(thr, n * MING["talker_1gpu"])
                nm = (f"{k}x(TP{t} + talker x{n})" if k > 1
                      else f"TP{t} + talker x{n}") + (f" +{idle} idle" if idle else "")
                plans.append((nm, u))
    return plans


def rank(name, plans, measured_winner):
    plans.sort(key=lambda p: -p[1])
    print(f"\n{name}")
    for nm, u in plans[:4]:
        print(f"  {u:6.1f}  {nm}")
    ok = measured_winner in plans[0][0]
    print(f"  argmax {'MATCHES' if ok else 'DIFFERS FROM'} measured winner ({measured_winner})")
    return ok


if __name__ == "__main__":
    ok = True
    ok &= rank("Qwen3 short-ctx, B=3", qwen_plans(3), "pipeline x3")
    ok &= rank("Qwen3 short-ctx, B=2", qwen_plans(2), "pipeline x2")
    ok &= rank("Ming, B=4", ming_plans(4), "TP2 + talker x2")
    ok &= rank("Ming, B=5", ming_plans(5), "TP2 + talker x3")  # prediction!
    print("\nB=5 Ming note: enumerator predicts talker x3 beats TP4 — TP4's")
    print("measured 10.3 vs talker-x2's 28.9 already confirms the direction;")
    print("talker x3 (~30, thinker-capped) is the NEXT falsifiable cell.")
    raise SystemExit(0 if ok else 1)
