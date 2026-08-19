"""Restage: measurement-driven residency planning for multi-stage omni serving.

Plans WHERE each stage lives (GPUs, memory fractions, TP degree, replicas,
GPU-sharing mode) from a handful of measured constants instead of grid search.

Quick start:
    python -m sglang_omni.restage.plan --gpus 4 \
        --context-tokens 6331 --audio-seconds 4.5
    python -m sglang_omni.restage.plan --gpus 4 --context-tokens 2048 \
        --calibrate   # prints the two probe commands for an unmeasured workload

Core ideas (validated on 8 configurations across two engines, see the Restage
paper draft): the SLO-limited concurrency kappa is the water level of
    c/T_sat + (c + 2.33*sqrt(c)) * delta / D = 1
with T_sat from ONE saturation probe and delta (per-arrival prefill stall)
from ONE single-request probe; co-location on a shared GPU multiplies utility
by a sharing discount d(mode) -- time-slice d~=0.5 (invariant across models),
MPS d = 0.64-0.81 (workload-dependent).
"""
