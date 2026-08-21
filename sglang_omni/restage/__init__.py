"""Restage: measurement-driven residency planning for multi-stage omni serving.

Plans WHERE each stage lives (GPUs, memory fractions, TP degree, replicas,
GPU-sharing mode) from a handful of measured constants instead of grid search.

Quick start:
    python -m sglang_omni.restage.plan --gpus 4 \
        --context-tokens 6343 --audio-seconds 4.5
    python -m sglang_omni.restage.plan --gpus 4 --context-tokens 2048 \
        --calibrate   # prints the two probe commands for an unmeasured workload

Core ideas (kappa validated on 10 cells, all on vLLM-Omni -- Qwen3-Omni-30B on
4xH100 and Qwen3-TTS-1.7B on 1xH100; the sglang-omni 2xH100 line has three
measured arms, see the Restage paper draft): the SLO-limited concurrency kappa
is the water level of
    c/T_sat + (c + 2.33*sqrt(c)) * delta / D = 1
with T_sat from ONE saturation probe and delta (per-arrival prefill stall)
from ONE single-request probe; co-location on a shared GPU multiplies utility
by a sharing discount d(mode) -- measured on vLLM-Omni: time-slice d~=0.51
(both verified models), MPS d = 0.64-0.81 (load-dependent); on sglang-omni
(sgl-dm 20057386, matched c8) time-slice 0.58, MPS 0.85 -- both inside the
pre-registered bands (throughput ratios; consolidated rtf99 > 1 at c8).
"""
