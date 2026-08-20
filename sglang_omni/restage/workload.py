"""Workload spec + constant prediction laws.

A workload is (context tokens L, audio seconds D, SLO). For a measured
workload the planner uses stored constants; for a new one it PREDICTS them
from two laws fitted on the vllm-omni H100 campaign (and flags them as
predictions until the two calibration probes confirm):

  service time   s(L) = a + b*L + c*L^2      (attention-quadratic prefill)
  arrival stall  delta(L) = max(delta_floor, L / prefill_rate)

Both laws are engine-family-shaped; their coefficients are stack-specific.
For sglang-omni run the calibration probes (see plan.py --calibrate) to fit
a/b/c and prefill_rate before trusting cross-workload extrapolation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    context_tokens: int          # L: prompt tokens per request (unique prefix!)
    audio_seconds: float         # D: mean audio seconds per response
    slo_rtf: float = 1.0         # rtf ceiling
    slo_pct: int = 99            # percentile the ceiling applies to
    name: str = ""


@dataclass(frozen=True)
class ServiceLaw:
    a: float                     # fixed seconds/request
    b: float                     # linear prefill s/token
    c: float                     # attention-quadratic s/token^2
    delta_floor_s: float         # per-arrival stall floor
    prefill_rate: float          # tok/s for the stall chunk
    source: str = ""

    def s(self, L):
        return self.a + self.b * L + self.c * L * L

    def T_sat(self, D, L):
        return D / self.s(L)

    def delta(self, L):
        return max(self.delta_floor_s, L / self.prefill_rate)


# vllm-omni / Qwen3-Omni-30B / 1xH100-per-thinker, 3-point fit (L=512/2048/6331)
VLLM_OMNI_QWEN3_LAW = ServiceLaw(
    a=0.057, b=1.55e-6, c=2.7e-9,
    delta_floor_s=0.027, prefill_rate=36_000,
    source="laxis/laxis2/laxis512 (jobs 19844919/19861831/19879559)",
)

# sglang-omni: NOT yet fitted -- run the calibration probes. Placeholder reuses
# the vllm-omni coefficients UNCHANGED (no sglang-specific scaling). Measured
# sglang-omni facts so far (2xH100, L=6343): no-TP hand-tuned 15.33 > default
# 12.96 > TP2 9.97 at QoS -- TP2 LOSES there (compute-bound). Every number
# derived from this law is marked PREDICTED by the planner.
SGLANG_OMNI_QWEN3_LAW = ServiceLaw(
    a=0.057, b=1.55e-6, c=2.7e-9,
    delta_floor_s=0.027, prefill_rate=36_000,
    source="UNCALIBRATED: vllm-omni shape reused; run plan.py --calibrate",
)
