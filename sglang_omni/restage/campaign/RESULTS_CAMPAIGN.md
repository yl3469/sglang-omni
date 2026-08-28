# Restage serving campaign — 8x Blackwell 183GiB node (2026-08-25)

Goal (user-set): lower profiling time; better serving results under VoxServe
metrics (TTFA, streaming viability, goodput) across QPS and models; document
with analysis. Two coordinated Claude sessions split the node: peer session =
GPUs 0-2 (Qwen3-Omni VoxServe A/B, Qwen3-TTS-1.7B sweeps), this session =
GPUs 4-7 (calibration, repeats, third model family) + this consolidated doc.

## 1. Setup

- Branch `lisa/sglang-copy`; uv venv py3.12; sglang 0.5.16, torch 2.11+cu130,
  flashinfer 0.6.14 cu13. 8x Blackwell sm10.0 (183 GiB), CUDA 13.2.
- Models: Qwen/Qwen3-Omni-30B-A3B-Instruct (66 GiB), Qwen/Qwen3-TTS-12Hz-1.7B-Base,
  bosonai/higgs-audio-v3-tts-4b. Dataset: seed-tts-eval EN (unique target texts).
- Controls (mandatory, learned the hard way): per-request-UNIQUE system
  prefixes (RadixAttention dedup hides KV capacity); content gate D in
  [0.5,20] s + temperature 0.2 (degenerate 0.06 s audio blobs otherwise own
  every p99); a full throwaway ladder after boot (graph capture per batch
  size); SGLANG_OMNI_STARTUP_TIMEOUT=1800.
- Harness: exp/run_bench_uniq.py (session-header+line-rotation prefixes),
  exp/calibrate2.py (probes+ladders+NNLS fit), exp/vox_metrics.py (TTFA /
  streaming viability / goodput), exp/repeats/analyze_repeats.py
  (bootstrap-CI kappa, shared estimators across both sessions).

## 2. Calibration: the Blackwell service law (calib2-20260825)

Two-probe protocol (Restage deck), upgraded per the model critique:

- SERIAL law from 160 individual c=1 requests (4 usable L; L=9000 infeasible:
  thinker ships context_length=8192):
  s(L,D) = 0.2078 + 0*L + 2.634e-9*L^2 + 0.1076*D  (NNLS, b,c,d>=0)
  delta_floor 0.089 s, prefill_rate 95,712 tok/s (~2.8x the H100 fit).
- VALIDATED: kappa water-level prediction within +-2 of the gated measured
  frontier at every L including the held-out L=4000
  (pred 4.9/4.8/4.7/4.3 vs measured 5/4/7/3).
- KNOWN BIAS: T_sat from the serial law underpredicts saturated throughput
  (stages overlap under load); capacity uses MEASURED anchors:
  QoS-feasible best (gated, n=200, single-run): L512 23.2@c5, L2048 17.7@c4,
  L4000 21.5@c7, L6343 11.9@c3 audio-s/s per default 2-GPU pipeline.
- Do NOT fit a throughput law through frontier points: single-run frontier
  noise made the holdout error 75%.
- Landed in sglang_omni/restage/workload.py as the Blackwell
  SGLANG_OMNI_QWEN3_LAW (peer session's edit; identical fit).

## 3. Tail stability: the QoS frontier has +-1c error bars (exp/repeats/)

Three identical gated n=200 ladders (L=6343, c={3,4,5,7}) on a fresh server:
kappa = none / 3 / 4; p99-CI-upper straddles 1.0 at BOTH c3 (0.85-1.39) and
c4 (0.84-2.12). rep1 = first-ladder contamination (throwaway must cover every
batch size). Between clean repeats: kappa in {3,4}, T@frontier 11.8-14.1.

Consequences:
- Every single-run kappa / throughput-at-frontier claim (including the
  original H100 anchors marked "single run, marginal", the A/B verdict, and
  calib2's per-L kappas) carries +-1c / +-20%.
- The cheap c=1 law predicts the frontier as accurately as one expensive
  ladder measures it.
- Protocol: discard a full first ladder; kappa = largest c with
  MEDIAN-of-3 bootstrap-CI-upper <= 1.0.

## 4. Profiling cost (the "lower profiling time" result)

| step | wall (this box) | buys |
|---|---|---|
| boot, warm JIT | ~4-6 min | fixed cost PER CONFIG |
| boot, cold JIT | ~10-25 min | once per machine/config shape |
| c=1 probe (40 samples) | ~1 min/L | s(L,D), delta, TTFT; kappa +-2 |
| gated ladder cell n=200 | ~1.2 min | one (c, thr, p99CI) point |
| 3-repeat bracketing ladder | ~15 min | reproducible kappa +-1 |

Sampling is cheap; boots dominate. FAST PROFILE for a new workload/model =
one boot + 4-5 c=1 probes (~10 min) -> kappa within the noise of a full
ladder. Full defensible calibration = one boot + probes + one 3-repeat
bracketing ladder (~45 min incl. warm boot). The planner's economic value on
this box is pruning per-config boots, not per-config sampling.

## 5. A/B: restage H100 plan vs shipped default (see AB_BLACKWELL_2GPU.md)

The measured 2xH100 verdict (rebalance 0.82/0.40 wins +18%) does NOT transfer:
at 183 GiB the auto-partition already gives the thinker a 1.07M-token KV pool
(H100: 31k) and the rebalance SHRINKS it. Arms tie within (large) tail noise;
memory rebalancing is moot at this VRAM. Anchors are hardware-specific —
which the planner's own provenance discipline predicted.

## 6. VoxServe-metric serving comparison (Qwen3-Omni, 2 GPUs)

Workload: L=6343 unique prefixes, streaming, Poisson arrivals, gated,
TTFA SLO 1.0 s. Two arms at equal 2-GPU budget:
default2gpu (thinker GPU-a, tails GPU-b — the shipped split) vs coloc2x
(whole pipeline per GPU x2 replicas via qwen3_omni_colocated_h200.yaml —
the residency shape 183 GiB enables; launch REQUIRES the colocated yaml:
per-stage total_gpu_memory_fraction, not --*-mem-fraction-static flags).

coloc2x, repeat 2 (this session, GPUs 4-5, exp/omnivox-rep2/, n=60/cell/replica,
0 errors; a+b = node totals at the offered rate):

| offered qps | completed | goodput qps | TTFA p50/p99 (s) | viability |
|---|---|---|---|---|
| 1 | 0.79 | 0.73 | 0.40 / 0.84-0.94 | 93% |
| 2 | 1.93 | 1.75 | 0.41 / 0.94-1.01 | 93-95% |
| 3 | 2.86 | 2.84 | 0.41 / 0.69-0.75 | 98-100% |
| 4 | 4.04 | 3.74 | 0.50 / 1.10-1.23 | 95-97% |

Reading: the coloc pair sustains ~3 goodput-qps (~10 audio-s/s) with TTFA p99
under the SLO; at 4 qps offered the TTFA tail crosses 1 s first — TTFA, not
viability, is the binding constraint at this workload (per-chunk viability
never drops below 93%). The low-rate viability dip (93% at 1-2 qps vs
98-100% at 3) is an early-cell ordering artifact worth a repeat.

default2gpu (same protocol, run against the peer's healthy 8061 server after
its driver deadlocked; n=120/cell):

| offered qps | completed | goodput qps | TTFA p50/p99 (s) | viability |
|---|---|---|---|---|
| 1 | 0.88 | 0.82 | 0.36 / 1.02 | 95% |
| 2 | 1.78 | 1.69 | 0.36 / 1.25 | 98-99% |
| 3 | 2.59 | 2.22 | 0.41 / 1.76 | 97-98% |
| 4 | 4.30 | 3.83 | 0.56 / 1.28 | 99% |

VERDICT (single-run, +-tail-noise caveats apply): at equal 2-GPU budget the
coloc2x plan WINS the TTFA-QoS frontier. At 3 qps offered: goodput 2.84 vs
2.22 and TTFA p99 0.69-0.75 s vs 1.76 s. Under a strict TTFA-p99<=1 s gate,
default2gpu is infeasible at every measured rate (p99 1.02 already at
1 qps — the single shared thinker serializes 6.3k-token prefills behind
decodes), while coloc2x holds to 3 qps. At 4 qps both violate the tail and
goodput ties (~3.8). Mechanism: two independent thinkers halve the
prefill-behind-decode queueing that dominates TTFA at long context; the
183 GiB VRAM makes the shape feasible at all. This is the residency-planning
win restage predicts qualitatively — but it required the RECALIBRATED
hardware picture (KV unconstrained -> spend memory on replicas, not
fractions), not the H100 plan.

coloc2x REPEAT 1 (GPUs 0-1, exp/omnivox-rep1/, n=60/cell/replica, 0 errors)
— the n=2 cross-GPU spread:

| offered qps | rep1 goodput / TTFA p99 | rep2 goodput / TTFA p99 |
|---|---|---|
| 1 | 0.90 / 0.68-0.93 | 0.73 / 0.84-0.94 |
| 2 | 2.12 / 0.62-0.83 | 1.75 / 0.94-1.01 |
| 3 | 2.99 / 0.68-0.92 | 2.84 / 0.69-0.75 |
| 4 | 3.42 / 0.78-0.80 | 3.74 / 1.10-1.23 |

n=2 verdict: the qps<=3 regime is REPRODUCIBLE (goodput 2.84 vs 2.99 at 3
offered, ~5% spread; TTFA p99 comfortably under SLO in both). The 4-qps cell
straddles the SLO across repeats (p99 0.78-0.80 vs 1.10-1.23) — the frontier
carries the same run-to-run noise the closed-loop repeats quantified, now in
the TTFA domain. Both repeats beat default2gpu (2.22 goodput at 3 qps, p99
>= 1.0 at every rate). Headline unchanged and now n=2: coloc2x ~ +28-35%
goodput at the TTFA frontier.

(tts17 coloc d(m) phases running.)

## 7. Qwen3-TTS-1.7B: single vs co-located engines (GPU 2)

Single engine (clean rerun; earlier cells were double-driver-tainted):
goodput tracks offered through 8 qps — 7.28 goodput-qps / 27.5 audio-s/s at
8 offered, TTFA p99 0.37 s, 100% viability. Knee NOT reached at 8 qps.

Time-slice coloc x2 (0.42 mem fraction each; auto-fraction would OOM-fight —
the 1.7B engine hoards 157/183 GiB by default): aggregate 8.07 goodput-qps /
30.8 audio-s/s at 8 offered (4/replica), TTFA p99 0.43-0.68 s. In the
sub-saturation regime, two time-sliced engines MATCH OR EXCEED one engine at
equal offered load — no measurable sharing penalty below the knee. True
d(ts) at saturation remains unmeasured (both configs keep up at 8 qps;
higher-rate cells needed).

MPS coloc x2: NOT MEASURABLE on this node — sglang-omni's spawned stage
processes fail under the MPS control daemon (platform detection falls back
to a NotImplementedError stub; after a set_device fallback patch, stages
still see "No CUDA GPUs are available"), while plain torch clients work.
Engine-level incompatibility on driver 595.71/Blackwell, recorded as a
landmine; d(MPS) on Blackwell stays PRIOR (H100-sglang measured 0.85).

Recurring artifact across runs: isolated ~25 s stalls poison single cells
(s_qps6, ts_qps2_b, ts_qps4_a) — one replica freezes mid-cell then recovers;
neighboring rates clean. Engine-side hiccup worth an upstream issue; affected
cells excluded from the statements above.

## 8. Third model family: Higgs Audio v3 TTS 4B (GPU 6, this session)

Fast-profile recipe on a brand-new family: **5.6 min end-to-end**
(boot 241 s + throwaway 27 s + c=1 probe 36 s + confirm cell 31 s), zero
extra deps. Sweep cells 28-120 s each. exp/higgs/.

Open-loop Poisson sweep (streaming, seed-tts EN, seed 42, gated — 0 errors,
0 gated-out at every rate, ONE GPU):

| offered qps | completed qps | TTFA p50/p99 (s) | viability | audio-s/s |
|---|---|---|---|---|
| 1 | 1.08 | 0.147 / 0.172 | 100% | 4.1 |
| 4 | 3.92 | 0.145 / 0.183 | 100% | 14.9 |
| 8 | 8.51 | 0.143 / 0.255 | 100% | 32.4 |
| 12 | 13.26 | 0.174 / 0.229 | 100% | 50.4 |
| 16 | **13.58 (knee)** | 0.183 / 0.249 | 100% | 52.0 |

Analysis: capacity ~13.6 req/s ~ 52 audio-s/s per GPU; past the knee,
overload appears ONLY as admission queueing (completed < offered) — TTFA p99
stays <=0.25 s and per-chunk viability stays 100%, i.e. this pipeline
degrades exactly the way VoxServe's goodput metric rewards. Contrast with
Qwen3-Omni at L=6343: ~12-14 audio-s/s on TWO GPUs with a fragile rtf tail —
though the workloads differ (Higgs requests here are short-context TTS, no
6.3k-token prefix), so this is a deployment-shape observation, not a
model-quality comparison.

## 8b. Larger model: Ming-flash-omni-2.0 (214 GiB MoE, GPUs 3-5)

The first model whose backbone CANNOT fit one 183 GiB GPU — the planner's
TP-mandatory arithmetic wall, tested pre-registered (exp/ming/PREREG.md):
the throwaway-entry planner run emitted exactly one shape
(backbone TP2 + tails on a 3rd GPU) and a 6.5 audio-s/s serial lower bound
BEFORE any measurement. All three pre-registered claims passed:

- TP2(2 GPUs) + talker(1 GPU) boots and serves: cold 952 s / warm 171 s;
  444/444 requests clean (0 errors, 0 gated).
- Measured saturated throughput 10.2-10.8 audio-s/s — inside the
  pre-registered (6.5, 26) band, 1.6-1.7x the serial bound (consistent with
  the known serial bias).
- Gated ladder: kappa ~ 2 (c2 p99 0.81 feasible; c4 1.21; c8 1.46);
  Poisson saturation ~2.7 completed qps at offered 4. Per-GPU cost at short
  context: ~3.6 audio-s/s/GPU — ~3x Qwen3-Omni-30B's, ~14x Higgs 4B's.
- No public streaming config -> TTFA/viability unmeasurable; rtf/qps only.

Three one-line compatibility fixes were required to serve it at all
(upstream missing talker/data/voice_name.json — manifest crafted;
transformers-5.12 get_seq_length() tensor vs torch.arange — int() coercion;
torchaudio-2.11 forcing torchcodec/FFmpeg — soundfile decode). The stack's
LAUNCHABILITY, not just its constants, moves under dependency drift —
strengthening the versioned-constants argument.

### 8c. Ming TP4 vs TP2: scaling the wrong stage buys zero (pre-registered)

TP4 thinker (4 GPUs) + talker vs TP2 + talker, same protocol
(exp/ming-tp4/, 468/468 clean): saturated 10.3-11.2 vs 10.2-10.8 audio-s/s
(~1.0x), c=1 SLOWER (4.04 vs 5.11: allreduce latency), per-GPU -38%
(2.23 vs 3.61), kappa unchanged (~2). The e~0.88 TP prior falsified
DOWNWARD: the thinker was never the binding stage — the single-GPU talker
tail is, so doubling thinker GPUs adds nothing (the max-min objective's
"non-bottleneck capacity has zero marginal value", now measured). At >=5-GPU
budgets the right spend is a second tails/pipeline replica, not more TP.
Bottleneck migration is real and invisible to a single-aggregate law:
per-stage probes are the next instrument.

### 8d. Per-stage replication (upstream PR #1175), pre-registered A/B

Rebased to upstream main (branch lisa/sglang-1175; sglang 0.5.18/torch 2.13
— the stack itself is faster: default 2-GPU L6343 saturated ~21 vs ~14.6
audio-s/s, constants re-measured). Qwen3-Omni tails-replica2 (3 GPUs) vs
default (2 GPUs), both workloads, gated, open-loop Poisson for serving cells:

- SHORT context: 1.66x saturated (48.2 vs 29.0 audio-s/s) — inside the
  pre-registered (1.3, 2.0) band, matching upstream's 1.74x. METHODOLOGY
  CAVEAT (user-flagged): this is 3 GPUs vs 2 (+50% HW); per-GPU it is only
  +11% (16.1 vs 14.5). The honest 3-GPU head-to-head vs coloc x3 is
  Experiment C (8f). At 8 qps offered (pure open loop, uncapped): replica2
  holds rtf_p99 0.63 vs default 2.23 — the earlier 31.5 was partly a
  client-cap artifact.
- LONG context L=6343: +12.9% throughput for +50% hardware — under the
  pre-registered 15% bound (thinker binds). UNEXPECTED: replica2's p99
  tails are far cleaner at long context too (0.37-0.83 vs 0.41-3.22) —
  the tails stages contribute more to the tail than assumed.
- 0 errors in 1,440 replicated-pipeline requests (sticky binding holds).
- Landmine: the shipped replica2 example yaml uses the removed
  stage_overrides schema — ported to stages.<name>.gpu_memory_fraction.

Together with Ming TP4 (8c) this completes the two-sided test: scaling the
non-binding stage buys ~nothing (TP4 at short-ctx Ming, tails-replicas at
long-ctx Qwen); scaling the BINDING stage pays in full (tails-replicas at
short-ctx: 1.66x). The n_i axis is now actuated upstream — the planner needs
only per-stage constants to pick it automatically.

### 8e. Ming n_talker=2 (Experiment B): the planner's guidance, cashed out

On the rebased stack (sglang 0.5.18; baseline re-measured: 11.3-12.5
audio-s/s saturated, kappa~4): thinker TP2 + TWO talker replicas (4 GPUs,
one-parameter port of the Ming factory) reaches 28.9 audio-s/s at c8 with
rtf_p99 0.39 — >= 2.56x the baseline at matched concurrency, ABOVE the
pre-registered (1.5, 2.2) band, and still unsaturated. ISO-BUDGET NOTE:
this is a fair 4-GPU comparison — at exactly 4 GPUs the non-replica
feasible set is {TP2+talker+idle} (TP4 needs 5, two pipelines need 6) —
and at a 5-GPU budget replication beats TP4 28.9 vs 10.3. Open loop it keeps
up through 6 qps offered (p99 0.33) where the baseline collapses past ~3.
Per-GPU: 7.2 vs 3.8 audio-s/s/GPU — replicating the BINDING stage nearly
doubles GPU efficiency, vs TP4's -38% (8c). The overshoot beyond 2x says
the single-talker baseline was queue-collapsing at the tail stage, so the
second replica recovers capacity plus the queueing loss. 0 errors in 840
requests across both arms.

The full n_i story, ISO-BUDGET framing (comparisons only at equal GPU
count, vs the enumerated feasible set at that budget): TP4 vs replication
at 5 GPUs: 10.3 vs 28.9; tails replicas at thinker-bound long context:
+12.9% for a GPU that had no better use; Ming talkers at 4 GPUs: 2.56x
over the idle-padded alternative. The remaining open cell — replica2 vs
coloc x3 at exactly 3 GPUs, short context — is Experiment C (8f). Law
unchanged but stated correctly: AT A FIXED BUDGET, the plan that feeds the
binding stage wins; capacity anywhere else adds ~nothing.

### 8f. Experiment C: the iso-budget correction, closed empirically

At exactly 3 GPUs, short context: coloc x3 (one full colocated pipeline per
GPU) = 65.4 audio-s/s saturated (p99 0.88-1.05) > tails-replica2 48.2 >
default+idle 29.0. Pre-registered prediction (55-75) PASSED; 0/1080 errors.
The "1.66x" stage-replica headline was a non-iso artifact. Corrected law,
budget-constrained: AT FIXED B, rank the enumerated feasible set — here
full-pipeline replication wins because nothing is shared; stage replication
wins only when full replication is infeasible (Ming: weights > 1 GPU) or
memory-bound. This motivates formulation v2: an explicit capacity identity
sum_i n_i * t_i * p_i + idle = B alongside the per-device memory constraint,
with per-stage constants — making iso-budget comparison structural.

## 9. Analysis / takeaways (updated as arms land)

1. Metrics: rtf_p99-at-SLO is brittle (tail = max of ~2 samples at n<=200 and
   content-coupled). VoxServe's goodput (TTFA SLO + streaming viability at
   offered QPS) is a steadier target because it composes per-chunk deadlines
   instead of a single ratio tail; we adopt it for serving comparisons.
2. Profiling: the two-probe split (serial c=1 vs saturation) is real physics —
   serial predicts the QoS frontier, saturation gives capacity; conflating
   them (one ServiceLaw) is the planner's main modeling debt.
3. Memory planning at 183 GiB is trivial for 30B-class pipelines (KV never
   binds); the open axes are compute sharing (coloc d(m)) and replica count.
4. Content quality gates are not optional in TTS serving benchmarks: without
   them every tail metric measures degeneracy, not capacity.
5. Prefix methodology, code-level cross-check vs VoxServe (2026-08-27):
   our per-request-unique prefixes are NOT VoxServe methodology and don't
   need to be — VoxServe's engine has no cross-request KV reuse at all
   (per-request page alloc/free, vox_serve/worker/base.py:300/774) and its
   prompts carry no long shared prefix, so it gets the invariant "every
   request pays its full KV/prefill cost" BY ENGINE CONSTRUCTION; SGLang's
   RadixAttention shares prefixes from token 0 (radix_cache.py:355
   match_prefix), so we impose the same invariant BY WORKLOAD CONSTRUCTION
   (session header + line rotation). Consistent intent, different mechanism.
   Cross-risks: VoxServe's fixed-sentence default on SGLang would inflate
   capacity to a near-100%% cache-hit regime; a radix cache added to VoxServe
   would make its numbers workload-dependent.

## 10. Formulation v2: the budget-constrained planner (validated prototype)

User-driven correction: the v1 objective constrained only per-device memory;
the GPU budget was implicit, idle devices invisible, and TP degree vs
replica count incomparable. v2 adds the capacity identity

    sum_i n_i * t_i * p_i + idle = B     (co-location folds instances onto
                                          a device under measured d(m))

with per-stage constants (kappa_i, T_i) and utility
u_i = n_i * e(t_i) * d(m) * min(pool_i/(k_i L), kappa_i), objective
max min_i u_i at FIXED B. Prototype exp/planner_v2/budget_enum.py,
parameterized by measured constants, reproduces the measured argmax at
every tested budget: Qwen3 B=3 (pipeline x3 > tails-replica2 > idle-padded),
Ming B=4 (talker x2 > idle-padded), Ming B=5 (replicas > TP4). New
falsifiable prediction: Ming B=5 TP2 + talker x3 ~ 30 audio-s/s
(thinker-capped). Qwen3 B=2 coloc x2 at short context is a PREDICTION
(43.6), unmeasured. Enumeration remains exact: integer compositions of B.


## 11. The B=8 program: full-node plans, predicted then measured

Standing budget B=8 (user directive). The v2 enumerator's top plans were
pre-registered and then run (exp/replica/PREREG.md cells D/E1/E2):

| cell | plan | predicted | measured | verdict |
|---|---|---|---|---|
| D (B=5) | Ming TP2 + talker x3 | ~30 (thinker cap) | 34.6, p99 0.36 | pass within bars; thinker const -> 35 |
| E1 (B=8) | Ming 2x(TP2 + talker x2) | 45.2 | >=48.0, p99 <= 0.41 | pass (+6%, unsaturated) |
| E2 (B=8) | Qwen3 pipeline x8 | 174.4 | 191.5 @ agg c96, all replicas p99 < 1 | pass (110%; coloc linear x3 -> x8) |

Full-node headline: 8x Blackwell serves ~192 audio-s/s (~55 qps) of
QoS-clean short-context speech as 8 colocated pipelines, and ~48 audio-s/s
of Ming (214 GiB MoE) as 2 replicated TP2 pipelines — both shapes produced
by the budget-constrained enumerator BEFORE measurement. Cost of the B=8
engineering: 4 infra landmines, incl. an UPSTREAM bug — sglang-omni's
_NcclPortAllocator's deterministic 29500+ counter is a TOCTOU across
co-hosted pipelines (fixed locally to ephemeral allocation) — plus
sequential-boot, death-detection, and cleanup-on-fail rules for co-hosted
servers.

## 12. Default vs plan at B=8, all three models (latency vs output load)

Fair defaults (by-the-book config replicated to fill 8 GPUs) vs the v2
enumerator's plans, matched aggregate loads, all pre-registered:

| model | fair B=8 default | ours | saturated (default vs ours) | latency at matched load |
|---|---|---|---|---|
| Qwen3-Omni (short ctx) | 4x shipped 2-GPU split | pipeline x8 | 114.1 (QoS-infeasible, rtf_p99 1.5-5.7) vs 191.5 (all p99 < 1) = 1.68x | default cracks from agg 16 qps; ours clean to 24+ |
| Ming (214 GiB MoE) | cookbook TP4+talker (+3 idle) | 2x(TP2+talker x2) | 14.5 (collapses at 6 qps, lat_p99 8 s) vs 48.0 (p99 <= 0.41) = 3.3x | default knee ~3 qps; ours ~12+ |
| Higgs 4B (streaming) | 8x 1 engine/GPU | (same — coloc x16 REFUTED, +5% within bars, worse TTFA) | 339 vs 356 ~= tie | TTFA p99 0.2-0.26 both; default already optimal |

Reading: where the default strands capacity (split pipelines, oversized TP),
the plan wins 1.7-3.3x with clean tails at loads where the default
collapses. Where the default already saturates the hardware (Higgs), the
enumerator correctly returns it unchanged — a planner that knows when to do
nothing. 0 errors across all 4,240 matrix requests.
