# Pre-registration: per-stage replication (PR #1175) on Blackwell (2026-08-27)

Rebased branch lisa/sglang-1175 = upstream sglang-omni main (incl. merged
PR #1175 process replicas) + our 12 restage commits (clean rebase, 0
conflicts). Deps moved sglang 0.5.16 -> 0.5.18.

## Experiment A: Qwen3-Omni talker_ar+code2wav replica2 (shipped yaml, 3 GPUs)

Arms: default pipeline (thinker G0, tails G1 — 2 GPUs) vs
examples/configs/qwen3_omni_speech_replica2.yaml (thinker G0, tails@r0 G1,
tails@r1 G2 — 3 GPUs). Same gated protocol; TWO workloads.

Falsifiable predictions (the two-sided zero-marginal-value test — the mirror
image of Ming TP4):

1. LONG context (L=6343, our anchor): the THINKER binds (prefill-dominated;
   measured kappa~3-4 there). Tails replicas add a GPU to a non-bottleneck
   stage -> QoS-frontier goodput improves < 15% despite +50% hardware.
2. SHORT context (seed-tts native, ~60 tok): tails are a larger cost share;
   upstream measured 1.74x at c=64 on 3xH200. Prediction: saturated
   throughput ratio replica2/default in (1.3, 2.0); TTFA tail improves.
3. Sticky binding holds: 0 cross-replica request errors at all rates.

If (1) shows a LARGE win at L=6343, the thinker-binds model for Qwen3-Omni
long context is falsified; if (2) shows ~1.0x, the replica mechanism (or the
tails-bound short-context model) is falsified.

## Experiment B (follow-up, custom yaml): Ming n_talker=2

Thinker TP2 (G0,G1) + talker replicas (G2,G3). Prediction from the TP4
verdict: saturated throughput ~1.6-2.0x TP2 baseline (10.5 -> 17-21
audio-s/s) because the talker binds. Requires writing a Ming pipeline yaml
with processes.talker num_replicas=2 (no shipped example; placement code
already consumes replica_devices).

Protocol: gated (D in [0.5,20]s, temp 0.2), unique prefixes for the L=6343
workload, throwaway warms, ladder + Poisson, n=60-120/cell, single-run
error bars +-1c/+-20% acknowledged.

---

# VERDICT Experiment A (2026-08-27, exp/replica/results/, sglang 0.5.18)

All three claims PASS (claim 1 marginal), plus one unexpected finding.

1. LONG context L=6343 (thinker binds -> <15% throughput gain): saturated
   c6 23.80 vs 21.09 audio-s/s = +12.9% for +50% hardware. PASS (marginal).
   UNEXPECTED: replica2's rtf_p99 is dramatically cleaner at every long-ctx
   cell (0.37-0.83 vs 0.41-3.22): the tails stages contribute more to the
   long-context TAIL than the thinker-binds model assumed — replica2's c6
   is QoS-feasible (0.83) where default's violates (1.29). QoS-frontier
   framing: replica2 23.8 vs default ~11-21 (default's feasible set is
   tail-noise-limited). Single-run caveats apply; the throughput claim is
   what was pre-registered and it passed.
2. SHORT context (tails-heavier, predict 1.3-2.0x): saturated c32 48.20 vs
   28.98 = 1.66x — inside the band, matching upstream's 1.74x on H200.
   PASS. Open-loop (UNCAPPED rerun): at 8 qps offered replica2 holds rtf_p99
   0.63 vs default 2.23 (7.32 completed qps). The first capped run's 31.5
   was partly a client-cap artifact — corrected.
3. Sticky replica binding: 0 errors in 1,440 replica2 requests. PASS.

Protocol notes: all qps cells are open-loop Poisson (detached tasks,
exponential inter-arrivals). The --max-concurrency 64 client cap could bind
only in collapsed overload cells; default.Lshort_qps4/qps8 rerun with
--max-concurrency 0 (pure open loop, unlimited connector) — qps cells use
cap 0 going forward. Ladder (c=N) cells are closed-loop BY DESIGN: kappa is
defined closed-loop in the formulation; no serving conclusion uses them.
Stack note: sglang 0.5.18/torch 2.13 is itself faster than 0.5.16 (default
2-GPU L6343 saturated ~21 vs ~14.6) — constants are software-versioned.

## Experiment B amendment (before any B run): the 0.5.18 re-baseline

Stack moved (0.5.16 -> 0.5.18); the TP2 baseline is RE-MEASURED on the new
stack (arm ming_base) rather than compared to run-4 numbers. Prediction
restated: talker-replica2 saturated throughput / re-measured TP2 baseline
in (1.5, 2.2); if ~1.0x, the talker-binds model is falsified on 0.5.18.
All qps cells uncapped open loop.

---

# VERDICT Experiment B (2026-08-27, exp/ming-replica/, sglang 0.5.18)

Baseline (TP2 thinker + 1 talker, 3 GPUs): saturated 11.3-12.5 audio-s/s,
kappa ~4 (c4 p99 0.63; c8 1.56), Poisson saturation ~3.2 qps (p99 >3 at
offered 4-6). All cells 0 errors.

talker-replica2 (TP2 + talkers on 2 GPUs, 4 GPUs): c8 = 28.90 audio-s/s at
rtf_p99 0.39 (NOT yet saturated); open loop keeps up through 6 qps offered
(22.3 audio-s/s, p99 0.33). 0 errors.

Prediction (1.5, 2.2)x: measured >= 2.56x at matched c8 — the band FAILED
HIGH. The talker-binds hypothesis is confirmed with a bonus: the overshoot
means the single-talker baseline was losing more than raw capacity (its
past-kappa cells are tail-degraded by queue collapse at the talker), so
removing the bottleneck recovers both capacity AND the queueing loss.
Honest caveats: single-run; talker2 unsaturated at c8 so 2.56x is a LOWER
bound on the saturated ratio; baseline vs TP4 comparisons cross stack
versions (TP4 was 0.5.16).

Per-GPU at matched budget question (the planner's ranking, now measured on
one stack): talker-replica2 7.2 audio-s/s/GPU vs baseline 3.8 vs
(cross-stack) TP4 2.2. Replicating the binding stage is ~3x more
GPU-efficient than raising TP on the non-binding stage.

Ming port cost for replicas: ONE parameter (gpu_id on create_talker_executor)
+ the 4 compat fixes — the audit's estimate held.

---

# METHODOLOGY CORRECTION + Experiment C (2026-08-27, pre-registered)

User-flagged flaw: replica ratios compared unequal GPU counts. Corrected
rule: plans compare ONLY at equal GPU budget against the enumerated feasible
set at that budget; idle-padded alternatives stated explicitly; per-GPU
efficiency reported alongside.

Restated verdicts:
- Ming talker2 (4 GPUs): iso-budget WIN stands — at 4 GPUs the non-replica
  feasible set is {TP2+talker+idle} (TP4 needs 5, 2 pipelines need 6), so
  28.9 vs 11.3 is a fair 4-GPU comparison. Per-GPU 7.2 vs 3.8.
- Ming TP4 (5) vs talker2(4)+idle: at a 5-GPU budget replication wins 28.9
  vs 10.3 — iso framing STRENGTHENS the TP4 null.
- Qwen3 tails-replica2 short ctx: "1.66x" was 3-vs-2 GPUs (+50% HW);
  per-GPU only +11% (16.1 vs 14.5). The true 3-GPU alternative coloc x3
  (three colocated single-GPU pipelines) was UNMEASURED -> Experiment C.

## Experiment C: Qwen3-Omni short-context, THREE GPUs, head-to-head

Arms (both 3 GPUs, same stack 0.5.18, gated, uncapped open loop):
  (i) tails-replica2 (measured: sat 48.2 audio-s/s at agg c32; qps8 p99 0.63)
  (ii) coloc x3: one full colocated pipeline per GPU (colocated yaml),
       3 servers, load split 3 ways, disjoint dataset shards.
Falsifiable prediction: coloc x3 saturated aggregate in (1.2, 1.9)x the
single colocated pipeline x3 discount... stated directly: coloc x3 BEATS
tails-replica2 at short context (predicted 55-75 audio-s/s aggregate:
3 independent pipelines with no shared-thinker queueing, ~0.9x linear
scaling of a ~20-27 audio-s/s single-GPU colocated pipeline). If coloc x3
< 48.2, replica2 wins iso-GPU and the prediction is falsified.

# VERDICT Experiment C (2026-08-27, exp/coloc3/)

coloc x3 aggregate: c33 = 65.4 audio-s/s (rtf_p99 0.88-1.05, at the QoS
edge); c15 = 48.2 with p99 0.60-0.65; open loop keeps up through 9 qps agg.
0 errors / 1080 requests. Prediction (beats replica2, 55-75) PASSES.

ISO-3-GPU RANKING, short context: coloc x3 (65.4) > tails-replica2 (48.2)
> default+idle (29.0). The "1.66x" replica headline was a non-iso artifact;
at equal budget, full-pipeline replication > stage replication > idle.
Stage replication remains the right tool when full replication is
infeasible (weights don't fit: Ming) or when only one stage binds under a
memory-constrained budget. Per-GPU: coloc x3 21.8 > default 14.5 — the
default STRANDS tails-GPU compute.

## Experiment D (pre-registered before run): Ming B=5, talker x3

v2 enumerator's top B=5 plan: TP2 (2 GPUs) + 3 talker replicas (3 GPUs).
Prediction: saturated ~30 audio-s/s, capped by the thinker constant
(min(u_thinker~30, 3x11.3)). Falsification map: ~30 => per-stage model
confirmed twice (win AND its saturation); >>30 => thinker constant wrong;
~28.9 (no gain over x2) => talker x2 had already un-bound the pipeline.
Same protocol (gated, uncapped open loop qps + closed ladder).

## B=8 program (standing budget per user directive, pre-registered)

Prototype gap found & fixed while enumerating B=8: whole-pipeline
replication was missing from ming_plans (B=4/5 argmax unchanged after fix).

Cell E1 — Ming B=8: 2x(TP2 + talker x2), predicted 45.2 audio-s/s
  (2 x min(thinker 30, 2x11.3)); vs the best single-pipeline plan's 30
  plateau. Runnable as two independent servers (4 GPUs each), load split.
  Falsification: ~45 confirms pipeline-level linearity; ~30-38 implies
  cross-pipeline interference (host/NVLink) the model lacks.
Cell E2 — Qwen3 short-ctx B=8: pipeline x8 (colocated), predicted 174.4
  (8 x 21.8). Tests coloc scaling linearity from x3 to x8; sub-linear
  measurement quantifies the host-side interference term the formulation
  currently omits.
Both gated, uncapped open loop + closed ladder, after Experiment D frees
GPUs 3-7.

# VERDICT Experiment D (exp/ming-replica/talker3.*)

Measured: c8 = 34.62 audio-s/s (rtf_p99 0.36), qps6 keeps up (5.84
completed, p99 0.45). 0 errors. vs prediction ~30 (thinker-capped):
- Diminishing returns CONFIRMED: 3rd talker bought +5.7 (2nd bought ~+17)
  — not the naive +11.3; the pipeline is transitioning to thinker-bound.
- Magnitude within the +-20% single-run band of the prediction; thinker
  constant revised 30 -> ~35 (>=34.6, possibly unsaturated at c8).
- B=5 iso-budget ranking settled: talker x3 (34.6) > talker x2+idle (28.9)
  >> TP4+talker (10.3). Idle at B=5 is no longer optimal — the enumerator's
  reclaim prediction was right.

# VERDICT Cell E1 (exp/b8/, 2026-08-28)

2x(TP2 + talker x2) at B=8: aggregate 48.02 audio-s/s at agg c12
(p99 0.40, still unsaturated -> lower bound) vs predicted 45.2. PASS
(+6%, within single-run bars). Open loop keeps up through agg 6 qps
(p99 <= 0.37). 0 errors / 720 requests. No cross-pipeline interference
detected. Cost of getting here: 6 attempts, 4 infra landmines fixed —
(1) _NcclPortAllocator deterministic 29500+ counter is a TOCTOU across
co-hosted pipelines (UPSTREAM BUG; fixed to ephemeral allocation);
(2) co-hosted 214 GiB boots must be sequential; (3) health polls must
detect server death; (4) failed boots must clean up their own trees or
orphans poison the next attempt's ports.

# VERDICT Cell E2 (exp/b8/e2_*, 2026-08-28)

pipeline x8: aggregate 191.5 audio-s/s at agg c96 with EVERY replica
QoS-feasible (p99 0.74-0.99); 169.6 at agg c64; open loop 88.6 at agg
24 qps offered (p99 <= 0.69). 0 errors / 1920 requests. vs predicted
174.4: PASS at 110% — coloc scaling from x3 to x8 is linear within
single-run bars; no host-interference term needed at this scale.

## Cell E3 (pre-registered): the FAIR B=8 baseline — 4x shipped default

User correction: the baseline must also use all 8 GPUs. Default-on-8-GPUs =
what a practitioner deploys out of the box: 4 replicas of the shipped
2-GPU split (thinker GPU-a + tails GPU-b), router-style load split.
Prediction from constants: saturated aggregate ~4 x 29.0 = ~116 audio-s/s
(tails-bound per pair), vs ours (pipeline x8) 191.5 — i.e. the planner plan
should deliver ~1.65x the fair default at equal budget, with lower latency
at matched output load (the default's per-pair tails collapse past ~25).
Non-streaming: latency = full-response wall time (TTFA undefined here).

## B=8 default-vs-plan matrix, ALL THREE MODELS (pre-registered)

Fair default = the config a practitioner deploys by the book, replicated to
fill 8 GPUs (idle stated when the default itself cannot fill). Ours = the
v2 enumerator's B=8 plan. Matched aggregate loads per model.

| model | fair B=8 default | ours (enumerator) | prediction |
|---|---|---|---|
| Qwen3-Omni (E3 vs E2) | 4x shipped 2-GPU split | pipeline x8 | ~116 vs 191.5; default latency knee much earlier |
| Ming (M-def vs E1) | cookbook TP4 thinker + talker (+3 idle) | 2x(TP2+talker x2) | TP4 arm ~11-13 sat (0.5.18 re-measure) vs 48.0; default collapses past ~3 qps agg |
| Higgs (H-def vs H-coloc) | 8x one-engine-per-GPU (auto fraction) | 16x two-engines-per-GPU (0.42 fraction each) | default ~8x13.6=109 qps knee; coloc x16 wins ONLY if one engine cannot saturate a GPU — if ~equal, the enumerator's coloc candidate is refuted for Higgs and default IS the plan |

Higgs arms run STREAMING (TTFA = true VoxServe latency); Qwen/Ming arms are
non-streaming (latency = full-response wall time; stated on the slide).

# VERDICT Cell E3 (exp/b8/e3_*, 2026-08-28)

4x shipped default at B=8: saturated 114.1 audio-s/s (predicted ~116 —
on the nose) but QoS-INFEASIBLE there (per-pair rtf_p99 1.5-5.7 at agg c96;
1.02-14.6 at c64). At matched open-loop loads vs E2 pipeline x8: agg 16 qps
58.1 (p99 to 1.26, cracking) vs 55.7 (p99 <= 0.59); agg 24: 85.8 (p99 to
9.2, collapsing) vs 88.6 (p99 <= 0.69). At saturation ours is 1.68x AND
QoS-clean where the fair default collapses. 0 errors / 640. (lad_c8 cell
missing - one client cell did not produce results; non-blocking.)

# VERDICT Ming default arm (mdef_*, 2026-08-28)

Cookbook TP4 thinker + talker (+3 idle) on 0.5.18: saturated 14.4-14.6
audio-s/s (slightly above the 11-13 band — the stack bump lifted TP4 as
well); kappa~4; open loop collapses at agg 6 qps (p99 3.44, lat_p99 8.0 s).
vs OURS (E1, 2x(TP2+talker x2)): 48.0 at p99 <= 0.41 -> 3.3x the fair
default at B=8, QoS-clean where the default collapses. 0 errors / 360.

# VERDICT Higgs B=8 arms (hdef/hcol, 2026-08-28)

hdef (8x 1 engine/GPU): agg 339.3 audio-s/s / 84.0 qps at 96 offered,
TTFA p99 0.20-0.26. hcol (16x, 0.40 fraction): 356.0 / 88.2 with TTFA p99
0.32-0.44. Coloc gains ~5% (within single-run bars) and worsens the TTFA
tail -> the coloc candidate is REFUTED for Higgs, exactly per the
pre-registered branch: one engine saturates the GPU; the default IS the
plan. 0 errors / 2,880 requests total. Full-node Higgs: ~339 audio-s/s
(~84 qps) TTFA-clean streaming.

## Grid addendum (pre-registered): streaming reruns for the 3x3 metric grid

User request: uniform 3x3 (models x {rtf p99, latency p99, TTFA p99}).
TTFA requires streaming -> rerun B=8 Poisson cells with --stream:
e3s (4x default Qwen), e2s (pipeline x8 Qwen), mdefs (Ming TP4 streaming),
e1s (Ming 2x(TP2+talker_stream x2) streaming). Ming uses the now-in-tree
MingOmniStreamingSpeechPipelineConfig (previously not public) — if it fails
to serve, the Ming TTFA cells are annotated as unavailable rather than
substituted. Expectations: streaming adds TTFA visibility without changing
throughput ranking; Qwen plan TTFA << default TTFA at matched load
(prefill-behind-decode queueing, cf. section 6).
