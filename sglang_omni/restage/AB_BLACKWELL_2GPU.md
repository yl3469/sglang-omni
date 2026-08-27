# Restage vs no-restage A/B — Qwen3-Omni-30B on sglang-omni, 2x Blackwell 183 GiB

Date: 2026-08-25. Run dir: /home/scratch.yueyingl_coreai/exp/ab-20260825
(first attempt ab-20260825-attempt1 discarded: bench wrapper import bug, no data).

## TL;DR

The restage planner's measured 2xH100 verdict (rebalanced_notp 15.3 >
shipped_auto 13.0 audio-s/s) does NOT transfer to 183 GiB GPUs, and the server
logs show exactly why: the pathology restage exploits on H100 — the shipping
auto-partition starving the thinker KV pool (31k tokens on H100) — does not
exist at 183 GiB. Here the auto-partition already gives the thinker a
1,070,464-token pool (~169 concurrent L=6343 requests); the H100-tuned 0.82
fraction actually SHRINKS it to 931,392. Both arms are compute-bound, matched
within noise at equal concurrency, and the QoS verdict flips to default on
tail noise. This validates the planner's own provenance discipline ("anchors
at L~6343 measured on 2xH100; off-anchor, recalibrate first") — the anchors
are hardware-specific, not just workload-specific.

## Setup

- Branch: `lisa/sglang-copy` (sglang-omni + `sglang_omni/restage` planner).
- Env: `uv venv -p 3.12`; `uv pip install --prerelease=allow -e .`
  → sglang 0.5.16, torch 2.11.0+cu130, flashinfer 0.6.14 cu13.
- Hardware: 8x NVIDIA Blackwell (sm 10.0, 183 GiB); experiment pinned to
  GPUs 0-1 (CUDA_VISIBLE_DEVICES=0,1) to mirror the measured 2-GPU line.
- Model: Qwen/Qwen3-Omni-30B-A3B-Instruct (66 GiB HF snapshot).
- Workload = the measured anchor: seed-tts-eval EN, L=6343-token long-context
  system prompt (main:docs/longctx_system_prompt.txt), D~4.5 s audio,
  closed loop. prompt_tokens_mean logged 6343.0 in every cell — exact match.
- PER-REQUEST-UNIQUE system prefixes (exp/run_bench_uniq.py, canonical
  session-header + line-rotation scheme from main:patches/seed_tts_dataset.py):
  otherwise RadixAttention dedupes the shared 6.3k-token prefix from token 0
  (>99% of request KV) and the KV-capacity regime is invisible.
- Protocol per arm (replicates main:jobs/run_sgl_3arm.sbatch): boot, health
  wait, one DISCARDED warmup round (c=4, 8 samples, pays JIT/graph capture),
  then 48 samples per concurrency point; arm judged at its OWN QoS frontier
  (best audio-s/s with rtf_p99 <= 1); full GPU drain between arms.

## Arms

| arm | meaning | launch |
|---|---|---|
| default (no restage) | shipped auto per-stage fractions | thinker GPU0, tails GPU1, no overrides |
| restage | planner winner at `--gpus 2` (`rebalanced_notp`) | `--thinker-mem-fraction-static 0.82 --talker-mem-fraction-static 0.40`, `SGLOMNI_THINKER_ARGS='{"max_running_requests":64,"cuda_graph_max_bs":64}'` |

Planner: `python -m sglang_omni.restage.plan --gpus 2 --gpu-mem-gib 183` ranks
rebalanced_notp 15.3 > shipped_auto 13.0 > tp2_thinker 10.0 (all MEASURED,
2xH100 anchors). Note: for qwen3-omni-30b at gpus=2 the ranking ignores
--gpu-mem-gib entirely (hardcoded measured arms) — see gaps below.

## Results (48 samples/point, closed loop, unique 6343-tok prefixes)

| arm/conc | audio-s/s | rtf_p99 | rtf_mean | latency_p99_s |
|---|---|---|---|---|
| default_c1 | 4.94 | 1.032 | 0.234 | 1.49 |
| default_c2 | 9.12 | 1.618 | 0.278 | 1.52 |
| default_c4 | **14.55** | **0.409** | 0.283 | 1.17 |
| default_c8 | 16.56 | 2.154 | 0.676 | 3.14 |
| restage_c2 | 7.47 | 1.153 | 0.309 | 1.92 |
| restage_c4 | 14.49 | 1.188 | 0.329 | 1.98 |
| restage_c8 | 16.86 | 2.797 | 0.702 | 3.06 |
| restage_c12 | 19.57 | 29.349 | 2.078 | 3.19 |
| restage_c16 | 19.41 | 2.657 | 1.028 | 3.77 |

QoS frontier (rtf_p99 <= 1): **default 14.55 audio-s/s at c4; restage has NO
feasible point** (best rtf_p99 1.15 at c2). At matched concurrency the arms
are equal within noise (c4: 14.55 vs 14.49; c8: 16.56 vs 16.86) — the QoS
verdict is decided by tail stragglers.

### Why the H100 verdict inverts here (from server logs)

KV pools actually allocated:

| pool | default (auto) | restage (0.82/0.40) | H100 default (campaign) |
|---|---|---|---|
| thinker | 1,070,464 tok (2x49.0 GB, frac 0.879) | 931,392 tok (2x42.6 GB, frac 0.82) | 31,189 tok |
| talker  | 3,174,720 tok (2x72.7 GB, frac 0.862) | 1,395,264 tok (2x31.9 GB, frac 0.40) | 1,281,734 tok |

On H100-80GB the auto-partition starves the thinker (31k tokens ≈ 5 concurrent
L=6343 requests) and hand-rebalancing to 0.82/0.40 is worth +18%. On 183 GiB
the auto fraction (0.879) already exceeds the H100-tuned 0.82, the thinker
pool holds ~169 concurrent requests either way, and KV never binds at any
tested concurrency. The restage arm's absolute fractions BUY NOTHING and its
`max_running_requests=64` override only admits more in-flight requests into a
compute-bound engine, inflating tails (restage_c12 rtf_p99 29.3 = admission
overshoot; a straggler ran ~2 min).

### Caveats

- Single run per cell; with 48 samples, p99 ≈ worst sample, hence the
  non-monotonic tails (default_c2 1.62 vs default_c4 0.41). ROOT CAUSE
  identified post-hoc: the extreme rtf values are DEGENERATE 0.06s-audio
  outputs (rtf = latency/D explodes as D->0) — restage_c12's "rtf_p99 29.35"
  was two such blobs at normal ~1.7s latency, i.e. content failure, not
  queueing. Both arms' QoS frontiers here are content-noise-limited; a
  content gate (audio duration in [0.5, 20]s, temperature 0.2) is mandatory
  before citing any tail. Cross-arm deltas <2 audio-s/s are not resolvable.
- This is the 0%-prefix-hit (worst-case, multi-tenant) regime by design; a
  deployment with one shared cached system prompt sits at the best-case end.
- Prior anchors are 2xH100; this box is Blackwell sm10.0 with 2.3x the VRAM —
  the config ORDERING, not the absolute numbers, was under test.

## What restage should do on this hardware

The planner's own calibration path is the right answer: its memory-feasibility
arithmetic (pool_requests() at gpu_mem_gib=183) would show the auto partition
is KV-unconstrained, making fraction rebalancing moot; the win, if any, must
come from the compute side (e.g. DP replicas across the 8 GPUs, or the
dp3_consolidated shapes at --gpus 4). Concretely: run the two calibration
probes on this box (single-request probe → delta; c-sweep around kappa with
unique prefixes → T_sat), refit ServiceLaw for a Blackwell law, and re-rank.

## Planner/tooling gaps found

1. `--gpus 3` crashes (`IndexError: rows[0]`): plan.py candidates() yields
   nothing for gpus==3 (only 2 and >=4 handled). plan.py:64-90.
2. The 2-GPU winner emits no launch script: launcher_lines() covers
   dp3_consolidated/dp2_dedicated/tp2_thinker but not rebalanced_notp or
   shipped_auto, so `--launch-dir` writes nothing for the recommended plan.
3. For qwen3-omni-30b at gpus==2 the ranking ignores --gpu-mem-gib: the
   measured H100 arms are returned verbatim at any memory size. This A/B is
   the empirical demonstration that the verdict can invert; the entry should
   at least stamp its provenance with the anchor hardware and re-derive the
   memory arithmetic at the requested gpu-mem-gib.
4. No in-tree unique-prefix benchmark support: benchmark_omni_seedtts.py sends
   one fixed --system-prompt verbatim (the campaign wrapper run_bench_uniq.py
   was never committed; reimplemented in exp/run_bench_uniq.py).
5. First boot on this box exceeds the 600 s stage-readiness default (cold
   Triton JIT + graph capture; thinker alone ~330 s). Fixed via
   SGLANG_OMNI_STARTUP_TIMEOUT=1800.

## Methodology notes

- Why unique system prompts: RadixAttention prefix-shares from token 0; a
  shared 6.3k-token system prompt dedupes >99% of per-request KV, hiding the
  KV-capacity regime and flattering the baseline. Evidence: campaign job
  19554509 (shared prefix) had an EMPTY QoS-feasible region. Real long-context
  sessions carry per-request-unique history no cache can dedupe.
- VoxServe cross-check (github.com/vox-serve/vox-serve, arXiv:2602.00269):
  open-loop Poisson goodput sweeps over short TTS sentences (default: one
  fixed sentence, datasets sampled with replacement), short hardcoded
  model-internal system prompts, and a from-scratch engine with NO
  cross-request prefix caching — the shared-prefix problem cannot arise there
  by construction, so our control is orthogonal to (not inconsistent with)
  their methodology. Worth adopting from VoxServe later: per-chunk streaming
  viability as the QoS gate (catches mid-stream stalls end-to-end RTF hides),
  open-loop rate sweeps to complement closed-loop ladders, TTFA excluding the
  WAV header chunk.

## Repro

    cd /home/scratch.yueyingl_coreai/restage   # branch lisa/sglang-copy
    uv venv -p 3.12 .venv && uv pip install --prerelease=allow -e .
    bash /home/scratch.yueyingl_coreai/exp/run_ab.sh both
    python3 /home/scratch.yueyingl_coreai/exp/summarize.py exp/ab-<date>/results
