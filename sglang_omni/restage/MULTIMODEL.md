# Multi-model restage planner — worklog and evidence status

Session: 2026-08-23, autonomous hour (Claude Code on the AWS SLURM login node,
no GPUs on the node; cluster queue saturated — see "What is NOT verified").
Task: generalize the residency planner beyond Qwen3-Omni-30B to the SpeechLMs
of the VoxServe paper (arXiv:2602.00269) and verify what can be verified.

## What was done (steps)

1. Cherry-picked the 5 `restage:` planner commits from `restage-planner` onto
   current `main` (adds-only; clean).
2. Fetched real geometry for the seven VoxServe models from HuggingFace
   (2026-08-23): `config.json` + `model.safetensors.index.json`.
   - exact (DERIVED): glm-4-voice-9b (19,085,115,456 B; 40L / 2 kv-groups /
     kv_channels 128), step-audio-2-mini (16,630,358,528 B; 28L / 4 kv-heads /
     head_dim 128), zonos-v0.1 KV geometry (26L / 4 kv-heads / 128).
   - gated repos (PRIOR from published backbone): orpheus-3b (Llama-3.2-3B),
     csm-1b (Llama-3.2-1B + depth decoder).
   - non-transformers repos (PRIOR from model cards): cosyvoice2-0.5b
     (Qwen2-0.5B LM), chatterbox (t3 ~0.5B; KV geometry unpublished →
     pool arithmetic disabled for it).
3. Added `models.py` (registry; DATA with per-constant provenance),
   generalized `plan.py` (`--model`, `--list-models`, `--gpu-mem-gib`,
   memory-feasibility from KV geometry, per-model shape families, emission
   refused for non-servable models), added 28 CPU tests
   (`tests/unit_test/test_restage_multimodel.py`, all passing).

## Hypotheses encoded in the shape families (to falsify on GPU)

- H1 (régime law, carried over): TP only wins when one request's weights+KV
  cannot fit a GPU — an arithmetic wall, not a slope. Encoded as
  `tp2_backbone_mandatory`, emitted *only* at the wall. Measured support:
  vLLM-Omni 65G/62G and sglang-omni 2×H100 (TP2 9.97 < no-TP 15.33).
- H2 (consolidation): moving tails off backbone GPUs pays only when the tails
  would otherwise strand backbone memory (Qwen3-Omni: 59.5+6.8 GiB). For
  mid-size models (GLM-4-Voice 17.8 GiB, Step-Audio-2 15.5 GiB) tails fit
  beside the backbone → `dedicated_x4` should win. UNMEASURED prediction.
- H3 (co-location): n engines per GPU under MPS gains only when a single
  engine cannot saturate the GPU. Only measured case: vLLM-Omni TTS-1.7B
  (26.1 → 41.7, +60%, d(mps_x2)=0.64). Gate: footprint ≤ 0.15×GPU. All coloc
  rows print "UPPER BOUND … true d → 1/n if one engine already saturates".
- H4 (d(m) transfer): qwen3-omni tails d(ts)=0.58 / d(mps)=0.85 (MEASURED
  sgl-dm 20057386) are used as PRIOR bands for other models. The cross-stack
  spread (0.41–0.93) says this is the weakest constant — measure per model.

## What IS verified (CPU, tonight)

- The planner reproduces the recorded, content-gated sglang-omni A/B: chosen
  no-TP rebalance 15.33 > shipped auto-partition 12.96 (+18.3%, sgl-3arm
  19589494) and TP2 (9.97, sgl-solv5 19591990) ranks last. Pinned by tests.
- Every printed number carries MEASURED/PREDICTED/PRIOR + source; unmeasured
  models can never print a MEASURED throughput (test-enforced).
- Memory arithmetic matches the config-derived geometry exactly (test vs
  hand computation for glm-4-voice-9b).
- Launch artifacts are emitted only for servable models; MPS launcher pins
  the pipe dir to node-local /tmp.

## What is NOT verified (and why)

No new GPU measurement was made this session: the login node has no GPUs, the
cluster queue held ~1,500 jobs with zero idle nodes, no sglang-omni venv or
model weights exist on this cluster yet, and the house rules forbid quoting
un-run numbers. Therefore:

- All VoxServe-model throughputs are PREDICTED from an UNCALIBRATED law
  (vLLM-Omni H100 coefficients reused, flagged on every row).
- The seven VoxServe models are PLANNER-ONLY: sglang-omni has no serving
  implementation for them (see `sglang_omni/models/` for what it can load).
  Porting one (GLM-4-Voice is the closest fit: LM + separate flow-decoder
  tail) is the prerequisite for any measured verdict on them.

## Next steps (GPU verification protocol)

Per model, in order: (1) throwaway round after boot (graph capture), (2)
single-request probe → δ, weights, pool (server log), (3) closed-loop
saturation probe at 3 points around predicted κ, 60+ samples, per-request-
unique prefixes, (4) content gate (WER median ≈ 0, duration band) before any
number is cited, (5) refit `ServiceLaw`, flip PRIOR→MEASURED with the job id.
Pre-register the H2/H3 predictions above before the first run.

## Publishing

Commits are local on `main` (this clone tracks
github.com/zhumengzhiren/sglang-omni). No credentials on this machine, so to
put them on a fork:

```bash
gh repo fork zhumengzhiren/sglang-omni --clone=false   # or fork in the UI
git remote add myfork git@github.com:<you>/sglang-omni.git
git push myfork main
```

Hold the upstream PR until at least one GPU-measured verdict exists for a
non-Qwen model; the PR text must keep every PREDICTED/PRIOR tag.
