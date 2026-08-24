"""Model registry for the restage planner. This file is DATA, not arithmetic.

Every constant carries a provenance tag:
  MEASURED   from a content-gated run on the named stack (job id given)
  DERIVED    computed from the model's published config.json / safetensors
             index (fetch date given) -- exact geometry, no runtime claim
  PRIOR      from public model cards / papers when the config is gated or
             non-transformers-format -- replace via the calibration probes

KV bytes/token (bf16) = 2 (K+V) * layers * kv_heads * head_dim * 2 bytes.
Weights GiB (DERIVED) = safetensors index metadata.total_size / 2^30.

`servable` means sglang-omni has a serving implementation for the family
(see sglang_omni/models/). The seven VoxServe models (arXiv:2602.00269) are
planner-only entries: the planner ranks residency shapes for them from their
real geometry, but sglang-omni cannot launch them until the architecture is
ported. Emission is refused for non-servable entries.

Throughput anchors exist ONLY for qwen3-omni-30b (2xH100 line, sgl-3arm
19589494 / sgl-solv5 19591990). Nothing here transplants those numbers to
another model: all other entries rank via the uncalibrated law and print
PREDICTED/PRIOR on every number until the two probes are run.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .workload import Workload


@dataclass(frozen=True)
class ModelEntry:
    name: str                                # planner key (--model)
    hf_id: str
    servable: bool                           # sglang-omni implementation exists
    backbone_gib: Optional[float]            # LM backbone weights (None = unknown)
    tails_gib: Optional[float]               # decoder/vocoder stages (None = unknown)
    kv_bytes_per_tok: Optional[int]          # backbone KV, bf16 (None = unpublished)
    default_workload: Workload = field(default=Workload(250, 4.5))
    anchors: Optional[Dict[str, Tuple[float, str]]] = None  # MEASURED T_sat units
    provenance: str = ""
    notes: str = ""


REGISTRY: Dict[str, ModelEntry] = {}


def _reg(e: ModelEntry) -> ModelEntry:
    REGISTRY[e.name] = e
    return e


# --- sglang-omni-servable families -----------------------------------------

_reg(ModelEntry(
    name="qwen3-omni-30b",
    hf_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    servable=True,
    backbone_gib=59.5, tails_gib=6.8,
    kv_bytes_per_tok=98304,
    default_workload=Workload(6343, 4.5, name="long-ctx TTS (measured anchor)"),
    anchors={
        "default_auto": (12.96, "MEASURED sglang-omni 2xH100 sgl-3arm 19589494 (shipped auto-partition, c4)"),
        "handtuned_notp": (15.33, "MEASURED sglang-omni 2xH100 sgl-3arm 19589494 (0.82/0.40 no-TP; c8 rtf99 0.985 -- single run, marginal)"),
        "tp2": (9.97, "MEASURED sglang-omni 2xH100 sgl-solv5 19591990 (TP2 thinker @ fraction 0.62; loses at QoS -- compute-bound)"),
    },
    provenance="anchors MEASURED (sglang-omni 2xH100); weights/kv PRIOR from the "
               "vLLM-Omni engine logs (same checkpoint; sglang log values not recorded)",
    notes="the campaign's measured model; thinker + talker + code2wav stages",
))

_reg(ModelEntry(
    name="qwen3-tts-1.7b",
    hf_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    servable=True,
    backbone_gib=3.4, tails_gib=1.0,
    kv_bytes_per_tok=20480,
    default_workload=Workload(250, 4.2, name="short-text TTS"),
    provenance="PRIOR: weights/kv from the vLLM-Omni campaign logs (same "
               "checkpoint family); NO sglang-omni measurement exists -- "
               "never quote the vLLM-Omni 26.1->41.7 line for this stack",
    notes="single-LM TTS + code2wav tail; sglang_omni/models/qwen3_tts",
))

# --- VoxServe (arXiv:2602.00269) SpeechLMs -- planner-only entries ----------

_reg(ModelEntry(
    name="glm-4-voice-9b",
    hf_id="zai-org/glm-4-voice-9b",
    servable=False,
    backbone_gib=17.77, tails_gib=1.5,
    kv_bytes_per_tok=40960,   # 2*40 layers*2 kv-groups*128 kv_channels*2 B
    default_workload=Workload(2048, 4.5, name="speech-to-speech turn"),
    provenance="DERIVED config.json + safetensors index (fetched 2026-08-23): "
               "total_size 19,085,115,456 B; 40 layers, multi_query_group_num 2, "
               "kv_channels 128, bf16. Tail = glm-4-voice-decoder (CosyVoice-based "
               "flow) ~1.5 GiB PRIOR (separate repo)",
    notes="LM backbone + separate flow-decoder tail: maps directly onto "
          "thinker/tails residency shapes",
))

_reg(ModelEntry(
    name="step-audio-2-mini",
    hf_id="stepfun-ai/Step-Audio-2-mini",
    servable=False,
    backbone_gib=15.49, tails_gib=0.3,
    kv_bytes_per_tok=57344,   # 2*28 layers*4 kv-heads*128 head_dim*2 B
    default_workload=Workload(2048, 4.5, name="speech-to-speech turn"),
    provenance="DERIVED config.json + safetensors index (fetched 2026-08-23): "
               "total_size 16,630,358,528 B (incl. 32-layer audio encoder); "
               "text_config 28 layers, 4 kv heads, head_dim 3584/28=128, bf16",
    notes="audio encoder + LM in one checkpoint; token2wav tail is external",
))

_reg(ModelEntry(
    name="orpheus-3b",
    hf_id="canopylabs/orpheus-3b-0.1-ft",
    servable=False,
    backbone_gib=7.5, tails_gib=0.1,
    kv_bytes_per_tok=114688,  # 2*28 layers*8 kv-heads*128 head_dim*2 B
    default_workload=Workload(250, 4.5, name="short-text TTS"),
    provenance="PRIOR (HF repo gated 2026-08-23): Llama-3.2-3B backbone geometry "
               "(28 layers, 8 kv heads, head_dim 128) + extended audio vocab; "
               "SNAC decoder tail ~20M params",
    notes="MHA-era GQA geometry -> largest KV/token in this registry despite 3B "
          "size; pool-bound earlier than the 9B models",
))

_reg(ModelEntry(
    name="csm-1b",
    hf_id="sesame/csm-1b",
    servable=False,
    backbone_gib=3.1, tails_gib=0.2,
    kv_bytes_per_tok=32768,   # 2*16 layers*8 kv-heads*64 head_dim*2 B
    default_workload=Workload(500, 4.5, name="conversational TTS"),
    provenance="PRIOR (HF repo gated 2026-08-23): Llama-3.2-1B-shaped backbone "
               "(16 layers, 8 kv heads, head_dim 64) + ~100M depth decoder; "
               "Mimi codec tail",
))

_reg(ModelEntry(
    name="zonos-v0.1",
    hf_id="Zyphra/Zonos-v0.1-transformer",
    servable=False,
    backbone_gib=3.0, tails_gib=0.7,
    kv_bytes_per_tok=53248,   # 2*26 attn layers*4 kv-heads*128 head_dim*2 B
    default_workload=Workload(250, 4.5, name="short-text TTS"),
    provenance="kv DERIVED config.json (fetched 2026-08-23: n_layer 26, "
               "num_heads_kv 4, d_model/num_heads=128); weights PRIOR ~1.6B bf16 "
               "(no safetensors index in repo); DAC autoencoder tail PRIOR",
))

_reg(ModelEntry(
    name="cosyvoice2-0.5b",
    hf_id="FunAudioLLM/CosyVoice2-0.5B",
    servable=False,
    backbone_gib=0.9, tails_gib=1.0,
    kv_bytes_per_tok=12288,   # Qwen2-0.5B: 2*24 layers*2 kv-heads*64 head_dim*2 B
    default_workload=Workload(250, 4.5, name="short-text TTS"),
    provenance="PRIOR (repo is non-transformers-format, config.json empty): "
               "Qwen2-0.5B LM backbone geometry from its model card; flow + "
               "HiFT vocoder tails ~1 GiB",
    notes="tails heavier than the backbone -- consolidation arithmetic inverts "
          "vs the 30B case; good stress test for the break-even",
))

_reg(ModelEntry(
    name="chatterbox",
    hf_id="ResembleAI/chatterbox",
    servable=False,
    backbone_gib=1.1, tails_gib=1.0,
    kv_bytes_per_tok=None,    # t3 backbone geometry not published
    default_workload=Workload(250, 4.5, name="short-text TTS"),
    provenance="PRIOR (repo is non-transformers-format 2026-08-23): ~0.5B "
               "llama-style t3 backbone per model card; s3gen tail. KV geometry "
               "unpublished -- pool arithmetic disabled until calibrated",
))


def get(name: str) -> ModelEntry:
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit("unknown model %r; --list-models shows the registry" % name)
