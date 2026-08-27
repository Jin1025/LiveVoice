"""Checkpoint inspection and partial load helpers."""
from __future__ import annotations

import dataclasses

import torch


# Key under which the full LiveVoiceConfig is stored inside a checkpoint. Topology
# inference (the infer_* helpers below) cannot see settings that leave no distinctive
# parameter — speaker_encoder_type="codec_prompt" is the motivating case: it reuses the
# codebook embeddings and builds no speaker_proj/speaker_prefix_proj at all, so a
# codec_prompt checkpoint is indistinguishable from "no speaker encoder" and silently
# falls back to whatever the current config default happens to be. Storing the config
# removes the guesswork; the topology heuristics remain as a fallback for older ckpts.
CONFIG_CKPT_KEY = "livevoice_config"


def config_to_ckpt_dict(config) -> dict:
    """Serialise a LiveVoiceConfig (dataclass) to a plain dict for checkpoint storage."""
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return dict(getattr(config, "__dict__", {}))


# Fields whose CURRENT default differs from the behaviour every older checkpoint was
# produced with. Same reasoning as CODEC_PROMPT_FIELD_DEFAULTS below, opposite direction:
# peak normalisation used to be unconditional, so a stored config without the field was
# decoded WITH it. Filling in today's False would quietly decode old checkpoints at a
# different input scale than the one they were evaluated at.
LEGACY_FIELD_DEFAULTS = {
    "audio_peak_normalize": True,
}


def read_config_from_ckpt(ckpt_path: str) -> dict | None:
    """Return the stored config dict, or None for checkpoints saved before this existed.

    Missing fields listed in LEGACY_FIELD_DEFAULTS are filled with the value that
    reproduces how the checkpoint actually behaved, not the current dataclass default.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get(CONFIG_CKPT_KEY) if isinstance(obj, dict) else None
    if not isinstance(cfg, dict):
        return None
    for field, legacy in LEGACY_FIELD_DEFAULTS.items():
        cfg.setdefault(field, legacy)
    return cfg


def _stored(ckpt_path: str, field: str):
    cfg = read_config_from_ckpt(ckpt_path)
    return cfg.get(field) if cfg else None


# codec_prompt_* fields and the value that reproduces "this field did not exist yet".
# A field missing from a stored config means the checkpoint predates it, so the behaviour
# it controls was OFF — NOT whatever the current default happens to be. Letting today's
# default leak in silently evaluates an old checkpoint as if it had been trained with a
# feature it never saw.
CODEC_PROMPT_FIELD_DEFAULTS = {
    "codec_prompt_continuation": False,
    "codec_prompt_alibi_exempt": True,
    "codec_prompt_content": False,
    "codec_prompt_loss_weight": 0.0,
}
CODEC_PROMPT_FIELDS = tuple(CODEC_PROMPT_FIELD_DEFAULTS)


def infer_codec_prompt_flags_from_ckpt(ckpt_path: str) -> dict:
    """codec_prompt_* settings to rebuild a model the way this checkpoint was TRAINED.

    With a stored config, take the fields it has and treat every missing one as "did not
    exist yet" (see CODEC_PROMPT_FIELD_DEFAULTS). With no stored config at all the
    checkpoint predates every one of them.

    NOTE `codec_prompt_alibi_exempt` cannot be recovered for checkpoints trained before the
    ALiBi prefix-length fix: those saw only `speaker_prefix_len` exempt columns, which
    neither True nor False reproduces.
    """
    stored = read_config_from_ckpt(ckpt_path)
    if stored is None:
        return dict(CODEC_PROMPT_FIELD_DEFAULTS)
    return {
        k: stored.get(k, default)
        for k, default in CODEC_PROMPT_FIELD_DEFAULTS.items()
    }


def infer_content_source_from_ckpt(ckpt_path: str) -> str | None:
    """Guess content path from ``state_dict`` keys.

    Returns ``"sw2v"`` if the SW2V projection weights exist,
    ``"streamvoiceanon"`` if the StreamVoiceAnon token path exists,
    ``"mimi_semantic"`` if ``model.semantic_proj`` weights exist,
    ``"hubert"`` if HuBERT ``model.content_extractor`` exists, else ``None``.
    """
    stored = _stored(ckpt_path, "content_source")
    if stored:
        return str(stored)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    has_sw2v = any(
        k.startswith("model.sw2v_proj.") or k.startswith("model.sw2v_to_hidden.")
        for k in sd
    )
    has_streamvoiceanon = any(
        k.startswith("model.streamvoiceanon_to_hidden.")
        or k.startswith("model.content_extractor.code_embedding.")
        or k.startswith("model.content_extractor.out_norm.")
        for k in sd
    )
    has_mimi = any(k.startswith("model.semantic_proj.") for k in sd)
    has_hubert = any(k.startswith("model.content_extractor.") for k in sd)
    if has_sw2v:
        return "sw2v"
    if has_streamvoiceanon:
        return "streamvoiceanon"
    if has_mimi:
        return "mimi_semantic"
    if has_hubert:
        return "hubert"
    return None


def infer_speaker_conditioning_from_ckpt(ckpt_path: str) -> str | None:
    """Guess speaker conditioning from ``state_dict`` keys.

    ``crossattn`` and ``global_avg`` share the same parameter topology, so old
    checkpoints with decoder cross-attention are reported as ``"crossattn"``.
    Prefix checkpoints omit decoder cross-attention parameters.
    """
    stored = _stored(ckpt_path, "speaker_conditioning")
    if stored:
        return str(stored)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    has_decoder_cross = any(
        k.startswith("model.transformer.decoder_layers.") and ".cross_attn." in k
        for k in sd
    )
    has_encoder_layers = any(k.startswith("model.transformer.encoder_layers.") for k in sd)
    if has_decoder_cross or has_encoder_layers:
        return "crossattn"
    if any(k.startswith("model.speaker_proj.") for k in sd):
        return "prefix"
    return None


def load_model_weights_from_ckpt(
    model,
    ckpt_path: str,
    *,
    verbose: bool = True,
    log_prefix: str = "[ckpt]",
) -> tuple[list[str], list[str]]:
    """Load only ``model.*`` weights; skip ``_whisper_model.*`` baked into train ckpts."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    model_sd = {k[len("model.") :]: v for k, v in sd.items() if k.startswith("model.")}
    if not model_sd:
        raise RuntimeError(f"No model.* keys in checkpoint: {ckpt_path}")
    n_whisper = sum(1 for k in sd if k.startswith("_whisper_model."))
    if verbose and n_whisper:
        print(
            f"{log_prefix} skipping {n_whisper} _whisper_model.* keys in checkpoint "
            "(WER Whisper is loaded separately for eval)."
        )
    return model.load_state_dict(model_sd, strict=False)


def infer_content_fsq_from_ckpt(ckpt_path: str) -> tuple[int, ...] | None:
    """Return FSQ levels if the checkpoint has a content FSQ bottleneck, else None.

    The exact levels are stored as the ``model.content_fsq._levels`` buffer,
    so eval can rebuild the identical bottleneck without any CLI/config input.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    # Prefer the buffer even when a config is stored: it is the ground truth for the
    # bottleneck that was actually built (config.use_content_fsq may be off while an
    # older ckpt still carries the levels, and vice versa).
    levels = sd.get("model.content_fsq._levels")
    if isinstance(levels, torch.Tensor):
        return tuple(int(l) for l in levels.tolist())
    return None


def infer_speaker_encoder_from_ckpt(ckpt_path: str) -> str | None:
    """Speaker encoder type: from the stored config if present, else parameter topology.

    Topology CANNOT detect ``codec_prompt`` — it builds no speaker_proj,
    speaker_prefix_proj or speaker_encoder (it reuses the AR codebook embeddings), so it
    looks identical to a checkpoint with no speaker path and returns None. Checkpoints
    saved after CONFIG_CKPT_KEY was introduced answer this exactly.
    """
    stored = _stored(ckpt_path, "speaker_encoder_type")
    if stored:
        return str(stored)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    if any(k.startswith("model.spark_speaker_proj.") for k in sd):
        return "spark_global"
    if any(k.startswith("model.speaker_prefix_proj.") for k in sd):
        return "speechbrain_ecapa"
    w = sd.get("model.speaker_proj.weight")
    if isinstance(w, torch.Tensor):
        if w.dim() == 2 and w.shape[1] == 192:
            return "speechbrain_ecapa"
        return "codec"
    return None
