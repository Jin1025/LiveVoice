"""Checkpoint inspection and partial load helpers."""
from __future__ import annotations

import torch


def infer_content_source_from_ckpt(ckpt_path: str) -> str | None:
    """Guess content path from ``state_dict`` keys.

    Returns ``"sw2v"`` if the SW2V projection weights exist,
    ``"streamvoiceanon"`` if the StreamVoiceAnon token path exists,
    ``"mimi_semantic"`` if ``model.semantic_proj`` weights exist,
    ``"hubert"`` if HuBERT ``model.content_extractor`` exists, else ``None``.
    """
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
    levels = sd.get("model.content_fsq._levels")
    if isinstance(levels, torch.Tensor):
        return tuple(int(l) for l in levels.tolist())
    return None


def infer_speaker_encoder_from_ckpt(ckpt_path: str) -> str | None:
    """Guess speaker encoder type from checkpoint parameter topology."""
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
