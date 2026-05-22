"""Checkpoint inspection and partial load helpers."""
from __future__ import annotations

import torch


def infer_content_source_from_ckpt(ckpt_path: str) -> str | None:
    """Guess content path from ``state_dict`` keys.

    Returns ``"mimi_semantic"`` if ``model.semantic_proj`` weights exist,
    ``"hubert"`` if ``model.content_extractor`` exists, else ``None``.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    has_mimi = any(k.startswith("model.semantic_proj.") for k in sd)
    has_hubert = any(k.startswith("model.content_extractor.") for k in sd)
    if has_mimi:
        return "mimi_semantic"
    if has_hubert:
        return "hubert"
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
