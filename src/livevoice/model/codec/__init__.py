from .mimi_model import MimiCodec
from .jhcodec_model import JHCodecModel


def build_codec(config):
    """Instantiate the right codec from config.codec ('mimi' or 'jhcodec')."""
    name = str(getattr(config, "codec", "mimi")).lower()
    if name == "jhcodec":
        return JHCodecModel(config)
    if name == "mimi":
        return MimiCodec(config)
    raise ValueError(f"Unknown codec {name!r}; expected 'mimi' or 'jhcodec'.")


__all__ = ["MimiCodec", "JHCodecModel", "build_codec"]
