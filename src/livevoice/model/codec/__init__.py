from .dac_model import DACModel
from .mimi_model import MimiCodec


def build_codec(config):
    """Instantiate the right codec from config.codec ('dac' or 'mimi')."""
    name = str(getattr(config, "codec", "dac")).lower()
    if name == "mimi":
        return MimiCodec(config)
    return DACModel(config)


__all__ = ["DACModel", "MimiCodec", "build_codec"]
