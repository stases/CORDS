"""CORDS: discrete objects ↔ continuous density and feature fields.

Plotting, notebooks and training are imported only when requested.
"""

__version__ = "0.1.0"
__all__ = ["CORDSTransform", "FieldSamples", "DecodedSet", "FixedSigmaGMM",
           "CORDSMoleculeDenoiser", "CORDSDetector"]


def __getattr__(name):
    from importlib import import_module
    modules = {"CORDSTransform": "reconstruction", "FieldSamples": "fields",
               "DecodedSet": "fields", "FixedSigmaGMM": "gmm",
               "CORDSMoleculeDenoiser": "models", "CORDSDetector": "detection"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f".{modules[name]}", __name__), name)
