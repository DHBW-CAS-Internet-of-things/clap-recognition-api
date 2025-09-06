from dataclasses import dataclass
import numpy as np
import torch

def _get_device() -> str:
    """
    Get device name
    :return: Return device name
    """
    return (
        "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

@dataclass
class CFG:
    sr: int = 16000
    duration_s: float = 1.0
    n_mels: int = 40
    n_fft: int = 1024
    hop_length: int = 160
    fmin: int = 60
    fmax: int = 7600
    device: str = _get_device()
    volume_level_threshold: int = 1250
    webhook_url = "http://localhost:8080"
    p_clap_threshold = 0.6


config = CFG()