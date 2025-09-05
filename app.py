import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
CKPT_PATH = BASE / "best_clap_cnn.pt" # path to the .pt file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Clap CNN Inference (checkpoint-compatible)")

# ----------------------------- Utilities ----------------------------- #


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handle checkpoints saved with DataParallel ('module.' prefix)
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.split("module.", 1)[1]: v for k, v in sd.items()}
    return sd


def _infer_conv_channels(sd: Dict[str, torch.Tensor]) -> List[int]:
    """
    Find conv layers under 'features.*.weight' with 4D tensors and return
    their out_channels in ascending index order. E.g., [16, 32, 64].
    """
    conv_entries = []
    for k, v in sd.items():
        if not k.startswith("features."):
            continue
        if not k.endswith(".weight"):
            continue
        if v.ndim == 4:  # Conv2d weight: [out, in, kH, kW]
            try:
                idx = int(k.split(".")[1])
            except Exception:
                continue
            conv_entries.append((idx, v.shape[0]))
    conv_entries.sort(key=lambda x: x[0])
    channels = [c for _, c in conv_entries]
    if not channels:
        raise RuntimeError("Could not infer conv channels from state_dict.")
    return channels


class CompatSmallCNN(nn.Module):
    """
    Minimal CNN that matches the checkpoint's parameter names and shapes:
    features: (Conv2d -> BN -> ReLU -> MaxPool2d) repeated per block
    Then AdaptiveAvgPool2d to (1,1) and Linear to num_classes.
    """

    def __init__(self, channels: List[int], num_classes: int):
        super().__init__()
        assert len(channels) >= 1
        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _load_checkpoint_and_model(
    ckpt_path: Path,
) -> Tuple[nn.Module, dict, Optional[List[str]]]:
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(str(ckpt_path), map_location=DEVICE)
    if not isinstance(raw, dict) or "model" not in raw:
        raise RuntimeError("Expected a checkpoint dict with keys: ['model', 'cfg'].")

    cfg = raw.get("cfg", {})
    sd = raw["model"]
    sd = _strip_module_prefix(sd)

    # Infer channels from conv weights and classes from classifier weight
    channels = _infer_conv_channels(sd)
    if "classifier.weight" in sd:
        num_classes = sd["classifier.weight"].shape[0]
    else:
        num_classes = cfg.get("num_classes") or len(cfg.get("labels", [])) or 2

    model = CompatSmallCNN(channels=channels, num_classes=num_classes)
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()

    labels = cfg.get("labels")
    return model, cfg, labels


# Preprocessing builders from cfg
_resamplers: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}


def _get_resampler(src: int, dst: int):
    if src == dst:
        return None
    key = (src, dst)
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(
            orig_freq=src, new_freq=dst
        )
    return _resamplers[key]


def _build_mel_and_db(c: dict):
    sr = c.get("sr", 16000)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=c.get("n_fft", 1024),
        win_length=c.get("win_length", c.get("n_fft", 1024)),
        hop_length=c.get("hop_length", 256),
        f_min=c.get("fmin", 0),
        f_max=c.get("fmax", sr // 2),
        n_mels=c.get("n_mels", 64),
        power=c.get("power", 2.0),
        center=True,
        pad_mode="reflect",
        norm=None,
        mel_scale="htk",
    )
    db = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=c.get("top_db", 80)
    )
    return mel, db


def _wav_bytes_to_waveform(b: bytes, expected_sr: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(io.BytesIO(b))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decode error: {e}")
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    rs = _get_resampler(sr, expected_sr)
    if rs is not None:
        wav = rs(wav)
    return wav


def _waveform_to_input(
    wav: torch.Tensor, c: dict, mel_tx, db_tx
) -> torch.Tensor:
    x = mel_tx(wav)
    x = db_tx(x)

    mm, ms = c.get("mel_mean"), c.get("mel_std")
    if mm is not None and ms is not None:
        mm_t = torch.tensor(mm, device=x.device, dtype=x.dtype)
        ms_t = torch.tensor(ms, device=x.device, dtype=x.dtype)
        x = (x - mm_t) / (ms_t + 1e-6)
    else:
        x = (x - x.mean()) / (x.std(unbiased=False) + 1e-6)

    target_frames = c.get("target_frames")
    if target_frames is None and c.get("target_sec") is not None:
        target_frames = int(
            c["target_sec"] * c.get("sr", 16000) / c.get("hop_length", 256)
        )
    if target_frames is not None:
        T = x.size(-1)
        if T < target_frames:
            pad = target_frames - T
            x = torch.nn.functional.pad(x, (0, pad), value=0.0)
        elif T > target_frames:
            x = x[..., target_frames]

    if x.dim() == 3:
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected mel shape {tuple(x.shape)}")

    x = x.to(DEVICE)
    return x


def _topk_from_logits(
    logits: torch.Tensor, k: int
) -> Tuple[List[int], List[float]]:
    probs = torch.softmax(logits, dim=1)
    k = min(k, probs.size(1))
    s, i = torch.topk(probs, k=k, dim=1)
    return i[0].tolist(), [float(v) for v in s[0].tolist()]


# ------------------------------ Schemas ------------------------------ #
class PredictB64Request(BaseModel):
    audio_b64: str
    top_k: int = 2


class PredictResponse(BaseModel):
    top_indices: List[int]
    top_scores: List[float]
    top_labels: Optional[List[str]] = None


# ----------------------------- App state ----------------------------- #
model: Optional[nn.Module] = None
cfg: dict = {}
labels: Optional[List[str]] = None
mel_tx = None
db_tx = None


@app.on_event("startup")
def _startup():
    global model, cfg, labels, mel_tx, db_tx
    m, c, labs = _load_checkpoint_and_model(CKPT_PATH)
    model, cfg, labels = m, c, labs
    mel_tx, db_tx = _build_mel_and_db(cfg)


# ---------------------------- Endpoints ----------------------------- #
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "labels": labels,
        "sr": cfg.get("sr"),
        "n_mels": cfg.get("n_mels"),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...), top_k: int = 2):
    audio_bytes = await file.read()
    wav = _wav_bytes_to_waveform(audio_bytes, expected_sr=cfg["sr"])
    x = _waveform_to_input(wav, cfg, mel_tx, db_tx)
    with torch.inference_mode():
        logits = model(x)
    idx, scr = _topk_from_logits(logits, top_k)
    labs = [labels[i] for i in idx] if labels else None
    return PredictResponse(top_indices=idx, top_scores=scr, top_labels=labs)


@app.post("/predict_b64", response_model=PredictResponse)
def predict_b64(req: PredictB64Request):
    b64 = req.audio_b64
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        audio_bytes = base64.b64decode(b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"base64 decode error: {e}")
    wav = _wav_bytes_to_waveform(audio_bytes, expected_sr=cfg["sr"])
    x = _waveform_to_input(wav, cfg, mel_tx, db_tx)
    with torch.inference_mode():
        logits = model(x)
    idx, scr = _topk_from_logits(logits, req.top_k)
    labs = [labels[i] for i in idx] if labels else None
    return PredictResponse(top_indices=idx, top_scores=scr, top_labels=labs)