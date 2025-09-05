import os
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from cnn import SmallCNN
from config import config

app = FastAPI(title="Clap CNN Inference API")


def load_mono(path: Path, sr: int) -> torch.Tensor:
    wav, in_sr = torchaudio.load(str(path))
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    wav = wav.mean(dim=0, keepdim=True)
    return wav


mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.sr,
    n_fft=config.n_fft,
    hop_length=config.hop_length,
    f_min=config.fmin,
    f_max=config.fmax,
    n_mels=config.n_mels,
    power=2.0,
)
to_db = torchaudio.transforms.AmplitudeToDB(stype="power")


def pre_emphasis(x: torch.Tensor, a: float) -> torch.Tensor:
    return torch.cat([x[:, :1], x[:, 1:] - a * x[:, :-1]], dim=1)


def logmel_window(x: torch.Tensor) -> torch.Tensor:
    # x: [1, T_win]
    x = pre_emphasis(x, 0.97)
    spec = to_db(mel_tf(x))
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.unsqueeze(0)


def make_windows(x: torch.Tensor, sr: int, win_s: float, hop_s: float):
    T = x.shape[1]
    win = int(sr * win_s)
    hop = max(1, int(sr * hop_s))
    if T <= win:
        rep = math.ceil(win / T)
        xx = x.repeat(1, rep)[:, :win]
        return [(0.0, xx)]
    windows = []
    t = 0
    while t + win <= T:
        windows.append((t / sr, x[:, t: t + win]))
        t += hop
    if windows and windows[-1][0] * sr + win < T:
        windows.append(((T - win) / sr, x[:, T - win: T]))
    return windows


class ClassProb(BaseModel):
    label: str
    prob: float


class BestWindow(BaseModel):
    p_clap: float
    t: float


class TimelineEntry(BaseModel):
    t: float
    p_clap: float
    p_noise: float


class PredictResponse(BaseModel):
    filename: str
    label2idx: Dict[str, int]
    clap_idx: int
    threshold: float
    best: BestWindow
    top_k: List[ClassProb]
    timeline: List[TimelineEntry]
    num_windows: int

DEFAULT_THRESHOLD = 0.5
DEFAULT_CKPT_PATH = Path(os.getenv("CKPT_PATH", "best_clap_cnn.pt"))


def infer_path(
        wav_path: Path,
        ckpt_path: Path = DEFAULT_CKPT_PATH,
        threshold: float = DEFAULT_THRESHOLD,
):
    ckpt = torch.load(str(ckpt_path), map_location=config.device)
    model = SmallCNN(config.n_mels, 2).to(config.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    label2idx = ckpt.get("label2idx", {"noise": 0, "clap": 1})
    clap_idx = label2idx.get("clap", 1)

    x = load_mono(wav_path, config.sr)

    wins = make_windows(x, config.sr, config.duration_s, config.hop_length)

    best = {"p_clap": -1.0, "t": 0.0}
    best_probs = None
    timeline: List[TimelineEntry] = []

    with torch.no_grad():
        for t0, w in wins:
            spec = logmel_window(w).to(config.device)
            logits = model(spec)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            p_clap = float(probs[clap_idx])
            p_noise = float(probs[1 - clap_idx])

            timeline.append(
                TimelineEntry(
                    t=round(float(t0), 3), p_clap=p_clap, p_noise=p_noise
                )
            )
            if p_clap > best["p_clap"]:
                best = {"p_clap": p_clap, "t": float(t0)}
                best_probs = probs

    if best_probs is None:
        # Shouldn't happen but guard anyway
        best_probs = np.array([0.0, 0.0], dtype=np.float32)

    return {
        "label2idx": label2idx,
        "clap_idx": clap_idx,
        "threshold": threshold,
        "best": BestWindow(p_clap=best["p_clap"], t=best["t"]),
        "timeline": timeline[:500],  # match script's truncation
        "num_windows": len(timeline),
        "best_probs": best_probs,
    }


@app.post("/predict")
async def predict_file(file: UploadFile = File(...), top_k: int = 2):
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        if not DEFAULT_CKPT_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Checkpoint not found at {DEFAULT_CKPT_PATH}",
            )

        result = infer_path(tmp_path, DEFAULT_CKPT_PATH, DEFAULT_THRESHOLD)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {str(e)}"
        ) from e
    finally:
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    label2idx = result["label2idx"]
    idx2label = {v: k for k, v in label2idx.items()}
    probs = result["best_probs"]
    n_classes = probs.shape[0]
    k = min(top_k, n_classes)

    top = sorted(
        [{"label": idx2label.get(i, str(i)), "prob": float(probs[i])} for i in range(n_classes)],
        key=lambda x: x["prob"],
        reverse=True,
    )[:k]

    top_models = [ClassProb(**t) for t in top]

    return PredictResponse(
        filename=file.filename or "uploaded.wav",
        label2idx=label2idx,
        clap_idx=result["clap_idx"],
        threshold=result["threshold"],
        best=result["best"],
        top_k=top_models,
        timeline=result["timeline"],
        num_windows=result["num_windows"],
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
