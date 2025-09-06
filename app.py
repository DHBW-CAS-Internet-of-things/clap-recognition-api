import logging
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
# NEU: WebSocket-Imports
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from cnn import SmallCNN
from config import config

import urllib.request

app = FastAPI(title="Clap CNN Inference API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def trigger_webhook():
    result = urllib.request.urlopen(config.webhook_url).read()
    logger.info(result)


def load_mono(path: Path, sr: int) -> torch.Tensor:
    try:
        wav, in_sr = torchaudio.load(str(path))
        if in_sr != sr:
            wav = torchaudio.functional.resample(wav, in_sr, sr)
        wav = wav.mean(dim=0, keepdim=True)
        return wav
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load audio file: {str(e)}"
        ) from e


# Hilfsfunktion: Int16-PCM-Block -> Tensor (1, T) float32 [-1,1]
def _int16_block_to_tensor_mono(block: bytes) -> torch.Tensor:
    arr = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(arr).unsqueeze(0)


# Lazy-Initialisierung des Modells (einmal pro Prozess)
_model = None
_label2idx = None
_clap_idx = None

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


def get_model():
    global _model, _label2idx, _clap_idx
    if _model is None:
        ckpt = torch.load(str(DEFAULT_CKPT_PATH), map_location=config.device)
        m = SmallCNN(config.n_mels, 2).to(config.device)
        m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        m.eval()
        _model = m
        _label2idx = ckpt.get("label2idx", {"noise": 0, "clap": 1})
        _clap_idx = _label2idx.get("clap", 1)
        logger.info("Model loaded for WebSocket streaming.")
    return _model, _label2idx, _clap_idx


def pre_emphasis(x: torch.Tensor, a: float) -> torch.Tensor:
    return torch.cat([x[:, :1], x[:, 1:] - a * x[:, :-1]], dim=1)


def logmel_window(x: torch.Tensor) -> torch.Tensor:
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


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """
    Erwartet binäre Frames mit Int16-PCM (mono, 16 kHz).
    Pro empfangenem ~1s-Block wird ein Ergebnis (p_clap/p_noise) zurückgeschickt.
    """
    await ws.accept()
    try:
        model, _, clap_idx = get_model()
        t_blocks = 0.0

        while True:
            message = await ws.receive()
            data = message.get("bytes", None)
            if data is None:
                # Textframes optional für Pings/Steuerung ignorieren
                continue

            # Optional: Minimalprüfung auf erwartete Länge (1s-Block = 32000 Bytes)
            if len(data) < 32000:
                await ws.send_json({"error": "block_too_small", "expected_bytes": 32000, "got": len(data)})
                continue

            # PCM -> Tensor -> Spec -> Modell
            x = _int16_block_to_tensor_mono(data)  # (1, T) float32
            spec = logmel_window(x).to(config.device)

            with torch.no_grad():
                logits = model(spec)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            p_clap = float(probs[clap_idx])
            p_noise = float(probs[1 - clap_idx])

            await ws.send_json({
                "t": t_blocks,
                "p_clap": p_clap,
                "p_noise": p_noise,
                "threshold": DEFAULT_THRESHOLD
            })

            if(p_clap >= config.p_clap_threshold):
                trigger_webhook()

            t_blocks += 1.0
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        try:
            await ws.send_json({"error": f"{type(e).__name__}: {str(e)}"})
        finally:
            await ws.close()


@app.post("/predict/volume")
async def predict_level(levels: List[int]):
    clap_recognized = any(l >= config.volume_level_threshold for l in levels)
    if (clap_recognized):
        trigger_webhook()
    return clap_recognized


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
