#!/usr/bin/env python3
import argparse, math, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import torchaudio
from dataclasses import dataclass

from cnn import SmallCNN
from config import config

def load_mono(path: Path, sr: int) -> torch.Tensor:
    wav, in_sr = torchaudio.load(str(path))
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    wav = wav.mean(dim=0, keepdim=True)
    return wav

mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.sr, n_fft=config.n_fft, hop_length=config.hop_length,
    f_min=config.fmin, f_max=config.fmax, n_mels=config.n_mels, power=2.0)
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
    win = int(sr*win_s)
    hop = max(1, int(sr*hop_s))
    if T <= win:
        rep = math.ceil(win / T)
        xx = x.repeat(1, rep)[:, :win]
        return [(0.0, xx)]
    windows = []
    t = 0
    while t + win <= T:
        windows.append((t/sr, x[:, t:t+win]))
        t += hop
    if windows and windows[-1][0]*sr + win < T:
        windows.append(((T-win)/sr, x[:, T-win:T]))
    return windows

def infer_file(wav_path: Path, ckpt_path: Path, threshold: float = 0.5, clap_idx: int = None, json_out: Path = None):
    ckpt = torch.load(str(ckpt_path), map_location=config.device)
    model = SmallCNN(config.n_mels, 2).to(config.device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    label2idx = ckpt.get("label2idx", {"noise":0, "clap":1})
    if clap_idx is None:
        clap_idx = label2idx.get("clap", 1)

    x = load_mono(wav_path, config.sr)

    wins = make_windows(x, config.sr, config.duration_s, config.hop_length)

    best = {"p_clap": -1.0, "t": 0.0}
    timeline = []

    with torch.no_grad():
        for t0, w in wins:
            spec = logmel_window(w).to(config.device)
            logits = model(spec)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            p_clap = float(probs[clap_idx])
            p_noise = float(probs[1 - clap_idx])
            timeline.append({"t": round(t0,3), "p_clap": p_clap, "p_noise": p_noise})
            if p_clap > best["p_clap"]:
                best = {"p_clap": p_clap, "t": float(t0)}

    # Ausgabe
    print(f"\nDatei: {wav_path}")
    print(f"Mapping: {label2idx}  (clap_idx={clap_idx})")
    print(f"Fenster: {config.duration_s:.2f}s, Hop: {config.hop_length:.2f}s, SR: {config.sr} Hz, n_mels={config.n_mels}")
    print(f"Max. clap-Wahrscheinlichkeit: {best['p_clap']:.3f} @ t={best['t']:.2f}s")
    verdict = "CLAP erkannt" if best["p_clap"] >= threshold else "kein Clap"
    print(f"Entscheidung (Schwelle {threshold:.2f}): {verdict}")

    if len(timeline) <= 20:
        print("\nTimeline (t[s] -> p_clap):")
        print(", ".join([f"{e['t']:.2f}:{e['p_clap']:.2f}" for e in timeline]))
    else:
        print(f"\nTimeline hat {len(timeline)} Fenster (nur Max gezeigt).")

    if json_out:
        payload = {
            "file": str(wav_path),
            "config": config.__dict__,
            "label2idx": label2idx,
            "clap_idx": clap_idx,
            "threshold": threshold,
            "best": best,
            "timeline": timeline[:500]
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON gespeichert: {json_out}")

# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Teste ein trainiertes Clap-CNN auf einem Audiofile.")
    ap.add_argument("--wav", required=True, type=Path, help="Pfad zu .wav")
    ap.add_argument("--ckpt", default="best_clap_cnn.pt", type=Path, help="Checkpoint vom Training")
    ap.add_argument("--threshold", type=float, default=0.5, help="Schwellwert für 'Clap erkannt'")
    ap.add_argument("--clap-idx", type=int, default=None, help="Index der 'clap'-Klasse (falls nicht im Checkpoint)")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional: Timeline/Ergebnis als JSON speichern")
    args = ap.parse_args()
    infer_file(args.wav, args.ckpt, args.threshold, args.clap_idx, args.json_out)
