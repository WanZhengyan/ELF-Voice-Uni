#!/usr/bin/env python
"""Cache LibriTTS (OpenSLR 60) for the TTS rows: log-mel + char ids + t5 ids +
speaker/chapter, in the same index.pt / latents.f16 layout as
preprocess_libri.py, so LibriCache and Tagged read it unchanged.

What differs from the LibriSpeech cache, and why:

1. 24 kHz NATIVE. LibriTTS is 24 kHz, the rate Vocos was trained at, so the mel
   is taken with no resampling and keeps the 8-12 kHz band that LibriSpeech's
   16 kHz audio does not have. F5-TTS small's LibriSpeech-PC column (Table 9)
   is trained on exactly these three splits.
2. NORMALIZED TEXT. *.normalized.txt (cased, punctuated, numbers spelled out)
   lower-cased, the same char table as every other cache. A clip whose text
   has a char outside that table ([ ] { } / ...) is EXCLUDED and counted, not
   mapped to UNK.
3. NOTHING IS TRUNCATED. Clips outside 1-16 s, or whose T5 sequence (with EOS)
   exceeds 112 tokens, are excluded and counted; the model never sees audio
   paired with a text it was cut from.

Excluded counts, original/kept hours and speakers go into index.pt["filter"]
and READY.json next to it.

  python preprocess_libritts.py --split train-clean-360 --out cache_libritts_train_clean_360
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS = " abcdefghijklmnopqrstuvwxyz'\".,;:!?-()"      # = preprocess_libri.py
PAD, UNK = 0, 1


def log(*a):
    print(*a, flush=True)


def walk(root):
    """-> [(uid, wav, normalized text, speaker, chapter)]"""
    rows = []
    for spk in sorted(os.listdir(root), key=lambda x: (len(x), x)):
        sd = os.path.join(root, spk)
        if not os.path.isdir(sd):
            continue
        for ch in sorted(os.listdir(sd), key=lambda x: (len(x), x)):
            cd = os.path.join(sd, ch)
            if not os.path.isdir(cd):
                continue
            for f in sorted(os.listdir(cd)):
                if f.endswith(".normalized.txt"):
                    uid = f[:-len(".normalized.txt")]
                    w = os.path.join(cd, uid + ".wav")
                    if os.path.exists(w):
                        t = open(os.path.join(cd, f), encoding="utf-8").read().strip()
                        rows.append((uid, w, t, int(spk), int(ch)))
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(HERE, "data", "LibriTTS"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_s", type=float, default=1.0)
    ap.add_argument("--max_s", type=float, default=16.0)
    ap.add_argument("--max_t5", type=int, default=112)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    from vocos.feature_extractors import MelSpectrogramFeatures
    fe = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256,
                                n_mels=100, padding="center").to(dev).eval()
    D, FR = 100, 24000 / 256
    stoi = {c: i + 2 for i, c in enumerate(CHARS)}
    rows = walk(os.path.join(a.root, a.split))
    if a.limit:
        rows = rows[:a.limit]
    log(f"LibriTTS {a.split}: {len(rows)} clips, {len({r[3] for r in rows})} speakers")

    ex = {"duration": 0, "chars_outside_table": 0, "t5_too_long": 0, "not_24k": 0}
    hours_all = 0.0
    chunks, recs, total = [], [], 0
    t0 = time.time()
    for i, (uid, path, text, spk, ch) in enumerate(rows):
        info = torchaudio.info(path)
        dur = info.num_frames / info.sample_rate
        hours_all += dur / 3600
        low = text.lower().strip()
        if info.sample_rate != 24000:
            ex["not_24k"] += 1; continue
        if not (a.min_s <= dur <= a.max_s):
            ex["duration"] += 1; continue
        if any(c not in stoi for c in low):
            ex["chars_outside_table"] += 1; continue
        t5 = tk(low).input_ids
        if len(t5) > a.max_t5:
            ex["t5_too_long"] += 1; continue
        w, sr = torchaudio.load(path)
        x = w.to(dev)
        x = x.mean(0, keepdim=True) if x.shape[0] > 1 else x
        m = fe(x)[0].transpose(0, 1).float()              # (T, 100)
        T = m.shape[0]
        chunks.append(m.half().cpu().numpy())
        recs.append({"id": uid, "offset": total, "T": T, "spk": spk, "chap": ch, "dur": dur,
                     "chars": np.array([stoi[c] for c in low], dtype=np.int16),
                     "t5": np.array(t5, dtype=np.int32), "code": np.zeros(0, dtype=np.int16)})
        total += T
        if (i + 1) % 5000 == 0:
            el = time.time() - t0
            log(f"  {i+1}/{len(rows)}  kept {len(recs)}  {el:.0f}s ({(i+1)/el:.1f} clip/s)")

    path = os.path.join(a.out, "latents.f16")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(total, D))
    at = 0
    for c in chunks:
        mm[at:at + c.shape[0]] = c; at += c.shape[0]
    mm.flush(); del mm
    mm = np.memmap(path, dtype=np.float16, mode="r", shape=(total, D))
    samp = mm[np.random.default_rng(0).choice(total, min(400_000, total), replace=False)].astype(np.float32)
    Ts = np.array([r["T"] for r in recs]); Ks = np.array([len(r["t5"]) for r in recs])
    Ns = np.array([len(r["chars"]) for r in recs])
    spks = sorted({r["spk"] for r in recs})
    flt = {"min_s": a.min_s, "max_s": a.max_s, "max_t5_with_eos": a.max_t5, "excluded": ex,
           "clips_all": len(rows), "clips_kept": len(recs),
           "hours_all": round(hours_all, 2), "hours_kept": round(float(Ts.sum()) / FR / 3600, 2),
           "speakers_all": len({r[3] for r in rows}), "speakers_kept": len(spks)}
    meta = {"records": recs, "total_frames": total, "dim": D,
            "char_vocab_size": len(CHARS) + 2, "stoi": stoi, "t5_vocab_size": 32100,
            "audio_vocab_size": 2, "frame_rate": FR,
            "latent_mean": float(samp.mean()), "latent_std": float(samp.std()),
            "speakers": spks, "n_speakers": len(spks),
            "frames": {"mean": float(Ts.mean()), "p95": float(np.percentile(Ts, 95)), "max": int(Ts.max())},
            "t5_tokens": {"mean": float(Ks.mean()), "p99": float(np.percentile(Ks, 99)), "max": int(Ks.max())},
            "frames_per_char": float((Ts / np.maximum(Ns, 1)).mean()),
            "frames_per_t5": float((Ts / np.maximum(Ks, 1)).mean()),
            "target": "mel", "dataset": "libritts", "split": a.split, "sample_rate": 24000,
            "text": "normalized.txt, lower-cased", "filter": flt}
    with open(os.path.join(a.out, "index.pt"), "wb") as f:
        pickle.dump(meta, f)
    json.dump(flt | {"latent_mean": meta["latent_mean"], "latent_std": meta["latent_std"]},
              open(os.path.join(a.out, "READY.json"), "w"), indent=1)
    log(f"kept {len(recs)}/{len(rows)} clips, {flt['hours_kept']} / {flt['hours_all']} h, "
        f"{len(spks)} speakers; excluded {ex}; mel mean {meta['latent_mean']:+.4f} std {meta['latent_std']:.4f}")


if __name__ == "__main__":
    main()
