#!/usr/bin/env python
"""Cache LibriSpeech as log-mel + char ids + t5 ids + speaker id.

One cache serves both directions of the unified model:

  TTS   chars (upsampled) -> mel          needs chars, mel, speaker
  S2T   mel -> t5 text latent             needs mel, t5 ids

Three things differ from the LJSpeech cache and each one matters.

1. SPEAKER IDS. train-clean-100 has 251 speakers, so text -> speech is
   one-to-many in speaker identity and an L2 flow with no voice prompt would
   regress to the mean speaker. The cache keeps the speaker so a prompt can be
   drawn from the same speaker (and so SIM-o becomes measurable at all, which
   it is not on the single-speaker LJSpeech run).

2. LOWERCASING BEFORE T5. LibriSpeech transcripts are upper case and t5's
   sentencepiece fragments upper case badly -- measured on one sentence,
   "THE QUICK BROWN FOX" is 10 tokens upper and 6 lower, and the upper-case
   pieces are junk. Lowercasing also makes the char vocab identical to the
   LJSpeech cache, so the two datasets share one char embedding.

3. 16 kHz SOURCE. LibriSpeech is 16 kHz flac; Vocos wants 24 kHz, so each clip
   is resampled before the mel. Using Vocos's own MelSpectrogramFeatures keeps
   n_fft/hop/padding matched to the pretrained vocoder -- any mismatch silently
   degrades the audio.
"""
import argparse, os, pickle, time

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))

# identical to preprocess_mel.py so the two caches share one char embedding
CHARS = " abcdefghijklmnopqrstuvwxyz'\".,;:!?-()"
PAD, UNK = 0, 1


def log(*a):
    print(*a, flush=True)


def walk_split(root):
    """-> [(uid, flac_path, transcript, speaker)] for one LibriSpeech split."""
    rows = []
    for spk in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, spk)
        if not os.path.isdir(spk_dir):
            continue
        for chap in sorted(os.listdir(spk_dir)):
            ch_dir = os.path.join(spk_dir, chap)
            tr = os.path.join(ch_dir, f"{spk}-{chap}.trans.txt")
            if not os.path.exists(tr):
                continue
            with open(tr, encoding="utf-8") as f:
                for line in f:
                    uid, _, text = line.strip().partition(" ")
                    p = os.path.join(ch_dir, uid + ".flac")
                    if text and os.path.exists(p):
                        rows.append((uid, p, text, int(spk)))
    return rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech"))
    ap.add_argument("--split", default="train-clean-100")
    ap.add_argument("--out", default="")
    ap.add_argument("--min_s", type=float, default=1.0)
    ap.add_argument("--max_s", type=float, default=16.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, "cache_libri_" + args.split.replace("-", "_"))
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")

    from vocos.feature_extractors import MelSpectrogramFeatures
    fe = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256,
                                n_mels=100, padding="center").to(dev).eval()
    D, FR = 100, 24000 / 256
    rs = torchaudio.transforms.Resample(16000, 24000).to(dev)
    log(f"mel: {D} bins, hop 256 @ 24 kHz -> {FR:.2f} Hz (Vocos config)")

    stoi = {c: i + 2 for i, c in enumerate(CHARS)}
    rows = walk_split(os.path.join(args.libri_root, args.split))
    log(f"{args.split}: {len(rows)} clips, "
        f"{len({r[3] for r in rows})} speakers")
    if args.limit:
        rows = rows[:args.limit]

    chunks, recs, total, skipped = [], [], 0, 0
    t0 = time.time()
    for i, (uid, path, text, spk) in enumerate(rows):
        w, sr = torchaudio.load(path)
        dur = w.shape[-1] / sr
        if not (args.min_s <= dur <= args.max_s):
            skipped += 1
            continue
        x = w.to(dev)
        x = x.mean(0, keepdim=True) if x.shape[0] > 1 else x
        if sr != 16000:                      # all of LibriSpeech is 16 kHz
            x = torchaudio.functional.resample(x, sr, 16000)
        m = fe(rs(x))[0].transpose(0, 1).float()        # (T, 100)
        T = m.shape[0]
        low = text.lower().strip()
        chunks.append(m.half().cpu().numpy())
        recs.append({
            "id": uid, "offset": total, "T": T, "spk": spk, "dur": dur,
            "chars": np.array([stoi.get(c, UNK) for c in low], dtype=np.int16),
            "t5": np.array(tk(low).input_ids, dtype=np.int32),
            "code": np.zeros(0, dtype=np.int16),       # no discrete audio target
        })
        total += T
        if (i + 1) % 2000 == 0:
            el = time.time() - t0
            log(f"  {i+1}/{len(rows)}  kept {len(recs)}  frames {total}  "
                f"{el:.0f}s ({(i+1)/el:.1f} clip/s)")

    path = os.path.join(out, "latents.f16")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(total, D))
    at = 0
    for a in chunks:
        mm[at:at + a.shape[0]] = a; at += a.shape[0]
    mm.flush(); del mm
    log(f"wrote {path} ({total*D*2/1e9:.2f} GB)")

    mm = np.memmap(path, dtype=np.float16, mode="r", shape=(total, D))
    samp = mm[np.random.default_rng(0).choice(total, min(400_000, total),
                                              replace=False)].astype(np.float32)
    lm, ls = float(samp.mean()), float(samp.std())
    Ts = np.array([r["T"] for r in recs])
    Ns = np.array([len(r["chars"]) for r in recs])
    Ks = np.array([len(r["t5"]) for r in recs])
    spks = sorted({r["spk"] for r in recs})
    meta = {
        "records": recs, "total_frames": total, "dim": D,
        "char_vocab_size": len(CHARS) + 2, "stoi": stoi,
        "t5_vocab_size": 32100, "audio_vocab_size": 2,
        "frame_rate": FR, "latent_mean": lm, "latent_std": ls,
        "speakers": spks, "n_speakers": len(spks),
        "frames": {"mean": float(Ts.mean()), "p95": float(np.percentile(Ts, 95)),
                   "p99": float(np.percentile(Ts, 99)), "max": int(Ts.max())},
        "t5_tokens": {"mean": float(Ks.mean()), "p99": float(np.percentile(Ks, 99)),
                      "max": int(Ks.max())},
        "frames_per_char": float((Ts / np.maximum(Ns, 1)).mean()),
        "frames_per_t5": float((Ts / np.maximum(Ks, 1)).mean()),
        "target": "mel", "dataset": "librispeech", "split": args.split,
    }
    with open(os.path.join(out, "index.pt"), "wb") as f:
        pickle.dump(meta, f)

    log("")
    log(f"kept {len(recs)} clips, skipped {skipped}, {total} frames, "
        f"{len(spks)} speakers")
    log(f"  latent_mean  {lm:+.5f}")
    log(f"  latent_std   {ls:.5f}")
    log(f"  frames/clip  mean {Ts.mean():.0f}  p95 {np.percentile(Ts,95):.0f}  "
        f"p99 {np.percentile(Ts,99):.0f}  max {Ts.max()}")
    log(f"  t5 tokens    mean {Ks.mean():.1f}  p99 {np.percentile(Ks,99):.0f}  "
        f"max {Ks.max()}")
    log(f"  frames/char  {meta['frames_per_char']:.2f}   "
        f"frames/t5 token {meta['frames_per_t5']:.2f}")
    # The unified sequence is mel frames + text tokens side by side, so the
    # transformer length is the sum, not the max.
    L = int(np.percentile(Ts, 95) + np.percentile(Ks, 99))
    log(f"  suggested max_length (mel p95 + t5 p99) = {L//16*16+16}")


if __name__ == "__main__":
    main()
