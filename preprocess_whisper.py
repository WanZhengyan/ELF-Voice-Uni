#!/usr/bin/env python
"""Precompute frozen Whisper encoder features for the S2T audio condition.

Why this exists. ELF-S2T's 10.50% WER rests on the audio condition being a
frozen speech encoder's output, not raw mel: the audio arrives already in a
representation where phonetic content is close to linearly readable, and a
linear projector maps it into the text latent space. Feeding raw mel instead
asks the flow transformer to learn an acoustic model at the same time as it
learns the flow, and measured over five configurations it does not: token
accuracy sat at 0.7-4.9% against 0.003% chance.

The hybrid layout made this worse before it made it better. It fixed length
control (S2T WER stopped exploding: 1333% -> 157% at step 2000) but took away
the redundancy the channel layout had -- an upsampled target gave every token
~27 positions to be learned from, while the token-resolution target gives one.
Sparser target plus raw mel is the hardest combination, and token accuracy went
flat at ~1%. ELF-S2T runs the sparse target too; the difference is the encoder.

Frame rates do not match and that is handled explicitly. Whisper runs at 16 kHz
with hop 160 and its encoder halves the rate, so its output is 50 Hz. Our mel
is 24 kHz with hop 256, i.e. 93.75 Hz. The features are stored at Whisper's own
50 Hz and the index floor(j * 50/93.75) maps them onto our frames at train time
-- the same nearest-neighbour gather the text upsampling already uses, so no
new machinery and no interpolation inventing frames that were never encoded.

Whisper's feature extractor pads everything to a fixed 30 s window, so the
encoder always returns 1500 frames; only the first n_samples/160/2 are real and
only those are stored.
"""
import argparse, os, pickle, time

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_lj_uni"))
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech"))
    ap.add_argument("--split", default="train-clean-100")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--suffix", default="",
                    help="写到 whisper<suffix>.f16，让不同 encoder 的特征共存")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.cache, "index.pt"), "rb") as f:
        meta = pickle.load(f)
    recs = meta["records"]
    dataset = meta.get("dataset", "librispeech")
    log(f"cache {os.path.basename(args.cache)}: {len(recs)} clips, {dataset}")

    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    # whisper-large-v3 ships fp16 weights while the feature extractor emits
    # fp32, which fails as "Input type (float) and bias type (c10::Half)".
    # Force fp32 rather than halving the inputs: these features are computed
    # once and cached, so precision is worth more than speed here.
    enc = (WhisperModel.from_pretrained(args.model, dtype=torch.float32)
           .encoder.float().to(dev).eval())
    for p in enc.parameters():
        p.requires_grad_(False)
    D = enc.config.d_model
    log(f"{args.model}: frozen encoder, d_model {D}, "
        f"{sum(p.numel() for p in enc.parameters())/1e6:.1f}M params")
    log(f"  output rate {fe.sampling_rate/fe.hop_length/2:.0f} Hz vs our mel "
        f"{meta['frame_rate']:.2f} Hz")

    if dataset == "ljspeech":
        src_sr = 22050
        wav_path = lambda u: os.path.join(args.lj_root, "wavs", u + ".wav")
    else:
        src_sr = 16000
        def wav_path(u):
            s, c, _ = u.split("-")
            return os.path.join(args.libri_root, args.split, s, c, u + ".flac")
    rs = torchaudio.transforms.Resample(src_sr, 16000).to(dev)

    # one pass to lay out the memmap: the real length is a function of duration
    lens = []
    for r in recs:
        n16 = int(round(r["dur"] * 16000))
        lens.append(max(1, n16 // 160 // 2))
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    total = int(offs[-1])
    fname = f"whisper{args.suffix}.f16"
    path = os.path.join(args.cache, fname)
    log(f"  {total} frames total -> {total*D*2/1e9:.2f} GB")
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(total, D))

    t0 = time.time()
    for s in range(0, len(recs), args.batch):
        chunk = recs[s:s + args.batch]
        waves = []
        for r in chunk:
            w, sr = torchaudio.load(wav_path(r["id"]))
            w = w.to(dev)
            w = w.mean(0) if w.shape[0] > 1 else w[0]
            waves.append((rs(w) if sr == src_sr
                          else torchaudio.functional.resample(w, sr, 16000))
                         .cpu().numpy())
        feats = fe(waves, sampling_rate=16000, return_tensors="pt").input_features
        h = enc(feats.to(dev)).last_hidden_state.float().cpu().numpy()
        for k, r in enumerate(chunk):
            i = s + k
            L = lens[i]
            mm[offs[i]:offs[i + 1]] = h[k, :L].astype(np.float16)
        if (s + args.batch) % (args.batch * 40) == 0:
            el = time.time() - t0
            log(f"  {s+len(chunk)}/{len(recs)}  {el:.0f}s "
                f"({(s+len(chunk))/el:.1f} clip/s)")
    mm.flush(); del mm
    log(f"wrote {path}")

    for i, r in enumerate(recs):
        r["w_offset"] = int(offs[i])
        r["w_T"] = int(lens[i])
    meta["whisper_file"] = fname
    meta["whisper_dim"] = D
    meta["whisper_model"] = args.model
    meta["whisper_rate"] = fe.sampling_rate / fe.hop_length / 2
    meta["whisper_total"] = total
    with open(os.path.join(args.cache, "index.pt"), "wb") as f:
        pickle.dump(meta, f)

    mmr = np.memmap(path, dtype=np.float16, mode="r", shape=(total, D))
    samp = mmr[np.random.default_rng(0).choice(total, min(200_000, total),
                                               replace=False)].astype(np.float32)
    log("")
    log(f"frames/clip  mean {np.mean(lens):.0f}  max {max(lens)}")
    log(f"feature std  {samp.std():.4f}   mean {samp.mean():+.4f}")
    log(f"-> the projector has to map this into ELF's latent scale "
        f"(x0_t std ~0.77), so it is trained, not frozen")


if __name__ == "__main__":
    main()
