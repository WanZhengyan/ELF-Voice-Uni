#!/usr/bin/env python
"""Precompute a speaker embedding per clip from a pretrained x-vector model.

Three diagnostics said the model is learning the mapping fine -- the uniform
alignment error is the same on the corpus where TTS works (1.07 word durations)
as where it fails (1.04); the exposure gap is WORSE on the working one (3.63 vs
3.54); the flow-matching objective is well optimised on both (teacher-forced
error 0.30 vs 0.26). The one diagnostic that separated them is SIM-o: the model
reaches 47% of the way from "ignored the prompt" (0.7017, a different speaker)
to the vocoder's ceiling (0.9592). The prompt is the only speaker information
in a zero-shot setting, and it is half used.

STTATTS -- the only baseline at our data scale, 155M on LibriSpeech 100h, TTS
CER 2.10-6.22 -- does not ask its backbone to infer the speaker: "the speaker
embedding vector of the target speaker extracted using x-vectors is
concatenated to the output". We have 0.368 hours per speaker; learning speaker
extraction from that is a different and much harder problem than using a vector
from a model trained on thousands of speakers.

The embedding is taken from the FIRST 3 SECONDS of the clip, because that is
exactly what the evaluation protocol hands the model as the prompt -- taking it
from the whole clip would train on information inference does not have.

The encoder here is UniSpeechSat, deliberately NOT the WavLM model that scores
SIM-o: conditioning on the same model that grades would be optimising the
metric.
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(" ", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_lj_uni"))
    ap.add_argument("--model", default="microsoft/unispeech-sat-base-plus-sv")
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech"))
    ap.add_argument("--split", default="train-clean-100")
    ap.add_argument("--libritts_root", default=os.path.join(HERE, "data", "LibriTTS"))
    ap.add_argument("--prompt_sec", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.cache, "index.pt"), "rb") as f:
        meta = pickle.load(f)
    recs = meta["records"]
    dataset = meta.get("dataset", "librispeech")
    log(f"cache {os.path.basename(args.cache)}: {len(recs)} clips, {dataset}")

    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    enc = AutoModelForAudioXVector.from_pretrained(args.model).to(dev).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    D = enc.config.xvector_output_dim
    log(f"{args.model}: frozen, x-vector dim {D}, "
        f"{sum(p.numel() for p in enc.parameters())/1e6:.1f}M params")
    log(f"  embedding taken from the first {args.prompt_sec}s, which is what the "
        f"evaluation hands the model as the prompt")
    log(f"  NOT the WavLM model used for SIM-o, so conditioning does not "
        f"optimise the metric")

    if dataset == "ljspeech":
        src_sr = 22050
        wav_path = lambda u: os.path.join(args.lj_root, "wavs", u + ".wav")
    elif dataset == "libritts":
        # preprocess_libritts.py caches: uid = spk_chap_para_sent, 24 kHz wav
        src_sr = 24000

        def wav_path(u):
            s, c = u.split("_")[:2]
            return os.path.join(args.libritts_root, meta["split"], s, c, u + ".wav")
    else:
        src_sr = 16000

        def wav_path(u):
            s, c, _ = u.split("-")
            return os.path.join(args.libri_root, args.split, s, c, u + ".flac")

    out = np.zeros((len(recs), D), dtype=np.float32)
    n16 = int(round(args.prompt_sec * 16000))
    t0 = time.time()
    for s in range(0, len(recs), args.batch):
        chunk = recs[s:s + args.batch]
        waves = []
        for r in chunk:
            # only the prompt is embedded, so decode only the prompt: a 12 s
            # flac decoded in full is four times the work for the same vector
            w, sr = torchaudio.load(wav_path(r["id"]),
                                    num_frames=int(args.prompt_sec * src_sr) + src_sr // 10)
            w = w.mean(0) if w.shape[0] > 1 else w[0]
            if sr != 16000:
                w = torchaudio.functional.resample(w, sr, 16000)
            waves.append(w[:n16].numpy())
        iv = fe(waves, sampling_rate=16000, padding=True,
                return_tensors="pt").input_values.to(dev)
        with torch.no_grad():
            e = enc(iv).embeddings.float().cpu().numpy()
        out[s:s + len(chunk)] = e
        if (s + args.batch) % (args.batch * 60) == 0:
            el = time.time() - t0
            log(f"  {s+len(chunk)}/{len(recs)}  {el:.0f}s "
                f"({(s+len(chunk))/el:.1f} clip/s)")

    path = os.path.join(args.cache, "spkemb.f32")
    out.tofile(path)
    for i, r in enumerate(recs):
        r["spk_row"] = i
    meta["spk_emb_file"] = "spkemb.f32"
    meta["spk_emb_dim"] = D
    meta["spk_emb_model"] = args.model
    meta["spk_emb_prompt_sec"] = args.prompt_sec
    with open(os.path.join(args.cache, "index.pt"), "wb") as f:
        pickle.dump(meta, f)
    log(f"wrote {path}  ({out.nbytes/2**20:.1f} MB)")

    # Does this embedding actually separate speakers on THIS data? If the
    # within-speaker similarity is not clearly above the between-speaker one,
    # the vector carries no usable identity and conditioning on it would be a
    # waste of a training run.
    nrm = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    spk = np.array([r["spk"] for r in recs])
    rng = np.random.default_rng(0)
    same, diff = [], []
    for _ in range(4000):
        i = int(rng.integers(len(recs)))
        same_pool = np.flatnonzero(spk == spk[i])
        diff_pool = np.flatnonzero(spk != spk[i])
        if len(same_pool) > 1:
            j = int(rng.choice(same_pool[same_pool != i]))
            same.append(float(nrm[i] @ nrm[j]))
        if len(diff_pool):
            k = int(rng.choice(diff_pool))
            diff.append(float(nrm[i] @ nrm[k]))
    log("")
    if same:
        log(f"同一说话人的余弦相似度  均值 {np.mean(same):.4f}  "
            f"std {np.std(same):.4f}  (n={len(same)})")
    if diff:
        log(f"不同说话人的余弦相似度  均值 {np.mean(diff):.4f}  "
            f"std {np.std(diff):.4f}  (n={len(diff)})")
    if same and diff:
        sep = (np.mean(same) - np.mean(diff)) / np.sqrt(
            0.5 * (np.var(same) + np.var(diff)) + 1e-9)
        log(f"可分性 (Cohen's d) {sep:.2f}   "
            f"<- 小于 1.0 说明这个向量区分不了说话人，别拿它去训练")
    log(f"embedding std {out.std():.4f}  mean {out.mean():+.4f}  "
        f"-> 投影层要把它映到 ELF 的 latent 尺度 (x0_t std ~0.77)")


if __name__ == "__main__":
    main()
