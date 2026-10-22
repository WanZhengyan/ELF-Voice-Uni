#!/usr/bin/env python
"""Score generated TTS with F5-TTS's own, unmodified scoring code.

Runs in .venv_f5eval (third_party/F5-TTS installed, commit pinned in the
output). Nothing here re-implements a metric; it only builds the
(gen_wav, prompt_wav, truth) list F5's functions take and averages their
per-utterance results the way their scripts do (np.mean):

  WER    f5_tts.eval.utils_eval.run_asr_wer  faster-whisper large-v3, beam 5,
         punctuation stripped, lower-cased, jiwer, mean of per-utterance WER
  SIM-o  f5_tts.eval.utils_eval.run_sim      WavLM-large ECAPA-TDNN
         (UniSpeech wavlm_large_finetune.pth), generated vs prompt recording
  UTMOS  f5_tts/eval/eval_utmos.py's loop    SpeechMOS utmos22_strong, librosa
         load at the file's own rate

libri uses get_librispeech_test (their list, their file layout), including its
eval_ground_truth=True path for the "Ground Truth" row. lj has no released F5
list, so the list comes from gen_f5_protocol.py's testset.jsonl and is passed
to the same functions.

  .venv_f5eval/bin/python score_f5_official.py --run out/f5/mix48_100000_libri --corpus libri
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F5 = os.path.join(HERE, "third_party", "F5-TTS")
WAVLM = os.path.join(F5, "checkpoints", "UniSpeech", "wavlm_large_finetune.pth")


def boot(x, n=1000, seed=0):
    x = np.asarray(x, dtype=np.float64); r = np.random.default_rng(seed)
    m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def utmos_dir(d):
    """eval_utmos.py's loop, verbatim apart from returning the scores."""
    import librosa
    import torch
    from pathlib import Path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev)
    out = {}
    for audio_path in sorted(Path(d).rglob("*.wav")):
        wav, sr = librosa.load(audio_path, sr=None, mono=True)
        wav_tensor = torch.from_numpy(wav).to(dev).unsqueeze(0)
        out[audio_path.stem] = predictor(wav_tensor, sr).item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="gen_f5_protocol.py output dir")
    ap.add_argument("--corpus", choices=["libri", "lj"], required=True)
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech", "test-clean"))
    ap.add_argument("--rows", default="gen,gt", help="gen and/or gt")
    a = ap.parse_args()
    from f5_tts.eval.utils_eval import get_librispeech_test, run_asr_wer, run_sim
    commit = subprocess.run(["git", "-C", F5, "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    metalst = os.path.join(F5, "data", "librispeech_pc_test_clean_cross_sentence.lst")
    res = {"run": a.run, "corpus": a.corpus, "f5_commit": commit,
           "asr": "faster-whisper large-v3 (F5 run_asr_wer)", "sim": "WavLM-large ECAPA-TDNN (F5 run_sim)",
           "utmos": "utmos22_strong (F5 eval_utmos.py)",
           "f5_table9": {"libri": {"ground_truth": [2.23, 0.69, 4.09], "100k": [29.5, 0.53, 3.78],
                                   "200k": [4.58, 0.59, 4.07], "400k": [2.44, 0.60, 4.11]},
                         "lj": {"ground_truth": [2.36, 0.72, 4.36], "100k": [5.64, 0.72, 4.17],
                                "200k": [2.93, 0.72, 4.18], "600k": [5.25, 0.69, 3.93]}}[a.corpus]}
    rows = [json.loads(l) for l in open(os.path.join(a.run, "testset.jsonl"))]
    for kind in a.rows.split(","):
        if a.corpus == "libri":
            (_, ts), = get_librispeech_test(metalst, os.path.join(a.run, "wavs"), [0], a.libri_root,
                                            eval_ground_truth=(kind == "gt"))
        else:
            d = os.path.join(a.run, "wavs")
            ts = [(r["gt_wav"] if kind == "gt" else os.path.join(d, r["gen_utt"] + ".wav"),
                   r["prompt_wav"], r["gen_txt"]) for r in rows]
        print(f"[{kind}] {len(ts)} utterances", flush=True)
        W = run_asr_wer((0, "en", ts, ""))
        S = run_sim((0, ts, WAVLM))
        U = utmos_dir(os.path.join(a.run, "wavs" if kind == "gen" else "gt_wavs"))
        wer = [r["wer"] * 100 for r in W]; sim = [r["sim"] for r in S]
        ut = [U[os.path.splitext(os.path.basename(g))[0]] for g, _, _ in ts]
        res[kind] = {"n": len(ts), "wer": boot(wer), "sim_o": boot(sim), "utmos": boot(ut),
                     "per_utt": [{"wav": w["wav"], "wer": w["wer"], "sim": s["sim"], "utmos": u,
                                  "truth": w["truth"], "hypo": w["hypo"]}
                                 for w, s, u in zip(W, S, ut)]}
        f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
        print(f"[{kind}] WER {f(res[kind]['wer'])}  SIM-o {f(res[kind]['sim_o'])}  "
              f"UTMOS {f(res[kind]['utmos'])}", flush=True)
    print("F5-TTS small Table 9:", res["f5_table9"], flush=True)
    out = os.path.join(a.run, "f5_official_scores.json")
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
