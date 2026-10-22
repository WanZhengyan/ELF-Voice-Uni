#!/usr/bin/env python
"""Gate before the multi-speaker run: every cache it reads must load, agree on
the conventions the trainer asserts (char table, frame rate, x-vector model),
carry what its role needs, and look like the data it claims to be.
Exit 1 on the first failure, with the reason; print a summary otherwise.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
from train_unified import LibriCache

TTS = ["cache_libritts_train_clean_100", "cache_libritts_train_clean_360", "cache_libritts_train_other_500"]
S2T = ["cache_libri_tc100_lv3", "cache_libri_train_clean_360", "cache_libri_train_other_500"]
DEV_TTS, DEV_S2T = "cache_libritts_dev_clean", "cache_libri_dev_clean_lv3"
# OpenSLR 60 (LibriTTS, before any filter): clips per split
OFFICIAL = {"train-clean-100": 33236, "train-clean-360": 116500, "train-other-500": 205044, "dev-clean": 5736}


def fail(msg):
    print("FAIL:", msg, flush=True); sys.exit(1)


ref = None
tot_h = 0.0
for c in TTS + S2T + [DEV_TTS, DEV_S2T]:
    if not os.path.exists(os.path.join(c, "DONE")) and c not in S2T:
        fail(f"{c}: no DONE marker")
    d = LibriCache(c, "all"); m = d.meta
    if ref is None:
        ref = m
    if m["stoi"] != ref["stoi"]:
        fail(f"{c}: char table differs")
    if abs(m["frame_rate"] - ref["frame_rate"]) > 1e-6:
        fail(f"{c}: frame rate {m['frame_rate']}")
    if m.get("spk_emb_model") != "microsoft/unispeech-sat-base-plus-sv" or m.get("spk_emb_dim") != 512:
        fail(f"{c}: x-vector {m.get('spk_emb_model')} {m.get('spk_emb_dim')}")
    it = d[len(d) // 2]
    if it["mel"].shape[1] != 100 or not np.isfinite(it["mel"]).all() or "spk_emb" not in it:
        fail(f"{c}: bad item")
    if not np.isfinite(d.spk[: min(2000, len(d))]).all() or np.abs(d.spk[: min(2000, len(d))]).sum(1).min() == 0:
        fail(f"{c}: x-vector rows non-finite or zero")
    T = np.array([r["T"] for r in d.recs]); h = T.sum() / m["frame_rate"] / 3600
    if T.max() > 1536:
        fail(f"{c}: clip of {T.max()} frames > 1536")
    K = max(len(r["t5"]) for r in d.recs)
    if K > 112:
        fail(f"{c}: {K} T5 tokens > 112")
    line = f"{c:34s} {len(d):7d} clips {h:7.1f} h  {m['n_speakers']:5d} spk  mel {m['latent_mean']:+.3f}/{m['latent_std']:.3f}"
    if c in TTS + [DEV_TTS]:
        f = m["filter"]
        if m.get("dataset") != "libritts" or m.get("sample_rate") != 24000:
            fail(f"{c}: not a 24 kHz LibriTTS cache")
        if f["clips_all"] != OFFICIAL[m["split"]]:
            fail(f"{c}: {f['clips_all']} clips on disk, OpenSLR says {OFFICIAL[m['split']]} -- incomplete extraction?")
        line += f"  kept {f['clips_kept']}/{f['clips_all']} ({f['hours_kept']}/{f['hours_all']} h) excluded {f['excluded']}"
        if c in TTS:
            tot_h += h
    if c in S2T + [DEV_S2T]:
        if m.get("whisper_model") != "openai/whisper-large-v3" or m.get("whisper_dim") != 1280 or "wh" not in it:
            fail(f"{c}: whisper-large-v3 features missing")
    print(line, flush=True)
# speakers: dev / test speakers must not be in any training split
tr_spk = set()
for c in TTS + S2T:
    tr_spk |= set(LibriCache(c, "all").meta["speakers"])
for c in (DEV_TTS, DEV_S2T, "cache_libri_test_clean_lv3"):
    ov = tr_spk & set(LibriCache(c, "all").meta["speakers"])
    if ov:
        fail(f"{c}: {len(ov)} speakers also in training data")
print(f"TTS LibriTTS kept total {tot_h:.1f} h; dev/test speakers disjoint from all training speakers")
print("CACHES OK")
