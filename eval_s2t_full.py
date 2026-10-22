#!/usr/bin/env python
"""S2T on ALL of LibriSpeech test-clean, ELF-S2T's protocol (arXiv 2606.10368).

ELF-S2T scores raw WER over the full test-clean, 2,620 utterances. Our
sequence holds 1,536 mel frames (16.38 s), so the cache stops at 2,423 and the
other 197 used to be dropped, which makes a number on a different, easier
subset. Here they are cut at silences into 10-16 s pieces, each piece gets its
mel and whisper-large-v3 features exactly as preprocess_libri.py /
preprocess_whisper.py make them for a whole clip, the pieces are decoded, and
their text is joined before scoring against the full transcript.

  references  the official .trans.txt, lower-cased (raw WER, no normaliser)
  WER         corpus level: total errors / total reference words
  sampling    EMA weights, Euler ODE, K steps, audio guidance w (ELF-S2T:
              SDE, K = 128, w = 2.0)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import S2T, Cfg
from train_unified import LibriCache, make_collate, wer_counts, TextEncoder
from eval_protocol import s2t_eval, boot_ci
from eval_lj import load_model


def log(*a):
    print(" ", *a, flush=True)


def cut_points(w16, lo_s=10.0, hi_s=16.0, sr=16000, min_tail_s=2.0):
    """Piece boundaries at the quietest 10 ms frame in [start+lo, start+hi],
    never leaving less than min_tail_s after the cut. Without that bound 7 of
    the 408 pieces were tails of 0.05-0.4 s (silence or a breath), which no
    training clip (1-16 s) resembles and which the x-vector encoder cannot
    even read (2026-10-20)."""
    hop = sr // 100
    e = np.array([float((w16[i:i + hop] ** 2).mean())
                  for i in range(0, len(w16) - hop + 1, hop)])
    cuts, s = [0], 0
    while (len(w16) - s) / sr > hi_s:
        a = s // hop + int(lo_s * 100)
        b = min(len(e), s // hop + int(hi_s * 100), len(w16) // hop - int(min_tail_s * 100))
        b = max(b, a + 1)
        s = (a + int(np.argmin(e[a:b]))) * hop
        cuts.append(s)
    return cuts + [len(w16)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_libri_test_clean_lv3"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech"))
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--nfe", type=int, default=128)
    ap.add_argument("--acfg", default="2.0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_s", type=float, default=16.0,
                    help="pieces never exceed this; must fit max_length")
    ap.add_argument("--out", required=True)
    # decode with a frozen standalone decoder (train_text_decoder.py) instead
    # of the backbone's CE head -- required for a model trained flow-only
    ap.add_argument("--ext_decoder", default="")
    # ode: ours. elf_sde: ELF-S2T's own sampler (SDE gamma 1.5, logit-normal
    # grid, zeroed self-cond + audio CFG at the t = 1 decode)
    ap.add_argument("--sampler", choices=["ode", "elf_sde"], default="ode")
    ap.add_argument("--sde_gamma", type=float, default=1.5)
    # ODE only: feed the last x-hat into the t = 1 decode, as before the
    # 2026-10-20 review. Training's decoder rows see zero there.
    ap.add_argument("--legacy_decode_sc", type=int, default=0)
    # smoke tests only: score a few utterances of each kind
    ap.add_argument("--limit_whole", type=int, default=0)
    ap.add_argument("--limit_long", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    model, _ = load_model(ck, device)
    ds = LibriCache(a.cache, "all")
    meta = ds.meta
    mel_stats = (meta["latent_mean"], meta["latent_std"])
    fr, wrate = meta["frame_rate"], meta["whisper_rate"]
    assert model.whisper_dim == meta["whisper_dim"], \
        f"model expects {model.whisper_dim}-d audio features, cache has {meta['whisper_dim']}"
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"], decoder_p_mean=ta["decoder_p_mean"])
    ext_dec = None
    if a.ext_decoder:
        from train_text_decoder import TextDecoder
        dk = torch.load(a.ext_decoder, map_location="cpu", weights_only=False)
        ext_dec = TextDecoder(**dk["cfg"]).to(device).eval(); ext_dec.load_state_dict(dk["model"])
    text_enc = TextEncoder(ta["text_latent_std"]).to(device)
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}

    ref = {}
    for tr in glob.glob(os.path.join(a.libri_root, a.split, "*", "*", "*.trans.txt")):
        for line in open(tr, encoding="utf-8"):
            u, _, t = line.strip().partition(" ")
            ref[u] = t.lower()
    if a.limit_whole:
        ds.recs = ds.recs[:a.limit_whole]
    have = {r["id"] for r in ds.recs}
    long_ids = sorted(u for u in ref if u not in {r["id"] for r in LibriCache(a.cache, "all").recs})
    if a.limit_long:
        long_ids = long_ids[:a.limit_long]
    if a.limit_whole or a.limit_long:
        ref = {u: t for u, t in ref.items() if u in have or u in set(long_ids)}
    log("=" * 70)
    log("S2T —— 整个 test-clean（ELF-S2T 的口径）")
    log("=" * 70)
    log(f"ckpt   {a.ckpt} @ step {ck['step']} (EMA)")
    log(f"官方 {len(ref)} 句 = cache 里 {len(have)} 句（≤16.38 s，整句解码）"
        f" + {len(long_ids)} 句长句（在静音处切成 10-{a.max_s:.0f} s 的段）")
    log(f"采样   {'Euler ODE（均匀网格）' if a.sampler == 'ode' else f'ELF-S2T 的 SDE（gamma {a.sde_gamma}，logit-normal 网格，解码步自条件置零、logits 也做 CFG）'}"
        f" {a.nfe} 步，音频引导 w = {a.acfg}，seed {a.seed}")
    log(f"出字   {'冻结的外置解码器 ' + a.ext_decoder if a.ext_decoder else '主干的交叉熵头（ELF）'}")

    # ---- the long utterances, cut and featurised like a training clip
    from vocos.feature_extractors import MelSpectrogramFeatures
    mfe = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256,
                                 n_mels=100, padding="center").to(device).eval()
    wfe = WhisperFeatureExtractor.from_pretrained(meta["whisper_model"])
    enc = (WhisperModel.from_pretrained(meta["whisper_model"], dtype=torch.float32)
           .encoder.float().to(device).eval())
    pieces, n_pieces = [], {}
    # a checkpoint trained with the x-vector block gets one for every piece too,
    # made the way preprocess_spk.py made the cache's: same encoder, the clip's
    # own first prompt_sec seconds at 16 kHz. Without it the long pieces ran
    # with the block zeroed, a state training never produced
    # (CODE_REVIEW_2026-10-20 item 3).
    xv = None
    if getattr(model, "spk_dim", 0):
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
        xv_name, xv_sec = meta["spk_emb_model"], meta["spk_emb_prompt_sec"]
        xv_fe = AutoFeatureExtractor.from_pretrained(xv_name)
        xv = AutoModelForAudioXVector.from_pretrained(xv_name).to(device).eval()
        log(f"音色   长句每段的 x-vector：{xv_name}，取该段前 {xv_sec} s（与缓存一致）")
    with torch.no_grad():
        for u in long_ids:
            s, c, _ = u.split("-")
            w, sr = torchaudio.load(os.path.join(a.libri_root, a.split, s, c, u + ".flac"))
            w = (w.mean(0) if w.shape[0] > 1 else w[0])
            w16 = torchaudio.functional.resample(w, sr, 16000).numpy()
            cuts = cut_points(w16, hi_s=a.max_s)
            n_pieces[u] = len(cuts) - 1
            for k in range(len(cuts) - 1):
                seg = w16[cuts[k]:cuts[k + 1]]
                x24 = torchaudio.functional.resample(torch.from_numpy(seg), 16000,
                                                     24000).to(device)
                mel = mfe(x24.unsqueeze(0))[0].transpose(0, 1).float().cpu().numpy()
                L = max(1, len(seg) // 160 // 2)
                f = wfe([seg], sampling_rate=16000, return_tensors="pt").input_features
                wh = enc(f.to(device)).last_hidden_state[0, :L].float().cpu().numpy()
                pc = {"mel": mel.astype(np.float16), "chars": np.array([2], np.int16),
                      "t5": np.array([1], np.int32), "spk": 0,
                      "uid": f"{u}#{k}", "wh": wh.astype(np.float16),
                      "task": S2T}
                if xv is not None:
                    # the piece's own first xv_sec s, as for every cached clip;
                    # a piece under 1 s (shorter than any training clip) takes
                    # the utterance's first xv_sec s instead -- same speaker,
                    # same recording
                    src = seg if len(seg) >= 16000 else w16
                    iv = xv_fe([src[:int(round(xv_sec * 16000))]], sampling_rate=16000,
                               padding=True, return_tensors="pt").input_values.to(device)
                    pc["spk_emb"] = xv(iv).embeddings[0].float().cpu().numpy()
                pieces.append(pc)
    del enc
    durs = [p["mel"].shape[0] / fr for p in pieces]
    log(f"长句切成 {len(pieces)} 段，段长 {min(durs):.1f}-{max(durs):.1f} s，"
        f"平均每句 {len(pieces)/max(len(long_ids),1):.1f} 段")
    assert max(durs) <= (ta["max_length"] - ta["max_text"]) / fr, "piece longer than the sequence"

    coll = make_collate(ta["max_length"], 1.0, (0.0, 0.0), rng_seed=7,
                        max_text=ta["max_text"], itos=itos, tk=tk,
                        layout=ta["layout"], whisper_rate=wrate, frame_rate=fr)
    res = {"ckpt": a.ckpt, "step": ck["step"], "n_official": len(ref),
           "n_whole": len(have), "n_long": len(long_ids), "n_pieces": len(pieces),
           "nfe": a.nfe, "seed": a.seed, "sampler": a.sampler,
           "decode_sc": "last x-hat (legacy)" if a.legacy_decode_sc else "zero (as training)",
           "spk_for_long_pieces": bool(getattr(model, "spk_dim", 0)), "runs": {}}
    for w in [float(x) for x in a.acfg.split(",")]:
        hyp = {}
        for src in (ds, pieces):
            torch.manual_seed(a.seed)
            dl = DataLoader(src, batch_size=a.batch, collate_fn=coll)
            r = s2t_eval(model, text_enc, dl, cfg, device, mel_stats, tk,
                         ta["layout"], a.nfe, w, len(src), ext_dec=ext_dec,
                         sampler=a.sampler, sde_gamma=a.sde_gamma, seed=a.seed,
                         legacy_decode_sc=bool(a.legacy_decode_sc))
            for uid, _, h in r["all"]:
                hyp[uid] = h
        joined = {}
        for u in long_ids:
            joined[u] = " ".join(hyp[f"{u}#{k}"] for k in range(n_pieces[u])).strip()
        rows = []
        for u in sorted(ref):
            h = hyp.get(u) if u in have else joined[u]
            c = wer_counts(ref[u], h.lower())
            rows.append((u, c))
        P = [(c["err"], c["n"]) for _, c in rows]
        Ps = [(c["err"], c["n"]) for u, c in rows if u in have]
        Pl = [(c["err"], c["n"]) for u, c in rows if u not in have]
        full, short, lng = boot_ci(P), boot_ci(Ps), boot_ci(Pl)
        D = {k: sum(c[k] for _, c in rows) for k in ("sub", "dele", "ins")}
        res["runs"][str(w)] = {"wer_full": full, "wer_whole_2423": short,
                               "wer_long_197": lng, "decomp": D,
                               # every (ref, hyp), so the score can be re-read
                               # with ELF-S2T's own jiwer.wer call
                               "pairs": {u: [ref[u], hyp.get(u) if u in have else joined[u]]
                                         for u in sorted(ref)},
                               "examples": [(u, ref[u], hyp.get(u) or joined.get(u))
                                            for u in long_ids[:3]]}
        log("")
        log(f"w={w}:  全部 {len(rows)} 句 WER {full[0]:.2f} [{full[1]:.2f}, {full[2]:.2f}]"
            f"   其中 ≤16 s 的 {len(Ps)} 句 {short[0]:.2f}，长句 {len(Pl)} 句 {lng[0]:.2f}")
        log(f"        错误分解 sub {D['sub']} del {D['dele']} ins {D['ins']}")
        for u in long_ids[:2]:
            log(f"        ref  {ref[u][:100]}")
            log(f"        hyp  {joined[u][:100]}")
    log("")
    log("ELF-S2T（ELF-B 105.9M，960 h，whisper-large-v3）test-clean WER 10.50")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
