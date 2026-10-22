#!/usr/bin/env python
"""Generate TTS samples for F5-TTS's OWN scoring scripts (score_f5_official.py).

Everything a paper number depends on outside the model is taken from F5-TTS
(third_party/F5-TTS, src/f5_tts/eval/utils_eval.py, get_inference_prompt and
get_librispeech_test_clean_metainfo), line for line:

  list      libri: data/librispeech_pc_test_clean_cross_sentence.lst, ALL 1127
            (byte-identical to F5's repo copy). lj: F5 did not release its LJ
            list ("1000 samples organized with 4 to 7 seconds to infer with an
            around 3-second prompt"), so ours follows that sentence.
  text      gen_text = " " + gen_txt (metainfo); prompt_text gets a trailing
            " " when its last char is one byte; model text = prompt + gen.
  duration  use_truth_duration = False:
              total = ref_mel_len + int(ref_mel_len / bytes(prompt_text)
                                        * bytes(gen_text) / speed), speed 1
            with ref_mel_len = frames of the 24 kHz prompt at hop 256
            (center padding: len // 256 + 1).
  output    the generated span only (mel[ref_mel_len:total]) through Vocos,
            saved at 24 kHz as <gen_utt>.wav, the file layout their
            eval_librispeech_test_clean.py / eval_utmos.py read.
  sampling  the model's own: EMA, Euler ODE, NFE 32, CFG 2 (F5: 32 / 2 too).

Model-side choices (ours, printed): the prompt mel is the cached mel of the
prompt recording without F5's RMS boost (our model never saw boosted audio),
the x-vector comes from the prompt clip, and the model reads the text the way
it was trained (LibriSpeech: lower case, letters and apostrophes).

  python gen_f5_protocol.py --ckpt runs/uni_mix48/ckpt_100000.pt --corpus libri \
      --out out/f5/mix48_100000_libri
"""
import argparse
import json
import math
import os
import pickle
import re
import sys

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, Cfg
from train_unified import TextEncoder, make_collate
from sample_unified import tts_rollout
from eval_lj import load_model

HOP, SR = 256, 24000


def log(*a):
    print(" ", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", choices=["lj", "libri"], required=True)
    ap.add_argument("--list", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech", "test-clean"))
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt", type=int, default=1, help="also write the ground-truth targets as wavs")
    # before/after control for CODE_REVIEW_2026-10-20 item 1: hand Vocos the
    # reference-space mel as every result before the fix did
    ap.add_argument("--legacy_mel_ref", type=int, default=0)
    # timbre diagnostics (diag runs only; the defaults are the paper protocol).
    # "other" = the prompt row of a DIFFERENT speaker in the same subset.
    ap.add_argument("--every", type=int, default=1, help="use rows[::every]")
    ap.add_argument("--prompt_from", choices=["self", "other"], default="self")
    ap.add_argument("--xvec_from", choices=["self", "other"], default="self")
    ap.add_argument("--sway", type=float, default=None, help="F5 Sway Sampling coefficient, e.g. -1")
    # auto: joint CFG (text + prompt + x-vector dropped together) for a model
    # trained with --joint_cond_drop, whose text-dropped rows never kept the
    # prompt; the old text-only CFG for every earlier model
    ap.add_argument("--cfg_mode", choices=["auto", "text", "joint"], default="auto")
    # vocoder ceiling: the target's TRUE cached mel through Vocos, no model
    ap.add_argument("--resynth_gt", type=int, default=0)
    ap.add_argument("--no_prompt", type=int, default=0,
                    help="no prompt audio (K = 0) and no prompt text; length still from the F5 estimate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    LIB = a.corpus == "libri"
    a.list = a.list or os.path.join(HERE, "data", *(
        ("f5_eval", "librispeech_pc_test_clean_cross_sentence.lst") if LIB
        else ("lj_eval", "lj_inset_1000.lst")))
    a.cache = a.cache or os.path.join(HERE, "cache_libri_test_clean_lv3" if LIB else "cache_lj13k")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    model, meta = load_model(ck, device)
    mk = ck["meta_keys"]
    mel_stats = (mk["latent_mean"], mk["latent_std"])
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"], decoder_p_mean=ta["decoder_p_mean"])
    if a.cfg_mode == "auto":
        a.cfg_mode = "joint" if ta.get("joint_cond_drop") else "text"
    text_enc = TextEncoder(ta["text_latent_std"]).to(device)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    stoi = meta["stoi"]

    C = pickle.load(open(os.path.join(a.cache, "index.pt"), "rb"))
    by_id = {r["id"]: r for r in C["records"]}
    lat = np.memmap(os.path.join(a.cache, "latents.f16"), dtype=np.float16, mode="r",
                    shape=(C["total_frames"], C["dim"]))
    sc = mel_stats[1] / C["latent_std"]; sh = mel_stats[0] - C["latent_mean"] * sc
    spk_tab = None
    if model.spk_dim:
        spk_tab = np.memmap(os.path.join(a.cache, C["spk_emb_file"]), dtype=np.float32,
                            mode="r", shape=(len(C["records"]), C["spk_emb_dim"]))

    def wav_path(uid):
        if LIB:
            s_, c_, _ = uid.split("-")
            return os.path.join(a.libri_root, s_, c_, uid + ".flac")
        return os.path.join(a.lj_root, "wavs", uid + ".wav")

    # the text the model was TRAINED on decides how the list's text is fed:
    # LibriSpeech transcripts are letters + apostrophes only, so those models
    # read LibriSpeech-PC with punctuation stripped; a model whose TTS rows came
    # from LibriTTS (normalized.txt, punctuated) reads it punctuated, with any
    # char outside the shared table turned into a space.
    tts_src = (ck.get("data") or {}).get("tts_cache", "") or ta.get("cache", "")
    punct = LIB and ("libritts" in tts_src)
    if LIB:
        log(f"文本     {'小写，保留标点（模型的 TTS 训练文本是 LibriTTS，带标点）' if punct else '小写，只留字母和撇号（模型在 LibriSpeech 转写上训练）'}")

    def model_text(t):
        if not LIB:                    # LJ: exactly as before
            return re.sub(r"\s+", " ", t.lower()).strip()
        if punct:
            t = "".join(c if c in stoi else " " for c in t.lower())
            return re.sub(r"\s+", " ", t).strip()
        return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", t.lower())).strip()

    rows = [l.rstrip("\n").split("\t") for l in open(a.list, encoding="utf-8")][::a.every]
    spk_of = lambda u: u.split("-")[0] if LIB else "LJ"
    other = []
    for i, r in enumerate(rows):
        j = (i + len(rows) // 2) % len(rows)
        while spk_of(rows[j][0]) == spk_of(r[0]) and len(rows) > 1 and LIB:
            j = (j + 1) % len(rows)
        other.append(j)
    gen_dir = os.path.join(a.out, "wavs"); os.makedirs(gen_dir, exist_ok=True)
    log("=" * 70)
    log(f"F5-TTS 推理协议生成  {a.corpus}  {len(rows)} 对  -> {gen_dir}")
    log(f"ckpt {a.ckpt} @ {ck['step']} (EMA)  NFE {a.nfe}  CFG {a.cfg}（{a.cfg_mode}）  seed {a.seed}")
    log("=" * 70)

    cap = ta["max_length"] - ta["max_text"]
    items, trunc, kdiff = [], [], []
    for ri, (pu0, _, ptxt0, gu, _, gtxt) in enumerate(rows):
        pb = rows[other[ri]]
        pu, ptxt = (pb[0], pb[2]) if a.prompt_from == "other" else (pu0, ptxt0)
        xr = by_id[pb[0] if a.xvec_from == "other" else pu0]
        pr = by_id[pu]
        w, sr = torchaudio.load(wav_path(pu))
        # torchaudio Resample's output length: ceil(new_sr * length / orig_sr)
        n24 = math.ceil(SR * w.shape[-1] / sr) if sr != SR else w.shape[-1]
        ref_mel_len = n24 // HOP + 1                     # F5 MelSpec, center=True
        prompt_text = ptxt + " " if len(ptxt[-1].encode("utf-8")) == 1 else ptxt
        gen_text = " " + gtxt
        total = ref_mel_len + int(ref_mel_len / len(prompt_text.encode("utf-8"))
                                  * len(gen_text.encode("utf-8")) / 1.0)
        pm = np.asarray(lat[pr["offset"]:pr["offset"] + pr["T"]], dtype=np.float32) * sc + sh
        kdiff.append(pm.shape[0] - ref_mel_len)
        # the prompt occupies exactly ref_mel_len frames, as in F5
        if pm.shape[0] >= ref_mel_len:
            pm = pm[:ref_mel_len]
        else:
            pm = np.concatenate([pm, np.repeat(pm[-1:], ref_mel_len - pm.shape[0], 0)])
        Tg = total - ref_mel_len
        if total > cap:
            trunc.append((gu, total, cap)); Tg = cap - ref_mel_len
        if a.no_prompt:
            pm = pm[:0]; ref_mel_len = 0
        full = model_text(gen_text if a.no_prompt else prompt_text + gen_text)
        it = {"mel": np.concatenate([pm, np.zeros((Tg, pm.shape[1]), np.float32)]),
              "chars": np.array([stoi.get(c, 1) for c in full], dtype=np.int16),
              "t5": np.asarray(tk(full).input_ids, dtype=np.int32),
              "spk": 0, "uid": gu, "task": TTS}
        if spk_tab is not None:
            it["spk_emb"] = np.asarray(spk_tab[xr["spk_row"]])
        items.append((it, ref_mel_len, Tg, gu, (pm - sh) / sc))
    kd = np.abs(np.array(kdiff))
    log(f"prompt 帧数：缓存 vs F5 公式 len//256+1，最大差 {kd.max()} 帧（{(kd > 0).sum()} 条不同，已截/补到 F5 的帧数）")
    log(f"超过模型窗口 {cap} 帧而被截短的：{len(trunc)} 条" + (f"  例 {trunc[:3]}" if trunc else ""))

    coll = make_collate(ta["max_length"], 0.0, (0.0, 0.0), rng_seed=7, max_text=ta["max_text"],
                        itos={v: k for k, v in stoi.items()}, tk=tk, layout=ta["layout"],
                        whisper_rate=0.0, frame_rate=meta["frame_rate"])
    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    torch.manual_seed(a.seed)
    secs = []
    for s in range(0, len(items), a.batch):
        chunk = items[s:s + a.batch]
        b = coll([x[0] for x in chunk])
        b["prompt_v"].zero_()
        for i, (_, K, _, _, _) in enumerate(chunk):
            b["prompt_v"][i, :K] = 1.0
            b["n_prompt"][i] = K
        with torch.no_grad():
            mel = tts_rollout(model, b, cfg, device, text_enc, mel_stats, steps=a.nfe, cfg_w=a.cfg,
                              sway=a.sway, cfg_mode=a.cfg_mode)
            # tts_rollout returns z * sd_ref + mu_ref, the REFERENCE space the
            # prompt was mapped into (m_ref = m * sc + sh). The vocoder needs this
            # source's own log-mel, so undo that map: m = (m_ref - sh) / sc.
            # Without it a LibriSpeech output from a LJ-referenced model reached
            # Vocos as 0.802 m + 0.082 (CODE_REVIEW_2026-10-20 item 1).
            if not a.legacy_mel_ref:
                mel = (mel - sh) / sc
            if a.resynth_gt:
                for i, (_, K, Tg, gu, _) in enumerate(chunk):
                    tr = by_id[gu]
                    m = torch.from_numpy(np.asarray(lat[tr["offset"]:tr["offset"] + tr["T"]],
                                                    dtype=np.float32)).to(device)
                    g = voc.decode(m.T.unsqueeze(0))[0].cpu()
                    torchaudio.save(os.path.join(gen_dir, gu + ".wav"), g.unsqueeze(0), SR)
                    secs.append(tr["T"] * HOP / SR)
                continue
            for i, (_, K, Tg, gu, _) in enumerate(chunk):
                # the prompt span is pinned to x0, so it must come back as the
                # recording's own cached mel -- checked on every clip
                if not a.legacy_mel_ref and K > 0:
                    err = float((mel[i, :K].cpu() - torch.from_numpy(chunk[i][4])).abs().max())
                    assert err < 1e-3, f"{gu}: prompt mel not restored (max err {err})"
                g = voc.decode(mel[i:i + 1, K:K + Tg].transpose(1, 2))[0].cpu()
                torchaudio.save(os.path.join(gen_dir, gu + ".wav"), g.unsqueeze(0), SR)
                secs.append(Tg * HOP / SR)
        if (s // a.batch) % 10 == 0:
            log(f"  {s + len(chunk)}/{len(items)}")
    log(f"生成完：{len(secs)} 条，平均 {np.mean(secs):.2f} s")

    if a.gt:
        gd = os.path.join(a.out, "gt_wavs"); os.makedirs(gd, exist_ok=True)
        for _, _, _, gu, _, _ in rows:
            w, sr = torchaudio.load(wav_path(gu))
            torchaudio.save(os.path.join(gd, gu + ".wav"), w, sr)   # lossless copy as wav
        log(f"真实录音目标写到 {gd}（无损转 wav，采样率不变）")
    # what score_f5_official.py needs for a list F5's code does not ship (LJ)
    with open(os.path.join(a.out, "testset.jsonl"), "w") as f:
        for ri, (pu, _, ptxt, gu, _, gtxt) in enumerate(rows):
            pb = rows[other[ri]][0]
            used = pb if (a.prompt_from == "other" and not a.no_prompt) else pu
            f.write(json.dumps({"gen_utt": gu, "prompt_wav": wav_path(used), "gen_txt": gtxt,
                                "gt_wav": wav_path(gu), "prompt_a_wav": wav_path(pu),
                                "prompt_b_wav": wav_path(pb)}) + "\n")
    json.dump({"ckpt": a.ckpt, "step": ck["step"], "corpus": a.corpus, "list": a.list,
               "n": len(rows), "nfe": a.nfe, "cfg": a.cfg, "seed": a.seed,
               "truncated": len(trunc), "prompt_frame_maxdiff": int(kd.max()),
               "every": a.every, "prompt_from": a.prompt_from, "xvec_from": a.xvec_from,
               "no_prompt": a.no_prompt, "sway": a.sway, "resynth_gt": a.resynth_gt,
               "cfg_mode": a.cfg_mode,
               "mel_out": "reference space (legacy, pre-fix)" if a.legacy_mel_ref
                          else "source log-mel (fixed)"},
              open(os.path.join(a.out, "gen_meta.json"), "w"), indent=1)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
