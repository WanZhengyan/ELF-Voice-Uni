#!/usr/bin/env python
"""TTS on LJSpeech under F5-TTS small's protocol (arXiv 2410.06885, Table 9).

  list      data/lj_eval/lj_inset_1000.lst: 1000 in-set pairs, target 4-7 s,
            prompt a DIFFERENT ~3 s clip (build_lj_eval_list.py; F5 did not
            release its list, so ours follows the same rules)
  input     prompt mel + "prompt text target text"; the target length is
            F5's own estimate, not the true duration (use_truth_duration =
            False in their eval_infer_batch.py):
              T_gen = int(K_prompt / bytes(prompt_text) * bytes(target_text))
  sampling  EMA weights, Euler ODE, NFE 32, voice-channel CFG w (F5: 2.0)
  scoring   the generated span ONLY -- F5 cuts the prompt off before saving
  WER       Whisper-large-v3, beam 5; punctuation stripped and lower-cased,
            then the MEAN of per-utterance WER, exactly F5's run_asr_wer and
            np.mean. (F5 runs the same weights through faster-whisper; this is
            the transformers implementation.) The pooled WER is printed too.
  SIM-o     generated span vs the prompt's ORIGINAL recording, F5's pairing.
            Ours is WavLM-base-plus-sv; F5's is a WavLM-large ECAPA-TDNN, so
            the level is not digit-comparable -- the GT row brackets it.
  UTMOS     utmos22_strong on the generated span

Two reference rows on the same 1000 targets: the real recording (F5's
"Ground Truth" row reads 2.36 / 0.72 / 4.36) and the true mel through Vocos,
the ceiling no mel generator can beat.
"""
import argparse
import json
import os
import pickle
import string
import sys

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, Cfg, UnifiedELF
from train_unified import TextEncoder, make_collate, wer_counts
from sample_unified import tts_rollout
from eval_protocol import SpeakerSim, boot_ci, boot_mean_ci


def log(*a):
    print(" ", *a, flush=True)


def f5_norm(s):
    """F5's run_asr_wer for English: drop punctuation, one pass of double
    spaces, lower-case."""
    for x in string.punctuation:
        s = s.replace(x, "")
    return s.replace("  ", " ").lower().strip()


def utt_wer(ref, hyp):
    c = wer_counts(ref, hyp)
    return c["err"] / max(c["n"], 1), c


def load_model(ck, device):
    """Rebuild from the weights themselves, so single-cache and two-source
    checkpoints (whose args['cache'] may lack Whisper fields) load alike."""
    ta, sd = ck["args"], ck["ema"]
    # a released checkpoint (tools/export_hf.py) carries its own char table and
    # frame rate, so it loads without the training caches; every other
    # checkpoint reads them from the cache it was trained with, as before
    meta = ck.get("vocab") or pickle.load(open(os.path.join(ta["cache"], "index.pt"), "rb"))
    wdim = sd["whisper_proj.weight"].shape[1] if "whisper_proj.weight" in sd else 0
    sdim = sd["spk_proj.weight"].shape[1] if "spk_proj.weight" in sd else 0
    m = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                   max_length=ta["max_length"], whisper_dim=wdim,
                   use_cond=not ta.get("no_cond", 0), spk_dim=sdim,
                   spk_norm=bool(ta.get("spk_norm", 0))).to(device)
    m.load_state_dict(sd)
    return m.eval(), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    # lj:    F5 Table 9, LJSpeech column (in-set, our list under F5's rules)
    # libri: F5's own LibriSpeech-PC test-clean cross-sentence list (1127
    #        pairs, unseen speakers): the multi-speaker zero-shot protocol
    ap.add_argument("--corpus", choices=["lj", "libri"], default="lj")
    ap.add_argument("--list", default="")
    ap.add_argument("--lj_cache", default="", help="cache holding prompt/target mel")
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech", "test-clean"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", default="2.0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_ref", type=int, default=0,
                    help="skip the GT / vocoder rows (model-independent)")
    ap.add_argument("--save_wavs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    LIB = a.corpus == "libri"
    a.list = a.list or os.path.join(HERE, "data", *(
        ("f5_eval", "librispeech_pc_test_clean_cross_sentence.lst") if LIB
        else ("lj_eval", "lj_inset_1000.lst")))
    a.lj_cache = a.lj_cache or os.path.join(
        HERE, "cache_libri_test_clean_lv3" if LIB else "cache_lj13k")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    model, meta = load_model(ck, device)
    mk = ck["meta_keys"]
    mel_stats = (mk["latent_mean"], mk["latent_std"])
    fr = meta["frame_rate"]
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"], decoder_p_mean=ta["decoder_p_mean"])
    text_enc = TextEncoder(ta["text_latent_std"]).to(device)
    from transformers import AutoTokenizer, pipeline
    tk = AutoTokenizer.from_pretrained("t5-small")
    stoi = meta["stoi"]
    itos = {v: k for k, v in stoi.items()}

    lj = pickle.load(open(os.path.join(a.lj_cache, "index.pt"), "rb"))
    by_id = {r["id"]: r for r in lj["records"]}
    lat = np.memmap(os.path.join(a.lj_cache, "latents.f16"), dtype=np.float16,
                    mode="r", shape=(lj["total_frames"], lj["dim"]))
    # the model reads each source standardised by its own stats (Tagged in
    # train_unified.py): map the cache's mel onto the model's reference space
    sc = mel_stats[1] / lj["latent_std"]
    sh = mel_stats[0] - lj["latent_mean"] * sc
    spk_tab = None
    if model.spk_dim:
        spk_tab = np.memmap(os.path.join(a.lj_cache, lj["spk_emb_file"]), dtype=np.float32,
                            mode="r", shape=(len(lj["records"]), lj["spk_emb_dim"]))
    import re as _re

    def model_text(t):
        # LibriSpeech TTS rows were trained on its transcripts: lower case,
        # letters and apostrophes only. The PC list is cased and punctuated.
        if not LIB:
            return t.lower().strip()
        return _re.sub(r"\s+", " ", _re.sub(r"[^a-z' ]", " ", t.lower())).strip()
    rows = [l.rstrip("\n").split("\t") for l in open(a.list, encoding="utf-8")][:a.n]

    log("=" * 70)
    log("LibriSpeech-PC test-clean cross-sentence —— F5-TTS 的 zero-shot 协议" if LIB
        else "LJSpeech TTS —— F5-TTS small 的 Table 9 协议（in-set）")
    log("=" * 70)
    log(f"ckpt     {a.ckpt} @ step {ck['step']} (EMA)")
    log(f"列表     {a.list}: {len(rows)} 对" + ("（同一说话人的另一句当 prompt，说话人训练时没见过）" if LIB
        else "（目标 4-7 s，prompt 约 3 s 的另一句）"))
    if spk_tab is not None:
        log(f"音色     x-vector（{lj['spk_emb_model']}）取自 prompt 那句话的前 {lj['spk_emb_prompt_sec']} s")
    log(f"采样     Euler ODE NFE {a.nfe}, CFG {a.cfg}, seed {a.seed}")
    log(f"时长     F5 的估计：K_prompt / bytes(prompt) * bytes(target)，不用真实时长")

    def item(r):
        pu, _, pt, tu, _, tt = r
        pr = by_id[pu]
        pm = np.asarray(lat[pr["offset"]:pr["offset"] + pr["T"]], dtype=np.float32) * sc + sh
        K = pm.shape[0]
        sep = " " if len(pt[-1].encode("utf-8")) == 1 else ""
        Tg = int(K / len(pt.encode("utf-8")) * len(tt.encode("utf-8")))
        full = model_text(pt + sep + tt)
        cap = ta["max_length"] - ta["max_text"] - K
        if Tg > cap:
            # a model trained on a shorter window cannot hold prompt + target;
            # squeezing the text into fewer frames is a different task, so it
            # is counted and shouted about rather than silently scored
            TRUNC.append((tu, Tg, cap))
            Tg = cap
        mel = np.concatenate([pm, np.zeros((Tg, pm.shape[1]), np.float32)])
        it = {"mel": mel,
              "chars": np.array([stoi.get(c, 1) for c in full], dtype=np.int16),
              "t5": np.asarray(tk(full).input_ids, dtype=np.int32),
              "spk": 0, "uid": tu, "task": TTS}
        if spk_tab is not None:
            it["spk_emb"] = np.asarray(spk_tab[pr["spk_row"]])
        return it, K, Tg

    TRUNC = []
    coll = make_collate(ta["max_length"], 0.0, (0.0, 0.0), rng_seed=7,
                        max_text=ta["max_text"], itos=itos, tk=tk,
                        layout=ta["layout"], whisper_rate=0.0, frame_rate=fr)

    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                   device=0 if device.type == "cuda" else -1,
                   torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    gk = {"language": "en", "num_beams": 5}
    um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                        trust_repo=True).to(device).eval()
    sv = SpeakerSim(device)
    log(f"SIM-o    {sv.name} vs prompt 原始录音（F5 用 WavLM-large ECAPA-TDNN）")

    def wav16(path):
        w, sr = torchaudio.load(path)
        w = w.mean(0) if w.shape[0] > 1 else w[0]
        return torchaudio.functional.resample(w, sr, 16000).numpy()

    def lj_wav(uid):
        if LIB:
            s_, c_, _ = uid.split("-")
            return os.path.join(a.libri_root, s_, c_, uid + ".flac")
        return os.path.join(a.lj_root, "wavs", uid + ".wav")

    def score(tag, auds, refs, prompts, res):
        hyps = [x["text"] for x in asr([w.copy() for w in auds], batch_size=16,
                                       generate_kwargs=gk)]
        W, P, D = [], [], {"sub": 0, "dele": 0, "ins": 0}
        for h, rf in zip(hyps, refs):
            w, c = utt_wer(f5_norm(rf), f5_norm(h))
            W.append(w); P.append((c["err"], c["n"]))
            for k in D:
                D[k] += c[k]
        mos = [float(um(torch.from_numpy(x.copy()).float().to(device).unsqueeze(0), 16000))
               for x in auds]
        sims = [sv.sim(x, p) for x, p in zip(auds, prompts)]
        out = {"wer_mean_utt": boot_mean_ci([100 * w for w in W]),
               "wer_pooled": boot_ci(P), "sim_o": boot_mean_ci(sims),
               "utmos": boot_mean_ci(mos), "decomp": D, "n": len(W),
               "examples": [{"ref": rf, "hyp": h, "wer": round(100 * w, 1)}
                            for rf, h, w in list(zip(refs, hyps, W))[:20]]}
        res[tag] = out
        f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
        log(f"  {tag:<10} WER(逐句平均) {f(out['wer_mean_utt'])}  "
            f"WER(合并) {out['wer_pooled'][0]:.2f}  SIM-o {f(out['sim_o'])}  "
            f"UTMOS {f(out['utmos'])}  (n={len(W)})")
        log(f"             错误分解 sub {D['sub']} del {D['dele']} ins {D['ins']}")
        return hyps

    res = {"ckpt": a.ckpt, "step": ck["step"], "list": a.list, "n": len(rows),
           "nfe": a.nfe, "seed": a.seed, "asr": "openai/whisper-large-v3 (transformers, beam 5)",
           "corpus": a.corpus, "sim_model": sv.name,
           "f5_table9_libritts": {"ground_truth": [2.23, 0.69, 4.09], "100k": [29.5, 0.53, 3.78],
               "200k": [4.58, 0.59, 4.07], "400k": [2.44, 0.60, 4.11]},
           "f5_table9_lj": {
               "ground_truth": [2.36, 0.72, 4.36], "100k": [5.64, 0.72, 4.17],
               "200k": [2.93, 0.72, 4.18], "600k": [5.25, 0.69, 3.93]}}
    refs = [r[5] for r in rows]
    prompts16 = [wav16(lj_wav(r[0])) for r in rows]
    if not a.skip_ref:
        log("")
        log("参照行（与模型无关）")
        gt = [wav16(lj_wav(r[3])) for r in rows]
        score("真实录音", gt, refs, prompts16, res)
        cei = []
        for r in rows:
            tr = by_id[r[3]]
            m = torch.from_numpy(np.asarray(lat[tr["offset"]:tr["offset"] + tr["T"]],
                                            dtype=np.float32)).to(device)
            with torch.no_grad():
                w = voc.decode(m.T.unsqueeze(0))[0].cpu()
            cei.append(torchaudio.functional.resample(w, 24000, 16000).numpy())
        score("Vocos上限", cei, refs, prompts16, res)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    wav_dir = os.path.splitext(a.out)[0] + "_wavs"
    for w in [float(x) for x in a.cfg.split(",")]:
        torch.manual_seed(a.seed)
        gens, tg_s = [], []
        for s in range(0, len(rows), a.batch):
            its = [item(r) for r in rows[s:s + a.batch]]
            b = coll([x[0] for x in its])
            b["prompt_v"].zero_()
            for i, (_, K, Tg) in enumerate(its):
                b["prompt_v"][i, :K] = 1.0
                b["n_prompt"][i] = K
            with torch.no_grad():
                mel = tts_rollout(model, b, cfg, device, text_enc, mel_stats,
                                  steps=a.nfe, cfg_w=w)
                # back from the model's reference space to this source's own
                # log-mel before the vocoder (CODE_REVIEW_2026-10-20 item 1)
                mel = (mel - sh) / sc
                for i, (_, K, Tg) in enumerate(its):
                    g = voc.decode(mel[i:i + 1, K:K + Tg].transpose(1, 2))[0].cpu()
                    gens.append(torchaudio.functional.resample(g, 24000, 16000).numpy())
                    tg_s.append(Tg / fr)
        log("")
        log(f"模型 CFG w={w}：生成区平均 {np.mean(tg_s):.2f} s（估计时长）")
        nt = len({u for u, _, _ in TRUNC})
        res[f"truncated_cfg{w}"] = nt
        if nt:
            log(f"  !!! {nt} 条的 prompt+目标超过模型的 {ta['max_length'] - ta['max_text']} 帧，"
                f"目标被截短 —— 这些数字不能和 F5 比")
        TRUNC.clear()
        hyps = score(f"cfg{w}", gens, refs, prompts16, res)
        if a.save_wavs:
            os.makedirs(wav_dir, exist_ok=True)
            for i in range(min(a.save_wavs, len(gens))):
                torchaudio.save(os.path.join(wav_dir, f"cfg{w}_{rows[i][3]}.wav"),
                                torch.from_numpy(gens[i]).unsqueeze(0), 16000)
        for rf, h in list(zip(refs, hyps))[:3]:
            log(f"      ref  {rf[:90]}")
            log(f"      hyp  {h.strip()[:90]}")

    log("")
    if LIB:
        log("F5-TTS small（Table 9，LibriTTS 训练，LibriSpeech-PC 测）  WER / SIM-o / UTMOS")
        log("  真实录音 2.23 / 0.69 / 4.09   100K 步 29.5 / 0.53 / 3.78   "
            "200K 步 4.58 / 0.59 / 4.07   400K 步 2.44 / 0.60 / 4.11")
    else:
        log("F5-TTS small（Table 9，LJSpeech in-set）  WER / SIM-o / UTMOS")
        log("  真实录音 2.36 / 0.72 / 4.36   100K 步 5.64 / 0.72 / 4.17   "
            "200K 步 2.93 / 0.72 / 4.18")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
