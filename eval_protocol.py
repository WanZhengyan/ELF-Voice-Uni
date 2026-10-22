#!/usr/bin/env python
"""Evaluate both directions under the published protocols, not ours.

Our own evaluation drifted from the literature in ways that each moved the
number, so this script fixes the protocol and PRINTS every choice next to the
result. The protocol below is F5-TTS's (arXiv 2410.06885), which follows
Voicebox and VALL-E, plus ELF-S2T's for the recognition direction.

TTS (zero-shot, "continuation"):
  * LibriSpeech test-clean filtered to 4-10 s. F5-TTS releases such a subset
    (1127 samples of LibriSpeech-PC) exactly because earlier English models each
    used a different unlisted subset, which made comparison impossible.
  * a 3-SECOND prompt, absolute, not a fraction of the clip. A fraction makes
    the prompt short on short clips and long on long ones, so the WER mixes
    model quality with the clip-length distribution.
  * Voicebox reports both prompting tasks: cross-sentence (prompt is a
    different utterance by the same speaker) and continuation (prompt is the
    start of the target utterance). Ours is continuation, so the comparable
    Voicebox row is 2.0 WER / 0.593 SIM-o, not the 1.9 / 0.662 cross-sentence
    row that usually gets quoted.
  * WER from Whisper-large-v3, SIM-o as the cosine between WavLM speaker
    embeddings of the generated and the ground-truth target, UTMOS for
    naturalness. Vocos vocoder, EMA weights, Euler ODE.

S2T: raw WER on test-clean. ELF-S2T scores all 2,620 utterances; our cache has
2,423 because preprocessing dropped everything over 16 s, so the excluded count
is printed with the result instead of being quietly dropped.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, build_targets,
                           make_schedule, collapse, wer_pair, wer_counts,
                           audio_cond, spk_cond, T5_EOS, T5_PAD)
from sample_unified import tts_rollout


def log(*a):
    print(" ", *a, flush=True)


def boot_ci(pairs, seed=0, n=1000, lo=2.5, hi=97.5):
    """Bootstrap CI for a pooled ratio sum(num)/sum(den) over clips.

    Every comparison in this project so far was a single number against another
    single number, and the sampler alone has a measured spread of +-2.58 WER
    across seeds. Without an interval there is no way to tell a real difference
    from a draw, so two settings whose intervals overlap must not be ranked.
    """
    if not pairs:
        return None, None, None
    a = np.asarray(pairs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    num = a[:, 0][idx].sum(1)
    den = np.maximum(a[:, 1][idx].sum(1), 1e-9)
    r = 100.0 * num / den
    point = 100.0 * a[:, 0].sum() / max(a[:, 1].sum(), 1e-9)
    return point, float(np.percentile(r, lo)), float(np.percentile(r, hi))


def boot_mean_ci(vals, seed=0, n=1000, lo=2.5, hi=97.5):
    if not vals:
        return None, None, None
    a = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    m = a[idx].mean(1)
    return float(a.mean()), float(np.percentile(m, lo)), float(np.percentile(m, hi))


# ---------------------------------------------------------------- SIM-o
class SpeakerSim:
    """Cosine similarity between WavLM speaker embeddings.

    F5-TTS and Voicebox use a WavLM-large speaker-verification model. The
    public checkpoint here is WavLM-base-plus-sv, so SIM-o is comparable in
    trend but not digit-for-digit with their table; the model name is printed
    with the number so that is visible rather than implied.
    """

    def __init__(self, device, name="microsoft/wavlm-base-plus-sv"):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.m = WavLMForXVector.from_pretrained(name).to(device).eval()
        self.device = device
        self.name = name

    @torch.no_grad()
    def emb(self, wav_16k):
        x = self.fe(wav_16k, sampling_rate=16000,
                    return_tensors="pt").input_values.to(self.device)
        return self.m(x).embeddings[0].float()

    def sim(self, a_16k, b_16k):
        ea, eb = self.emb(a_16k), self.emb(b_16k)
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=0))


# ---------------------------------------------------------------- S2T
@torch.no_grad()
def s2t_eval(model, text_enc, loader, cfg, device, mel_stats, tk, layout,
             steps, acfg, n_max, ext_dec=None, sampler="ode", sde_gamma=1.5,
             seed=0, legacy_decode_sc=False):
    """sampler="ode": ours (uniform grid, Euler, CE head on the last x-hat).
    sampler="elf_sde": ELF-S2T's own _sample_batch (elf_s2t/eval_utils.py), so
    the S2T number is read under exactly their recipe: a logit-normal time grid
    redrawn per batch, an SDE step that re-noises z back to alpha*t before each
    forward (alpha = 1 - gamma*h), a final ODE step, then the t = 1 decode with
    the self-conditioning input ZEROED -- which is also what our decoder rows
    see in training -- and audio CFG applied to the decode logits too.
    Both samplers decode with the self-conditioning input zeroed, as training's
    decoder rows do ((1 - dv) zeroes sc_t and sc_v); legacy_decode_sc=True
    restores the old ODE behaviour (last x-hat fed in) to reproduce numbers
    reported before CODE_REVIEW_2026-10-20 item 2."""
    err = ln = tok = tokn = seen = 0
    pairs, every = [], []
    gen = torch.Generator(device=device).manual_seed(seed)
    for b in loader:
        if seen >= n_max:
            break
        chars = b["chars"].to(device); valid = b["valid"].to(device)
        v_act = b["v_act"].to(device).unsqueeze(-1)
        t_act = b["t_act"].to(device).unsqueeze(-1)
        p_v, p_t = b["prompt_v"].to(device), b["prompt_t"].to(device)
        task = b["task"].to(device)
        x0_t, x0_v = build_targets(b, text_enc, device, mel_stats)
        ac, am = audio_cond(b, device)
        se = spk_cond(b, device)
        B = x0_t.shape[0]
        ns = cfg.denoiser_noise_scale
        ac_u = torch.zeros_like(ac) if ac is not None else None
        z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
        z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
        xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
        sc = torch.ones(B, device=device)

        def fwd(zt, zv, pt_, pv_, t):
            xh_t, xh_v, _ = model(zt, zv, pt_, pv_, chars, t, task,
                                  attention_mask=valid, self_cond_cfg_scale=sc,
                                  audio_cond=ac, audio_cond_mask=am,
                              spk_emb=se)
            if acfg != 1.0:
                u_t, _, _ = model(zt, torch.zeros_like(zv), pt_,
                                  torch.zeros_like(pv_), chars, t, task,
                                  attention_mask=valid,
                                  self_cond_cfg_scale=sc,
                                  audio_cond=ac_u, audio_cond_mask=am,
                              spk_emb=se)
                xh_t = u_t + acfg * (xh_t - u_t)
            return xh_t, xh_v

        if sampler == "ode":
            ts = make_schedule(steps, cfg, device)
            for i in range(steps):
                t = torch.full((B,), float(ts[i]), device=device)
                xh_t, xh_v = fwd(z_t, z_v, xp_t, xp_v, t)
                h = float(ts[i + 1]) - float(ts[i])
                z_t = restore(z_t + h * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
                z_v = restore(z_v + h * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
                xp_t = restore(xh_t, x0_t, p_t) * t_act
                xp_v = restore(xh_v, x0_v, p_v) * v_act
            dec_pt, dec_pv = ((xp_t, xp_v) if legacy_decode_sc
                              else (torch.zeros_like(xp_t), torch.zeros_like(xp_v)))
        elif sampler == "elf_sde":
            zz = (torch.randn(steps - 1, generator=gen, device=device)
                  * cfg.denoiser_p_std + cfg.denoiser_p_mean)
            ts = torch.cat([torch.zeros(1, device=device), torch.sort(torch.sigmoid(zz)).values,
                            torch.ones(1, device=device)])
            for i in range(steps - 1):
                tv, tn = float(ts[i]), float(ts[i + 1])
                alpha = max(0.0, min(1.0, 1.0 - sde_gamma * (tn - tv)))
                tb = alpha * tv
                e_t = torch.randn(z_t.shape, generator=gen, device=device) * ns
                e_v = torch.randn(z_v.shape, generator=gen, device=device) * ns
                zb_t = restore(alpha * z_t + (1 - alpha) * e_t, x0_t, p_t) * t_act
                zb_v = restore(alpha * z_v + (1 - alpha) * e_v, x0_v, p_v) * v_act
                t = torch.full((B,), tb, device=device)
                xh_t, xh_v = fwd(zb_t, zb_v, xp_t, xp_v, t)
                z_t = restore(zb_t + (tn - tb) * net_to_v(xh_t, zb_t, t), x0_t, p_t) * t_act
                z_v = restore(zb_v + (tn - tb) * net_to_v(xh_v, zb_v, t), x0_v, p_v) * v_act
                xp_t = restore(xh_t, x0_t, p_t) * t_act
                xp_v = restore(xh_v, x0_v, p_v) * v_act
            tv = float(ts[-2]); t = torch.full((B,), tv, device=device)
            xh_t, xh_v = fwd(z_t, z_v, xp_t, xp_v, t)
            z_t = restore(z_t + (1.0 - tv) * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
            z_v = restore(z_v + (1.0 - tv) * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
            dec_pt, dec_pv = torch.zeros_like(xp_t), torch.zeros_like(xp_v)
        else:
            raise ValueError(sampler)
        t1 = torch.ones(B, device=device)
        _, _, logits = model(z_t, z_v, dec_pt, dec_pv, chars, t1, task,
                             attention_mask=valid, self_cond_cfg_scale=sc,
                             decoder_step_active=True,
                             audio_cond=ac, audio_cond_mask=am,
                              spk_emb=se)
        if sampler == "elf_sde" and acfg != 1.0:
            _, _, lu = model(z_t, torch.zeros_like(z_v), dec_pt, dec_pv, chars, t1, task,
                             attention_mask=valid, self_cond_cfg_scale=sc,
                             decoder_step_active=True,
                             audio_cond=ac_u, audio_cond_mask=am,
                              spk_emb=se)
            logits = acfg * logits + (1.0 - acfg) * lu
        pred = logits.argmax(-1)
        if ext_dec is not None:
            # Multimodal-Flow style: a frozen decoder reads the generated
            # token-region latent alone; the backbone's CE head is not used
            s0 = int(b["text_start"][0]); Lt = b["t5_dense"].shape[1]
            pred = pred.clone()
            pred[:, s0:s0 + Lt] = ext_dec(z_t[:, s0:s0 + Lt]).argmax(-1)
        for i in range(B):
            n = int(b["n_text"][i]); st = int(b["text_start"][i])
            ref_ids = [x for x in b["t5_dense"][i, :n].tolist()
                       if x not in (T5_EOS, T5_PAD)]
            if layout == "channel":
                hyp_ids = collapse(pred[i].tolist())
            else:
                raw = pred[i, st:st + b["t5_dense"].shape[1]].tolist()
                if T5_EOS in raw:
                    raw = raw[:raw.index(T5_EOS)]
                hyp_ids = [x for x in raw if x != T5_PAD]
            ref = tk.decode(ref_ids, skip_special_tokens=True)
            hyp = tk.decode(hyp_ids, skip_special_tokens=True)
            e, l = wer_pair(ref, hyp)
            err += e; ln += l
            k = min(len(ref_ids), len(hyp_ids))
            tok += sum(1 for j in range(k) if ref_ids[j] == hyp_ids[j])
            tokn += len(ref_ids)
            if len(pairs) < 5:
                pairs.append((ref, hyp))
            # every hypothesis, keyed by clip: a chunked long utterance is
            # scored only after its pieces are joined back together
            every.append((b["uid"][i], ref, hyp))
        seen += B
    return dict(wer=100.0 * err / max(ln, 1), tok_acc=tok / max(tokn, 1),
                n=seen, pairs=pairs, all=every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="libri_tc100_uni_ddp")
    # a kept milestone (runs/<run>/ckpt_100000.pt) instead of the rolling ckpt.pt
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--cache", default=os.path.join(HERE, "cache_libri_test_clean"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech"))
    ap.add_argument("--libri_split", default="test-clean")
    ap.add_argument("--dur_lo", type=float, default=4.0)
    ap.add_argument("--dur_hi", type=float, default=10.0)
    ap.add_argument("--prompt_sec", type=float, default=3.0)
    ap.add_argument("--tts_n", type=int, default=300)
    ap.add_argument("--s2t_n", type=int, default=2423)
    ap.add_argument("--nfe", type=int, default=32)
    # SDE churn. Re-noising is the standard remedy for the exposure gap in
    # diffusion models, and this knob existed but had never been turned on:
    # every number in this project so far came from gamma=0, a pure ODE.
    ap.add_argument("--gamma", type=float, default=0.0)
    # The decisive test for "does the model use the text at all": run the
    # same clean protocol with the text zeroed. If the gen-only WER barely
    # moves, the mel loss is being satisfied by the acoustic envelope the
    # prompt already supplies, and no amount of capacity or data fixes that
    # -- the objective has to force text dependence.
    ap.add_argument("--drop_text", type=int, default=0)
    ap.add_argument("--tts_cfg", default="2.0")
    ap.add_argument("--s2t_acfg", default="1.0")
    ap.add_argument("--s2t_nfe", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_tts", type=int, default=0)
    ap.add_argument("--skip_s2t", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "protocol.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or os.path.join(HERE, "runs", args.run, "ckpt.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]

    log("=" * 68)
    log("协议对齐评测 —— 每一项选择都和结果一起打印")
    log("=" * 68)
    log(f"ckpt            {args.run} @ step {ck['step']} (EMA 权重)")
    log(f"训练数据        {os.path.basename(ta['cache'])}, "
        f"max_length {ta['max_length']} ({ta['max_length'] - ta['max_text']} mel "
        f"+ {ta['max_text']} text)")
    log(f"训练 prompt     prompt_lo {ta['prompt_lo']} ~ prompt_hi {ta['prompt_hi']} "
        f"(比例，不是秒)")
    log(f"采样            NFE {args.nfe}   gamma(SDE churn) {args.gamma}"
        f"   TTS CFG {args.tts_cfg}")
    if args.drop_text:
        log("文本            【已清零】—— 这一跑在测「模型到底用不用文本」")

    ds_all = LibriCache(args.cache, "all")
    meta = ds_all.meta
    fr = meta["frame_rate"]
    n_official = len(
        [p for p in os.listdir(os.path.join(args.libri_root, args.libri_split))
         if not p.startswith(".")]) if os.path.isdir(
             os.path.join(args.libri_root, args.libri_split)) else 0
    import glob
    n_flac = len(glob.glob(os.path.join(args.libri_root, args.libri_split,
                                        "*", "*", "*.flac")))
    log("")
    log(f"官方 {args.libri_split} 的 flac 数     {n_flac}")
    log(f"我们 cache 里的条数            {len(ds_all)}"
        f"   —— 少 {n_flac - len(ds_all)} 条，因为预处理 --max_s 16.0 丢掉了 >16s 的")
    log(f"  cache 内时长 {min(r['dur'] for r in ds_all.recs):.2f}s "
        f"~ {max(r['dur'] for r in ds_all.recs):.2f}s")

    mel_stats = (meta["latent_mean"], meta["latent_std"])
    itos = {v: k for k, v in meta["stoi"].items()}
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"],
              decoder_p_mean=ta["decoder_p_mean"])
    # a two-source checkpoint's args["cache"] is the TTS cache (no Whisper
    # fields); its own "data" record says what the S2T rows were trained with
    D = ck.get("data") or {}
    wdim = D.get("whisper_dim", meta.get("whisper_dim", 0)) if ta.get("whisper") else 0
    wrate = D.get("whisper_rate", meta.get("whisper_rate", 0.0)) if ta.get("whisper") else 0.0
    model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                       max_length=ta["max_length"], whisper_dim=wdim,
                       use_cond=not ta.get("no_cond", 0),
                       spk_dim=(meta.get("spk_emb_dim", 0)
                                if ta.get("spk") else 0),
                       # trained with --spk_norm: without it the strict load
                       # fails on spk_norm.weight (CODE_REVIEW 12.1)
                       spk_norm=bool(ta.get("spk_norm", 0))).to(device)
    model.load_state_dict(ck["ema"]); model.eval()
    text_enc = TextEncoder(ta["text_latent_std"]).to(device)
    res = {"run": args.run, "step": ck["step"],
           "n_flac_official": n_flac, "n_cache": len(ds_all),
           "excluded_over_16s": n_flac - len(ds_all)}

    # ------------------------------------------------------------ S2T
    if not args.skip_s2t:
        log("")
        log("-" * 68)
        log(f"S2T  —— 整个 cache（无时长筛选），{args.s2t_nfe} 步 Euler ODE")
        log("-" * 68)
        coll = make_collate(ta["max_length"], 1.0, (0.0, 0.0), rng_seed=7,
                            max_text=ta["max_text"], itos=itos, tk=tk,
                            layout=ta["layout"], whisper_rate=wrate,
                            frame_rate=fr)
        res["s2t"] = {}
        for w in [float(x) for x in args.s2t_acfg.split(",")]:
            torch.manual_seed(args.seed)
            dl = DataLoader(ds_all, batch_size=args.batch, collate_fn=coll)
            r = s2t_eval(model, text_enc, dl, cfg, device, mel_stats, tk,
                         ta["layout"], args.s2t_nfe, w, args.s2t_n)
            res["s2t"][str(w)] = {k: v for k, v in r.items() if k != "pairs"}
            log(f"  ACFG w={w:<4}  WER {r['wer']:6.2f}%   token acc "
                f"{r['tok_acc']*100:5.2f}%   (n={r['n']} / 官方 {n_flac})")
            for ref, hyp in r["pairs"][:2]:
                log(f"      ref  {ref[:88]}")
                log(f"      hyp  {hyp[:88]}")

    # ------------------------------------------------------------ TTS
    if not args.skip_tts:
        log("")
        log("-" * 68)
        log(f"TTS  —— 文献协议：{args.dur_lo}-{args.dur_hi}s 子集，"
            f"{args.prompt_sec}s 绝对 prompt（continuation）")
        log("-" * 68)
        ds = LibriCache(args.cache, "all", dur_range=(args.dur_lo, args.dur_hi))
        lo, hi, before, after = ds.dur_filter
        log(f"  时长筛选 {lo}-{hi}s: {before} -> {after} 条"
            f"   (F5-TTS 发布的 LibriSpeech-PC 4-10s 子集是 1127 条)")
        log(f"  注意：>16s 的截断不触及 {lo}-{hi}s 区间，所以这个子集没被污染")

        from vocos import Vocos
        voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
        from transformers import pipeline
        from transformers.models.whisper.english_normalizer import \
            EnglishTextNormalizer
        asr = pipeline("automatic-speech-recognition",
                       model="openai/whisper-large-v3",
                       device=0 if device.type == "cuda" else -1,
                       torch_dtype=torch.float16 if device.type == "cuda"
                       else torch.float32)
        norm = EnglishTextNormalizer({})
        try:
            um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                trust_repo=True).to(device).eval()
        except Exception as e:
            log(f"  UTMOS 不可用: {e}"); um = None
        try:
            sv = SpeakerSim(device)
            log(f"  SIM-o 用 {sv.name}（文献用 WavLM-large 微调的 SV 模型，"
                f"所以 SIM-o 趋势可比、数字不逐位可比）")
        except Exception as e:
            log(f"  SIM-o 不可用: {e}"); sv = None

        # A SIM-o of 0.82 means nothing without the two numbers that bracket it:
        # what a DIFFERENT speaker scores (ignore the prompt entirely) and what
        # the ceiling scores (true mel through the same vocoder). Voicebox's
        # 0.593 is only interpretable against their own bracket, so report ours.
        by_spk = {}
        uid2spk = {}
        for r in ds.recs:
            by_spk.setdefault(r["spk"], []).append(r["id"])
            uid2spk[r["id"]] = r["spk"]
        spks = sorted(by_spk)
        rng_other = np.random.default_rng(0)

        def other_speaker_clip(spk):
            cand = [x for x in spks if x != spk]
            if not cand:
                return None
            o = cand[int(rng_other.integers(len(cand)))]
            ids = by_spk[o]
            return ids[int(rng_other.integers(len(ids)))]

        id2text = {}
        for tr in glob.glob(os.path.join(args.libri_root, args.libri_split,
                                         "*", "*", "*.trans.txt")):
            with open(tr, encoding="utf-8") as f:
                for line in f:
                    uid, _, txt = line.strip().partition(" ")
                    id2text[uid] = txt
        log(f"  参考转录 {len(id2text)} 条（纯 LibriSpeech，非 PC：无标点/大小写）")

        coll_t = make_collate(ta["max_length"], 0.0, (0.0, 0.0), rng_seed=7,
                              max_text=ta["max_text"], itos=itos, tk=tk,
                              layout=ta["layout"], whisper_rate=wrate,
                              frame_rate=fr, prompt_sec=args.prompt_sec)
        res["tts"] = {}
        for w in [float(x) for x in args.tts_cfg.split(",")]:
            torch.manual_seed(args.seed)
            dl = DataLoader(ds, batch_size=args.batch, collate_fn=coll_t)
            E = {"gen": [0, 0], "ceil": [0, 0], "orig": [0, 0],
                 "gen_only": [0, 0]}
            DEC = {"sub": 0, "dele": 0, "ins": 0}
            PAIRS = []
            MOS = {"gen": [], "ceil": [], "orig": [], "gen_only": []}
            SIM, SIM_FULL = [], []
            SIM_CEIL, SIM_FLOOR = [], []
            pk = []
            seen = 0
            for b in dl:
                if seen >= args.tts_n:
                    break
                mel = tts_rollout(model, b, cfg, device, text_enc, mel_stats,
                                  steps=args.nfe, gamma=args.gamma, cfg_w=w,
                                  drop_text=bool(args.drop_text))
                pk += b["n_prompt"].tolist()
                for i in range(mel.shape[0]):
                    uid = b["uid"][i] if "uid" in b else None
                    if uid is None:
                        continue
                    T = int(b["v_act"][i].sum())
                    spk, chap, _ = uid.split("-")
                    path = os.path.join(args.libri_root, args.libri_split,
                                        spk, chap, uid + ".flac")
                    if not os.path.exists(path) or uid not in id2text:
                        continue
                    ref = norm(id2text[uid])
                    wv, sr = torchaudio.load(path)
                    wv = wv.mean(0) if wv.shape[0] > 1 else wv[0]
                    w16 = torchaudio.functional.resample(wv, sr, 16000).numpy()
                    # The rollout holds the prompt positions at their CLEAN
                    # value (unified.restore), so mel[:K] is the ground-truth
                    # mel, not the model's output. Scoring the whole utterance
                    # therefore measures 3 s of real audio alongside the
                    # generation: it inflated SIM-o to 0.96 (Voicebox reports
                    # 0.593) and flattered WER. Score the generated span only.
                    K = int(b["n_prompt"][i])
                    g_full = voc.decode(mel[i:i + 1, :T].transpose(1, 2))[0].cpu()
                    g_gen = voc.decode(
                        mel[i:i + 1, K:T].transpose(1, 2))[0].cpu()
                    g16 = torchaudio.functional.resample(
                        g_full, 24000, 16000).numpy()
                    gen16 = torchaudio.functional.resample(
                        g_gen, 24000, 16000).numpy()
                    # ceiling: the TRUE mel through the same vocoder, the bound
                    # no generation can beat
                    tm = (torch.from_numpy(
                        np.asarray(b["mel"][i, :T])).to(device).unsqueeze(0))
                    c = voc.decode(tm.transpose(1, 2))[0].cpu()
                    c16 = torchaudio.functional.resample(c, 24000, 16000).numpy()
                    # reference words that start after the prompt, from
                    # Whisper word timestamps on the REAL recording -- the only
                    # way to score the generated span without a forced aligner
                    t_cut = K / fr
                    try:
                        wt = asr(w16.copy(), return_timestamps="word",
                                 generate_kwargs={"language": "en"})
                        # `ch`, not `c`: `c` is the ceiling waveform five
                        # lines up, and shadowing it here is a trap even though
                        # a comprehension has its own scope
                        tail = " ".join(
                            ch["text"] for ch in wt.get("chunks", [])
                            if ch.get("timestamp")
                            and ch["timestamp"][0] is not None
                            and ch["timestamp"][0] >= t_cut)
                        ref_tail = norm(tail)
                    except Exception:
                        ref_tail = ""
                    if ref_tail:
                        hyp_tail = norm(asr(gen16.copy(), generate_kwargs={
                            "language": "en"})["text"])
                        wc = wer_counts(ref_tail, hyp_tail)
                        E["gen_only"][0] += wc["err"]
                        E["gen_only"][1] += wc["n"]
                        for kk in ("sub", "dele", "ins"):
                            DEC[kk] += wc[kk]
                        PAIRS.append((wc["err"], wc["n"]))
                        if um is not None:
                            a = torch.from_numpy(gen16.copy()).float().to(device)
                            MOS["gen_only"].append(float(um(a.unsqueeze(0), 16000)))
                    for tag, aud in (("gen", g16), ("ceil", c16), ("orig", w16)):
                        hyp = norm(asr(aud.copy(), generate_kwargs={
                            "language": "en"})["text"])
                        e, l = wer_pair(ref, hyp)
                        E[tag][0] += e; E[tag][1] += l
                        if um is not None:
                            a = torch.from_numpy(aud.copy()).float().to(device)
                            MOS[tag].append(float(um(a.unsqueeze(0), 16000)))
                    if sv is not None:
                        # the generated span against the SAME span of the real
                        # recording: comparing full-to-full would be measuring
                        # the copied prompt
                        t_gen = voc.decode(
                            tm[:, K:T].transpose(1, 2))[0].cpu()
                        t16 = torchaudio.functional.resample(
                            t_gen, 24000, 16000).numpy()
                        SIM.append(sv.sim(gen16, t16))
                        SIM_FULL.append(sv.sim(g16, w16))
                        # ceiling: the vocoder's own cost on speaker identity
                        SIM_CEIL.append(sv.sim(t16, w16))
                        # floor: a different speaker entirely. This is what
                        # "ignored the prompt" scores, and it is the number that
                        # decides whether 0.82 is a result or a coincidence.
                        # the collate does not carry spk, so resolve it from
                        # the record. Passing a sentinel instead would let the
                        # "other speaker" be the SAME speaker and quietly lift
                        # the floor, which is the one number that must be honest.
                        ou = other_speaker_clip(uid2spk[uid])
                        if ou is not None:
                            osp, och, _ = ou.split("-")
                            op = os.path.join(args.libri_root, args.libri_split,
                                              osp, och, ou + ".flac")
                            if os.path.exists(op):
                                ow, osr = torchaudio.load(op)
                                ow = ow.mean(0) if ow.shape[0] > 1 else ow[0]
                                o16 = torchaudio.functional.resample(
                                    ow, osr, 16000).numpy()
                                SIM_FLOOR.append(sv.sim(o16, w16))
                    seen += 1
                    if seen >= args.tts_n:
                        break
            pm = float(np.mean(pk)) / fr if pk else 0.0
            row = {"n": seen, "prompt_sec_actual": pm,
                   "sim_o": float(np.mean(SIM)) if SIM else None,
                   "sim_o_contaminated": (float(np.mean(SIM_FULL))
                                          if SIM_FULL else None),
                   "sim_ceiling": (float(np.mean(SIM_CEIL))
                                   if SIM_CEIL else None),
                   "sim_floor": (float(np.mean(SIM_FLOOR))
                                 if SIM_FLOOR else None),
                   "decomp": dict(DEC)}
            pt, clo, chi = boot_ci(PAIRS)
            row["gen_only_ci"] = [pt, clo, chi]
            sm, slo, shi = boot_mean_ci(SIM)
            row["sim_o_ci"] = [sm, slo, shi]
            for tag in ("gen", "ceil", "orig", "gen_only"):
                row[tag] = {"wer": 100.0 * E[tag][0] / max(E[tag][1], 1),
                            "utmos": float(np.mean(MOS[tag])) if MOS[tag] else None}
            res["tts"][str(w)] = row
            log("")
            log(f"  CFG w={w}   n={seen}   实际 prompt {pm:.2f}s")
            log(f"    {'轨道':<8} {'WER↓':>8} {'UTMOS↑':>8}")
            for tag, name in (("orig", "真音频"), ("ceil", "天花板"),
                              ("gen", "整句(含真prompt,污染)"),
                              ("gen_only", "仅生成区(干净)")):
                u = row[tag]["utmos"]
                log(f"    {name:<8} {row[tag]['wer']:8.2f} "
                    f"{(f'{u:8.3f}' if u is not None else '       -')}")
            pt, clo, chi = row["gen_only_ci"]
            if pt is not None:
                log(f"    仅生成区 WER 的 95% 置信区间  "
                    f"{pt:.2f}  [{clo:.2f}, {chi:.2f}]"
                    f"   宽度 {chi - clo:.1f} 点")
                log(f"    -> 两个设置的差距小于约 {chi - clo:.0f} 点时不可排序")
            d = row["decomp"]
            tot = max(d["sub"] + d["dele"] + d["ins"], 1)
            log(f"    误差构成   替换 {d['sub']} ({100*d['sub']/tot:.0f}%)  "
                f"删除 {d['dele']} ({100*d['dele']/tot:.0f}%)  "
                f"插入 {d['ins']} ({100*d['ins']/tot:.0f}%)"
                f"   <- 以删除为主=没声/以插入为主=乱说")
            if row["sim_o"] is not None:
                fl, ce = row["sim_floor"], row["sim_ceiling"]
                sm, slo, shi = row["sim_o_ci"]
                log("")
                log(f"    SIM-o 的三个数（只有放在一起才可解读）:")
                log(f"      天花板 真mel→声码器 vs 原录音   "
                    f"{(f'{ce:.4f}' if ce is not None else '—')}"
                    f"   <- 声码器本身的上限")
                log(f"      我们   仅生成区                  {sm:.4f}"
                    f"  [{slo:.4f}, {shi:.4f}]")
                log(f"      地板   不同说话人 vs 原录音       "
                    f"{(f'{fl:.4f}' if fl is not None else '—')}"
                    f"   <- 完全无视 prompt 就能拿到这个")
                if fl is not None and ce is not None and ce > fl:
                    pos = (sm - fl) / (ce - fl)
                    log(f"      -> 我们落在地板到天花板的 {100*pos:.0f}% 处")
                    row["sim_o_normalized"] = float(pos)
                log(f"    SIM-o 整句(含真prompt)  "
                    f"{row['sim_o_contaminated']:.4f}  <- 被污染，不可比")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    log("")
    log(f"wrote {args.out}")
    log("")
    log("对标（continuation 任务，不是常被引用的 cross-sentence 行）:")
    log("  Voicebox   continuation   WER 2.0   SIM-o 0.593")
    log("  Voicebox   cross-sentence WER 1.9   SIM-o 0.662")
    log("  F5-TTS     LibriSpeech-PC WER 2.42  (32 NFE)")
    log("  SpeechT5   统一模型        TTS WER 5.91  UTMOS 3.32  ASR 4.4")
    log("  UniVoice   统一模型        TTS WER 4.06  UTMOS 3.72  ASR 3.0")
    log("  ELF-S2T    ELF-B 105.9M   ASR 10.50 (960h + whisper-large-v3)")


if __name__ == "__main__":
    main()
