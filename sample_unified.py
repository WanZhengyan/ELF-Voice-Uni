#!/usr/bin/env python
"""Vocode the unified model's TTS output and measure WER + UTMOS.

This is the measurement that decides, and it had never been run. Every
pass/fail call in the debugging so far used ||z_hat - x0|| / ||x0|| against a
trivial baseline of 1.000 -- but that is a RECONSTRUCTION metric, and TTS is
generation: the same text spoken with different prosody or a different speaker
realisation is correct and still lands far from one particular reference
recording in L2. The standalone mel run that produced intelligible audio
reported, at step 100000,

    rollout/z_err_ratio   1.0282   (1.0667 with no prompt)
    rollout/beats_trivial 0.0
    exposure_gap/t0.7     4.4279

i.e. it "failed" the same test, more badly than the unified model does
(0.811 and 1.39 at step 4000). So the test does not separate the two.

Four tracks, the same shape the standalone run used, so the numbers are
comparable:

  orig      the real recording. Floor for this dataset and this ASR.
  ceiling   original -> mel -> Vocos -> waveform. The upper bound for anything
            that generates mel, since the vocoder's own loss is in it.
  gen       text -> flow -> mel -> Vocos. What the model actually does.
  uncond    the same with the text removed. If gen is not far below this, the
            text is not doing the work.

WER uses Whisper-large-v3 with Whisper's own EnglishTextNormalizer. The gap that
matters is gen vs ceiling, not gen vs orig.
"""
import argparse, base64, io, json, math, os, sys

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import TTS, S2T, Cfg, UnifiedELF, net_to_v, restore
from train_unified import (LibriCache, TextEncoder, make_collate, build_targets,
                           make_schedule, wer_pair)


def log(*a):
    print(*a, flush=True)


def wav_b64(w, sr=24000):
    buf = io.BytesIO()
    torchaudio.save(buf, w.cpu(), sr, format="wav", encoding="PCM_S",
                    bits_per_sample=16)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@torch.no_grad()
def tts_rollout(model, b, cfg, device, text_enc, mel_stats, *, steps, gamma=0.0,
                drop_text=False, cfg_w=1.0, sway=None, cfg_mode="text"):
    """cfg_w > 1 turns on classifier-free guidance on the VOICE channel.

    `gamma` is SDE churn (re-noising), not guidance -- the two are unrelated
    knobs and TTS here ran with no guidance at all until cfg_w was added. The
    unconditional branch is the same one `drop_text` already builds: text latent
    zeroed, char block zeroed, drop_cond set. cfg_w == 1.0 takes the
    single-pass path, so it is bit-for-bit the old sampler."""
    chars = b["chars"].to(device); valid = b["valid"].to(device)
    v_act = b["v_act"].to(device).unsqueeze(-1)
    t_act = b["t_act"].to(device).unsqueeze(-1)
    p_v, p_t = b["prompt_v"].to(device), b["prompt_t"].to(device)
    task = b["task"].to(device)
    x0_t, x0_v = build_targets(b, text_enc, device, mel_stats)
    from train_unified import spk_cond
    se = spk_cond(b, device)
    B = x0_v.shape[0]
    ns = cfg.denoiser_noise_scale
    if drop_text:
        # the text channel is the condition for TTS, so removing it means
        # zeroing it AND the upsampled char block
        x0_t = torch.zeros_like(x0_t)
        chars = torch.zeros_like(chars)
    dc = torch.ones(B, device=device) if drop_text else None

    z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
    z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
    xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
    sc_cfg = torch.ones(B, device=device)
    # cfg_mode "joint": the unconditional branch drops text AND the audio prompt
    # AND the x-vector -- the one "no condition" state a model trained with
    # --joint_cond_drop has seen (its text-dropped rows lost all three), and
    # F5-TTS's own CFG (drop_audio_cond + drop_text together). The prompt span
    # then has no clean value in that branch: it gets the forward-noised prompt
    # at the current t (a fixed draw, so it is one path) and, as self-condition,
    # that branch's own previous estimate there, which is what such a training
    # row sees. "text" (default) is the old text-only guidance.
    assert cfg_mode in ("text", "joint")
    eps_p = torch.randn_like(x0_v) * ns if cfg_mode == "joint" else None
    xpu_prev = torch.zeros_like(x0_v)
    ts = make_schedule(steps, cfg, device)
    if sway is not None:
        # F5-TTS's Sway Sampling on the same t convention (0 noise, 1 data):
        # t <- t + s (cos(pi t / 2) - 1 + t); s = -1 packs steps near t = 0
        ts = ts + sway * (torch.cos(math.pi / 2 * ts) - 1 + ts)
    for i in range(steps):
        tv = float(ts[i]); tn = float(ts[i + 1]); h = tn - tv
        if gamma > 0.0:
            al = max(0.0, 1.0 - gamma * h)
            z_t = restore(al * z_t + (1 - al) * torch.randn_like(z_t) * ns,
                          x0_t, p_t) * t_act
            z_v = restore(al * z_v + (1 - al) * torch.randn_like(z_v) * ns,
                          x0_v, p_v) * v_act
            tv = al * tv
        t = torch.full((B,), tv, device=device)
        xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                              attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                              drop_cond=dc, spk_emb=se)
        if cfg_w != 1.0 and not drop_text:
            zt_u = torch.zeros_like(z_t)
            ch_u = torch.zeros_like(chars)
            dc_u = torch.ones(B, device=device)
            if cfg_mode == "joint":
                pv = p_v.unsqueeze(-1) > 0
                tt = t.view(-1, 1, 1)
                zv_u = torch.where(pv, tt * x0_v + (1 - tt) * eps_p, z_v) * v_act
                xpv_u = torch.where(pv, xpu_prev, xp_v) * v_act
                se_u = None if se is None else torch.zeros_like(se)
                _, u_v, _ = model(zt_u, zv_u, torch.zeros_like(xp_t), xpv_u, ch_u, t,
                                  task, attention_mask=valid,
                                  self_cond_cfg_scale=sc_cfg, drop_cond=dc_u, spk_emb=se_u)
                xpu_prev = u_v
            else:
                _, u_v, _ = model(zt_u, z_v, torch.zeros_like(xp_t), xp_v, ch_u, t,
                                  task, attention_mask=valid,
                                  self_cond_cfg_scale=sc_cfg, drop_cond=dc_u, spk_emb=se)
            # guide the mel prediction away from "what this model says with no
            # text at all"; the text channel is the condition, not a target
            xh_v = u_v + cfg_w * (xh_v - u_v)
        hh = tn - tv
        z_t = restore(z_t + hh * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
        z_v = restore(z_v + hh * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
        xp_t = restore(xh_t, x0_t, p_t) * t_act
        xp_v = restore(xh_v, x0_v, p_v) * v_act
    mu, sd = mel_stats
    return z_v * sd + mu                      # back to log-mel units


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "ch_tts", "ckpt.pt"))
    ap.add_argument("--cache", default=os.path.join(
        HERE, "cache_libri_train_clean_100"))
    ap.add_argument("--libri_root", default=os.path.join(HERE, "data", "LibriSpeech"))
    ap.add_argument("--lj_root", default=os.path.join(HERE, "data", "LJSpeech-1.1"))
    ap.add_argument("--split", default="train-clean-100")
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n_wav", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.0)  # SDE churn
    ap.add_argument("--cfg_w", type=float, default=1.0)  # voice-channel CFG
    ap.add_argument("--seed", type=int, default=0)
    # NOT --split: that name is already taken by the LibriSpeech split name
    # (train-clean-100 / test-clean), which is what locates the flac files.
    ap.add_argument("--cache_split", default="test",
                    help="'all' 用完整官方测试集（文献报的是这个）；"
                         "'test' 只切 300 条")
    # 251 speakers: with no voice prompt, text -> speech is one-to-many in
    # speaker identity and the L2 optimum is the speaker average. Training drew
    # prompt_frac ~ U(0, 0.3), so 0.3 is in distribution.
    ap.add_argument("--prompt", type=float, default=0.0)
    ap.add_argument("--asr", default="openai/whisper-large-v3")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "unified_wer.json"))
    ap.add_argument("--page", default=os.path.join(HERE, "listen_unified.html"))
    args = ap.parse_args()
    # the rollout starts from torch.randn, which was unseeded: the same
    # checkpoint and flags measured 50.63 and 93.07 WER on two runs. Seeding
    # here also makes a cfg_w sweep PAIRED -- the guided branch draws no
    # noise of its own, so every w starts from the same noise.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    cfg = Cfg(denoiser_p_mean=ta["denoiser_p_mean"])
    log(f"ckpt {ta['model']} @ step {ck['step']} ({args.weights}); "
        f"layout {ta['layout']}, p_s2t {ta['p_s2t']}, NFE {args.nfe}, "
        f"gamma {args.gamma}")

    from vocos import Vocos
    voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
    log("Vocos charactr/vocos-mel-24khz loaded")
    from transformers import pipeline
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    asr = pipeline("automatic-speech-recognition", model=args.asr,
                   device=0 if device.type == "cuda" else -1,
                   torch_dtype=torch.float16 if device.type == "cuda"
                   else torch.float32)
    norm = EnglishTextNormalizer({})
    try:
        um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                            trust_repo=True).to(device).eval()
    except Exception as e:
        log(f"UTMOS unavailable: {e}"); um = None

    ds = LibriCache(args.cache, args.cache_split)
    meta = ds.meta
    mel_stats = (meta["latent_mean"], meta["latent_std"])
    itos = {v: k for k, v in meta["stoi"].items()}
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    # pure TTS: no voice prompt, so nothing of the reference audio is reused
    # TTS-only eval, so every row is TTS and the audio condition mask is all
    # zero -- the Whisper features are inert here by construction.
    coll = make_collate(ta["max_length"], 0.0, (args.prompt, args.prompt),
                        rng_seed=7, max_text=ta["max_text"], itos=itos, tk=tk,
                        layout=ta["layout"],
                        whisper_rate=meta.get("whisper_rate", 0.0)
                        if ta.get("whisper") else 0.0,
                        frame_rate=meta["frame_rate"])
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0,
                    collate_fn=coll)

    # a two-source checkpoint's args["cache"] is the TTS cache (no Whisper
    # fields); its own "data" record says what the S2T rows were trained with
    D = ck.get("data") or {}
    wdim = D.get("whisper_dim", meta.get("whisper_dim", 0)) if ta.get("whisper") else 0
    # the architecture is read off the checkpoint, not hardcoded here, so a
    # cond-less run loads with the same command as a cond run
    model = UnifiedELF(ta["model"], char_vocab=meta["char_vocab_size"],
                       max_length=ta["max_length"], whisper_dim=wdim,
                       use_cond=not ta.get("no_cond", 0),
                       spk_dim=(meta.get("spk_emb_dim", 0)
                                if ta.get("spk") else 0),
                       # trained with --spk_norm: without it the strict load
                       # fails on spk_norm.weight (CODE_REVIEW 12.1)
                       spk_norm=bool(ta.get("spk_norm", 0))).to(device)
    model.load_state_dict(ck[args.weights]); model.eval()
    text_enc = TextEncoder(ta["text_latent_std"]).to(device)

    # the reference transcript and the source waveform for each cached clip
    dataset = meta.get("dataset", "librispeech")
    id2text = {}
    if dataset == "ljspeech":
        with open(os.path.join(args.lj_root, "metadata.csv"),
                  encoding="utf-8") as f:
            for line in f:
                q = line.strip().split("|")
                if len(q) >= 3:
                    id2text[q[0]] = q[2]
        src_sr = 22050

        def wav_path(uid):
            return os.path.join(args.lj_root, "wavs", uid + ".wav")
    else:
        import glob
        for tr in glob.glob(os.path.join(args.libri_root, args.split,
                                         "*", "*", "*.trans.txt")):
            with open(tr, encoding="utf-8") as f:
                for line in f:
                    uid, _, txt = line.strip().partition(" ")
                    id2text[uid] = txt
        src_sr = 16000

        def wav_path(uid):
            spk, chap, _ = uid.split("-")
            return os.path.join(args.libri_root, args.split, spk, chap,
                                uid + ".flac")
    log(f"dataset {dataset}, {meta.get('n_speakers')} speaker(s); "
        f"{len(id2text)} transcripts; {len(ds)} held-out clips")

    tracks = ["orig", "ceiling", "gen", "uncond"]
    errs = {t: 0 for t in tracks}; lens = {t: 0 for t in tracks}
    mos = {t: [] for t in tracks}; rows = []
    rs_src = None   # built once src_sr is known
    done = 0

    for b in dl:
        if done >= args.n:
            break
        mel_gen = tts_rollout(model, b, cfg, device, text_enc, mel_stats,
                              steps=args.nfe, gamma=args.gamma,
                              cfg_w=args.cfg_w)
        mel_unc = tts_rollout(model, b, cfg, device, text_enc, mel_stats,
                              steps=args.nfe, gamma=args.gamma, drop_text=True)
        mel_true = b["mel"].to(device)

        for i in range(mel_gen.shape[0]):
            if done >= args.n:
                break
            uid = b["uid"][i]
            ref_txt = id2text.get(uid, "")
            T = int(b["valid"][i].sum().item())
            if not ref_txt or T < 32:
                done += 1
                continue
            # the cropped span the model was actually asked to speak
            n = int(b["n_text"][i])
            ref_ids = b["t5_dense"][i, :n].tolist()
            ref = norm(tk.decode(ref_ids, skip_special_tokens=True))
            if not ref.strip():
                done += 1
                continue
            if rs_src is None:
                rs_src = torchaudio.transforms.Resample(src_sr, 24000).to(device)
            g = rs_src(torchaudio.load(wav_path(uid))[0].to(device))
            g = g.mean(0, keepdim=True) if g.shape[0] > 1 else g
            # the SAME span the mel crop used: the mel came from frame a, so
            # cutting from 0 compares against words the reference lacks
            a0 = int(b["crop_a"][i]); hop = 256
            g = g[:, a0 * hop:(a0 + T) * hop]
            waves = {
                "orig": g,
                "ceiling": voc.decode(mel_true[i:i + 1, :T].transpose(1, 2)),
                "gen": voc.decode(mel_gen[i:i + 1, :T].transpose(1, 2)),
                "uncond": voc.decode(mel_unc[i:i + 1, :T].transpose(1, 2)),
            }
            for t in tracks:
                a = waves[t].squeeze(0).float().clamp(-1, 1)
                out = asr({"sampling_rate": 24000, "raw": a.cpu().numpy()},
                          generate_kwargs={"language": "english"})
                e, nn = wer_pair(ref, norm(out["text"]))
                errs[t] += e; lens[t] += nn
                if um is not None:
                    mos[t].append(float(um(a.unsqueeze(0), 24000).item()))
            if len(rows) < args.n_wav:
                rows.append({"id": uid, "text": ref, "T": T,
                             "dur": T / meta["frame_rate"],
                             "b64": {t: wav_b64(
                                 waves[t].squeeze(0).unsqueeze(0).cpu())
                                 for t in tracks}})
            done += 1
            if done % 10 == 0:
                log(f"  {done}/{args.n}  " + "  ".join(
                    f"{t} {100*errs[t]/max(lens[t],1):.1f}" for t in tracks))

    log("")
    log(f"=== unified model, TTS direction, {done} held-out clips, "
        f"prompt {args.prompt:.0%} ===")
    n_scored = len(rows)
    if n_scored == 0:
        raise SystemExit(
            "0 clips scored -- nothing was measured, and printing zeros "
            "and nans here would look like a result. The usual cause is "
            f"--split: it names the LibriSpeech directory used to find "
            f"the flac files (got {args.split!r}), which must match the "
            "cache. --cache_split is a different thing.")
    log(f"{'track':<10} {'WER %':>8} {'UTMOS':>8}")
    res = {"n": done, "step": ck["step"], "layout": ta["layout"],
           "p_s2t": ta["p_s2t"], "nfe": args.nfe, "tracks": {}}
    for t in tracks:
        wv = 100 * errs[t] / max(lens[t], 1)
        uv = float(np.mean(mos[t])) if mos[t] else float("nan")
        log(f"{t:<10} {wv:>8.2f} {uv:>8.3f}")
        res["tracks"][t] = {"wer": wv, "utmos": uv}
    g = res["tracks"]["gen"]["wer"]; c = res["tracks"]["ceiling"]["wer"]
    u = res["tracks"]["uncond"]["wer"]
    log("")
    log(f"gen is {g - c:+.2f} WER points from its own ceiling, and "
        f"{u - g:+.2f} better than having no text.")
    log("the standalone mel run on LJSpeech measured gen 44.88 vs ceiling 38.72")
    log("(+6.16) with uncond at 99.19 -- that is the shape to compare against,")
    log("not the absolute number: different dataset, different crop length.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")

    # ---- listen page
    names = {"orig": "Real recording",
             "ceiling": "Mel ceiling (original mel through Vocos)",
             "gen": "Generated from text",
             "uncond": "Generated with the text removed"}
    html = ["<title>Unified Model Audio</title>",
            "<style>",
            ":root{color-scheme:light;--bg:#f4f5f7;--fg:#15181e;--mut:#79818d;",
            "--card:#fbfbfc;--rule:#d6dbe1}",
            "@media(prefers-color-scheme:dark){:root:not([data-theme=light]){",
            "color-scheme:dark;--bg:#10131a;--fg:#e6eaf0;--mut:#6f7883;",
            "--card:#171b23;--rule:#2a313b}}",
            ":root[data-theme=dark]{color-scheme:dark;--bg:#10131a;--fg:#e6eaf0;",
            "--mut:#6f7883;--card:#171b23;--rule:#2a313b}",
            "body{background:var(--bg);color:var(--fg);font:15px/1.6 ",
            "Georgia,serif;padding-inline:20px}",
            ".w{max-width:860px;margin:0 auto;padding-block:40px}",
            "h1{font:600 30px/1.15 'Helvetica Neue',Arial,sans-serif;margin:0 0 8px}",
            "h2{font:600 17px/1.3 'Helvetica Neue',Arial,sans-serif;margin:0 0 4px}",
            ".clip{background:var(--card);border:1px solid var(--rule);",
            "border-radius:6px;padding:16px 18px;margin-top:18px}",
            ".tx{color:var(--mut);font-size:14px;margin:0 0 12px}",
            ".tr{display:grid;grid-template-columns:1fr;gap:4px;",
            "padding:8px 0;border-top:1px solid var(--rule)}",
            ".tr:first-of-type{border-top:0}",
            ".lb{font:500 12px/1.4 ui-monospace,Menlo,monospace;",
            "letter-spacing:.05em;color:var(--mut)}",
            "audio{width:100%}",
            "table{border-collapse:collapse;margin-top:18px;font-size:14px}",
            "td,th{text-align:left;padding:6px 16px 6px 0;",
            "border-bottom:1px solid var(--rule)}",
            "th{font:500 11px/1.4 ui-monospace,Menlo,monospace;",
            "letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}",
            ".n{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:",
            "tabular-nums}",
            "</style>", "<div class=w>",
            "<h1>Unified model &mdash; TTS output</h1>",
            f"<p class=tx>ELF-B, {ta['layout']} layout, step {ck['step']}, "
            f"NFE {args.nfe}, no voice prompt. LibriSpeech held-out clips. "
            f"Compare <b>Generated from text</b> against <b>Mel ceiling</b>: "
            f"that gap is what the flow costs. <b>Text removed</b> is the "
            f"control &mdash; if it sounds as good, the text is not doing the "
            f"work.</p>",
            "<table><tr><th>track</th><th>WER %</th><th>UTMOS</th></tr>"]
    for t in tracks:
        html.append(f"<tr><td>{names[t]}</td>"
                    f"<td class=n>{res['tracks'][t]['wer']:.2f}</td>"
                    f"<td class=n>{res['tracks'][t]['utmos']:.3f}</td></tr>")
    html.append("</table>")
    for r in rows:
        html.append(f"<div class=clip><h2>{r['id']} "
                    f"<span class=lb>{r['dur']:.1f}s</span></h2>"
                    f"<p class=tx>{r['text']}</p>")
        for t in tracks:
            html.append(f"<div class=tr><span class=lb>{names[t]}</span>"
                        f"<audio controls preload=metadata "
                        f"src='data:audio/wav;base64,{r['b64'][t]}'></audio></div>")
        html.append("</div>")
    html.append("</div>")
    open(args.page, "w").write("\n".join(html))
    log(f"wrote {args.page}")


if __name__ == "__main__":
    main()
