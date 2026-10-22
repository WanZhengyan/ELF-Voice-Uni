#!/usr/bin/env python
"""Unified TTS + S2T on one ELF transformer, one objective, two channels.

Both directions are continuous-target flow matching, so unlike UniVoice (which
bridges an autoregressive ASR to a flow-matching TTS with dual-attention
masking) there is a single loss, a single attention pattern and a single
transformer. What differs between the two tasks is only which channel carries
the noise and which carries the condition.

SEQUENCE: one row holds the mel frames and the text tokens side by side, so
neither modality is resampled to the other's rate and no alignment is assumed.

    position      0 .. T-1  (mel frames)       T .. T+N-1  (text tokens)
    ---------------------------------------------------------------------
    z_V (100)     mel                          0
    z_T (512)     0                            T5 latent
    cond (512)    upsampled char emb (TTS)     0

    TTS   flow on z_V over mel positions;  text positions are clean condition
    S2T   flow on z_T over text positions; mel  positions are clean condition

The two tasks are the same computation with the clean/noisy roles swapped, which
is what makes one objective possible.

NO ENCODER AND NO DECODER ON THE SPEECH SIDE. Speech enters as raw log-mel in
its own 100-d channel: nothing is learned to produce it and nothing maps it to
tokens, because a continuous target has no tokens to predict. Text gets both,
and both are ELF's own: a frozen t5-small embedding lookup on the way in, and
ELF's pretrained decode head (vocab 32100) on the way out. That asymmetry is the
design, not an oversight -- it is also where ELF's late discretization finally
earns its place, since the S2T half has a genuinely discrete target.

CHANNEL ORDER is [z_T, sc_T, z_V, sc_V, cond]. The first 1024 columns are
exactly stock ELF's [z, x_pred], so load_elf_init's existing partial-copy rule
fills them from the checkpoint and zeroes the rest -- the model starts at ELF's
behaviour and grows the voice channel as a residual from zero.

CLEAN/NOISY is expressed entirely through the prompt mask that the TTS code
already has: prompt = 1 means "this position stays at its clean value", which
restore() enforces and the loss mask excludes. S2T is prompt = all mel frames;
TTS is prompt = all text tokens, plus the first K mel frames when a voice prompt
is wanted. No new conditioning machinery.
"""
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from elf_tts.model import ELF_models
from elf_tts.layers import RMSNorm

TTS, S2T = 0, 1
T_EPS = 0.05


def log(*a):
    print(*a, flush=True)


# ============================================================ model
class UnifiedELF(nn.Module):
    """ELF with a voice channel bolted on and a per-row task prefix."""

    def __init__(self, name="ELF-B", text_dim=512, voice_dim=100,
                 char_vocab=40, text_vocab=32100, max_length=512,
                 num_task_tokens=4, text_norm=True, whisper_dim=0,
                 use_cond=True, spk_dim=0, spk_norm=False, **kw):
        super().__init__()
        self.text_dim, self.voice_dim = text_dim, voice_dim
        # [z_T, sc_T, z_V, sc_V] and optionally cond.
        #
        # Dropping cond does three things at once. It removes 512 input columns
        # (1736 -> 1224) and 283,136 parameters; it raises the share of input
        # columns that carry ELF's pretrained weights from 59.0% to 83.7%,
        # leaving only z_V/sc_V new -- and those are the flow target, so they
        # get direct gradient, unlike cond which was measured at 1.14% of z_T's
        # scale early in training; and it makes a leak structurally impossible,
        # because the character ids simply never enter the model. The leak was
        # that `chars` was not gated by task, so every S2T row received the
        # character sequence of its own answer through this block.
        #
        # What TTS gives up is granularity, not per-frame conditioning: the
        # upsampled T5 token latent stays in z_T (about 28 frames per unit),
        # only the character-level signal (about 6.2 frames per unit) goes.
        self.use_cond = use_cond
        # A global speaker vector, broadcast to every position. Three diagnostics
        # said the backbone is learning the mapping fine -- the uniform-alignment
        # error is the same on the corpus where TTS works as where it fails
        # (1.07 vs 1.04 word durations), the exposure gap is WORSE on the working
        # one (3.63 vs 3.54), and the flow objective is well optimised on both.
        # The one measurement that separated them is SIM-o: 47% of the way from
        # "ignored the prompt" to the vocoder ceiling. The prompt is the only
        # speaker information in a zero-shot setting and it is half used.
        #
        # STTATTS, the only baseline at our data scale (155M, LibriSpeech 100h,
        # TTS CER 2.10-6.22), does not make its backbone infer the speaker: it
        # concatenates an x-vector from a pretrained speaker-verification model.
        # We have 0.368 hours per speaker; learning speaker extraction from that
        # is a separate and much harder problem than using such a vector.
        self.spk_dim = spk_dim
        self.input_width = (2 * text_dim + 2 * voice_dim
                            + (text_dim if use_cond else 0)
                            + (text_dim if spk_dim > 0 else 0))
        if use_cond:
            self.char_embed = nn.Embedding(char_vocab, text_dim, padding_idx=0)
            nn.init.normal_(self.char_embed.weight, std=0.02)
            with torch.no_grad():
                self.char_embed.weight[0].zero_()
        # z_T enters at per-dim std ~1; a std-0.02 table enters ~50x weaker and
        # the projection was measured not to compensate, so the cond block is
        # normalised onto the same footing. (Same finding as the TTS-only run.)
        self.cond_norm = (RMSNorm(text_dim, eps=1e-6)
                          if (text_norm and use_cond) else None)

        self.elf = ELF_models[name](
            text_encoder_dim=text_dim, max_length=max_length,
            vocab_size=text_vocab, input_width=self.input_width,
            n_input_blocks=2, bottleneck_dim=128, num_time_tokens=4,
            num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
            num_task_tokens=num_task_tokens, num_tasks=2, **kw)

        # The denoiser's output layer for the voice channel. 768 -> 512 is
        # ELF's pretrained final_layer; this is the 512 -> 100 that follows it.
        # Not an encoder: the hidden width has to reach 100 somehow, and the
        # TTS-only run that works has exactly this layer at 768 -> 100.
        # ELF-S2T's linear projector: a FROZEN speech encoder supplies the
        # audio condition and this maps its space into ELF's text latent space.
        # It has to be trained -- Whisper's features have std ~1 in their own
        # basis and no relation to ELF's latent (x0_t std ~0.77), so nothing
        # frozen could line them up. The encoder itself stays frozen, which is
        # the part that matters: no speech encoder is learned here.
        self.whisper_dim = whisper_dim
        if whisper_dim > 0:
            self.whisper_proj = nn.Linear(whisper_dim, text_dim)
            nn.init.zeros_(self.whisper_proj.bias)
            nn.init.normal_(self.whisper_proj.weight, std=whisper_dim ** -0.5)

        if spk_dim > 0:
            # same treatment as whisper_proj: a frozen encoder's space has no
            # relation to ELF's latent scale, so the map into it is trained
            self.spk_proj = nn.Linear(spk_dim, text_dim)
            nn.init.zeros_(self.spk_proj.bias)
            nn.init.normal_(self.spk_proj.weight, std=spk_dim ** -0.5)
        # The projected x-vector enters at per-dim std ~0.19 against z_T's ~0.77
        # and was measured to be nearly ignored (another speaker's vector moved
        # the x0 error by 2%). Same fix as the char block: RMS-normalise it.
        self.spk_norm = (RMSNorm(text_dim, eps=1e-6)
                         if (spk_dim > 0 and spk_norm) else None)

        self.voice_out = nn.Linear(text_dim, voice_dim)
        nn.init.zeros_(self.voice_out.bias)
        # std 0.02 over 512 inputs gives an output std of sqrt(512)*0.02 = 0.45,
        # but the mel target is normalised to std 1, so the voice channel starts
        # 2.2x too quiet at the only place it leaves the model. 1/sqrt(dim)
        # makes a std-1 input come out at std 1.
        nn.init.normal_(self.voice_out.weight, std=text_dim ** -0.5)

    def forward(self, z_t, z_v, sc_t, sc_v, char_ids, t, task_id, *,
                attention_mask=None, self_cond_cfg_scale=None,
                decoder_step_active=None, drop_cond=None,
                audio_cond=None, audio_cond_mask=None, spk_emb=None):
        """z_t,sc_t: (B,L,512)  z_v,sc_v: (B,L,100)  char_ids: (B,L)  t: (B,)

        audio_cond: (B,L,whisper_dim) frozen encoder features already gathered
        onto our frame positions; audio_cond_mask: (B,L) 1 where they apply
        (S2T rows, audio region). They are ADDED into the z_T channel, where
        that slot is zero for S2T rows -- so the audio condition rides ELF's
        pretrained text channel, exactly as ELF-S2T's projector does, rather
        than a new column block that would start untrained.

        Returns (x0_text (B,L,512), x0_voice (B,L,100), logits (B,L,V) or None).
        """
        if audio_cond is not None and self.whisper_dim > 0:
            a = self.whisper_proj(audio_cond.float())
            if audio_cond_mask is not None:
                a = a * audio_cond_mask.unsqueeze(-1).to(a.dtype)
            z_t = z_t + a
        blocks = [z_t, sc_t, z_v, sc_v]
        if self.spk_dim > 0:
            # one vector per clip, broadcast across the sequence -- the same
            # mechanism as cond, which demonstrably works (it acquired phonetic
            # structure with no phonetic supervision). Zero when not supplied,
            # so a checkpoint trained with it can still run without.
            if spk_emb is None:
                sp = z_t.new_zeros(z_t.shape[0], 1, self.text_dim)
            else:
                sp = self.spk_proj(spk_emb.float())
                if self.spk_norm is not None:
                    sp = self.spk_norm(sp)
                sp = sp.unsqueeze(1)
            blocks.append(sp.expand(-1, z_t.shape[1], -1))
        if self.use_cond:
            cond = self.char_embed(char_ids)
            if self.cond_norm is not None:
                cond = self.cond_norm(cond.float())
            if drop_cond is not None:                  # classifier-free guidance
                cond = cond * (1.0 - drop_cond.to(cond.dtype).view(-1, 1, 1))
            blocks.append(cond)

        x = torch.cat(blocks, dim=-1)
        out, logits = self.elf(
            x, t, attention_mask=attention_mask, deterministic=not self.training,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active, task_id=task_id)
        x0_text = out[0] if isinstance(out, tuple) else out
        return x0_text, self.voice_out(x0_text), logits


def load_elf_init(model, path, use_ema=True, verbose=True, init_frac=0.0):
    """Initialise from ELF-B-owt. Everything loads except what cannot.

    self_cond_proj  ELF concatenates 1024 input columns, we concatenate 1736.
                    ELF's go in the first 1024 (which are exactly [z_T, sc_T]).
    task_tokens     new (num_tasks, 4, hidden). Random init.
    voice_out       new. Random init.

    init_frac sets the scale of the 712 NEW input columns, as a fraction of the
    pretrained z_T block's per-column norm.

    0.0 zeroes them, so the model starts at exactly ELF's behaviour. Measured at
    step 2000 of a run that did this: the audio columns had grown to 14% of z_T
    and the audio condition moved the S2T prediction by 131%, but the TTS cond
    block was still at 1.14% and removing ALL text moved the TTS prediction by
    under 7%. The two differ because the audio is the only route for audio
    information, whereas for TTS the text is also reachable through ELF's
    pretrained z_T channel, so there is little pressure to grow the one block
    that carries the frame-level alignment -- the part TTS actually needs.

    A nonzero init_frac starts those columns at a scale the trunk already
    responds to. 0.25 is below the 14% -> 131% point the audio block reached on
    its own, so it is enough to matter, while staying small next to z_T.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    src = ck.get("ema_params1" if use_ema else "params", ck.get("params", ck))
    tgt = model.state_dict()
    loaded, partial, skipped = [], [], []
    with torch.no_grad():
        for k, v in src.items():
            ours = f"elf.{k}"
            if ours not in tgt:
                skipped.append((k, "no such tensor")); continue
            t = tgt[ours]
            if t.shape == v.shape:
                t.copy_(v); loaded.append(k)
            elif ("self_cond_proj.weight" in k and t.shape[0] == v.shape[0]
                  and t.shape[1] > v.shape[1]):
                t.zero_(); t[:, :v.shape[1]].copy_(v)
                n_new = t.shape[1] - v.shape[1]
                if init_frac > 0.0:
                    # match the pretrained z_T block's per-column norm, scaled
                    ref = v[:, :model.text_dim].norm(dim=0).mean()
                    nw = torch.randn(t.shape[0], n_new)
                    nw = nw / nw.norm(dim=0, keepdim=True) * ref * init_frac
                    t[:, v.shape[1]:].copy_(nw)
                    how = (f"{n_new} new cols at {init_frac:.2f} x z_T "
                           f"(|w|/col {nw.norm(dim=0).mean():.5f} vs {ref:.5f})")
                else:
                    how = f"{n_new} new cols zeroed"
                partial.append((k, f"{tuple(v.shape)} -> {tuple(t.shape)}, {how}"))
            else:
                skipped.append((k, f"shape {tuple(v.shape)} vs {tuple(t.shape)}"))
    model.load_state_dict(tgt)
    n_ok = sum(tgt[f"elf.{k}"].numel() for k in loaded)
    n_tot = sum(p.numel() for p in model.parameters())
    if verbose:
        log(f"ELF init from {os.path.basename(path)} ({'ema' if use_ema else 'raw'})")
        log(f"  loaded  {len(loaded)} tensors, {n_ok/1e6:.2f}M / {n_tot/1e6:.2f}M "
            f"({n_ok/n_tot*100:.1f}%)")
        for k, why in partial: log(f"  partial {k}: {why}")
        for k, why in skipped: log(f"  skipped {k}: {why}  (random init)")
    return {"loaded": len(loaded), "params_loaded": n_ok,
            "partial": [k for k, _ in partial], "skipped": [k for k, _ in skipped]}


# ============================================================ flow
@dataclass
class Cfg:
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 2.0
    decoder_p_mean: float = 0.8          # ELF-S2T lowers this to -0.5 ("audio
    decoder_p_std: float = 0.8           # forcing"); exposed so we can test it
    decoder_noise_scale: float = 5.0
    decoder_prob: float = 0.2            # S2T rows only: TTS has no tokens
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 1.0
    self_cond_cfg_max: float = 3.0
    label_drop_prob: float = 0.1
    # F5-TTS drops the text at 0.2 to force text dependence; we had 0.0,
    # so our TTS CFG guided away from an untrained branch.
    text_drop_prob: float = 0.0          # drops the CHAR cond block (ELF's own)
    # ELF-S2T: "we drop the entire audio condition on 10% of utterances during
    # training so that an unconditional branch is learned alongside the
    # conditional one", then guide with w = 2.0 at inference. Our
    # label_drop_prob dropped the upsampled-character block instead, so no such
    # branch existed and ACFG was impossible. This is the one.
    audio_drop_prob: float = 0.0
    t_eps: float = 0.05
    s2t_weight: float = 1.0              # UniVoice down-weights its ASR to 0.005
    # 0.999 leaves 13.5% of the init in the average at 2k steps, which made an
    # earlier run's first eval read near-zero accuracy on a model that was fine.
    ema_decay: float = 0.9995


def sample_timesteps(n, p_mean, p_std, device):
    return torch.sigmoid(torch.randn(n, device=device) * p_std + p_mean)


def restore(z, clean, mask):
    """Hold the conditioning positions at their clean value."""
    return torch.where(mask.unsqueeze(-1) > 0, clean, z)


def net_to_v(x0_hat, z, t):
    return (x0_hat - z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=T_EPS)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ELF-B")
    ap.add_argument("--init", default=os.path.join(
        HERE, "ckpt", "ELF-B-owt-torch", "checkpoint_95085"))
    ap.add_argument("--max_length", type=int, default=512)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = UnifiedELF(a.model, max_length=a.max_length).to(dev)
    log(f"input_width {m.input_width}  "
        f"= 2x512 (text z,sc) + 2x100 (voice z,sc) + 512 (cond)")
    n = sum(p.numel() for p in m.parameters())
    log(f"params {n/1e6:.2f}M  "
        f"(char_embed {m.char_embed.weight.numel()/1e3:.1f}K, "
        f"task_tokens {m.elf.task_tokens.numel()/1e3:.1f}K, "
        f"voice_out {sum(p.numel() for p in m.voice_out.parameters())/1e3:.1f}K)")
    if os.path.exists(a.init):
        load_elf_init(m, a.init)
    else:
        log(f"(no ckpt at {a.init})")

    # shapes + both task routes
    B, L = 2, a.max_length
    z_t = torch.randn(B, L, 512, device=dev)
    z_v = torch.randn(B, L, 100, device=dev)
    char = torch.randint(0, 40, (B, L), device=dev)
    t = torch.rand(B, device=dev)
    am = torch.ones(B, L, device=dev)
    sc_cfg = torch.ones(B, device=dev)
    for task, name in ((TTS, "TTS"), (S2T, "S2T")):
        tid = torch.full((B,), task, dtype=torch.long, device=dev)
        xt, xv, lg = m(z_t, z_v, torch.zeros_like(z_t), torch.zeros_like(z_v),
                       char, t, tid, attention_mask=am,
                       self_cond_cfg_scale=sc_cfg,
                       decoder_step_active=(task == S2T))
        log(f"  {name}: x0_text {tuple(xt.shape)}  x0_voice {tuple(xv.shape)}  "
            f"logits {tuple(lg.shape) if lg is not None else None}")

    # The task prefix must actually change the computation, or it is decoration.
    tid0 = torch.zeros(B, dtype=torch.long, device=dev)
    tid1 = torch.ones(B, dtype=torch.long, device=dev)
    with torch.no_grad():
        a0 = m(z_t, z_v, torch.zeros_like(z_t), torch.zeros_like(z_v), char, t,
               tid0, attention_mask=am, self_cond_cfg_scale=sc_cfg)[0]
        a1 = m(z_t, z_v, torch.zeros_like(z_t), torch.zeros_like(z_v), char, t,
               tid1, attention_mask=am, self_cond_cfg_scale=sc_cfg)[0]
    rel = (a0 - a1).norm().item() / a0.norm().item()
    log(f"  task token changes the output by {rel*100:.2f}% (relative)")
    log("  -> " + ("task routing is live" if rel > 1e-4 else
                   "TASK TOKEN IS INERT -- it is not reaching the trunk"))
