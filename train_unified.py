#!/usr/bin/env python
"""Train the unified TTS + S2T model. One transformer, one objective.

TWO LAYOUTS, selected by --layout.

  channel   both modalities occupy the SAME positions in two channels, with the
            text latent upsampled to the mel frame rate:

                position        0        1        2    ...  T-1
                z_V  (100)      mel      mel      mel       mel
                z_T  (512)      tok[0]   tok[0]   tok[1]    tok[N-1]

            TTS  z_V noisy (target), z_T clean (condition)
            S2T  z_V clean (condition), z_T noisy (target)

            Symmetric: same positions, one channel noisy and one clean, and the
            two tasks swap which is which. The alignment is given by position so
            attention never has to discover it, and the TTS half reduces to the
            per-position upsampled-text setup the standalone mel run worked with.

            Cost: the S2T target is the UPSAMPLED latent, so decoding collapses
            adjacent duplicate tokens (CTC-style), and the upsampling assumes a
            uniform char-to-frame rate -- measured against Whisper word
            timestamps the real rate varies 6.6x between its 5th and 95th
            percentile, so this alignment is coarse.

  hybrid    the sequence carries BOTH regions, and each direction gets the
            resolution it actually needs:

                position      0 .. 511  (audio frames)   512 .. 559  (tokens)
                ------------------------------------------------------------
                z_V  (100)    mel                        0
                z_T  (512)    upsampled text (TTS only)  T5 latent at token rate
                cond (512)    upsampled chars (TTS only) 0

            TTS  flow on z_V over the audio frames; text is clean in BOTH
                 regions, so TTS keeps the per-frame upsampled conditioning on
                 ELF's PRETRAINED z_T channel -- the thing that made the
                 standalone run reach UTMOS 3.692.
            S2T  flow on z_T over the TOKEN positions only; mel is clean and
                 conditions it. No upsampled target, so the uniform
                 char-to-frame assumption (whose real rate varies 6.6x) is no
                 longer a supervision signal, and no collapse, so output length
                 is not at the mercy of adjacent frames disagreeing.

            This makes each half structurally equal to the configuration known
            to work for that half: S2T is ELF-S2T's shape (audio as a
            conditioning region, text generated at its own positions), TTS is
            ours. Measured motivation: p_s2t 0 -> 0.5 under the channel layout
            cost TTS 27.47 -> 77.05 WER and 3.692 -> 2.462 UTMOS on the same
            data and step count, and the text's contribution fell from 165 WER
            points to 21 -- consistent with half the batch pushing the shared
            backbone toward a wrong target.

  token     the earlier layout: [mel frames | text tokens] side by side along
            the sequence, text at its own token resolution. Measured to fail --
            the per-position text rode a new zero-initialised channel that
            stayed at 1.14% of the pretrained z_T block's scale, TTS text
            sensitivity was 6%, and the rollout got WORSE with training
            (0.855 -> 2.204 over 8k steps) under three separate interventions
            (zero vs 0.25 init of the new columns, denoiser_p_mean -1.5 vs 0.0,
            and restoring ELF's self-conditioning CFG target).

CLEAN vs NOISY is per CHANNEL, not per position, because in the channel layout
both channels sit at the same positions in different states. Hence prompt_v and
prompt_t rather than one prompt mask.

TEXT ENCODER. t5-small's encoder, frozen, run on a dense block and gathered to
positions. This is the one encoder in the system and it is on the text side
only: speech enters as raw log-mel and leaves through the denoiser's own output
layer, with no learned speech encoder and no speech decode head, because a
continuous target has no tokens to predict.

DECODE HEAD. ELF's pretrained head (vocab 32100), on S2T rows only -- where
ELF's late discretization actually applies, since that target is discrete.
"""
import argparse, contextlib, json, math, os, pickle, sys, time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified import (TTS, S2T, Cfg, UnifiedELF, load_elf_init, net_to_v,
                     restore, sample_timesteps)

T_EPS = 0.05
T5_EOS, T5_PAD = 1, 0


RANK = int(os.environ.get("RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
DDP = WORLD > 1


def fmt_n(n):
    return f"{n:,}"


def log(*a):
    # only rank 0 writes, so a 4-GPU log reads exactly like a 1-GPU one
    if RANK == 0:
        print(*a, flush=True)


# ============================================================ data
class LibriCache(Dataset):
    def __init__(self, cache_dir, split="train", val_n=300, test_n=300, seed=0,
                 dur_range=None):
        """dur_range=(lo, hi) keeps only clips in that second range.

        The zero-shot-TTS literature evaluates on a duration-filtered
        subset -- F5-TTS releases a 4-to-10-second LibriSpeech-PC subset of
        1127 samples precisely because everyone was using different unlisted
        subsets of test-clean, so the numbers were not comparable.
        """
        with open(os.path.join(cache_dir, "index.pt"), "rb") as f:
            self.meta = pickle.load(f)
        self.dim = self.meta["dim"]
        self.lat = np.memmap(os.path.join(cache_dir, "latents.f16"),
                             dtype=np.float16, mode="r",
                             shape=(self.meta["total_frames"], self.dim))
        # the x-vector table, one row per clip. memmapped like the rest so the
        # DataLoader workers share it instead of each holding a copy.
        self.sdim = self.meta.get("spk_emb_dim", 0)
        self.spk = None
        if self.sdim:
            sf = self.meta.get("spk_emb_file", "spkemb.f32")
            self.spk = np.memmap(os.path.join(cache_dir, sf),
                                 dtype=np.float32, mode="r",
                                 shape=(len(self.meta["records"]), self.sdim))
        self.wdim = self.meta.get("whisper_dim", 0)
        self.wh = None
        if self.wdim:
            # the filename comes from the meta so several encoders' features can
            # coexist in one cache (whisper.f16, whisper_lv3.f16, ...)
            wf = self.meta.get("whisper_file", "whisper.f16")
            self.wh = np.memmap(os.path.join(cache_dir, wf),
                                dtype=np.float16, mode="r",
                                shape=(self.meta["whisper_total"], self.wdim))
        recs = self.meta["records"]
        idx = np.random.default_rng(seed).permutation(len(recs))
        if split == "all":
            # an official evaluation set is used whole: carving a val/test slice
            # out of it would make our WER incomparable to published numbers
            keep = np.arange(len(recs))
        elif split == "val":
            keep = idx[:val_n]
        elif split == "test":
            keep = idx[val_n:val_n + test_n]
        else:
            keep = idx[val_n + test_n:]
        self.recs = [recs[i] for i in keep]
        if dur_range is not None:
            lo, hi = dur_range
            before = len(self.recs)
            self.recs = [r for r in self.recs if lo <= r["dur"] <= hi]
            self.dur_filter = (lo, hi, before, len(self.recs))
        else:
            self.dur_filter = None

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        out = {"mel": np.asarray(self.lat[r["offset"]:r["offset"] + r["T"]]),
               "chars": r["chars"], "t5": r["t5"], "spk": r["spk"],
               "uid": r["id"], "i": i}
        if self.wh is not None:
            out["wh"] = np.asarray(
                self.wh[r["w_offset"]:r["w_offset"] + r["w_T"]])
        if self.spk is not None:
            out["spk_emb"] = np.asarray(self.spk[r["spk_row"]])
        return out


class Tagged(Dataset):
    """One source pinned to one task, its mel moved onto the reference stats.

    The two-dataset setup trains TTS on LJSpeech (F5-TTS small's single-speaker
    column in its Table 9) and S2T on LibriSpeech 960 h (ELF-S2T's data), so the
    task is a property of where a clip came from, not a coin flip in collate.
    build_targets standardises every mel with ONE (mean, std) -- the TTS
    cache's, because that is the space the vocoder decodes from -- so a clip
    from another source is first mapped affinely onto those stats. After
    build_targets it is then standardised with its OWN mean and std, exactly
    what each single-cache run fed the model.
    """

    def __init__(self, base, task, ref_stats):
        self.base, self.task, self.meta = base, task, base.meta
        mu, sd = base.meta["latent_mean"], base.meta["latent_std"]
        rmu, rsd = ref_stats
        self.scale = rsd / sd
        self.shift = rmu - mu * self.scale

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base[i]
        it["task"] = self.task
        it["mel"] = it["mel"].astype(np.float32) * self.scale + self.shift
        if self.task == TTS:
            # a TTS row never reads Whisper features, and an older cache may
            # carry another encoder's (cache_lj_uni has whisper-small, 768-d)
            it.pop("wh", None)
        return it


class TaskSampler(torch.utils.data.Sampler):
    """Infinite, one stream per rank: each draw picks the task with p_s2t, then
    a clip uniformly from that task's sources.

    The sources differ ~20x in size (LJSpeech 13,100 clips, LibriSpeech 960 h
    ~265k), so shuffling their union would make ~95% of the rows S2T. The task
    split has to be applied per draw. Indices address a ConcatDataset laid out
    as [TTS clips | S2T clips].
    """

    def __init__(self, n_tts, n_s2t, p_s2t, seed, tts_groups=None, fixed_s2t=-1, block=0):
        self.n_tts, self.n_s2t, self.p, self.seed = n_tts, n_s2t, p_s2t, seed
        # fixed_s2t >= 0 (--fixed_s2t_rows): every `block` consecutive draws --
        # one per-rank micro-batch -- hold EXACTLY fixed_s2t S2T rows, the rest
        # TTS, instead of a coin per row. With 112 supervised S2T slots per row
        # the S2T denominator is then the same in every micro-batch, so
        # averaging micro-batch losses equals the merged-batch loss for S2T
        # (CODE_REVIEW 14.4). -1 = the per-row coin, as before.
        self.fixed, self.block = fixed_s2t, block
        # (offset, size, weight) per TTS group: a TTS row first picks a group
        # by weight, then a clip uniformly inside it. One group = the old
        # uniform-over-clips behaviour.
        self.groups = tts_groups or [(0, n_tts, 1.0)]
        w = np.array([g[2] for g in self.groups], dtype=np.float64)
        self.gp = w / w.sum()

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        if self.fixed >= 0:
            while True:
                for k in range(self.block):
                    if k < self.fixed:
                        yield self.n_tts + int(rng.integers(self.n_s2t))
                    else:
                        g = (self.groups[int(rng.choice(len(self.groups), p=self.gp))]
                             if len(self.groups) > 1 else self.groups[0])
                        yield g[0] + int(rng.integers(g[1]))
        while True:
            if rng.random() < self.p:
                yield self.n_tts + int(rng.integers(self.n_s2t))
            else:
                g = (self.groups[int(rng.choice(len(self.groups), p=self.gp))]
                     if len(self.groups) > 1 else self.groups[0])
                yield g[0] + int(rng.integers(g[1]))

    def __len__(self):
        return 2 ** 31            # never exhausted: --steps ends the run


class Interleave(Dataset):
    """a[0], b[0], a[1], b[1], ... so every eval batch holds both tasks."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n = 2 * min(len(a), len(b))

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        return (self.a if k % 2 == 0 else self.b)[k // 2]


class SpeakerPack(Dataset):
    """A TTS row of SEVERAL clips by one speaker, back to back (multi-speaker runs
    only; --tts_pack_prob, default 0 = off).

    A LibriTTS clip averages about 5.8 s, a third of the 1,536-frame mel region,
    so two thirds of every TTS row was padding the backbone still attends over.
    With probability p a row is filled with further clips of the SAME speaker
    (same chapter first: same recording session, so the joins do not also change
    the room) while the mel stays within max_frames and the joined text within
    max_text T5 tokens; the text is re-tokenised whole, so it carries one EOS.
    The prompt is still drawn by collate as a fraction of the row, so a packed
    row with a short prompt asks for the next sentences in the voice of an
    earlier one -- the cross-sentence task the LibriSpeech-PC test poses. The
    x-vector stays the FIRST clip's, the one the prompt region covers.
    """

    def __init__(self, inner, max_frames, max_text, p, seed=0):
        from collections import defaultdict
        from transformers import AutoTokenizer
        self.inner, self.meta, self.task = inner, inner.meta, inner.task
        self.max_frames, self.max_text, self.p, self.seed = max_frames, max_text, p, seed
        recs = inner.base.recs
        self.stoi = self.meta["stoi"]
        self.itos = {v: k for k, v in self.stoi.items()}
        by = defaultdict(list)
        for i, r in enumerate(recs):
            by[r["spk"]].append(i)
        self.same = [by[r["spk"]] for r in recs]
        self.chap = [r.get("chap") for r in recs]
        self.T = [r["T"] for r in recs]
        self.chars = [r["chars"] for r in recs]
        self.tk = AutoTokenizer.from_pretrained("t5-small")
        self._st = {}

    def __len__(self):
        return len(self.inner)

    def _rng(self):
        wi = torch.utils.data.get_worker_info()
        key = None if wi is None else wi.seed
        if self._st.get("key", "unset") != key:
            self._st["key"] = key
            self._st["rng"] = np.random.default_rng(
                [self.seed, int(os.environ.get("RANK", 0)), 0 if wi is None else wi.seed % (2 ** 63)])
        return self._st["rng"]

    def _text(self, j):
        return "".join(self.itos[int(c)] for c in self.chars[j]).strip()

    def __getitem__(self, i):
        it = self.inner[i]
        rng = self._rng()
        if self.p <= 0.0 or rng.random() >= self.p:
            return it
        cand = [j for j in self.same[i] if j != i]
        if not cand:
            return it
        cand = [cand[k] for k in rng.permutation(len(cand))]
        cand.sort(key=lambda j: self.chap[j] != self.chap[i])     # stable
        texts, picked, T = [self._text(i)], [], it["mel"].shape[0]
        for j in cand:
            if T + self.T[j] > self.max_frames:
                continue
            tj = self._text(j)
            if len(self.tk(" ".join(texts + [tj])).input_ids) > self.max_text:
                continue
            texts.append(tj); picked.append(j); T += self.T[j]
            if T > 0.9 * self.max_frames:
                break
        if not picked:
            return it
        parts = [self.inner[j] for j in picked]
        text = " ".join(texts)
        out = dict(it)
        out["mel"] = np.concatenate([it["mel"]] + [x["mel"] for x in parts], 0)
        out["chars"] = np.array([self.stoi[c] for c in text], dtype=np.int16)
        out["t5"] = np.asarray(self.tk(text).input_ids, dtype=np.int32)
        out["uid"] = "+".join([it["uid"]] + [x["uid"] for x in parts])
        return out


def make_collate(max_length, p_s2t, prompt_frac, rng_seed=0, max_text=48,
                 prompt_sec=0.0,
                 itos=None, tk=None, layout="channel", whisper_rate=0.0,
                 frame_rate=93.75, whisper_dim=0):
    """Build the batch for either layout.

    Emits, for every sequence position, which dense-block slot supplies its text
    latent (t5_idx, -1 = none) so one gather serves both layouts, plus separate
    activity and clean masks per channel.
    """
    # augmentation RNG (crop offsets, prompt length, task coin). Built once here
    # it was copied into every DataLoader worker on every rank, so 4 workers x 4
    # ranks drew the SAME sequence (CODE_REVIEW_2026-10-20 item 5). In a worker
    # it is now reseeded from that worker's own torch seed, which PyTorch makes
    # distinct per worker and per iterator; the rank is mixed in explicitly. The
    # main process (every eval script) keeps default_rng(rng_seed) exactly.
    main_rng = np.random.default_rng(rng_seed)
    wstate = {}

    class _R:
        def __getattr__(self, name):
            wi = torch.utils.data.get_worker_info()
            if wi is None:
                return getattr(main_rng, name)
            if wstate.get("key") != wi.seed:
                wstate["key"] = wi.seed
                wstate["rng"] = np.random.default_rng(
                    [rng_seed, int(os.environ.get("RANK", 0)), wi.seed % (2 ** 63)])
            return getattr(wstate["rng"], name)

    rng = _R()
    max_mel = max_length if layout == "channel" else max_length - max_text
    # the token region sits at a FIXED offset so its RoPE positions do not move
    # with the audio length
    tok0 = max_mel

    def collate(items):
        B, L, D = len(items), max_length, items[0]["mel"].shape[1]
        mel = np.zeros((B, L, D), dtype=np.float32)
        t5_tok = np.zeros((B, L), dtype=np.int64)
        t5_idx = np.full((B, L), -1, dtype=np.int64)
        chars = np.zeros((B, L), dtype=np.int64)
        v_act = np.zeros((B, L), dtype=np.float32)
        t_act = np.zeros((B, L), dtype=np.float32)
        prompt_v = np.zeros((B, L), dtype=np.float32)
        prompt_t = np.zeros((B, L), dtype=np.float32)
        valid = np.zeros((B, L), dtype=np.float32)
        task = np.zeros((B,), dtype=np.int64)
        n_text = np.zeros((B,), dtype=np.int64)
        text_start = np.zeros((B,), dtype=np.int64)
        crop_a = np.zeros((B,), dtype=np.int64)
        n_prompt = np.zeros((B,), dtype=np.int64)
        t5_dense = np.zeros((B, max_text), dtype=np.int64)
        t5_dmask = np.zeros((B, max_text), dtype=np.float32)
        # all(), not items[0]: caches built at different times disagree on
        # whether they carry spkemb.f32, and mixing them in one batch would
        # KeyError on the first item that lacks it.
        has_spk = all("spk_emb" in it for it in items)
        if has_spk:
            Ds = items[0]["spk_emb"].shape[0]
            spk_emb = np.zeros((B, Ds), dtype=np.float32)
        # With two sources, items[0] may be a TTS clip that carries no Whisper
        # features, and a batch may hold no S2T row at all (0.5^8 = 0.4% per
        # rank per step). The block must still exist then: without it
        # whisper_proj sits out of the forward and DDP, which expects every
        # parameter to receive a gradient, fails the step. whisper_dim > 0
        # forces the (zero, fully masked) block into every batch.
        has_wh = whisper_rate > 0 and (
            whisper_dim > 0 or any("wh" in it for it in items))
        if has_wh:
            Dw = whisper_dim or next(it["wh"].shape[1] for it in items
                                     if "wh" in it)
            wh = np.zeros((B, L, Dw), dtype=np.float32)
            w_mask = np.zeros((B, L), dtype=np.float32)

        for b, it in enumerate(items):
            m, ch, tok = it["mel"], it["chars"], it["t5"]
            T_full = m.shape[0]
            nch = max(len(ch), 1)
            # upsample over the FULL clip, then crop: upsampling after a crop
            # would stretch the whole transcript onto the fragment
            jf = np.arange(T_full)
            ch_full = ch[np.minimum((jf * nch) // max(T_full, 1), nch - 1)]
            T = min(T_full, max_mel)
            a = int(rng.integers(0, T_full - T + 1)) if T_full > T else 0
            if T < T_full and tk is not None:
                # the transcript has to follow the audio crop, or S2T is asked
                # to write 12 s of text from 5 s of audio
                c0 = (a * nch) // max(T_full, 1)
                c1 = max(c0 + 1, ((a + T) * nch) // max(T_full, 1))
                sub = "".join(itos.get(int(c), "") for c in ch[c0:c1])
                # snap to word boundaries: a half word fragments into junk t5
                # pieces and inflates WER for a reason the model cannot fix
                if c0 > 0 and " " in sub:
                    sub = sub[sub.index(" ") + 1:]
                if c1 < nch and " " in sub:
                    sub = sub[:sub.rindex(" ")]
                tok = np.asarray(tk(sub.strip()).input_ids, dtype=np.int32)
            N = max(1, min(len(tok), max_text))

            crop_a[b] = a
            if has_spk:
                spk_emb[b] = it["spk_emb"]
            mel[b, :T] = m[a:a + T]
            valid[b, :T] = 1.0
            v_act[b, :T] = 1.0
            t5_dense[b, :N] = tok[:N]
            t5_dmask[b, :N] = 1.0
            n_text[b] = N

            if layout == "channel":
                j = np.arange(T)
                up = np.minimum((j * N) // max(T, 1), N - 1)
                t5_idx[b, :T] = up
                t5_tok[b, :T] = tok[:N][up]
                t_act[b, :T] = 1.0
                text_start[b] = 0
                chars[b, :T] = ch_full[a:a + T]
            elif layout == "hybrid":
                # a Tagged source fixes the task; a single cache flips a coin
                is_s2t_row = (it["task"] == S2T) if "task" in it \
                    else rng.random() < p_s2t
                # token region: the real target, at token resolution
                t5_idx[b, tok0:tok0 + N] = np.arange(N)
                t5_tok[b, tok0:tok0 + N] = tok[:N]
                # the whole block is text, not just the N real tokens: giving
                # the model exactly N would hand it the answer's length
                t_act[b, tok0:tok0 + max_text] = 1.0
                valid[b, tok0:tok0 + max_text] = 1.0
                text_start[b] = tok0
                if not is_s2t_row:
                    # TTS also gets the per-frame upsampled text, on the same
                    # pretrained z_T channel, AND the finer character signal.
                    # Both are gated here: leaving `chars` outside this branch
                    # was a leak -- every S2T row then received the character
                    # sequence of its own answer through the cond block.
                    j = np.arange(T)
                    up = np.minimum((j * N) // max(T, 1), N - 1)
                    t5_idx[b, :T] = up
                    t_act[b, :T] = 1.0
                    chars[b, :T] = ch_full[a:a + T]
                task[b] = S2T if is_s2t_row else TTS
            else:
                mel_end = T
                assert mel_end + max_text <= L
                t5_idx[b, mel_end:mel_end + N] = np.arange(N)
                t5_tok[b, mel_end:mel_end + N] = tok[:N]
                t_act[b, mel_end:mel_end + max_text] = 1.0
                valid[b, mel_end:mel_end + max_text] = 1.0
                text_start[b] = mel_end
                chars[b, :T] = ch_full[a:a + T]

            if layout != "hybrid":
                task[b] = S2T if rng.random() < p_s2t else TTS
            if has_wh and task[b] == S2T:
                assert "wh" in it, f"S2T clip {it.get('uid')} has no Whisper features"
                # Whisper runs at 50 Hz, our mel at 93.75 Hz, so map our
                # cropped frame p (absolute frame a+p) onto a Whisper frame by
                # nearest neighbour -- the same gather the text upsampling
                # uses, and no interpolation inventing frames never encoded.
                wT = it["wh"].shape[0]
                wi = np.minimum(
                    ((a + np.arange(T)) * (whisper_rate / frame_rate))
                    .astype(np.int64), wT - 1)
                wh[b, :T] = it["wh"][wi]
                w_mask[b, :T] = 1.0
            if task[b] == S2T:
                prompt_v[b] = v_act[b]          # audio is the condition
                # prompt_t stays 0 on the token region: that is the target
            else:
                prompt_t[b] = t_act[b]          # all text is the condition
                # voice prompt: with many speakers, no prompt makes text ->
                # speech one-to-many in speaker identity and the L2 optimum is
                # the speaker average
                # The zero-shot-TTS literature prompts with a fixed DURATION
                # (3 s in Voicebox / VALL-E / F5-TTS), not a fraction of the
                # clip. A fraction makes the prompt shorter on short clips and
                # longer on long ones, so WER then mixes the model's quality
                # with the clip-length distribution. prompt_sec > 0 selects the
                # literature's protocol; the fraction stays the training-time
                # augmentation it always was.
                if prompt_sec > 0.0:
                    K = min(T - 1, int(round(prompt_sec * frame_rate)))
                else:
                    K = int(T * rng.uniform(*prompt_frac))
                if K > 0:
                    prompt_v[b, :K] = 1.0
                    n_prompt[b] = K

        return {"mel": torch.from_numpy(mel),
                "n_prompt": torch.from_numpy(n_prompt),
                **({"spk_emb": torch.from_numpy(spk_emb)} if has_spk else {}),
                "t5_tok": torch.from_numpy(t5_tok),
                "t5_idx": torch.from_numpy(t5_idx),
                "t5_dense": torch.from_numpy(t5_dense),
                "t5_dmask": torch.from_numpy(t5_dmask),
                "chars": torch.from_numpy(chars),
                "v_act": torch.from_numpy(v_act),
                "t_act": torch.from_numpy(t_act),
                "prompt_v": torch.from_numpy(prompt_v),
                "prompt_t": torch.from_numpy(prompt_t),
                "valid": torch.from_numpy(valid),
                "task": torch.from_numpy(task),
                "n_text": torch.from_numpy(n_text),
                "text_start": torch.from_numpy(text_start),
                "crop_a": torch.from_numpy(crop_a),
                "uid": [it.get("uid", "") for it in items],
                **({"wh": torch.from_numpy(wh),
                    "w_mask": torch.from_numpy(w_mask)} if has_wh else {})}

    return collate


# ============================================================ text encoder
class TextEncoder(torch.nn.Module):
    """Frozen t5-small encoder. The only encoder in the system.

    latent_std 0.2 is ELF's own, and it is the ENCODER OUTPUT that it divides,
    not the embedding table: sampling unconditionally from ELF-B-owt produced
    coherent English with x0 std 0.81-0.88, and t5-small's encoder output has
    std 0.1585 (0.1585/0.2 = 0.79) while its embedding table has std 17.26.
    """

    def __init__(self, latent_std=0.2):
        super().__init__()
        from transformers import T5EncoderModel
        self.t5 = T5EncoderModel.from_pretrained("t5-small").eval()
        for p in self.t5.parameters():
            p.requires_grad_(False)
        self.latent_std = latent_std

    @torch.no_grad()
    def forward(self, ids, mask):
        h = self.t5(input_ids=ids, attention_mask=mask).last_hidden_state
        return h.float() / self.latent_std


def spk_cond(batch, device):
    """The x-vector for this batch, or None when the cache has none."""
    e = batch.get("spk_emb")
    return None if e is None else e.to(device, non_blocking=True)


def model_spk_dim(ck_args, meta):
    """spk_dim for rebuilding a checkpoint: the flag it was TRAINED with."""
    return meta.get("spk_emb_dim", 0) if ck_args.get("spk") else 0


def audio_cond(batch, device):
    """-> (features, mask) for the frozen-encoder audio condition, or (None, None)."""
    if "wh" not in batch:
        return None, None
    return (batch["wh"].to(device, non_blocking=True),
            batch["w_mask"].to(device, non_blocking=True))


def build_targets(batch, text_enc, device, mel_stats):
    """-> x0_t (B,L,512), x0_v (B,L,100), zero where their channel is off."""
    mel = batch["mel"].to(device, non_blocking=True)
    v_act = batch["v_act"].to(device, non_blocking=True)
    t_act = batch["t_act"].to(device, non_blocking=True)
    idx = batch["t5_idx"].to(device, non_blocking=True)
    mu, sd = mel_stats

    x0_v = ((mel - mu) / sd) * v_act.unsqueeze(-1)
    dm = batch["t5_dmask"].to(device)
    hd = text_enc(batch["t5_dense"].to(device), dm) * dm.unsqueeze(-1)
    # one gather serves both layouts: -1 means this position has no text
    safe = idx.clamp(min=0)
    x0_t = torch.gather(hd, 1, safe.unsqueeze(-1).expand(-1, -1, hd.shape[-1]))
    x0_t = x0_t * (idx >= 0).unsqueeze(-1).float() * t_act.unsqueeze(-1)
    return x0_t, x0_v


# ============================================================ step
def _amp(cfg):
    """bf16 autocast for the forwards when --bf16 (default off: fp32 as before)."""
    if getattr(cfg, "bf16", 0):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train_step(model, text_enc, ema, batch, cfg, opt, device, mel_stats,
               accum_i=0, accum_n=1):
    chars = batch["chars"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    va = batch["v_act"].to(device, non_blocking=True)
    taq = batch["t_act"].to(device, non_blocking=True)
    v_act, t_act = va.unsqueeze(-1), taq.unsqueeze(-1)
    p_v = batch["prompt_v"].to(device, non_blocking=True)
    p_t = batch["prompt_t"].to(device, non_blocking=True)
    task = batch["task"].to(device, non_blocking=True)
    t5_tok = batch["t5_tok"].to(device, non_blocking=True)
    x0_t, x0_v = build_targets(batch, text_enc, device, mel_stats)
    ac, am = audio_cond(batch, device)
    se = spk_cond(batch, device)
    B, L, _ = x0_v.shape
    is_s2t = (task == S2T).float()

    # ---- condition dropout, BEFORE z is built ----
    # Both drops used to run AFTER z_v / z_t had been constructed from x0, so
    # zeroing x0 never reached the model's main input: z_V kept the full mel
    # under audio_drop and z_T kept the full T5 latent under text_drop. Only the
    # self-conditioning copies (rebuilt from the zeroed x0 by restore()) and the
    # Whisper/char/cond blocks went to zero, which left the model seeing "full
    # text in z_T, zero text in sc_T" -- a contradictory state that exists in
    # neither training nor inference, instead of an unconditional input. Doing
    # the drop here, before z, makes every downstream copy agree.
    #
    # audio_drop (S2T rows): x0_v is pure condition there and never a target, so
    # zeroing it corrupts no loss and removes the mel from z_V for real.
    drop_audio = ((torch.rand(B, device=device) < cfg.audio_drop_prob).float()
                  * is_s2t)
    if cfg.audio_drop_prob > 0.0:
        keep = (1.0 - drop_audio).view(-1, 1, 1)
        x0_v = x0_v * keep
        if ac is not None:
            ac = ac * keep
    # text_drop (TTS rows): zero the T5 latent and the char block together --
    # exactly the unconditional branch the TTS CFG uses at inference. F5-TTS
    # drops text at 0.2: "the two-stage control of CFG training may have the
    # model learn more with text alignment". x0_t is pure condition on a TTS row
    # and never a target, so this corrupts no loss either.
    drop = (torch.rand(B, device=device) < cfg.label_drop_prob).float()
    if cfg.text_drop_prob > 0.0:
        td = ((torch.rand(B, device=device) < cfg.text_drop_prob).float()
              * (1 - is_s2t))
        x0_t = x0_t * (1 - td).view(-1, 1, 1)
        chars = chars * (1 - td).long().view(-1, 1)
        drop = torch.clamp(drop + td, max=1.0)
    # F5-TTS's two-stage CFG dropout for TTS rows (multi-speaker runs only;
    # both default off). prompt_drop_prob: the row loses its audio prompt -- the
    # whole mel becomes the target -- which teaches the model the prompt-free
    # branch a prompt-side guidance needs. joint_cond_drop: the text-dropped rows
    # also lose the prompt AND the x-vector, so "no condition at all" exists as
    # one state, like F5's drop of text with the masked speech together.
    pd_p, jd = getattr(cfg, "prompt_drop_prob", 0.0), getattr(cfg, "joint_cond_drop", 0)
    if pd_p > 0.0 or jd:
        pdrop = (torch.rand(B, device=device) < pd_p).float() * (1 - is_s2t)
        if jd and cfg.text_drop_prob > 0.0:
            pdrop = torch.clamp(pdrop + td, max=1.0)
            if se is not None:
                se = se * (1 - td).view(-1, 1)
        p_v = p_v * (1 - pdrop).view(-1, 1)

    # the channel that carries noise, per row; the other is the condition
    tgt = (is_s2t.view(-1, 1) * taq * (1 - p_t)
           + (1 - is_s2t).view(-1, 1) * va * (1 - p_v))

    t = sample_timesteps(B, cfg.denoiser_p_mean, cfg.denoiser_p_std, device)
    tt = t.view(-1, 1, 1)
    ns = cfg.denoiser_noise_scale
    z_v = restore(tt * x0_v + (1 - tt) * torch.randn_like(x0_v) * ns,
                  x0_v, p_v) * v_act
    z_t = restore(tt * x0_t + (1 - tt) * torch.randn_like(x0_t) * ns,
                  x0_t, p_t) * t_act

    sc_cfg = torch.empty(B, device=device).uniform_(
        cfg.self_cond_cfg_min, cfg.self_cond_cfg_max)
    d = torch.bernoulli(torch.full((B,), cfg.decoder_prob, device=device)) * is_s2t

    # ELF's target is the SC-CFG-sharpened velocity: one no_grad forward with
    # zero self-conditioning gives x_u, a second conditioned on x_u gives the
    # sharpened direction. Dropping it still trains a valid flow model, but not
    # ELF's, and ACFG at sampling time would have nothing to use.
    use_sc = (torch.rand(B, device=device) < cfg.self_cond_prob).float().view(-1, 1, 1)
    # the unwrapped module for the no-grad passes: DDP only needs to see the
    # one forward that feeds the loss, and this keeps its bookkeeping out of
    # the target computation entirely
    core = getattr(model, "module", model)
    with torch.no_grad():
        zt_, zv_ = torch.zeros_like(z_t), torch.zeros_like(z_v)
        with _amp(cfg):
            ou_t, ou_v, _ = core(z_t, z_v, zt_, zv_, chars, t, task,
                                  attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                                  drop_cond=drop, audio_cond=ac, audio_cond_mask=am,
                                  spk_emb=se)
        ou_t, ou_v = ou_t.float(), ou_v.float()
        xu_t = restore(ou_t, x0_t, p_t) * t_act
        xu_v = restore(ou_v, x0_v, p_v) * v_act
        with _amp(cfg):
            oc_t, oc_v, _ = core(z_t, z_v, xu_t, xu_v, chars, t, task,
                                  attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                                  drop_cond=drop, audio_cond=ac, audio_cond_mask=am,
                                  spk_emb=se)
        oc_t, oc_v = oc_t.float(), oc_v.float()
        g = 1.0 - 1.0 / sc_cfg.view(-1, 1, 1)
        gui_t = g * (net_to_v(oc_t, z_t, t) - net_to_v(ou_t, z_t, t))
        gui_v = g * (net_to_v(oc_v, z_v, t) - net_to_v(ou_v, z_v, t))
        gui_t = torch.where(use_sc.bool(), gui_t, torch.zeros_like(gui_t))
        gui_v = torch.where(use_sc.bool(), gui_v, torch.zeros_like(gui_v))
        sc_t, sc_v = xu_t * use_sc, xu_v * use_sc

    # decoder rows see a near-clean text latent at t = 1, as in ELF
    lam = torch.sigmoid(torch.randn((B, L, 1), device=device)
                        * cfg.decoder_p_std + cfg.decoder_p_mean)
    dec_t = (lam * x0_t + (1 - lam) * torch.randn_like(x0_t)
             * cfg.decoder_noise_scale) * t_act
    dv = d.view(-1, 1, 1)
    z_t_in = dv * dec_t + (1 - dv) * z_t
    t_in = d * torch.ones_like(t) + (1 - d) * t
    sc_t = restore(sc_t, x0_t, p_t) * t_act * (1 - dv)
    sc_v = restore(sc_v, x0_v, p_v) * v_act * (1 - dv)

    # gradient accumulation: on every micro-batch but the last, the forward
    # that builds the graph AND its backward run inside DDP's no_sync(); DDP
    # requires the forward inside too, otherwise each backward still
    # all-reduces (CODE_REVIEW 14.3). The no-grad passes above use `core`.
    last = accum_i == accum_n - 1
    nosync = contextlib.ExitStack()
    if DDP and not last:
        nosync.enter_context(model.no_sync())
    with _amp(cfg):
        xh_t, xh_v, logits = model(z_t_in, z_v, sc_t, sc_v, chars, t_in, task,
                                   attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                                   decoder_step_active=d, drop_cond=drop,
                                   audio_cond=ac, audio_cond_mask=am,
                                  spk_emb=se)
    xh_t, xh_v = xh_t.float(), xh_v.float()

    v_tgt_t = (net_to_v(x0_t, z_t, t) + gui_t).detach()
    v_tgt_v = (net_to_v(x0_v, z_v, t) + gui_v).detach()
    l2_t = ((net_to_v(xh_t, z_t, t) - v_tgt_t) ** 2).mean(-1)
    l2_v = ((net_to_v(xh_v, z_v, t) - v_tgt_v) ** 2).mean(-1)
    l2 = is_s2t.view(-1, 1) * l2_t + (1 - is_s2t).view(-1, 1) * l2_v

    with torch.no_grad():
        # the trivial predictor, xhat0 = 0, scores ||x0/(1-t)||^2. The RATIO is
        # the readable number: 1.0 = no better than generating nothing.
        inv = 1.0 / torch.clamp(1.0 - t.view(-1, 1, 1), min=T_EPS)
        triv = (is_s2t.view(-1, 1) * ((x0_t * inv) ** 2).mean(-1)
                + (1 - is_s2t).view(-1, 1) * ((x0_v * inv) ** 2).mean(-1))

    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                         t5_tok.reshape(-1), reduction="none").view(B, L)

    m_ce = tgt * d.view(-1, 1)
    m_l2 = tgt * (1 - d).view(-1, 1)
    w = (1.0 + (cfg.s2t_weight - 1.0) * is_s2t).view(-1, 1)
    # DDP averages gradients across ranks. Normalising by a PER-RANK denom would
    # make that the mean of per-rank means, which is not the global mean when the
    # ranks hold different numbers of target positions -- the effective learning
    # rate would then drift with the batch composition. All-reduce the count so
    # the loss is normalised by the global denominator and 4 GPUs reproduce 1.
    num = (ce * m_ce * w).sum() + (l2 * m_l2 * w).sum()
    denom = (tgt * w).sum()
    if getattr(cfg, "task_loss_norm", 0):
        # --task_loss_norm (CODE_REVIEW 14.2): each task is averaged over its OWN
        # target positions (global over ranks), then weighted explicitly:
        #   loss = w_s2t * S2T_sum / S2T_targets + (1 - w_s2t) * TTS_sum / TTS_targets
        # The single shared denominator below gives S2T the weight
        # S2T_targets / all_targets instead, which a 7x larger TTS batch would
        # cut from ~6-7 % to ~0.6 %. Equal to the old objective whenever w_s2t
        # equals that fraction.
        ws = float(cfg.s2t_loss_weight)
        sm = is_s2t.view(-1, 1)
        num_s = (ce * m_ce * sm).sum() + (l2 * m_l2 * sm).sum()
        num_t = (l2 * m_l2 * (1 - sm)).sum()
        q = torch.stack([num_s.detach(), (tgt * sm).sum(), num_t.detach(), (tgt * (1 - sm)).sum()])
        if DDP:
            torch.distributed.all_reduce(q)
        Ds, Dt = q[1].clamp(min=1.0), q[3].clamp(min=1.0)
        loss = (WORLD if DDP else 1) * (ws * num_s / Ds + (1 - ws) * num_t / Dt)
        loss_log = float(ws * q[0] / Ds + (1 - ws) * q[2] / Dt)
        d_local, d_global = float(denom.detach()), float(q[1] + q[3])
    elif DDP:
        # reduce the numerator too, purely so the logged number is the GLOBAL
        # loss: rank 0's own num over the global denom is unbiased but noisy,
        # and at per-rank batch 8 a rank has no CE row at all 10% of the time
        # (0.75^8) versus 1e-4 at batch 32, which made the 4-GPU log read ~20%
        # below the 1-GPU log for no real reason.
        tot = torch.stack([num.detach(), denom])
        torch.distributed.all_reduce(tot)
        num_g, denom_g = tot[0], tot[1]
        loss = num * WORLD / denom_g.clamp(min=1.0)
        loss_log = (num_g / denom_g.clamp(min=1.0)).item()
        d_local, d_global = float(denom.detach()), float(denom_g)
    else:
        loss = num / denom.clamp(min=1.0)
        loss_log = loss.item()
        d_local = d_global = float(denom.detach())

    # gradient accumulation (--grad_accum N, default 1 = one backward per step,
    # exactly as before): N micro-batches feed one optimizer step. Each micro
    # loss is already normalised by its own global denominator, so the step's
    # gradient is the mean of the N micro-batch gradients (equal-size packed
    # micro-batches make that the global mean up to their size spread). Only the
    # last micro-batch all-reduces gradients across ranks (DDP no_sync).
    # what this micro-batch actually trained on, summed over ranks (CODE_REVIEW
    # 14.5): S2T rows / target text tokens; TTS rows / source clips packed in /
    # mel frames incl. prompt (no padding) / supervised mel frames
    with torch.no_grad():
        ttsm = 1 - is_s2t
        n_clips = sum(u.count("+") + 1 for u, k in zip(batch["uid"], batch["task"].tolist()) if k == TTS)
        ex = torch.stack([is_s2t.sum(), (batch["n_text"].to(device).float() * is_s2t).sum(),
                          ttsm.sum(), torch.tensor(float(n_clips), device=device),
                          (va.sum(-1) * ttsm).sum(), (tgt.sum(-1) * ttsm).sum()])
        if DDP:
            torch.distributed.all_reduce(ex)
        expo = [float(x) for x in ex]
    if accum_i == 0:
        opt.zero_grad(set_to_none=True)
    (loss / accum_n if accum_n > 1 else loss).backward()
    nosync.close()
    if not last:
        return {"loss": loss_log, "partial": True, "expo": expo}
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.mul_(cfg.ema_decay).add_(pm.detach(), alpha=1 - cfg.ema_decay)

        m_s2t = m_l2 * is_s2t.view(-1, 1)
        m_tts = m_l2 * (1 - is_s2t).view(-1, 1)
        # Every ratio here is num/den over the rows this rank happens to hold.
        # At per-rank batch 8 that is badly undersampled: a row joins m_s2t only
        # if it is S2T and did not take the decoder draw (p = 0.25), so rank 0
        # holds no such row 0.75^8 = 10% of the time and logged s2t/tr as a flat
        # 0.0000 -- which it did, on 4 of 37 points. At batch 32 that is 1e-4.
        # Reduce numerators and denominators so each logged ratio is the global
        # one; the gradient was always global, only the log was not.
        def ratio_parts(mm):
            return torch.stack([(l2 * mm).sum(), (triv * mm).sum()])
        parts = torch.stack([
            ratio_parts(m_l2), ratio_parts(m_tts), ratio_parts(m_s2t),
            torch.stack([((logits.argmax(-1) == t5_tok) * m_ce).sum().float(),
                         m_ce.sum().float()]),
            torch.stack([(ce * m_ce).sum(), m_ce.sum().float()]),
            torch.stack([(l2 * m_l2).sum(), m_l2.sum().float()]),
            torch.stack([is_s2t.sum(), torch.full_like(is_s2t.sum(),
                                                       float(is_s2t.numel()))]),
        ])
        if DDP:
            torch.distributed.all_reduce(parts)
        rr = lambda i: float(parts[i, 0] / parts[i, 1].clamp(min=1e-9))
        r_all, r_tts_v, r_s2t_v = rr(0), rr(1), rr(2)
        acc, ce_m, l2_m, frac_s2t = rr(3), rr(4), rr(5), rr(6)
    return {"loss": loss_log, "grad_norm": float(gn),
            "denom_local": d_local, "denom_global": d_global,
            "l2_ratio": r_all, "r_s2t": r_s2t_v, "r_tts": r_tts_v,
            "ce": ce_m, "l2": l2_m,
            "ce_acc": acc, "frac_s2t": frac_s2t, "expo": expo}


# ============================================================ eval
def wer_counts(ref, hyp):
    """Levenshtein with the edit types kept apart.

    WER alone stops ordering models once it passes 100%: "hallucinates twice the
    words" and "three times" are both just a big number. The split says WHICH
    failure it is -- all deletions means silence, all insertions means babble,
    and those need opposite fixes.
    """
    r, h = ref.split(), hyp.split()
    nr, nh = len(r), len(h)
    d = np.zeros((nr + 1, nh + 1), dtype=np.int32)
    # op: 0 none, 1 sub, 2 del (ref word dropped), 3 ins (hyp word added)
    op = np.zeros((nr + 1, nh + 1), dtype=np.int8)
    d[:, 0] = np.arange(nr + 1); op[1:, 0] = 2
    d[0, :] = np.arange(nh + 1); op[0, 1:] = 3
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]; op[i, j] = 0
                continue
            cs, cd, ci = d[i - 1, j - 1] + 1, d[i - 1, j] + 1, d[i, j - 1] + 1
            best = min(cs, cd, ci)
            d[i, j] = best
            op[i, j] = 1 if best == cs else (2 if best == cd else 3)
    i, j = nr, nh
    sub = dele = ins = 0
    while i > 0 or j > 0:
        o = op[i, j]
        if o == 0:
            i -= 1; j -= 1
        elif o == 1:
            sub += 1; i -= 1; j -= 1
        elif o == 2:
            dele += 1; i -= 1
        else:
            ins += 1; j -= 1
    return dict(sub=sub, dele=dele, ins=ins, err=int(d[nr, nh]), n=nr)


def wer_pair(ref, hyp):
    """Word error rate by Levenshtein on word tokens."""
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return int(d[len(r), len(h)]), len(r)


def collapse(ids):
    """CTC-style. The channel layout predicts one token per frame and a token
    spans ~27 frames, so adjacent duplicates are one token, not many."""
    out = []
    for t in ids:
        if t == T5_EOS:
            break
        if t != T5_PAD and (not out or t != out[-1]):
            out.append(t)
    return out


def make_schedule(n_steps, cfg, device, logit_normal=False):
    """Uniform by default, and that is deliberate.

    A logit-normal grid matched to the TRAINING t density is wrong:
    sigmoid(N(-1.5, 0.8)) puts every interior point below about 0.5, so 32 steps
    covered t up to 0.26 and the last step jumped from there to 1.0 in one Euler
    step. ODE integration has to cover [0, 1] regardless of where the training
    mass sits, and the one unconditional sample that produced coherent English
    from ELF-B-owt used a uniform grid.
    """
    if not logit_normal:
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    s = torch.sort(sample_timesteps(n_steps - 1, cfg.denoiser_p_mean,
                                    cfg.denoiser_p_std, device)).values
    return torch.cat([torch.zeros(1, device=device), s,
                      torch.ones(1, device=device)])


@torch.no_grad()
def rollout(model, text_enc, batch, cfg, device, mel_stats, steps=32, acfg=1.0):
    """Integrate from noise, holding the conditioning channel clean the whole
    way. This is the generation the task asks for, not a teacher-forced
    reconstruction."""
    chars = batch["chars"].to(device); valid = batch["valid"].to(device)
    v_act = batch["v_act"].to(device).unsqueeze(-1)
    t_act = batch["t_act"].to(device).unsqueeze(-1)
    p_v, p_t = batch["prompt_v"].to(device), batch["prompt_t"].to(device)
    task = batch["task"].to(device)
    x0_t, x0_v = build_targets(batch, text_enc, device, mel_stats)
    ac, am = audio_cond(batch, device)
    se = spk_cond(batch, device)
    B = x0_t.shape[0]
    ns = cfg.denoiser_noise_scale

    z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
    z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
    xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
    sc_cfg = torch.full((B,), acfg, device=device)
    ts = make_schedule(steps, cfg, device)
    for i in range(steps):
        t = torch.full((B,), float(ts[i]), device=device)
        xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                              attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                              audio_cond=ac, audio_cond_mask=am,
                              spk_emb=se)
        h = float(ts[i + 1]) - float(ts[i])
        z_t = restore(z_t + h * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
        z_v = restore(z_v + h * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
        xp_t = restore(xh_t, x0_t, p_t) * t_act
        xp_v = restore(xh_v, x0_v, p_v) * v_act

    t1 = torch.ones(B, device=device)
    # decode with the self-conditioning input zeroed: training's decoder rows
    # see exactly that ((1 - dv) zeroes sc_t, sc_v), and so does ELF-S2T
    _, _, logits = model(z_t, z_v, torch.zeros_like(xp_t), torch.zeros_like(xp_v),
                         chars, t1, task,
                         attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                         decoder_step_active=True,
                         audio_cond=ac, audio_cond_mask=am,
                              spk_emb=se)
    return z_t, z_v, logits, x0_t, x0_v


@torch.no_grad()
def evaluate(model, text_enc, loader, cfg, device, mel_stats, tk, layout,
             n_batches=4, steps=32, ext_dec=None):
    """Both directions, each against the baseline that makes its number mean
    something: on the error ratio, 1.000 is what generating nothing scores."""
    model.eval()
    A = dict(tn=0.0, td=0.0, sn=0.0, sd=0.0, err=0, ln=0, ntts=0, ns2t=0,
             tok=0, tokn=0)
    samples = []
    for bi, b in enumerate(loader):
        if bi >= n_batches:
            break
        z_t, z_v, logits, x0_t, x0_v = rollout(model, text_enc, b, cfg, device,
                                               mel_stats, steps=steps)
        task = b["task"].to(device)
        m_t = (b["t_act"].to(device) * (1 - b["prompt_t"].to(device))).unsqueeze(-1)
        m_v = (b["v_act"].to(device) * (1 - b["prompt_v"].to(device))).unsqueeze(-1)
        pred = logits.argmax(-1)
        if ext_dec is not None:
            s0 = int(b["text_start"][0]); Lt = b["t5_dense"].shape[1]
            pred = pred.clone()
            pred[:, s0:s0 + Lt] = ext_dec(z_t[:, s0:s0 + Lt]).argmax(-1)
        for i in range(z_t.shape[0]):
            if task[i].item() == S2T:
                A["ns2t"] += 1
                A["sn"] += ((z_t[i] - x0_t[i]) * m_t[i]).norm().item()
                A["sd"] += (x0_t[i] * m_t[i]).norm().item()
                n = int(b["n_text"][i]); st = int(b["text_start"][i])
                ref_ids = [t for t in b["t5_dense"][i, :n].tolist()
                           if t not in (T5_EOS, T5_PAD)]
                if layout == "channel":
                    # one token per frame, so adjacent duplicates are one token
                    hyp_ids = collapse(pred[i].tolist())
                else:
                    # hybrid / token: the token region IS the answer, read it
                    # straight. No collapse, so length is the model's own call
                    # via EOS rather than a function of frame-to-frame noise.
                    raw = pred[i, st:st + b["t5_dense"].shape[1]].tolist()
                    if T5_EOS in raw:
                        raw = raw[:raw.index(T5_EOS)]
                    hyp_ids = [t for t in raw if t != T5_PAD]
                ref = tk.decode(ref_ids, skip_special_tokens=True)
                hyp = tk.decode(hyp_ids, skip_special_tokens=True)
                e, l = wer_pair(ref, hyp)
                A["err"] += e; A["ln"] += l
                k = min(len(ref_ids), len(hyp_ids))
                A["tok"] += sum(1 for j in range(k) if ref_ids[j] == hyp_ids[j])
                A["tokn"] += len(ref_ids)
                if len(samples) < 3:
                    samples.append((ref, hyp))
            else:
                A["ntts"] += 1
                A["tn"] += ((z_v[i] - x0_v[i]) * m_v[i]).norm().item()
                A["td"] += (x0_v[i] * m_v[i]).norm().item()
    model.train()
    return {"tts_err_ratio": A["tn"] / max(A["td"], 1e-9),
            "s2t_err_ratio": A["sn"] / max(A["sd"], 1e-9),
            "s2t_wer": 100.0 * A["err"] / max(A["ln"], 1),
            "s2t_tok_acc": A["tok"] / max(A["tokn"], 1),
            "n_tts": A["ntts"], "n_s2t": A["ns2t"], "samples": samples}


@torch.no_grad()
def text_use(model, text_enc, bt, bs, device, mel_stats, ts=(0.1, 0.3), seed=0):
    """Does the TTS head read the text? x0_hat error on the generated mel when
    each clip gets ANOTHER clip's transcript, over the error with its own,
    same noisy input. 1.0 = the words are ignored; the LJ model that spoke
    at 28% WER reads 2.0 / 1.6 at t = 0.1 / 0.3.

    The rollout mel error above cannot answer this: a correct utterance with
    different timing is far from the recording frame by frame (two samples
    of one distribution sit ~1.4 apart on that ratio), so it rewards blur.
    """
    model.eval()
    out = {}
    for t in ts:
        g = torch.Generator().manual_seed(seed + int(1000 * t))
        errs, ref, eps = [], None, None
        for b in (bt, bs):
            x0_t, x0_v = build_targets(b, text_enc, device, mel_stats)
            if ref is None:
                ref = x0_v
                eps = torch.randn(x0_v.shape, generator=g).to(device)
            B = x0_v.shape[0]
            va = b["v_act"].to(device).unsqueeze(-1)
            tav = b["t_act"].to(device).unsqueeze(-1)
            p_v, p_t = b["prompt_v"].to(device), b["prompt_t"].to(device)
            z_v = restore(t * x0_v + (1 - t) * eps * 2.0, x0_v, p_v) * va
            z_t = restore(x0_t, x0_t, p_t) * tav
            _, xh_v, _ = model(z_t, z_v, torch.zeros_like(z_t), torch.zeros_like(z_v),
                               b["chars"].to(device), torch.full((B,), t, device=device),
                               b["task"].to(device), attention_mask=b["valid"].to(device),
                               self_cond_cfg_scale=torch.ones(B, device=device),
                               spk_emb=spk_cond(b, device))
            m = (b["v_act"].to(device) * (1 - p_v)).unsqueeze(-1)
            errs.append(((xh_v - ref) ** 2 * m).sum() / ((ref ** 2) * m).sum())
        out[t] = float(errs[1] / errs[0])
    model.train()
    return out


@torch.no_grad()
def exposure_gap(model, text_enc, batch, cfg, device, mel_stats, steps=32):
    """x_hat0 error on the model's own trajectory over the same error on the
    true noisy path at the same t, AND both absolute. The ratio alone cannot
    tell "both good" from "both useless", and those need opposite fixes -- it
    read 1.00 at every t once, on a model predicting the mean in both regimes.
    """
    chars = batch["chars"].to(device); valid = batch["valid"].to(device)
    v_act = batch["v_act"].to(device).unsqueeze(-1)
    t_act = batch["t_act"].to(device).unsqueeze(-1)
    p_v, p_t = batch["prompt_v"].to(device), batch["prompt_t"].to(device)
    task = batch["task"].to(device)
    x0_t, x0_v = build_targets(batch, text_enc, device, mel_stats)
    ac, am = audio_cond(batch, device)
    se = spk_cond(batch, device)
    B = x0_t.shape[0]
    s = (task == S2T).float().view(-1, 1, 1)
    m_t = (batch["t_act"].to(device) * (1 - p_t)).unsqueeze(-1) * s
    m_v = (batch["v_act"].to(device) * (1 - p_v)).unsqueeze(-1) * (1 - s)
    n_t = (x0_t * m_t).norm().item()
    n_v = (x0_v * m_v).norm().item()
    ns = cfg.denoiser_noise_scale
    sc_cfg = torch.ones(B, device=device)

    z_t = restore(torch.randn_like(x0_t) * ns, x0_t, p_t) * t_act
    z_v = restore(torch.randn_like(x0_v) * ns, x0_v, p_v) * v_act
    xp_t, xp_v = torch.zeros_like(z_t), torch.zeros_like(z_v)
    ts = make_schedule(steps, cfg, device)
    out = {}
    for i in range(steps):
        tv = float(ts[i])
        t = torch.full((B,), tv, device=device)
        xh_t, xh_v, _ = model(z_t, z_v, xp_t, xp_v, chars, t, task,
                              attention_mask=valid, self_cond_cfg_scale=sc_cfg,
                              audio_cond=ac, audio_cond_mask=am,
                              spk_emb=se)
        if i % max(steps // 4, 1) == 0:
            tt = t.view(-1, 1, 1)
            tf_t = restore(tt * x0_t + (1 - tt) * torch.randn_like(x0_t) * ns,
                           x0_t, p_t) * t_act
            tf_v = restore(tt * x0_v + (1 - tt) * torch.randn_like(x0_v) * ns,
                           x0_v, p_v) * v_act
            yh_t, yh_v, _ = model(tf_t, tf_v, torch.zeros_like(tf_t),
                                  torch.zeros_like(tf_v), chars, t, task,
                                  attention_mask=valid,
                                  self_cond_cfg_scale=sc_cfg,
                                  audio_cond=ac, audio_cond_mask=am,
                              spk_emb=se)
            for nm, xh, yh, x0, mm, nn in (
                    ("s2t", xh_t, yh_t, x0_t, m_t, n_t),
                    ("tts", xh_v, yh_v, x0_v, m_v, n_v)):
                er = ((xh - x0) * mm).norm().item() / max(nn, 1e-9)
                et = ((yh - x0) * mm).norm().item() / max(nn, 1e-9)
                out[f"{nm}_t{tv:.2f}"] = er / max(et, 1e-9)
                out[f"ABS_{nm}_t{tv:.2f}_roll"] = er
                out[f"ABS_{nm}_t{tv:.2f}_tf"] = et
        h = float(ts[i + 1]) - tv
        z_t = restore(z_t + h * net_to_v(xh_t, z_t, t), x0_t, p_t) * t_act
        z_v = restore(z_v + h * net_to_v(xh_v, z_v, t), x0_v, p_v) * v_act
        xp_t = restore(xh_t, x0_t, p_t) * t_act
        xp_v = restore(xh_v, x0_v, p_v) * v_act
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_cache", default="",
                    help="评测用的另一个 cache（如 LibriSpeech 官方 test-clean）；"
                         "留空则从 --cache 里切 val")
    ap.add_argument("--cache", default=os.path.join(
        HERE, "cache_libri_train_clean_100"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "uni"))
    ap.add_argument("--layout", choices=["hybrid", "channel", "token"],
                    default="hybrid")
    ap.add_argument("--model", default="ELF-B")
    ap.add_argument("--init", default=os.path.join(
        HERE, "ckpt", "ELF-B-owt-torch", "checkpoint_95085"))
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_text", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--p_s2t", type=float, default=0.5)
    ap.add_argument("--prompt_lo", type=float, default=0.0)
    ap.add_argument("--prompt_hi", type=float, default=0.3)
    ap.add_argument("--text_latent_std", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--eval_batches", type=int, default=4)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--init_frac", type=float, default=0.0)
    ap.add_argument("--denoiser_p_mean", type=float, default=-1.5)
    # ELF-S2T's "audio forcing". The decoder branch sees
    # z = lam*x0 + (1-lam)*noise with lam = sigmoid(N(decoder_p_mean, 0.8)).
    # At ELF's 0.8 that is lam ~ 0.69, a nearly clean latent the decode head
    # can read the answer off without consulting the audio at all. ELF-S2T
    # lowers it to -0.5 (lam ~ 0.38) so the head has to use the audio -- aimed
    # exactly at our S2T failure mode.
    ap.add_argument("--decoder_p_mean", type=float, default=0.8)
    # UniVoice weights its ASR loss at lambda = 0.005 to keep it from damaging
    # the generative half. Ours is 1.0; lowering it is the direct protection
    # for TTS if the two tasks turn out to fight over the shared backbone.
    ap.add_argument("--s2t_weight", type=float, default=1.0)
    # ELF-S2T trains the two objectives in a "1/1 split"; ELF's own default is
    # 0.2. Under a single denominator this IS the weight on the CE term.
    ap.add_argument("--decoder_prob", type=float, default=0.2)
    # fraction of S2T rows that lose the whole audio condition, which is what
    # creates the unconditional branch ACFG needs at inference
    ap.add_argument("--audio_drop_prob", type=float, default=0.0)
    # TTS rows: zero the T5 latent AND the char block together, which is
    # exactly the unconditional branch the TTS CFG uses at inference.
    ap.add_argument("--text_drop_prob", type=float, default=0.0)
    # drop the cond block entirely: 1736 -> 1224 input columns, 283,136 fewer
    # parameters, ELF's pretrained share of the input rising 59.0% -> 83.7%,
    # and the chars leak made structurally impossible. TTS keeps its per-frame
    # conditioning via the upsampled T5 latent in z_T, only the finer
    # character-level signal goes.
    ap.add_argument("--no_cond", type=int, default=0)
    ap.add_argument("--grad_ckpt", type=int, default=0)
    # continue a finished run instead of restarting. Checkpoints written before
    # this flag existed carry no optimizer state, so Adam's moments restart --
    # it re-warms within a few hundred steps, but the log will say so.
    ap.add_argument("--resume", default="")
    # use the cached frozen-encoder features as the S2T audio condition
    ap.add_argument("--whisper", type=int, default=0)
    # a pretrained x-vector as the speaker condition. STTATTS (155M,
    # LibriSpeech 100h) does exactly this; we were asking the backbone to
    # infer the speaker from 0.368 hours per speaker instead.
    ap.add_argument("--spk", type=int, default=0)
    # Two datasets, one backbone: TTS rows from --tts_cache, S2T rows from the
    # comma-separated --s2t_caches. Leave empty for the single-cache runs.
    ap.add_argument("--tts_cache", default="")
    ap.add_argument("--s2t_caches", default="")
    # TTS clips for the monitor eval; empty = in-set clips of the first TTS
    # cache (the LJSpeech protocol). Multi-speaker runs point it at test-clean.
    ap.add_argument("--tts_eval_cache", default="")
    # weights of the ";"-separated TTS groups, e.g. "0.5,0.5"; empty = by size
    ap.add_argument("--tts_group_weights", default="")
    # RMS-normalise the projected x-vector (unified.UnifiedELF spk_norm)
    ap.add_argument("--spk_norm", type=int, default=0)
    # monitor-eval S2T through a frozen standalone decoder (for flow-only runs
    # trained with --decoder_prob 0, whose CE head is never trained)
    ap.add_argument("--ext_decoder", default="")
    # also keep ckpt_<step>.pt (weights + EMA, no optimizer) every N steps;
    # ckpt.pt is overwritten at every eval and cannot answer "how was 100k?"
    ap.add_argument("--keep_every", type=int, default=0)
    # ---- multi-speaker TTS run (2026-10-21). All default to off, so every
    # earlier configuration trains exactly as before.
    # F5-TTS's CFG dropout: audio prompt dropped at 0.3, prompt + text (+ our
    # x-vector) together at --text_drop_prob (0.2)
    ap.add_argument("--prompt_drop_prob", type=float, default=0.0)
    ap.add_argument("--joint_cond_drop", type=int, default=0)
    # fill TTS rows with several clips of one speaker (SpeakerPack)
    ap.add_argument("--tts_pack_prob", type=float, default=0.0)
    # bf16 autocast for the forwards / TF32 matmuls
    ap.add_argument("--bf16", type=int, default=0)
    ap.add_argument("--tf32", type=int, default=0)
    # micro-batches per optimizer step (effective batch = batch x GPUs x this)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--log_mem", type=int, default=0)
    # CODE_REVIEW section 14 (all default off)
    ap.add_argument("--fixed_s2t_rows", type=int, default=-1)
    ap.add_argument("--task_loss_norm", type=int, default=0)
    ap.add_argument("--s2t_loss_weight", type=float, default=0.07)
    # a resume segment must not silently start from step 0
    ap.add_argument("--resume_strict", type=int, default=0)
    # constant after warmup (every run so far) or ELF-S2T's cosine to zero
    ap.add_argument("--lr_schedule", choices=["constant", "cosine"],
                    default="constant")
    # step where the cosine decay starts. -1: right after warmup for a fresh
    # run; when resuming a constant-lr checkpoint it becomes the resumed step
    # (decaying from a point already passed would drop the lr in one jump),
    # and resuming a cosine checkpoint keeps that run's own value.
    ap.add_argument("--cosine_from", type=int, default=-1)
    a = ap.parse_args()

    if RANK == 0:
        os.makedirs(a.out, exist_ok=True)
    if DDP:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(LOCAL_RANK)
        dev = torch.device("cuda", LOCAL_RANK)
        log(f"DDP: {WORLD} ranks, per-rank batch {a.batch}, "
            f"effective batch {a.batch * WORLD}")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # every rank draws the same init; the sampler is what makes the DATA differ
    torch.manual_seed(0)

    dual = bool(a.tts_cache)
    data_info = {}
    if dual:
        assert a.layout == "hybrid", "two-source training is built for the hybrid layout"
        assert a.s2t_caches and a.eval_cache, \
            "--tts_cache needs --s2t_caches and an --eval_cache for S2T"
        # every eval script rebuilds the model and the decode stats from
        # args["cache"], and the vocoder decodes the TTS cache's mel space
        # LJSpeech is used WHOLE: F5-TTS small trains on all of it and reports
        # "in-set" tests, so carving out a val slice would train on less data
        # than the paper we compare to. LibriSpeech 960 h is used whole too, as
        # in ELF-S2T; S2T is scored on the official test-clean.
        # several TTS caches are allowed (LibriSpeech 960 h is three); the
        # first one's statistics are the model's mel space
        # ";" separates TTS groups (sampled by --tts_group_weights), ","
        # separates caches inside a group (uniform over their clips)
        tts_groups_c = [g.split(",") for g in a.tts_cache.split(";")]
        tts_list = [c for g in tts_groups_c for c in g]
        tts_bases = [LibriCache(c, "all") for c in tts_list]
        tts_base = tts_bases[0]
        a.cache = tts_list[0]
        meta = tts_base.meta
        ref = (meta["latent_mean"], meta["latent_std"])
        s2t_bases = [LibriCache(c, "all") for c in a.s2t_caches.split(",")]
        ev_s2t_base = LibriCache(a.eval_cache, "all")
        if a.spk:
            # every source must carry the x-vector, or a batch drawn without it
            # leaves spk_proj out of the graph and DDP fails the step
            sk = {(b_.meta.get("spk_emb_model"), b_.meta.get("spk_emb_dim"))
                  for b_ in tts_bases + s2t_bases + [ev_s2t_base]}
            assert len(sk) == 1 and None not in next(iter(sk)), \
                f"--spk needs the same x-vector in every cache: {sk}"
        for b_ in tts_bases + s2t_bases + [ev_s2t_base]:
            assert b_.meta["stoi"] == meta["stoi"], "character tables differ"
            assert abs(b_.meta["frame_rate"] - meta["frame_rate"]) < 1e-6
        wkeys = {(b_.meta.get("whisper_model"), b_.meta.get("whisper_dim"),
                  b_.meta.get("whisper_rate")) for b_ in s2t_bases + [ev_s2t_base]}
        assert len(wkeys) == 1 and None not in next(iter(wkeys)), \
            f"S2T caches disagree on, or lack, Whisper features: {wkeys}"
        wmeta = s2t_bases[0].meta
        if a.tts_pack_prob > 0.0:
            tts_ds = ConcatDataset([SpeakerPack(Tagged(b_, TTS, ref), a.max_length - a.max_text,
                                                a.max_text, a.tts_pack_prob, seed=11)
                                    for b_ in tts_bases])
            log(f"TTS 行按同一说话人拼接：概率 {a.tts_pack_prob}，每行最多 "
                f"{a.max_length - a.max_text} 帧 / {a.max_text} 个 T5 token")
        else:
            tts_ds = ConcatDataset([Tagged(b_, TTS, ref) for b_ in tts_bases])
        s2t_ds = [Tagged(b_, S2T, ref) for b_ in s2t_bases]
        ds = ConcatDataset([tts_ds] + s2t_ds)
        n_tts, n_s2t = len(tts_ds), sum(len(d_) for d_ in s2t_ds)
        gw = ([float(x) for x in a.tts_group_weights.split(",")]
              if a.tts_group_weights else None)
        groups, off, k = [], 0, 0
        for gi, g in enumerate(tts_groups_c):
            size = sum(len(tts_bases[k + j]) for j in range(len(g)))
            groups.append((off, size, gw[gi] if gw else float(size)))
            off += size; k += len(g)
        assert off == n_tts and (gw is None or len(gw) == len(groups))
        tsamp = TaskSampler(n_tts, n_s2t, a.p_s2t, seed=0,   # reseeded below
                            tts_groups=groups, fixed_s2t=a.fixed_s2t_rows, block=a.batch)
        if a.fixed_s2t_rows >= 0:
            log(f"每卡每个 micro-batch 固定 {a.fixed_s2t_rows} 行 S2T + {a.batch - a.fixed_s2t_rows} 行 TTS"
                f"（不再逐行抛硬币；--p_s2t 不再使用）")
        if len(groups) > 1:
            tot = sum(g[2] for g in groups)
            log("TTS 分组：" + "；".join(
                f"{'+'.join(os.path.basename(c) for c in g)} {fmt_n(sz)} 条，占 TTS 行 {100*w/tot:.0f}%"
                for g, (_, sz, w) in zip(tts_groups_c, groups)))
        # the monitor eval: in-set LJSpeech clips of 4-7 s (F5's LJSpeech
        # test range) alternating with test-clean, so each batch has both
        ev_tts = Tagged(LibriCache(a.tts_eval_cache or a.cache, "all",
                                   dur_range=(4.0, 7.0)), TTS, ref)
        vs = Interleave(ev_tts, Tagged(ev_s2t_base, S2T, ref))
        hrs = lambda b_: sum(r["T"] for r in b_.recs) / b_.meta["frame_rate"] / 3600
        for c_, b_ in zip(tts_list, tts_bases):
            log(f"TTS 数据 {os.path.basename(c_)}: {len(b_)} clips, "
                f"{hrs(b_):.1f} h, {b_.meta['n_speakers']} speaker(s); "
                f"mel mean {b_.meta['latent_mean']:+.4f} std {b_.meta['latent_std']:.4f}")
        log(f"  TTS 合计 {n_tts} clips；模型的 mel 参照统计量 {ref[0]:+.4f} / {ref[1]:.4f}"
            f"（{os.path.basename(a.cache)}）")
        for c_, b_ in zip(a.s2t_caches.split(","), s2t_bases):
            log(f"S2T 数据 {os.path.basename(c_)}: {len(b_)} clips, "
                f"{hrs(b_):.1f} h, {b_.meta['n_speakers']} speakers; "
                f"mel mean {b_.meta['latent_mean']:+.4f} std "
                f"{b_.meta['latent_std']:.4f} -> 仿射映射到参照统计量")
        log(f"  S2T 合计 {n_s2t} clips, {sum(hrs(b_) for b_ in s2t_bases):.1f} h; "
            f"每行按 p_s2t={a.p_s2t} 选任务，再在该任务的数据里均匀抽 clip")
        log(f"评测：{os.path.basename(a.tts_eval_cache or a.cache)} 4-7 s 的 {len(ev_tts)} 条 TTS 与 "
            f"{os.path.basename(a.eval_cache)} 的 {len(ev_s2t_base)} 条交替")
        data_info = {"tts_cache": a.tts_cache, "tts_eval_cache": a.tts_eval_cache,
                     "s2t_caches": a.s2t_caches.split(","),
                     "eval_cache": a.eval_cache, "n_tts": n_tts, "n_s2t": n_s2t,
                     "ref_stats": list(ref),
                     "s2t_stats": [[b_.meta["latent_mean"], b_.meta["latent_std"]]
                                   for b_ in s2t_bases],
                     "whisper_model": wmeta["whisper_model"],
                     "whisper_dim": wmeta["whisper_dim"],
                     "whisper_rate": wmeta["whisper_rate"]}
    else:
        ds = LibriCache(a.cache, "train")
        if a.eval_cache:
            vs = LibriCache(a.eval_cache, "all")
            log(f"评测集来自 {os.path.basename(a.eval_cache)}（整个集合，{len(vs)} clips）")
        else:
            vs = LibriCache(a.cache, "val")
        meta = wmeta = ds.meta
        log(f"cache {os.path.basename(a.cache)}: {len(ds)} train / {len(vs)} val "
            f"clips, {meta['n_speakers']} speakers")
    log(f"  mel mean {meta['latent_mean']:+.4f} std {meta['latent_std']:.4f}; "
        f"frames/clip mean {meta['frames']['mean']:.0f}; "
        f"frames per t5 token {meta['frames_per_t5']:.1f}")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("t5-small")
    itos = {v: k for k, v in meta["stoi"].items()}
    if a.layout == "channel":
        log(f"  LAYOUT channel: all {a.max_length} positions carry BOTH the mel "
            f"and the text latent upsampled to the frame rate")
        log(f"  S2T decodes one token per frame then collapses duplicates "
            f"(~{meta['frames_per_t5']:.0f} frames per token)")
    else:
        log(f"  LAYOUT token: {a.max_length - a.max_text} mel frames "
            f"+ {a.max_text} text positions")

    wrate = wmeta.get("whisper_rate", 0.0) if a.whisper else 0.0
    wdim = wmeta.get("whisper_dim", 0) if a.whisper else 0
    # dual: force the Whisper block into every batch (see make_collate)
    force_wdim = wdim if dual else 0
    coll = make_collate(a.max_length, a.p_s2t, (a.prompt_lo, a.prompt_hi),
                        max_text=a.max_text, itos=itos, tk=tk, layout=a.layout,
                        whisper_rate=wrate, frame_rate=meta["frame_rate"],
                        whisper_dim=force_wdim)
    coll_ev = make_collate(a.max_length, 0.5, (a.prompt_hi, a.prompt_hi),
                           rng_seed=1234, max_text=a.max_text, itos=itos, tk=tk,
                           layout=a.layout, whisper_rate=wrate,
                           frame_rate=meta["frame_rate"], whisper_dim=force_wdim)
    tu_bt = tu_bs = None
    if dual:
        # fixed probe for text_use(): 8 in-set LJ clips with their own
        # transcripts, and the same clips with the next clip's transcript.
        # Two collates with one seed give both batches the same prompt length.
        pick = np.random.default_rng(0).choice(len(ev_tts), 8, replace=False)
        tu_base = [ev_tts[int(i)] for i in pick]
        tu_shuf = [{**it, "chars": tu_base[(k + 1) % 8]["chars"],
                    "t5": tu_base[(k + 1) % 8]["t5"]} for k, it in enumerate(tu_base)]
        tu_coll = lambda: make_collate(a.max_length, 0.0, (0.2, 0.2), rng_seed=3,
                                       max_text=a.max_text, itos=itos, tk=tk,
                                       layout=a.layout, whisper_rate=0.0,
                                       frame_rate=meta["frame_rate"])
        tu_bt, tu_bs = tu_coll()(tu_base), tu_coll()(tu_shuf)
    if dual:
        samp = None
        dl = DataLoader(ds, batch_size=a.batch, sampler=tsamp, num_workers=4,
                        collate_fn=coll, drop_last=True, pin_memory=True)
    elif DDP:
        from torch.utils.data.distributed import DistributedSampler
        samp = DistributedSampler(ds, num_replicas=WORLD, rank=RANK,
                                  shuffle=True, drop_last=True)
        dl = DataLoader(ds, batch_size=a.batch, sampler=samp, num_workers=4,
                        collate_fn=coll, drop_last=True, pin_memory=True)
    else:
        samp = None
        dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4,
                        collate_fn=coll, drop_last=True, pin_memory=True)
    dv = DataLoader(vs, batch_size=a.batch, shuffle=False, num_workers=2,
                    collate_fn=coll_ev, drop_last=True)

    sdim = meta.get("spk_emb_dim", 0) if a.spk else 0
    # gradient checkpointing is ELF's own flag: at the LibriSpeech sequence
    # length (1536 mel + 112 text) batch 32 OOMs on an 80 GB H100, and halving
    # the batch was measured to cost real accuracy, so trade compute for memory
    # instead and keep the batch size the ablations were run at.
    model = UnifiedELF(a.model, char_vocab=meta["char_vocab_size"],
                       max_length=a.max_length, whisper_dim=wdim,
                       use_cond=not a.no_cond,
                       spk_dim=sdim, spk_norm=bool(a.spk_norm),
                       gradient_checkpointing=bool(a.grad_ckpt)).to(dev)
    log(f"  input_width {model.input_width}"
        f"{'  (cond 块已移除)' if a.no_cond else ''}")
    if sdim:
        log(f"  frozen {meta['spk_emb_model']} x-vector as the speaker "
            f"condition: {sdim}-d from the first "
            f"{meta['spk_emb_prompt_sec']}s, broadcast to all positions, "
            f"projector {sdim}x512 = "
            f"{sum(p.numel() for p in model.spk_proj.parameters()):,} params")
    if wdim:
        log(f"  frozen {wmeta['whisper_model']} features as the S2T audio "
            f"condition: {wdim}-d at {wmeta['whisper_rate']:.0f} Hz, "
            f"projector {wdim}x{512} = "
            f"{sum(p.numel() for p in model.whisper_proj.parameters()):,} params")
    if os.path.exists(a.init):
        load_elf_init(model, a.init, init_frac=a.init_frac)
    else:
        log(f"WARNING no ckpt at {a.init} -- training from scratch")
    import copy
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    # wrap AFTER the EMA copy so the EMA stays a plain module, and keep `core`
    # as the handle for eval and for saving: a DDP state_dict prefixes every key
    # with "module." and every eval script in this repo would fail to load it.
    core = model
    if DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        core = model.module
    text_enc = TextEncoder(a.text_latent_std).to(dev)
    log(f"trainable {sum(p_.numel() for p_ in model.parameters())/1e6:.2f}M; "
        f"t5 encoder frozen "
        f"{sum(p_.numel() for p_ in text_enc.parameters())/1e6:.2f}M")

    cfg = Cfg(denoiser_p_mean=a.denoiser_p_mean,
              decoder_p_mean=a.decoder_p_mean,
              s2t_weight=a.s2t_weight,
              decoder_prob=a.decoder_prob,
              audio_drop_prob=a.audio_drop_prob,
              text_drop_prob=a.text_drop_prob)
    cfg.prompt_drop_prob, cfg.joint_cond_drop, cfg.bf16 = a.prompt_drop_prob, a.joint_cond_drop, a.bf16
    cfg.task_loss_norm, cfg.s2t_loss_weight = a.task_loss_norm, a.s2t_loss_weight
    if a.task_loss_norm:
        log(f"损失按任务分别归一化：S2T 权重 {a.s2t_loss_weight}，TTS 权重 {1 - a.s2t_loss_weight:.2f}")
    log(f"cfg: {json.dumps(asdict(cfg))}")
    if a.prompt_drop_prob or a.joint_cond_drop or a.bf16 or a.tf32 or a.tts_pack_prob:
        log(f"多说话人选项：prompt_drop {a.prompt_drop_prob}  joint_cond_drop {a.joint_cond_drop}  "
            f"tts_pack {a.tts_pack_prob}  bf16 {a.bf16}  tf32 {a.tf32}")
    ext_dec = None
    if a.ext_decoder and RANK == 0:
        from train_text_decoder import TextDecoder
        dk = torch.load(a.ext_decoder, map_location="cpu", weights_only=False)
        ext_dec = TextDecoder(**dk["cfg"]).to(dev).eval(); ext_dec.load_state_dict(dk["model"])
        for p_ in ext_dec.parameters():
            p_.requires_grad_(False)
        log(f"monitor S2T decodes with the frozen external decoder {a.ext_decoder}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0,
                            betas=(0.9, 0.95))
    def lr_lambda(st):
        if st < a.warmup:
            return (st + 1) / max(a.warmup, 1)
        if a.lr_schedule == "cosine":
            # ELF-S2T: linear warmup, then cosine decay to zero at the last
            # step. A run cut at a budget without it is a model caught mid-
            # training, not a finished one at that budget. The decay starts
            # at cosine_from (read at call time, so a resume can set it).
            c0 = a.cosine_from if a.cosine_from >= 0 else a.warmup
            if st < c0:
                return 1.0
            p = min(1.0, (st - c0) / max(1, a.steps - c0))
            return 0.5 * (1.0 + math.cos(math.pi * p))
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    log(f"lr schedule: {a.lr_schedule}, peak {a.lr:g}, warmup {a.warmup}"
        + (f", cosine to 0 at step {a.steps}" if a.lr_schedule == "cosine" else ""))
    mel_stats = (meta["latent_mean"], meta["latent_std"])

    log("")
    log("l2/triv is the loss over the loss of predicting nothing: 1.000 means no")
    log("better than generating nothing. It equals the squared relative xhat0")
    log("error only while the target is unguided, and it is NOT comparable")
    log("across runs with different t distributions, so the rollout eval decides.")
    log("")
    log(f"{'step':>7}{'loss':>9}{'l2':>9}{'l2/triv':>9}{'tts/tr':>8}"
        f"{'s2t/tr':>8}{'ce':>8}{'ce_acc':>8}{'gnorm':>7}{'s/it':>6}")

    step = 0
    EXPO = ["s2t_rows", "s2t_text_tokens", "tts_rows", "tts_clips", "tts_frames", "tts_supervised_frames"]
    expo_tot = {k: 0.0 for k in EXPO}
    hist_prev = []
    if a.resume and os.path.exists(a.resume):
        rk = torch.load(a.resume, map_location="cpu", weights_only=False)
        expo_tot.update(rk.get("exposure", {}))
        hp = os.path.join(a.out, "history.json")
        if os.path.exists(hp):
            # keep the earlier segment's rows (CODE_REVIEW 14.5): up to the
            # resumed step, so a segment killed after its last save adds nothing twice
            hist_prev = [h for h in json.load(open(hp)) if h.get("step", 0) <= int(rk["step"])]
        # load into core, not model: under DDP `model` is the wrapper and expects
        # every key prefixed "module.", but checkpoints are saved from
        # core.state_dict() exactly so eval scripts can read them. Loading into
        # `model` would crash the first resumed segment of a chained run.
        core.load_state_dict(rk["model"]); ema.load_state_dict(rk["ema"])
        step = int(rk["step"])
        if a.lr_schedule == "cosine" and a.cosine_from < 0:
            prev = rk.get("args", {})
            if prev.get("lr_schedule", "constant") == "constant" and step > a.warmup:
                a.cosine_from = step
            elif prev.get("lr_schedule") == "cosine":
                a.cosine_from = prev.get("cosine_from", -1)
            log(f"cosine decay from step {a.cosine_from if a.cosine_from >= 0 else a.warmup} "
                f"(resumed checkpoint was trained with "
                f"{prev.get('lr_schedule', 'constant')} lr)")
        if "opt" in rk:
            opt.load_state_dict(rk["opt"]); sched.load_state_dict(rk["sched"])
            log(f"resumed from {a.resume} at step {step}, "
                f"optimizer and schedule restored; lr now "
                f"{a.lr * lr_lambda(step):.3g}")
        else:
            for _ in range(step):
                sched.step()
            log(f"resumed from {a.resume} at step {step}; that checkpoint has "
                f"NO optimizer state, so Adam's moments restart from zero "
                f"(the schedule was fast-forwarded)")
    elif a.resume:
        if a.resume_strict:
            raise SystemExit(f"--resume {a.resume} 不存在；续训段（--resume_strict）不能从第 0 步重新开始")
        log(f"WARNING --resume {a.resume} does not exist, starting fresh")

    step0 = step                      # for the s/it average after a resume
    if dual:
        # one stream per rank, and a resumed segment must not replay the
        # stream the first segment already consumed
        tsamp.seed = [RANK, step, 17]
    epoch = 0
    if samp is not None:
        samp.set_epoch(epoch)
    hist, it, t0 = hist_prev, iter(dl), time.time()
    if hist_prev:
        log(f"续训：保留之前的 {len(hist_prev)} 条 history；累计曝光 {expo_tot}")
    msk_log = a.grad_accum > 1 or a.task_loss_norm or a.fixed_s2t_rows >= 0
    while step < a.steps:
        try:
            b = next(it)
        except StopIteration:
            epoch += 1
            # without set_epoch a DistributedSampler reshuffles identically every
            # epoch, so each rank would see the same order for the whole run
            if samp is not None:
                samp.set_epoch(epoch)
            it = iter(dl); continue
        m = train_step(model, text_enc, ema, b, cfg, opt, dev, mel_stats,
                       accum_i=0, accum_n=a.grad_accum)
        ml, ex_step = [m["loss"]], list(m["expo"])
        for k_ in range(1, a.grad_accum):        # --grad_accum > 1 only
            while True:
                try:
                    b = next(it); break
                except StopIteration:
                    epoch += 1
                    if samp is not None:
                        samp.set_epoch(epoch)
                    it = iter(dl)
            m = train_step(model, text_enc, ema, b, cfg, opt, dev, mel_stats,
                           accum_i=k_, accum_n=a.grad_accum)
            ml.append(m["loss"]); ex_step = [x + y for x, y in zip(ex_step, m["expo"])]
        m["loss"] = sum(ml) / len(ml)            # the whole step, not its last micro-batch
        m.pop("expo", None)
        for k_, v_ in zip(EXPO, ex_step):
            expo_tot[k_] += v_
        sched.step()
        step += 1
        if a.log_mem and step == 20 and torch.cuda.is_available():
            log(f"峰值显存（rank {RANK}，前 20 步）{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

        if step % a.log_every == 0 or step == 1:
            el = (time.time() - t0) / max(step - step0, 1)
            log(f"{step:>7}{m['loss']:>9.4f}{m['l2']:>9.4f}{m['l2_ratio']:>9.4f}"
                f"{m['r_tts']:>8.4f}{m['r_s2t']:>8.4f}{m['ce']:>8.4f}"
                f"{m['ce_acc']*100:>7.1f}%{m['grad_norm']:>7.2f}{el:>6.2f}")
            hist.append({"step": step, **m, **({"exposure": dict(expo_tot)} if msk_log else {})})
            if msk_log:
                log(f"      曝光（第 {step} 步，全部卡、全部 micro-batch）S2T {ex_step[0]:.0f} 行 / {ex_step[1]:.0f} token；"
                    f"TTS {ex_step[2]:.0f} 行 / {ex_step[3]:.0f} 句 / {ex_step[4]:,.0f} 帧（含 prompt）/ 监督 {ex_step[5]:,.0f} 帧"
                    f" | 累计 S2T {expo_tot['s2t_rows']:,.0f} 行，TTS {expo_tot['tts_frames']/1e9:.3f}B 帧")

        if step % a.eval_every == 0 or step == a.steps:
          if RANK == 0:
            for tag, net in (("raw", core), ("ema", ema)):
                e = evaluate(net, text_enc, dv, cfg, dev, mel_stats, tk,
                             a.layout, n_batches=a.eval_batches, steps=a.nfe,
                             ext_dec=ext_dec)
                log(f"  [eval {step} {tag}] "
                    f"TTS mel err/trivial {e['tts_err_ratio']:.4f} "
                    f"({'beats' if e['tts_err_ratio'] < 1 else 'WORSE THAN'} "
                    f"nothing, n={e['n_tts']}) | "
                    f"S2T text err/trivial {e['s2t_err_ratio']:.4f} "
                    f"WER {e['s2t_wer']:.2f}% tok_acc {e['s2t_tok_acc']*100:.1f}% "
                    f"(n={e['n_s2t']}, NFE {a.nfe})")
                if tag == "ema":
                    if tu_bt is not None:
                        tu = text_use(net, text_enc, tu_bt, tu_bs, dev, mel_stats)
                        log("      TTS text use (wrong-text err / own-text err; "
                            "1.0 = text ignored): "
                            + "  ".join(f"t={k} {v:.3f}" for k, v in tu.items()))
                    g = exposure_gap(net, text_enc, next(iter(dv)), cfg, dev,
                                     mel_stats, steps=a.nfe)
                    gap = {k: v for k, v in g.items() if not k.startswith("ABS")}
                    ab = {k[4:]: v for k, v in g.items() if k.startswith("ABS")}
                    log("      gap (rollout / teacher-forced xhat0 err; "
                        "1.0 = no gap): "
                        + "  ".join(f"{k} {v:.2f}" for k, v in sorted(gap.items())))
                    log("      xhat0 err / ||x0||  (1.0 = predicting the mean): "
                        + "  ".join(f"{k} {v:.2f}" for k, v in sorted(ab.items())))
                    for ref, hyp in e["samples"]:
                        log(f"      ref  {ref[:110]}")
                        log(f"      hyp  {hyp[:110]}")
            keep = {"model": core.state_dict(), "ema": ema.state_dict(),
                    "args": vars(a), "cfg": asdict(cfg), "step": step,
                    "data": data_info, "exposure": dict(expo_tot),
                    "meta_keys": {k: meta[k] for k in
                                  ("dim", "char_vocab_size", "latent_mean",
                                   "latent_std", "n_speakers",
                                   "frames_per_t5")}}
            # written to a temp file and renamed: a job killed mid-save (the
            # 48 h wall limit) must leave the previous ckpt.pt loadable for the
            # resumed segment, not a truncated one
            torch.save({**keep, "opt": opt.state_dict(),
                        "sched": sched.state_dict()},
                       os.path.join(a.out, "ckpt.pt.tmp"))
            os.replace(os.path.join(a.out, "ckpt.pt.tmp"), os.path.join(a.out, "ckpt.pt"))
            if a.keep_every and step % a.keep_every == 0:
                torch.save(keep, os.path.join(a.out, f"ckpt_{step:06d}.pt"))
                log(f"  kept {a.out}/ckpt_{step:06d}.pt")
            with open(os.path.join(a.out, "history.json"), "w") as f:
                json.dump(hist, f)
          if DDP:
              # rank 0 just spent minutes in eval; hold the others here instead
              # of letting them queue up inside the next all-reduce
              torch.distributed.barrier()

    log("")
    log("done")
    if DDP:
        torch.distributed.destroy_process_group()
