# ELF-Voice

One 106M-parameter transformer that does **speech recognition** (speech → text) and **zero-shot speech synthesis** (text + a few seconds of a voice → speech) with the same weights. It is built on [ELF](https://github.com/lillian039/ELF) (Embedded Language Flows), a flow-matching language model.

Weights: **[ZhengyanWan/ELF-Voice](https://huggingface.co/ZhengyanWan/ELF-Voice)**

## Model

Speech and text sit side by side in one sequence, and the model learns to generate whichever half is missing, starting from noise (flow matching).

```
           1,536 positions                        112 positions
  [ speech: mel spectrogram, 93.75 frames/s ][ text: T5-small embeddings ]

  recognition:  speech in  ->  denoise the text part  ->  text
  synthesis:    text + voice prompt in  ->  denoise the speech part  ->  Vocos  ->  waveform
```

- Backbone: ELF-B (12 layers, width 768), initialised from `embedded-language-flows/ELF-B-owt-torch`; both tasks share all parameters.
- Recognition also reads frozen Whisper-large-v3 encoder features. Synthesis also reads a speaker x-vector of the voice prompt.
- Mel spectrograms are turned into audio with [Vocos](https://huggingface.co/charactr/vocos-mel-24khz).

## Results

Checkpoint `ms_f5x_100k`, 100K training steps. It saw about as much synthesis audio as F5-TTS small did in its 100K steps (30.5B vs 30.7B mel frames). Evaluated with each paper's own protocol: synthesis follows F5-TTS's inference rules and is scored with F5-TTS's official scoring code (our ground-truth row matches F5-TTS Table 9); recognition uses ELF-S2T's sampler and scoring.

**Zero-shot TTS** — LibriSpeech-PC test-clean, 1,127 pairs, unseen speakers

| | WER ↓ | Speaker similarity ↑ | UTMOS ↑ |
|---|---|---|---|
| **ELF-Voice** (mean of 3 generation seeds) | 6.37 | 0.231 | 2.66 |
| F5-TTS small, 100K steps | 29.5 | 0.53 | 3.78 |
| F5-TTS small, 200K steps | 4.58 | 0.59 | 4.07 |
| Real recordings | 2.22 | 0.695 | 4.10 |

**Speech recognition** — LibriSpeech test-clean, all 2,620 utterances

| | WER ↓ |
|---|---|
| **ELF-Voice** (ELF-S2T's SDE sampler) | 14.06 |
| **ELF-Voice** (Euler ODE sampler) | 12.98 |
| ELF-S2T, ELF-B (recognition only) | 10.50 |

Other checkpoints (an earlier unified model with a tenth of the synthesis audio, and a single-speaker LJSpeech model) and their results are on the [Hugging Face page](https://huggingface.co/ZhengyanWan/ELF-Voice).

## Use the checkpoint

```python
import torch
from eval_lj import load_model
ck = torch.load("ms_f5x_100k/ckpt.pt", map_location="cpu", weights_only=False)
model, vocab = load_model(ck, torch.device("cuda"))
```

## Training pipeline

**1. Environment**

```bash
git clone https://github.com/WanZhengyan/TTS-ELF.git && cd TTS-ELF
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt vocos
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('embedded-language-flows/ELF-B-owt-torch', local_dir='ckpt/ELF-B-owt-torch')"
```

**2. Data** — put under `data/`:

| Dataset | Used for |
|---|---|
| [LibriSpeech](https://www.openslr.org/12/) train (960 h), dev-clean, test-clean | recognition training, monitoring, testing |
| [LibriTTS](https://www.openslr.org/60/) train-clean-100/360, train-other-500, dev-clean | synthesis training and monitoring |
| F5-TTS's `librispeech_pc_test_clean_cross_sentence.lst` → `data/f5_eval/` | synthesis testing |

**3. Preprocess** — mel spectrograms, Whisper features and x-vectors, cached per split (clips of 1–16 s):

```bash
# LibriSpeech, for each split
python preprocess_libri.py   --split train-clean-360 --out cache_libri_train_clean_360
python preprocess_whisper.py --cache cache_libri_train_clean_360 --split train-clean-360 --model openai/whisper-large-v3
python preprocess_spk.py     --cache cache_libri_train_clean_360 --split train-clean-360
# LibriTTS (24 kHz), for each split
python preprocess_libritts.py --split train-clean-360 --out cache_libritts_train_clean_360
python preprocess_spk.py      --cache cache_libritts_train_clean_360 --split train-clean-360
python tools/validate_multispk_caches.py
```

**4. Train** — 4 GPUs, 100K steps. Each step has 224 rows (14 per GPU × 4 GPUs × 4 gradient-accumulation steps): 16 recognition rows and 208 synthesis rows, each synthesis row packed with several sentences of one speaker. That is about 305K mel frames of synthesis audio per step, as in F5-TTS small (307,200). The two tasks' losses are averaged separately and weighted 0.07 (recognition) and 0.93 (synthesis).

```bash
MODE=f5x sbatch slurm/submit_multispk_train.sh
```

The full `torchrun ... train_unified.py` command and its options are in that script.

**5. Evaluate**

```bash
# synthesis: generate the 1,127 pairs, then score with F5-TTS's official code
# (repeat with --seed 1 and --seed 2; the table reports the mean)
python gen_f5_protocol.py --ckpt runs/ms_f5x/ckpt_100000.pt --corpus libri --cfg 3 --sway -1 --seed 0 \
       --out out/f5/ms_f5x_libri_s0
python score_f5_official.py --run out/f5/ms_f5x_libri_s0 --corpus libri
# recognition: all of test-clean with ELF-S2T's sampler (or --sampler ode), then ELF-S2T's scoring
python eval_s2t_full.py --ckpt runs/ms_f5x/ckpt_100000.pt --sampler elf_sde --nfe 128 --acfg 2.0 \
       --out out/paper/ms_f5x_s2t.json
python score_elf_s2t.py out/paper/ms_f5x_s2t.json
```

`score_f5_official.py` runs in a separate environment with [F5-TTS](https://github.com/SWivid/F5-TTS) installed under `third_party/F5-TTS` and its WavLM-large speaker-verification checkpoint (see the F5-TTS evaluation README).

## Acknowledgements

[ELF](https://github.com/lillian039/ELF), [ELF-S2T](https://github.com/Sslnon/ELF-S2T), [F5-TTS](https://github.com/SWivid/F5-TTS), [Vocos](https://github.com/gemelo-ai/vocos). Data: LibriSpeech and LibriTTS (CC BY 4.0).
