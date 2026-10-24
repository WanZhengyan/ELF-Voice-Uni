#!/bin/bash
#SBATCH --job-name=vf-ms
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --output=logs/ms-%j.out
#SBATCH --error=logs/ms-%j.err
# Multi-speaker unified run (2026-10-21): S2T on LibriSpeech 960 h (the
# existing lv3 caches, ELF-S2T's data), TTS on LibriTTS 585 h (24 kHz, F5-TTS
# small's LibriSpeech-PC column), monitored on dev-clean. A NEW script: the
# single-speaker runs keep slurm/submit_uni_train.sh unchanged.
#   MODE=unified   16 S2T + 32 TTS rows/step (= uni_mix48's split; S2T budget = ELF-B)
#   MODE=ttsonly   the SAME 32 TTS rows/step, no S2T rows (the A/B control)
#   STEPS (100000) RUN (runs/ms_<mode>) SMOKE=1: 40 steps, eval at 20, into runs/_smoke_ms_<mode>
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$R"; source .venv/bin/activate
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
export TOKENIZERS_PARALLELISM=false   # SpeakerPack tokenises inside DataLoader workers
MODE="${MODE:?unified|ttsonly|f5x}"
STEPS="${STEPS:-100000}"; EVERY=2000; KEEP=25000
OUT="${RUN:-$R/runs/ms_$MODE}"
LOGEVERY=100
EXTRA=""
if [ "${SMOKE:-0}" = 1 ]; then STEPS="${SMOKE_STEPS:-40}"; EVERY=20; KEEP=0; LOGEVERY=10; OUT="$R/runs/_smoke_ms_$MODE"; fi
# each mode sets ALL of its own values here; only f5x reads BATCH / ACCUM from
# the caller's environment. (A blanket ACCUM=1 above this case silently beat
# the caller's ACCUM=4: the 2026-10-22 f5x smoke ran at accum 1.)
case "$MODE" in
  unified) BATCH=12; ACCUM=1; PACK=0.5; PS2T=0.3333333 ;;
  ttsonly) BATCH=8; ACCUM=1; PACK=0.5; PS2T=0.0 ;;
  # f5x (2026-10-22): F5-TTS small's audio exposure per step on 4 GPUs.
  # 14 rows/GPU x 4 GPUs x 4 micro-batches = 224 rows/step: 16 S2T (S2T total
  # = ELF-B's 1.6M over 100K steps) + 208 TTS rows, every TTS row packed
  # (mean 1,466 frames) -> ~305K frames/step vs F5's 307,200.
  # CODE_REVIEW 14: exactly 1 S2T row per GPU per micro-batch (16 per step at
  # 14 x 4 x 4) and the loss averaged per task, S2T weighted 0.07 (its share of
  # target positions in uni_mix48) instead of ~0.6 % under one shared denominator.
  f5x) BATCH="${BATCH:-14}"; ACCUM="${ACCUM:-4}"; PACK=1.0
       PS2T=$(python3 -c "print(16/($BATCH*4*$ACCUM))")
       EXTRA="--fixed_s2t_rows 1 --task_loss_norm 1 --s2t_loss_weight ${S2TW:-0.07}" ;;
  *) echo "MODE?"; exit 1 ;;
esac
LT="$R/cache_libritts_train_clean_100,$R/cache_libritts_train_clean_360,$R/cache_libritts_train_other_500"
LS="$R/cache_libri_tc100_lv3,$R/cache_libri_train_clean_360,$R/cache_libri_train_other_500"
echo "=== multi-speaker $MODE  job ${SLURM_JOB_ID:-local} $(hostname) $(date -Is)"
echo "  OUT $OUT  STEPS $STEPS  per-GPU batch $BATCH x 4 GPUs x accum $ACCUM = $((BATCH*4*ACCUM)) rows/step  p_s2t $PS2T  pack $PACK"
echo "  RESUME ${RESUME:-(none)}   extra: ${EXTRA:-(none)}"
echo "  TTS  $LT"
echo "  S2T  $LS"
echo "  monitor: S2T cache_libri_dev_clean_lv3, TTS cache_libritts_dev_clean (dev, not test)"
echo "  GPUs $(nvidia-smi -L | wc -l)"
torchrun --standalone --nproc_per_node=4 train_unified.py \
  --cache "$R/cache_libritts_train_clean_100" --out "$OUT" --steps "$STEPS" \
  --batch "$BATCH" --max_length 1648 --max_text 112 --layout hybrid \
  --p_s2t "$PS2T" --lr 1e-4 --lr_schedule cosine --init_frac 0.25 \
  --denoiser_p_mean -1.5 --decoder_p_mean -0.5 --decoder_prob 0.5 \
  --whisper 1 --spk 1 --spk_norm 1 --prompt_hi 0.3 \
  --audio_drop_prob 0.1 --text_drop_prob 0.2 \
  --prompt_drop_prob 0.3 --joint_cond_drop 1 --tts_pack_prob "$PACK" \
  --grad_accum "$ACCUM" --log_mem 1 --resume "${RESUME:-}" ${RESUME:+--resume_strict 1} $EXTRA \
  --bf16 1 --tf32 1 \
  --eval_every "$EVERY" --eval_batches 11 --keep_every "$KEEP" --log_every "$LOGEVERY" \
  --eval_cache "$R/cache_libri_dev_clean_lv3" \
  --tts_cache "$LT" --s2t_caches "$LS" --tts_eval_cache "$R/cache_libritts_dev_clean"
echo "finished $(date -Is)"
