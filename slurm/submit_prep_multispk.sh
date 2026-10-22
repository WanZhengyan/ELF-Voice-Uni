#!/bin/bash
#SBATCH --job-name=vf-prepms
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/prepms-%j.out
#SBATCH --error=logs/prepms-%j.err
# Caches for the multi-speaker run, each a NEW directory (no existing cache is
# touched). Steps strictly in sequence per cache: preprocess_whisper.py and
# preprocess_spk.py both rewrite index.pt.
#   KIND=libritts  SPLIT=train-clean-100|train-clean-360|train-other-500|dev-clean
#                  -> cache_libritts_<split>: 24 kHz mel + x-vector (TTS rows / TTS monitor)
#   KIND=libri     SPLIT=dev-clean -> cache_libri_dev_clean_lv3: mel + whisper-large-v3
#                  + x-vector (S2T monitor; replaces test-clean for monitoring)
set -uo pipefail
R="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TMPDIR="$R/tmp"; mkdir -p "$TMPDIR"
source "$R/.venv/bin/activate"; cd "$R"
: "${KIND:?}" "${SPLIT:?}"
S=${SPLIT//-/_}
echo "=== multi-speaker cache  $KIND $SPLIT  job ${SLURM_JOB_ID:-local} $(hostname) $(date -Is)"
if [ "$KIND" = libritts ]; then
  C=cache_libritts_$S
  [ -f "$C/index.pt" ] && { echo "$C 已存在，不覆盖"; exit 1; }
  python preprocess_libritts.py --split "$SPLIT" --out "$C" || { echo "mel 失败"; exit 1; }
  python preprocess_spk.py --cache "$C" --split "$SPLIT" --prompt_sec 3.0 || { echo "x-vector 失败"; exit 1; }
elif [ "$KIND" = libri ]; then
  C=cache_libri_${S}_lv3
  [ -f "$C/index.pt" ] && { echo "$C 已存在，不覆盖"; exit 1; }
  python preprocess_libri.py --split "$SPLIT" --out "$C" || { echo "mel 失败"; exit 1; }
  python preprocess_whisper.py --cache "$C" --split "$SPLIT" --model openai/whisper-large-v3 || { echo "whisper 失败"; exit 1; }
  python preprocess_spk.py --cache "$C" --split "$SPLIT" --prompt_sec 3.0 || { echo "x-vector 失败"; exit 1; }
fi
touch "$C/DONE"
echo "done $C $(date -Is)"
