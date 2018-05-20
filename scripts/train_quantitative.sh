#/bin/bash
# Usage:
# ./scripts/train_quantitative.sh GPU
#
# Example:
# ./scripts/train_quantitative.sh 1

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1

LOG="log/quantitative.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

CUDA_VISIBLE_DEVICES=${GPU_ID} ./quantitative/train.py