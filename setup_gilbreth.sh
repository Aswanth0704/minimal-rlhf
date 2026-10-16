#!/usr/bin/env bash
# One-time setup on a SLURM cluster login node (written for Purdue's Gilbreth; idempotent).
#   cd /path/to/minimal-rlhf && bash setup_gilbreth.sh
# Override the defaults with environment variables:
#   SCRATCH   where the conda env lives       (default /scratch/gilbreth/$USER)
#   ENV       path of the conda env           (default $SCRATCH/envs/rlhf)
set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="${SCRATCH:-/scratch/gilbreth/$USER}"
ENV="${ENV:-$SCRATCH/envs/rlhf}"
export HF_HOME="$PROJ/hf_cache"        # the notebook picks these folders up automatically
export NLTK_DATA="$PROJ/nltk_data"     # (keeps the 2 GB dataset off a small home quota)
mkdir -p "$HF_HOME" "$NLTK_DATA" "$PROJ/out"

module load conda 2>/dev/null || true

if [ ! -d "$ENV" ]; then
  echo ">> creating conda env at $ENV"
  conda create -y -p "$ENV" python=3.11
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo ">> installing python packages"
pip install --upgrade pip
pip install -r "$PROJ/requirements.txt"

echo ">> registering a Jupyter kernel (for Open OnDemand / JupyterLab)"
if command -v conda-env-mod >/dev/null 2>&1; then
  conda-env-mod kernel -p "$ENV" || python -m ipykernel install --user --name rlhf --display-name "rlhf"
else
  python -m ipykernel install --user --name rlhf --display-name "rlhf"
fi

echo ">> pre-downloading TinyStories (~2 GB) and the VADER lexicon"
python - <<'PY'
import datasets, nltk
datasets.load_dataset("roneneldan/TinyStories")
nltk.download("vader_lexicon")
print("downloads complete")
PY

echo ">> done. Launch JupyterLab on a GPU node, open rlhf_tinystories.ipynb with the 'rlhf' kernel, Run All."
