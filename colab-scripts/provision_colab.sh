# Run this on the Colab VM to set it up
set -euo pipefail
nvidia-smi      # make sure there's a GPU attached
cd toddlerbot
uv venv --python 3.11 --clear
source .venv/bin/activate
uv sync
