set -euo pipefail
nvidia-smi      # make sure there's a GPU attached
git clone https://github.com/orangishcat/toddlerbot
cd toddlerbot
uv venv --python 3.11 --clear
source .venv/bin/activate
uv sync
