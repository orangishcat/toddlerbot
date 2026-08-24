# Run this on the Colab VM to set it up
set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}       # repair the linker config (necessary for colab only)
apt-get update -qq
apt-get install -y libegl1        # MuJoCo's headless EGL backend needs the GLVND EGL loader
uv venv --python 3.11 --clear       # some packages don't have 3.12
source .venv/bin/activate
