# Run this on the Colab VM to set it up
set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}       # repair the linker config
python - <<'PY'
import jax
print(jax.devices())
print(jax.default_backend())
PY     # make sure there's a GPU attached
uv venv --python 3.11 --clear       # some packages don't have 3.12
uv sync
