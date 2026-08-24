uv pip install jax[cuda12]
python - <<'PY'
import jax
print(jax.devices())
print(jax.default_backend())
PY
# make sure there's a GPU attached
uv sync
