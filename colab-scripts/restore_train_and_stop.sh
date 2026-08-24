SKILL="${1:-cartwheel}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

SKILL="${1:-cartwheel}"
python toddlerbot/locomotion/train_mjx.py --gin-file $SKILL --env $SKILL --gin-config "PPOConfig.render_nums=50" --restore "${2}"


curl -X POST \
  "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
