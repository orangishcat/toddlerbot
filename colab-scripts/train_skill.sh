export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

SKILL="${1:-cartwheel}"
python toddlerbot/locomotion/train_mjx.py --gin-file $SKILL --env $SKILL --gin-config "PPOConfig.render_nums=50"
