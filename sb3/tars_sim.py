import math
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("tars_with_barrel.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # apply "init" keyframe
mujoco.mj_forward(model, data)
data.ctrl[:] = [0, 0, 0, 0, 0, 0, 320, 320, 200, 200]
# data.ctrl[:] = data.qpos[7:]
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
