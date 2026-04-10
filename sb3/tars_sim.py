import math
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("tars_fused.xml")
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
