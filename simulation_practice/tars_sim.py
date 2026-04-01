import math
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("tars.xml")
data = mujoco.MjData(model)

# Index of the ml_hinge_pos actuator control
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ml_hinge_pos")



with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
