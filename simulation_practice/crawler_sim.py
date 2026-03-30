import math
import mujoco
import mujoco.viewer
import time
model = mujoco.MjModel.from_xml_path("crawler.xml")
data = mujoco.MjData(model)

# Index of the ml_hinge_pos actuator control
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ml_hinge_pos")

target_angle = 0.0   # degrees
STEP = 5.0           # degrees per key press

# def key_callback(keycode):
#     global target_angle
#     # Left arrow (263) → negative, Right arrow (262) → positive
#     if keycode == 262:
#         target_angle = min(90.0, target_angle + STEP)
#     elif keycode == 263:
#         target_angle = max(-90.0, target_angle - STEP)
#     data.ctrl[act_id] = target_angle
#     print(f"ml_hinge target: {target_angle:.1f}°")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
