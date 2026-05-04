import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("tars_meshed.xml")
data = mujoco.MjData(model)
print(model.body_mass)
# got this from chat to check the COMs
# 🔥 IMPORTANT: update kinematics BEFORE reading COM
mujoco.mj_forward(model, data)

# print COMs (optional: skip world body)
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(name, data.xipos[i])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))