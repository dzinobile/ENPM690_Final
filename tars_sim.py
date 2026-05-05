import math
import mujoco
import mujoco.viewer
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--xml", default="tars_with_human.xml")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(args.xml)
data = mujoco.MjData(model)

if args.xml == "tars_with_barrel.xml":
    mujoco.mj_resetDataKeyframe(model, data, 0)  # apply "init" keyframe
    mujoco.mj_forward(model, data)
    data.ctrl[:] = [0, 0, 0, 0, 0, 0, 320, 320, 200, 200]

if args.xml == "tars_with_human.xml":
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = [0, 0, 0, 0, 0, 0, 320, 320, 200, 200]

if args.xml == "tars_meshed.xml":
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = [0, 0, 0, -114, 228, 228, 0, -114, 228, 228]

if args.xml == "tars2_meshed.xml":
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = [0, 0, -114, 0, 228, 228, 0, 0, -114, 0, 228, 228]


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        # mujoco.mj_step(model, data, 0) #freezeframe spawn
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
