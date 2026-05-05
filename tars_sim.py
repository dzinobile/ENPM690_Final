import mujoco
import mujoco.viewer
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--xml", default="tars_with_human.xml")
parser.add_argument("--float", action="store_true", help="freeze robot body in air; joints still actuate")
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(args.xml)
data = mujoco.MjData(model)

if args.xml == "tars_with_human.xml":
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

if args.xml == "tars.xml":
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

# print(model.body_mass)
# # print COMs (optional: skip world body)
# for i in range(model.nbody):
#     name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
#     print(name, data.xipos[i])

# Pre-compute freejoint slice indices for the --float mode
if args.float:
    root_jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    root_qposadr = model.jnt_qposadr[root_jnt_id]   # 7 values: pos(3) + quat(4)
    root_dofadr  = model.jnt_dofadr[root_jnt_id]    # 6 values: ang_vel(3) + lin_vel(3)
    frozen_qpos  = data.qpos[root_qposadr:root_qposadr + 7].copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data) # add a comma 0 to pause the robot in simulation in mid-air
        if args.float:
            data.qpos[root_qposadr:root_qposadr + 7] = frozen_qpos
            data.qvel[root_dofadr:root_dofadr + 6]   = 0
            mujoco.mj_forward(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))
