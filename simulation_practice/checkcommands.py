import mujoco, numpy as np
from train_tars_ppo import JOINT_NAMES, action_to_target
import math
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("tars.xml")
data  = mujoco.MjData(model)

# Send a max-rotation action to every joint
test_action = np.ones(10, dtype=np.float32)
data.ctrl[:] = action_to_target(test_action)

print("ctrl values:")
for name, val in zip(JOINT_NAMES, data.ctrl):
    print(f"  {name:12s}: {val:.4f}")

mujoco.mj_step(model, data)
print("\njoint positions after 1 step:")
for name, val in zip(JOINT_NAMES, [data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES]):
    print(f"  {name:12s}: {val:.4f}")
