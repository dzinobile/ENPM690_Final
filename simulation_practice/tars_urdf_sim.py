import mujoco
import mujoco.viewer
import os
import tempfile
import time

_DIR      = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(_DIR, "TARS", "urdf", "TARS.urdf")
TARS_DIR  = os.path.join(_DIR, "TARS")

# Resolve package:// URIs to absolute paths so MuJoCo can find the STL meshes.
# Write to a temp file next to the originals so from_xml_path has a filesystem
# context to load the mesh files from.
with open(URDF_PATH) as f:
    urdf_xml = f.read()

urdf_xml = urdf_xml.replace("package://TARS/", TARS_DIR + "/")
# urdf_xml = urdf_xml.replace(".STL", ".OBJ")

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf",
                                  dir=os.path.join(TARS_DIR, "urdf"),
                                  delete=False)
tmp.write(urdf_xml)
tmp.close()

try:
    model = mujoco.MjModel.from_xml_path(tmp.name)
finally:
    os.unlink(tmp.name)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0.0, model.opt.timestep - (time.time() - step_start)))
