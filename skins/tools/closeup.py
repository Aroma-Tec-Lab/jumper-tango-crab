"""Close-up renders of an assembly: usage closeup.py <assembly-dir> <out.png>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import DESIGN
sys.path.append(str(DESIGN / 'scripts'))
import mujoco
import numpy as np
import imageio.v3 as iio
from preview_simulation import load_package

model = load_package(Path(sys.argv[1]))
data = mujoco.MjData(model)
if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
tiles = []
with mujoco.Renderer(model, height=450, width=600) as r:
    for az, el in [(145, -25), (180, -12), (90, -8), (35, -35)]:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, .17]
        cam.distance, cam.elevation, cam.azimuth = .48, el, az
        r.update_scene(data, cam)
        tiles.append(r.render())
top = np.concatenate(tiles[:2], 1); bot = np.concatenate(tiles[2:], 1)
iio.imwrite(sys.argv[2], np.concatenate([top, bot], 0))
