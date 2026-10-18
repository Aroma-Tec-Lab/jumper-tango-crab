"""Render the tango crab for X: the trained jumper.dance_tango replay on the battle stage, a camera
keyed to the choreography (close on the face for the abrazo, wide for the sideways runs, from the
front-left for the quebrada so the raised claw does not hide the face), fade in and out.

usage: tango_render.py <replay.npz> <skin.xml> <length> <out.mp4>
"""
import sys

import mujoco
import numpy as np
import imageio

import battle_stage as bs
import roll_search as rs

z = np.load(sys.argv[1]); Q, dt = z['qpos'], float(z['dt'])
t0, length = 0.0, float(sys.argv[3])
m = bs.stage(sys.argv[2]); d = mujoco.MjData(m)
free, joints = bs.qpos_slices(m)['A_']
src = mujoco.MjModel.from_xml_path(str(rs.ROBOT / 'jumper/assets/jumper/jumper.xml'))
adr = {n: src.jnt_qposadr[mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in rs.HOME}
FPS, FADE = 30, 0.8
#        t     azimuth elevation distance   (azimuth 180 = in front; 225 = front-left of the crab)
CAM = [(0.0,   180, -10, 0.50),     # abrazo, head snaps: the face, lips and lashes
       (3.6,   180, -12, 0.55),
       (7.6,   160, -16, 0.70),     # caminata done, round to its right for the ochos
       (13.9,  200, -15, 0.65),     # boleo from the front-left: the leg whips out towards us
       (15.6,  180, -18, 0.85),     # wide for the sideways runs
       (19.6,  180, -18, 0.85),
       (21.6,  195, -12, 0.55),     # golpecitos close
       (23.0,  208, -10, 0.62),     # quebrada slightly from its left: face, high claw and the
       (28.7,  204, -8, 0.56)]      # stretched leg all in view (from the side the claw hides the face)


def cam_at(t):
    for (ta, *a), (tb, *b) in zip(CAM, CAM[1:]):
        if ta <= t <= tb:
            u = (t - ta) / (tb - ta); u = u * u * (3 - 2 * u)
            return [x + (y - x) * u for x, y in zip(a, b)]
    return CAM[-1][1:]
w = imageio.get_writer(sys.argv[4], fps=FPS, quality=8, macro_block_size=8)
look = None
with mujoco.Renderer(m, 1080, 1920) as r:
    for f in range(int(length * FPS)):
        tv = f / FPS
        q = Q[min(int(round((t0 + tv) / dt)), len(Q) - 1)]
        d.qpos[free:free + 7] = q[:7]
        for n, a in joints.items():
            d.qpos[a] = q[adr[n]]
        mujoco.mj_forward(m, d)
        look = q[:3].copy() if look is None else look + 0.05 * (q[:3] - look)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (look[0], look[1], 0.11)
        cam.azimuth, cam.elevation, cam.distance = cam_at(tv)
        opt = mujoco.MjvOption(); opt.geomgroup[1] = opt.geomgroup[3] = 0
        r.update_scene(d, cam, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        img = r.render().astype(np.float32)
        g = min(1.0, tv / FADE, (length - tv) / FADE)
        w.append_data((img * max(0.0, g)).astype(np.uint8))
w.close()
