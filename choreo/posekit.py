"""Pose helpers for the Ballerina choreography: mirroring, FK, and still renders."""
import sys
from pathlib import Path

import mujoco
import numpy as np

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT / 'jumper'))
from tasks.jumper.common.constants import HOME, STAND_Z  # noqa: E402

SKIN_XML = ROBOT / 'skins/ballerina/train-model/jumper_ballerina_v4.xml'
# Right side mirrors left: arms negate J0, J2, J3, J4 and keep J1; legs negate all three
# (read off HOME, where every right joint is the left one under exactly this rule).
ARM_SIGN = {0: -1, 1: 1, 2: -1, 3: -1, 4: -1}


def load(xml=SKIN_XML, floor=True):
    spec = mujoco.MjSpec.from_file(str(xml))
    if floor:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                                rgba=[0.83, 0.80, 0.78, 1])
        spec.worldbody.add_light(pos=[0.5, -0.5, 1.5], dir=[-0.3, 0.3, -1], diffuse=[0.6, 0.6, 0.6])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    return spec.compile()


def arms(j, side='both'):
    """Joint dict for both arms from the LEFT arm's five angles."""
    out = {}
    for k, v in enumerate(j):
        if v is None:
            continue
        out[f'LF_J{k}_joint'] = v
        out[f'RF_J{k}_joint'] = ARM_SIGN[k] * v
    return out


def legs(leg, j):
    """Joint dict for a left/right pair ('M' or 'R') from the LEFT leg's three angles."""
    out = {}
    for k, v in enumerate(j):
        out[f'L{leg}_J{k}_joint'] = v
        out[f'R{leg}_J{k}_joint'] = -v
    return out


def set_state(m, d, joints, base=(0, 0, STAND_Z), rpy=(0, 0, 0)):
    d.qpos[:] = 0
    d.qpos[:3] = base
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.asarray(rpy, dtype=float), 'xyz')
    d.qpos[3:7] = q
    full = dict(HOME); full.update(joints)
    for n, v in full.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        lo, hi = m.jnt_range[j]
        d.qpos[m.jnt_qposadr[j]] = np.clip(v, lo, hi)
    mujoco.mj_forward(m, d)


def render(m, d, renderer, azimuth=180, elevation=-12, distance=0.75, lookat=(0, 0, 0.13)):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = opt.geomgroup[3] = 0   # hide collision shapes, as play.py's video does
    renderer.update_scene(d, cam, scene_option=opt)
    return renderer.render()
