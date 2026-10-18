"""Two crabs on one stage: a MuJoCo scene holding two skinned robots (prefixed A_ and B_),
and the splicing that turns separately simulated segments into one timeline per crab."""
import sys
from pathlib import Path

import mujoco
import numpy as np

ROBOT = Path(__file__).resolve().parents[1]
FOG = (0.59, 0.59, 0.61)            # the floor's far colour = the backdrop at the horizon


def stage(skin_a, skin_b=None):
    """The stage with one robot (A_) or two (A_, B_)."""
    spec = mujoco.MjSpec()
    # a light concrete battle floor with a darker circle, lit from above and both sides
    # The floor's far edge used to show as a ragged, flickering 'horizon' (an 8 m plane against the
    # black background, at grazing depth precision). Now: a 40 m floor that fades into fog of the
    # backdrop's horizon colour, so there is no edge to see; the circle sits 2 mm up (no z-fighting).
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], rgba=[0.62, 0.62, 0.64, 1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[1.6, 0.001, 0], pos=[0, 0, 0.001],
                            rgba=[0.45, 0.45, 0.48, 1], contype=0, conaffinity=0)
    spec.add_texture(name='backdrop', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=list(FOG),   # one colour: no seam where the floor fades out
                     rgb2=list(FOG), width=512, height=3072)
    # directional lights add up over the whole floor (the spot lights fell off): about half the
    # old intensities keep the floor near its colour instead of blowing out to white
    for pos, dirn, dif in (([0, 0, 3.5], [0, 0, -1], 0.28), ([0, -3, 2.5], [0, 1, -0.8], 0.24),
                           ([0, 3, 2.5], [0, -1, -0.8], 0.14), ([-3, 0, 2.0], [1, 0, -0.6], 0.12)):
        # directional: a spot light leaves the far floor unlit, a dark band under the horizon
        spec.worldbody.add_light(pos=pos, dir=dirn, diffuse=[dif, dif, dif], castshadow=False,
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.headlight.ambient = [0.30, 0.30, 0.30]
    spec.visual.headlight.diffuse = [0.30, 0.30, 0.30]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    for prefix, xml in (('A_', skin_a), ('B_', skin_b)):
        if xml is None:
            continue
        child = mujoco.MjSpec.from_file(str(xml))
        frame = spec.worldbody.add_frame()
        frame.attach_body(child.body('base_link'), prefix, '')
    m = spec.compile()
    ext = m.stat.extent                              # MuJoCo's map distances are in units of it
    m.vis.map.znear, m.vis.map.zfar = 0.03 / ext, 60.0 / ext
    m.vis.map.fogstart, m.vis.map.fogend = 3.0 / ext, 14.0 / ext
    m.vis.rgba.fog[:] = [*FOG, 1]
    return m


def qpos_slices(m):
    """For each robot prefix, the qpos addresses of its free joint and of its 22 joints by name."""
    out = {}
    for p in ('A_', 'B_'):
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + 'floating_base') < 0:
            continue
        free = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + 'floating_base')]
        joints = {}
        for j in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n and n.startswith(p) and n != p + 'floating_base':
                joints[n[len(p):]] = m.jnt_qposadr[j]
        out[p] = (free, joints)
    return out


def yaw_of(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def place(Q, xy, yaw):
    """Rigidly move a segment (qpos rows of one robot) so it starts at `xy` with heading `yaw`."""
    Q = Q.copy()
    d_yaw = yaw - yaw_of(Q[0, 3:7])
    c, s = np.cos(d_yaw), np.sin(d_yaw)
    rot = np.zeros(4); mujoco.mju_axisAngle2Quat(rot, np.array([0, 0, 1.0]), d_yaw)
    p0 = Q[0, :2].copy()
    for i in range(len(Q)):
        dx, dy = Q[i, 0] - p0[0], Q[i, 1] - p0[1]
        Q[i, 0], Q[i, 1] = xy[0] + c * dx - s * dy, xy[1] + s * dx + c * dy
        qq = np.zeros(4); mujoco.mju_mulQuat(qq, rot, Q[i, 3:7]); Q[i, 3:7] = qq
    return Q


def splice(segments, dt, start_xy, start_yaw, blend_s=0.1):
    """Chain segments (lists of qpos rows at `dt`) end to start, each placed where the last ended."""
    out = []
    xy, yaw = np.array(start_xy, float), start_yaw
    for S in segments:
        S = place(np.asarray(S), xy, yaw)
        if out:
            nb = min(int(blend_s / dt), len(S) - 1)
            last = out[-1]
            for k in range(nb):                                  # cross-fade the pose mismatch away
                u = (k + 1) / (nb + 1)
                S[k, :3] = (1 - u) * last[:3] + u * S[k, :3]
                S[k, 7:] = (1 - u) * last[7:] + u * S[k, 7:]
                qq = (1 - u) * last[3:7] + u * S[k, 3:7]; S[k, 3:7] = qq / np.linalg.norm(qq)
        out.extend(S)
        xy, yaw = np.array(out[-1][:2]), yaw_of(out[-1][3:7])
    return np.array(out)
