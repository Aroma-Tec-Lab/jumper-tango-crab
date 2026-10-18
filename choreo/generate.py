"""Ballerina choreography -> dance clip (.npz) for the jumper dance-tracking framework.

Keyframes (choreography.py) set the base pose, both arms, the rear legs' lifted pose,
and two blend channels:
  com   0..1  how far the CoM is pulled over the middle feet (1 = over the LM/RM line)
  lift  0..1  rear legs from planted (IK) to their keyframed lifted pose
Support legs stay planted on their HOME footprint: their joints are solved by IK every frame,
so the clip is one kinematically consistent posture (the check the dance loader runs).
Channels are interpolated with quintic smoothstep between keyframes (zero velocity and
acceleration at every keyframe), which is what makes the motion read as slow and sustained.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT / 'jumper'))
from tasks.jumper.common.constants import HOME, STAND_Z  # noqa: E402
from tasks.jumper.common.dance.motion import LEG_JOINTS, _rpy_to_quat  # noqa: E402

TRAIN_XML = ROBOT / 'jumper/assets/jumper/jumper.xml'
ARM_SIGN = {0: -1, 1: 1, 2: -1, 3: -1, 4: -1}
DT = 0.01


def smooth(u):
    return u * u * u * (10 - 15 * u + 6 * u * u)


def channel(keys, name, t):
    """Quintic-smoothstep interpolation of one channel; a key without it holds the last value."""
    pts = []
    for k in keys:
        if name in k:
            pts.append((k['t'], np.asarray(k[name], dtype=float)))
    out = np.empty((len(t),) + pts[0][1].shape)
    for i, ti in enumerate(t):
        if ti <= pts[0][0]:
            out[i] = pts[0][1]; continue
        if ti >= pts[-1][0]:
            out[i] = pts[-1][1]; continue
        j = max(k for k in range(len(pts)) if pts[k][0] <= ti)
        (t0, a), (t1, b) = pts[j], pts[j + 1]
        out[i] = a + (b - a) * smooth((ti - t0) / (t1 - t0))
    return out


class Rig:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(TRAIN_XML))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        self.site = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in ('LF', 'RF', 'LM', 'RM', 'LR', 'RR')}
        self.set(HOME, (0, 0, STAND_Z), (0, 0, 0))
        self.foot_home = {n: self.d.site_xpos[s].copy() for n, s in self.site.items()}
        self.home_z = float(np.mean([self.foot_home[n][2] for n in ('LM', 'RM', 'LR', 'RR')]))

    def set(self, joints, pos, rpy):
        d, m = self.d, self.m
        d.qpos[:3] = pos
        d.qpos[3:7] = _rpy_to_quat(np.asarray([rpy], float))[0]
        for n, v in joints.items():
            j = self.jid[n]
            d.qpos[m.jnt_qposadr[j]] = np.clip(v, *m.jnt_range[j])
        mujoco.mj_forward(m, d)

    def q(self, name):
        return float(self.d.qpos[self.m.jnt_qposadr[self.jid[name]]])

    def ik_leg(self, leg, target, iters=60):
        m, d = self.m, self.d
        names = [f'{leg}_J{k}_joint' for k in range(3)]
        dofs = [m.jnt_dofadr[self.jid[n]] for n in names]
        adrs = [m.jnt_qposadr[self.jid[n]] for n in names]
        jac = np.zeros((3, m.nv))
        for _ in range(iters):
            err = target - d.site_xpos[self.site[leg]]
            if np.linalg.norm(err) < 1e-6:
                break
            mujoco.mj_jacSite(m, d, jac, None, self.site[leg])
            Jl = jac[:, dofs]
            dq = Jl.T @ np.linalg.solve(Jl @ Jl.T + 1e-6 * np.eye(3), err)
            for a, n, v in zip(adrs, names, dq):
                d.qpos[a] = np.clip(d.qpos[a] + v, *m.jnt_range[self.jid[n]])
            mujoco.mj_kinematics(m, d)
        return float(np.linalg.norm(target - d.site_xpos[self.site[leg]]))

    def com(self):
        mujoco.mj_comPos(self.m, self.d)
        return self.d.subtree_com[1].copy()     # body 1 = base_link subtree = whole robot


def arm_dict(left, right):
    out = {}
    for k in range(5):
        out[f'LF_J{k}_joint'] = left[k]
        out[f'RF_J{k}_joint'] = ARM_SIGN[k] * right[k]
    return out


def build(keys, duration):
    rig = Rig()
    t = np.arange(0, duration, DT)
    ch = {n: channel(keys, n, t) for n in ('base', 'larm', 'rarm', 'rear', 'com', 'lift')}
    out_q = {n: np.zeros(len(t)) for n in HOME}
    out_pos = np.zeros((len(t), 3)); out_rpy = np.zeros((len(t), 3))
    worst_ik = 0.0; com_err = []
    for i in range(len(t)):
        dx, dy, dz, roll, pitch, yaw = ch['base'][i]
        joints = dict(HOME); joints.update(arm_dict(ch['larm'][i], ch['rarm'][i]))
        pos = np.array([dx, dy, STAND_Z + dz]); rpy = (roll, pitch, yaw)
        c, s = float(ch['com'][i]), float(ch['lift'][i])
        rear_key = ch['rear'][i]
        for _ in range(4):                       # CoM correction needs re-solving the legs
            # Rear legs: solve the planted angles from a fixed seed (HOME) every time, so a
            # lifted pose never seeds the IK and the solution cannot flip between branches.
            for leg in ('LR', 'RR'):
                for k in range(3):
                    joints[f'{leg}_J{k}_joint'] = HOME[f'{leg}_J{k}_joint']
            rig.set(joints, pos, rpy)
            for leg in ('LM', 'RM'):
                worst_ik = max(worst_ik, rig.ik_leg(leg, rig.foot_home[leg]))
            for leg in ('LR', 'RR'):
                e = rig.ik_leg(leg, rig.foot_home[leg])
                if s < 1e-6:
                    worst_ik = max(worst_ik, e)
            planted = {f'{leg}_J{k}_joint': rig.q(f'{leg}_J{k}_joint')
                       for leg in ('LM', 'RM', 'LR', 'RR') for k in range(3)}
            joints.update({n: v for n, v in planted.items() if n[1] == 'M'})
            for k in range(3):
                for leg, sign in (('LR', 1), ('RR', -1)):
                    n = f'{leg}_J{k}_joint'
                    joints[n] = (1 - s) * planted[n] + s * sign * rear_key[k]
            rig.set(joints, pos, rpy)
            if c > 1e-6:
                mid = 0.5 * (rig.d.site_xpos[rig.site['LM']] + rig.d.site_xpos[rig.site['RM']])
                err = (mid[:2] - rig.com()[:2]) * c
                pos[:2] += err
        if c > 0.5:
            mid = 0.5 * (rig.d.site_xpos[rig.site['LM']] + rig.d.site_xpos[rig.site['RM']])
            com_err.append(float(np.linalg.norm(mid[:2] - rig.com()[:2])))
        for n in HOME:
            out_q[n][i] = rig.q(n)
        out_pos[i] = rig.d.qpos[:3]; out_rpy[i] = rpy
    return t, out_q, out_pos, out_rpy, dict(worst_support_ik_m=worst_ik,
                                            max_com_offset_two_leg_m=max(com_err) if com_err else None)


def write(path, t, q, pos, rpy):
    data = {'time': t.astype(np.float32), 'dt': np.float64(DT), 'phase': np.ones(len(t), np.float32)}
    for prefix in ('', 'meas_'):
        for k, a in enumerate('xyz'):
            data[f'{prefix}body_{a}'] = pos[:, k].astype(np.float32)
        for k, a in enumerate(('roll', 'pitch', 'yaw')):
            data[f'{prefix}body_{a}'] = rpy[:, k].astype(np.float32)
        for leg, names in LEG_JOINTS.items():
            for j, n in enumerate(names):
                data[f'{prefix}L{leg}_j{j}'] = q[n].astype(np.float32)
    data['bpm'] = np.float64(0.0)
    np.savez_compressed(path, **data)


if __name__ == '__main__':
    import importlib
    spec = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else 'choreography')
    t, q, pos, rpy, report = build(spec.KEYS, spec.DURATION)
    out = Path(sys.argv[2] if len(sys.argv) > 2 else 'ballerina.npz')
    write(out, t, q, pos, rpy)
    speed = max(float(np.max(np.abs(np.diff(v)))) / DT for v in q.values())
    report.update(frames=len(t), seconds=float(t[-1] + DT), max_joint_speed_rad_s=speed, file=str(out))
    print(json.dumps(report, indent=1))
