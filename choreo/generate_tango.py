"""Tango choreography -> dance clip. generate2.py's machinery, extended by two things tango needs:

* `Step` -- a tripod stepping figure that may walk as well as turn: each full cycle moves the
  footprints by `dxy / cycles` (in the figure's start frame) and turns them by `dpsi / cycles`.
  generate2's `Turn` is the special case dxy = 0. Figures compose: each starts in the frame the
  previous ones left, so a caminata followed by ochos walks on from where it stopped.
* one rear leg at a time: `lift_l` / `lift_r` (0 = planted, 1 = keyed pose) with `rear_l` /
  `rear_r` (left convention) -- a boleo or the final quebrada lifts one leg, not both.

Everything else (IK-planted feet, claws as keyed arms or planted feet, the CoM pull toward the
weighted support centroid) is generate2's, unchanged.

usage: generate_tango.py <spec module> <out.npz>
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

import generate as g
from generate import HOME, STAND_Z, DT, smooth, channel, arm_dict
from generate2 import LIMBS, TRIPOD, ik_arm


def rot2(v, psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Step:
    """Footprint plan of one tripod stepping figure starting at t0 (group LF/RM/LR swings first)."""

    def __init__(self, t0, cycles, half, lift, dpsi=0.0, dxy=(0.0, 0.0)):
        self.t0, self.cycles, self.half, self.lift = t0, cycles, half, lift
        self.dpsi, self.dxy = dpsi, np.asarray(dxy, float)
        self.t1 = t0 + 2 * half * cycles

    def _k(self, t):
        k = np.clip((t - self.t0) / self.half, 0, 2 * self.cycles)
        i = int(np.floor(k)) if k < 2 * self.cycles else 2 * self.cycles - 1
        return i, k - i

    def base(self, t):
        """(xy, yaw) of the base in the figure's start frame: half a step per half-step."""
        if t <= self.t0:
            return np.zeros(2), 0.0
        i, u = self._k(t)
        b = (i + smooth(u)) / 2 / self.cycles
        return b * self.dxy, b * self.dpsi

    def _local(self, home, n):
        """A footprint after n full steps, in the start frame."""
        f = n / self.cycles
        p = np.array(home, float)
        p[:2] = rot2(home[:2], f * self.dpsi) + f * self.dxy
        return p

    def foot(self, t, limb, home):
        i, u = self._k(t)
        group = 0 if limb in TRIPOD[0] else 1
        done = (i // 2) + (1 if (group == 0 and i % 2 == 1) else 0)
        if i % 2 == group:
            p0, p1 = self._local(home, done), self._local(home, done + 1)
            p = p0 + (p1 - p0) * smooth(u)
            p[2] += self.lift * np.sin(np.pi * u)
            return p
        return self._local(home, done)


def frame_at(figs, t):
    """World transform (xy, yaw) of the completed figures before t, and the active one."""
    P, Y = np.zeros(2), 0.0
    active = None
    for f in sorted(figs, key=lambda f: f.t0):
        if f.t1 <= t:
            P = P + rot2(f.dxy, Y); Y += f.dpsi
        elif f.t0 < t:
            active = f
    return P, Y, active


def to_world(P, Y, p):
    q = np.array(p, float); q[:2] = P + rot2(p[:2], Y)
    return q


def build(keys, duration, figs, foot_offset=None):
    """`foot_offset` moves limbs' footprints (xy, body frame) away from the HOME ones -- tango
    plants the claws wider, so a turning step cannot carry one claw onto the other."""
    rig = g.Rig()
    t = np.arange(0, duration, DT)
    names = ('base', 'larm', 'rarm', 'rear_l', 'rear_r', 'com', 'lift_l', 'lift_r', 'lf_plant',
             'rf_plant', 'support')
    ch = {n: channel(keys, n, t) for n in names}
    home = {L: p.copy() for L, p in rig.foot_home.items()}
    for L, (ox, oy) in (foot_offset or {}).items():
        home[L][:2] += (ox, oy)
    out_q = {n: np.zeros(len(t)) for n in HOME}
    out_pos = np.zeros((len(t), 3)); out_rpy = np.zeros((len(t), 3))
    worst = {'support_ik_m': 0.0, 'step_ik_m': 0.0, 'plant_ik_m': 0.0}
    prev = dict(HOME)
    prev_ik = dict(HOME)
    for i, ti in enumerate(t):
        dx, dy, dz, roll, pitch, yaw = ch['base'][i]
        P, Y, active = frame_at(figs, ti)
        bxy, byaw = active.base(ti) if active else (np.zeros(2), 0.0)
        xy = P + rot2(bxy + np.array([dx, dy]), Y)
        pos = np.array([xy[0], xy[1], STAND_Z + dz]); rpy = (roll, pitch, yaw + Y + byaw)
        c = float(ch['com'][i])
        lift = {'LR': float(ch['lift_l'][i]), 'RR': float(ch['lift_r'][i])}
        rear = {'LR': ch['rear_l'][i], 'RR': ch['rear_r'][i]}
        plant = {'LF': float(ch['lf_plant'][i]), 'RF': float(ch['rf_plant'][i])}
        w = np.asarray(ch['support'][i]); w = w / w.sum()
        joints = dict(HOME); joints.update(arm_dict(ch['larm'][i], ch['rarm'][i]))
        keyed_arm = {n: joints[n] for n in joints if n[1] == 'F'}
        target = {L: to_world(P, Y, active.foot(ti, L, home[L]) if active else home[L]) for L in LIMBS}
        for _ in range(4):
            for L in ('LR', 'RR'):
                for k in range(3):
                    n = f'{L}_J{k}_joint'
                    joints[n] = prev[n] if active else HOME[n]
            for L in ('LF', 'RF'):
                for k in range(5):
                    n = f'{L}_J{k}_joint'
                    joints[n] = prev_ik[n] if (active or plant[L] > 0) else keyed_arm[n]
            rig.set(joints, pos, rpy)
            for L in ('LM', 'RM', 'LR', 'RR'):
                e = rig.ik_leg(L, target[L])
                if L[1] == 'M' or lift.get(L, 0.0) < 1e-6:
                    key = 'step_ik_m' if active else 'support_ik_m'
                    worst[key] = max(worst[key], e)
            for L in ('LF', 'RF'):
                if active or plant[L] > 0:
                    e = ik_arm(rig, L, target[L])
                    if e > 1e-3:
                        for k in range(5):
                            n = f'{L}_J{k}_joint'
                            rig.d.qpos[rig.m.jnt_qposadr[rig.jid[n]]] = HOME[n]
                        mujoco.mj_kinematics(rig.m, rig.d)
                        e = ik_arm(rig, L, target[L])
                    if active or plant[L] > 0.999:
                        key = 'step_ik_m' if active else 'plant_ik_m'
                        worst[key] = max(worst[key], e)
            solved = {n: rig.q(n) for n in HOME}
            for L in ('LF', 'RF'):
                if active or plant[L] > 0:
                    for k in range(5):
                        prev_ik[f'{L}_J{k}_joint'] = solved[f'{L}_J{k}_joint']
            for n in HOME:
                L = n[:2]
                if L in ('LM', 'RM'):
                    joints[n] = solved[n]
                elif L in ('LR', 'RR'):
                    sign = 1 if L == 'LR' else -1
                    k = int(n[4])
                    s = 0.0 if active else lift[L]
                    joints[n] = (1 - s) * solved[n] + s * sign * rear[L][k]
                else:
                    p = 1.0 if active else plant[L]
                    joints[n] = (1 - p) * keyed_arm[n] + p * solved[n]
            rig.set(joints, pos, rpy)
            if c > 1e-6:
                goal = sum(w[j] * rig.d.site_xpos[rig.site[L]] for j, L in enumerate(LIMBS))
                pos[:2] += (goal[:2] - rig.com()[:2]) * c
        for n in HOME:
            out_q[n][i] = rig.q(n)
            prev[n] = rig.q(n)
        out_pos[i] = rig.d.qpos[:3]; out_rpy[i] = rpy
    return t, out_q, out_pos, out_rpy, worst


if __name__ == '__main__':
    import importlib
    spec = importlib.import_module(sys.argv[1])
    t, q, pos, rpy, report = build(spec.KEYS, spec.DURATION, spec.FIGS, getattr(spec, 'FOOT_OFFSET', None))
    out = Path(sys.argv[2])
    g.write(out, t, q, pos, rpy)
    speeds = {n: float(np.max(np.abs(np.diff(v)))) / DT for n, v in q.items()}
    worst_joint = max(speeds, key=speeds.get)
    report.update(frames=len(t), seconds=round(float(t[-1] + DT), 2),
                  max_joint_speed_rad_s=round(speeds[worst_joint], 2), at_joint=worst_joint,
                  final_xy=np.round(pos[-1, :2], 3).tolist(),
                  final_yaw_deg=round(float(np.degrees(rpy[-1, 2])), 1), file=str(out))
    print(json.dumps(report, indent=1))
