"""Ballerina choreography v3 -> dance clip: adds a stepping pirouette and claw-supported figures.

Every limb is, on each frame, in one of three states:
  * IK-driven to a world target -- planted on the footprint, or swinging between footprints
    during the turn (all six limbs: the claws are feet in the turn's tripods),
  * keyframed (arms in the air, rear legs in the arabesque), or
  * a blend of the two (`lf_plant` / `rf_plant`: 0 = keyframed arm, 1 = claw planted;
    `lift`: 0 = rear legs planted, 1 = keyframed lifted pose).
The CoM is pulled towards the weighted centroid of the support feet (`support` = six weights,
LF RF LM RM LR RR) by `com` in 0..1.

The pirouette is a tripod turn -- groups LF/RM/LR and RF/LM/RR, the robot's own walking
tripods, each balanced about the CoM -- so it needs no CoM shift: one group swings to the
footprint rotated by `dpsi` while the base turns half of it, then the other group follows.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

import generate as g
from generate import HOME, STAND_Z, ARM_SIGN, DT, smooth, channel, arm_dict

LIMBS = ('LF', 'RF', 'LM', 'RM', 'LR', 'RR')
TRIPOD = (('LF', 'RM', 'LR'), ('RF', 'LM', 'RR'))


def rotz(p, psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


class Turn:
    """Footprint plan of one tripod turn starting at t0."""

    def __init__(self, t0, total, cycles, half, lift):
        self.t0, self.cycles, self.half, self.lift = t0, cycles, half, lift
        self.dpsi = total / cycles
        self.t1 = t0 + 2 * half * cycles

    def yaw(self, t):
        """Base yaw added by the turn so far (0 before, `total` after)."""
        if t <= self.t0:
            return 0.0
        k = min((t - self.t0) / self.half, 2 * self.cycles)
        i = int(np.floor(k)) if k < 2 * self.cycles else 2 * self.cycles - 1
        return (i + smooth(k - i)) * self.dpsi / 2

    def foot(self, t, limb, home):
        """World target of a limb's foot site at t (the turn only)."""
        k = np.clip((t - self.t0) / self.half, 0, 2 * self.cycles)
        group = 0 if limb in TRIPOD[0] else 1
        i = int(np.floor(k)) if k < 2 * self.cycles else 2 * self.cycles - 1
        u = k - i
        # how many full dpsi steps this limb has completed before half-step i
        done = (i + (1 if group == 0 else 0)) // 2 if True else 0
        done = (i // 2) + (1 if (group == 0 and i % 2 == 1) else 0)
        if i % 2 == group:              # this limb swings in half-step i
            p0 = rotz(home, done * self.dpsi)
            p1 = rotz(home, (done + 1) * self.dpsi)
            p = p0 + (p1 - p0) * smooth(u)
            p[2] += self.lift * np.sin(np.pi * u)
            return p
        return rotz(home, done * self.dpsi)


def ik_arm(rig, limb, target, iters=200, pull=3e-4):
    """Claw site to `target` with J0..J3. The arm has one joint more than a point needs, so the
    solution is pulled gently toward HOME in the null space -- without it the IK drifts frame by
    frame into a twisted branch and later snaps back (measured: 87 rad/s spikes at a turn)."""
    m, d = rig.m, rig.d
    names = [f'{limb}_J{k}_joint' for k in range(4)]
    dofs = [m.jnt_dofadr[rig.jid[n]] for n in names]
    adrs = [m.jnt_qposadr[rig.jid[n]] for n in names]
    q_ref = np.array([HOME[n] for n in names])
    jac = np.zeros((3, m.nv))
    for _ in range(iters):
        err = target - d.site_xpos[rig.site[limb]]
        q = np.array([d.qpos[a] for a in adrs])
        if np.linalg.norm(err) < 1e-6 and np.linalg.norm(q - q_ref) < 1e-3:
            break
        mujoco.mj_jacSite(m, d, jac, None, rig.site[limb])
        Jl = jac[:, dofs]
        dq = np.linalg.solve(Jl.T @ Jl + pull * np.eye(4), Jl.T @ err + pull * (q_ref - q))
        for a, n, v in zip(adrs, names, dq):
            d.qpos[a] = np.clip(d.qpos[a] + v, *m.jnt_range[rig.jid[n]])
        mujoco.mj_kinematics(m, d)
    return float(np.linalg.norm(target - d.site_xpos[rig.site[limb]]))


def build(keys, duration, turns):
    rig = g.Rig()
    t = np.arange(0, duration, DT)
    names = ('base', 'larm', 'rarm', 'rear', 'com', 'lift', 'lf_plant', 'rf_plant', 'support')
    ch = {n: channel(keys, n, t) for n in names}
    home = rig.foot_home
    out_q = {n: np.zeros(len(t)) for n in HOME}
    out_pos = np.zeros((len(t), 3)); out_rpy = np.zeros((len(t), 3))
    worst = {'support_ik_m': 0.0, 'turn_ik_m': 0.0, 'plant_ik_m': 0.0}
    prev = dict(HOME)       # last frame's solution: the IK seed for legs in a turn
    prev_ik = dict(HOME)    # last *pure* IK solution of each arm -- never a blended pose, so a
                            # claw lifting off or coming down cannot reseed the IK onto another branch
    for i, ti in enumerate(t):
        dx, dy, dz, roll, pitch, yaw = ch['base'][i]
        psi = sum(tr.yaw(ti) for tr in turns)
        active = next((tr for tr in turns if tr.t0 < ti < tr.t1), None)
        pos = np.array([dx, dy, STAND_Z + dz]); rpy = (roll, pitch, yaw + psi)
        c, s = float(ch['com'][i]), float(ch['lift'][i])
        plant = {'LF': float(ch['lf_plant'][i]), 'RF': float(ch['rf_plant'][i])}
        w = np.asarray(ch['support'][i]); w = w / w.sum()
        joints = dict(HOME); joints.update(arm_dict(ch['larm'][i], ch['rarm'][i]))
        keyed_arm = {n: joints[n] for n in joints if n[1] == 'F'}
        # footprint targets on this frame
        target = {}
        # yaw already turned by every other turn (earlier ones are complete, later ones 0)
        psi_prev = sum(tr.yaw(ti) for tr in turns if tr is not active) if active else psi
        for L in LIMBS:
            target[L] = rotz(active.foot(ti, L, home[L]), psi_prev) if active else rotz(home[L], psi)
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
                if L[1] == 'M' or s < 1e-6:
                    worst['turn_ik_m' if active else 'support_ik_m'] = max(
                        worst['turn_ik_m' if active else 'support_ik_m'], e)
            for L in ('LF', 'RF'):
                if active or plant[L] > 0:
                    e = ik_arm(rig, L, target[L])
                    if e > 1e-3:              # warm start stuck: retry from HOME
                        for k in range(5):
                            n = f'{L}_J{k}_joint'
                            rig.d.qpos[rig.m.jnt_qposadr[rig.jid[n]]] = HOME[n]
                        mujoco.mj_kinematics(rig.m, rig.d)
                        e = ik_arm(rig, L, target[L])
                    if active or plant[L] > 0.999:
                        key = 'turn_ik_m' if active else 'plant_ik_m'
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
                    joints[n] = solved[n] if active else (1 - s) * solved[n] + s * sign * ch['rear'][i][k]
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
    t, q, pos, rpy, report = build(spec.KEYS, spec.DURATION, spec.TURNS)
    out = Path(sys.argv[2])
    g.write(out, t, q, pos, rpy)
    speeds = {n: float(np.max(np.abs(np.diff(v)))) / DT for n, v in q.items()}
    worst_joint = max(speeds, key=speeds.get)
    report.update(frames=len(t), seconds=round(float(t[-1] + DT), 2),
                  max_joint_speed_rad_s=round(speeds[worst_joint], 2), at_joint=worst_joint,
                  final_yaw_deg=round(float(np.degrees(rpy[-1, 2])), 1), file=str(out))
    print(json.dumps(report, indent=1))
