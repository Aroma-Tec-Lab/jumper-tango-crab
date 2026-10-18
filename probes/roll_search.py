"""Search for an open-loop joint-target schedule that makes Jumper really barrel-roll in physics.

Plain MuJoCo, the Train asset plus a floor, PD at kp=10 / kd=0.5 (the dance tasks' gains)
through the measured servo torque-speed curve (`common/actuator.py`). A schedule is a few
phases of joint targets; an evolution strategy tunes their angles and timing. The score is the
roll actually achieved (unwrapped, from the base's angular velocity) and how upright, on its
feet and still the robot is at the end. Whatever schedule wins, its simulated states are
physically possible by construction -- they are what MuJoCo produced.

usage: roll_search.py search [generations] | replay <params.json> <out.npz>
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT / 'jumper'))
from tasks.jumper.common.constants import HOME, STAND_Z  # noqa: E402
from tasks.jumper.common.actuator import (PLATEAU_TORQUE, CORNER_SPEED, DECAY_SPEED,  # noqa: E402
                                           CUTOFF_SPEED)

KP, KD, DT, T_END = 10.0, 0.5, 0.001, 4.5
AXIS_W = 5.0     # tumbling charge; 12 confined the search to tipping over and back (<= 90 deg)
NAMES = list(HOME)

# parameter vector: name -> (low, high)
SPACE = {
    't_wind': (0.2, 0.6),        # start of the rocking (after a crouch from 0)
    'd_rock': (0.12, 0.6),       # duration of each of the two rocks before the wind-up
    'rock_amp': (0.0, 1.0),      # rock amplitude, as a fraction of the wind-up's leg angles
    'd_wind': (0.15, 0.9),       # wind-up: the left side sinks, the right side rises
    'wind_l_j1': (0.5, 1.6),     # left legs fold (left convention) -- body tilts left
    'wind_r_j1': (-1.5, 0.5),    # right legs straighten -- the right side rises
    'd_push_gap': (0.0, 0.0001), # (kept fixed: the push follows the wind-up directly)
    'd_push': (0.08, 0.40),      # push duration
    'd_tuck': (0.2, 1.2),        # how long everything stays tucked after the push
    'push_j1': (-2.4, 0.5),      # left middle/rear legs during the push (left convention)
    'push_j2': (-2.4, 1.8),
    'push_j0': (-0.6, 0.9),
    'fold_j1': (-0.5, 1.6),      # right legs during the push (left convention): lift/fold
    'fold_j2': (-2.4, 0.0),
    'tuck_j1': (0.0, 1.6),       # all legs while rolling
    'tuck_j2': (-2.4, 0.0),
    'arm_l_j1': (-4.5, 1.2),     # left arm during the push (it can push too)
    'arm_r_j1': (-4.5, 1.2),     # right arm during the push
    'arm_tuck_j1': (-4.5, 1.2),
    'crouch': (0.0, 0.9),        # crouch depth (J1 offset, legs)
}
KEYS = list(SPACE)
LO = np.array([SPACE[k][0] for k in KEYS]); HI = np.array([SPACE[k][1] for k in KEYS])

_model = None


def model():
    global _model
    if _model is None:
        spec = mujoco.MjSpec.from_file(str(ROBOT / 'jumper/assets/jumper/jumper.xml'))
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], friction=[1, 0.01, 0.01])
        m = spec.compile(); m.opt.timestep = DT
        _model = (m, [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in NAMES],
                  [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in NAMES],
                  np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in NAMES]))
    return _model


def leg(prefix, side, j0, j1, j2):
    s = 1 if side == 'L' else -1
    return {f'{side}{prefix}_J0_joint': s * j0, f'{side}{prefix}_J1_joint': s * j1, f'{side}{prefix}_J2_joint': s * j2}


def arm(side, j1):
    return {f'{side}F_J1_joint': j1}


def targets(p, t):
    """Joint targets at time t for parameter dict p."""
    q = dict(HOME)
    hm = (HOME['LM_J0_joint'], HOME['LM_J1_joint'], HOME['LM_J2_joint'])
    hr = (HOME['LR_J0_joint'], HOME['LR_J1_joint'], HOME['LR_J2_joint'])
    tr = p['t_wind']; t0 = tr + 2 * p['d_rock']
    t1 = t0 + p['d_wind']; t2 = t1 + p['d_push']; t3 = t2 + p['d_tuck']
    if tr <= t < t0:                             # pump: rock left-down, then right-down
        a = p['rock_amp']; first = t < tr + p['d_rock']
        lj1 = hm[1] + a * ((p['wind_l_j1'] - hm[1]) if first else (p['wind_r_j1'] - hm[1]))
        rj1 = hm[1] + a * ((p['wind_r_j1'] - hm[1]) if first else (p['wind_l_j1'] - hm[1]))
        q.update(leg('M', 'L', hm[0], lj1, hm[2])); q.update(leg('R', 'L', hr[0], lj1, hr[2]))
        q.update(leg('M', 'R', hm[0], rj1, hm[2])); q.update(leg('R', 'R', hr[0], rj1, hr[2]))
        return q
    if t < tr:                                   # crouch
        for side in 'LR':
            q.update(leg('M', side, hm[0], hm[1] + p['crouch'], hm[2]))
            q.update(leg('R', side, hr[0], hr[1] + p['crouch'], hr[2]))
    elif t < t1:                                 # wind-up: sink to the left, load the left legs
        q.update(leg('M', 'L', hm[0], p['wind_l_j1'], hm[2])); q.update(leg('R', 'L', hr[0], p['wind_l_j1'], hr[2]))
        q.update(leg('M', 'R', hm[0], p['wind_r_j1'], hm[2])); q.update(leg('R', 'R', hr[0], p['wind_r_j1'], hr[2]))
    elif t < t2:                                 # push left, fold right
        q.update(leg('M', 'L', p['push_j0'], p['push_j1'], p['push_j2']))
        q.update(leg('R', 'L', hr[0] + p['push_j0'], p['push_j1'], p['push_j2']))
        q.update(leg('M', 'R', hm[0], p['fold_j1'], p['fold_j2']))
        q.update(leg('R', 'R', hr[0], p['fold_j1'], p['fold_j2']))
        q.update(arm('L', p['arm_l_j1'])); q.update(arm('R', p['arm_r_j1']))
    elif t < t3:                                 # tucked roll
        for side in 'LR':
            q.update(leg('M', side, hm[0], p['tuck_j1'], p['tuck_j2']))
            q.update(leg('R', side, hr[0], p['tuck_j1'], p['tuck_j2']))
        q.update(arm('L', p['arm_tuck_j1'])); q.update(arm('R', p['arm_tuck_j1']))
    return q                                    # after t3: HOME, unfold onto the feet


def torque_limit(w):
    lim = PLATEAU_TORQUE * np.exp(-np.clip(np.abs(w) - CORNER_SPEED, 0, None) / DECAY_SPEED)
    return np.where(np.abs(w) >= CUTOFF_SPEED, 0.0, lim)


def rollout(x, record=False):
    m, qadr, dadr, rng = model()
    d = mujoco.MjData(m)
    p = dict(zip(KEYS, x))
    d.qpos[2] = STAND_Z; d.qpos[3] = 1
    d.qpos[qadr] = [HOME[n] for n in NAMES]
    mujoco.mj_forward(m, d)
    roll = 0.0; prev_phi = 0.0; peak = 0.0; rec = []; axis_dev = 0.0; nax = 0
    n = int(T_END / DT)
    for i in range(n):
        t = i * DT
        if i % 2 == 0:                           # 500 Hz targets
            tq = targets(p, t)
            target = np.clip([tq[k] for k in NAMES], rng[:, 0], rng[:, 1])
        qd = d.qvel[dadr]
        tau = KP * (target - d.qpos[qadr]) - KD * qd
        lim = torque_limit(qd)
        d.qfrc_applied[:] = 0
        d.qfrc_applied[dadr] = np.clip(tau, -lim, lim)
        mujoco.mj_step(m, d)
        if i % 10 == 0:                           # roll from the body's orientation, unwrapped
            R = d.xmat[1].reshape(3, 3)
            phi = np.arctan2(R[2, 1], R[2, 2])
            roll += (phi - prev_phi + np.pi) % (2 * np.pi) - np.pi
            prev_phi = phi; peak = max(peak, abs(roll))
            axis_dev += 1.0 - R[0, 0]; nax += 1          # body x must stay along world x
        if not np.isfinite(d.qpos).all():
            return -1e3, None
        if record and i % 10 == 0:
            rec.append(d.qpos.copy())
    up = 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)          # body z axis . world z
    still = np.linalg.norm(d.qvel[:6])
    height = d.qpos[2]
    # completed roll (net, unwrapped) dominates; the furthest it got shapes the search before that
    score = (min(abs(roll), 2 * np.pi) / (2 * np.pi)) * 10 + (min(peak, 2 * np.pi) / (2 * np.pi)) * 4 \
        + 4 * up - 0.5 * still - 20 * abs(height - STAND_Z) * (up > 0.8) \
        - 3 * max(0.0, abs(roll) - 2 * np.pi - 0.6) \
        - AXIS_W * axis_dev / max(nax, 1) - 4 * (1 - d.xmat[1][0])
    # a barrel roll turns about the long axis only: tumbling (pitch) or ending turned (yaw)
    # both pull the body's x axis off the world's x axis and are charged above
    info = dict(roll_deg=round(float(np.degrees(roll)), 1), peak_deg=round(float(np.degrees(peak)), 1),
                upright=round(float(up), 3), axis_dev=round(axis_dev / max(nax, 1), 3), heading=round(float(d.xmat[1][0]), 3), height=round(float(height), 3), still=round(float(still), 3))
    return float(score), (info, np.array(rec) if record else None)


def evaluate(x):
    s, out = rollout(x)
    return s, (out[0] if out else None)


def search(gens=40, pop=84, seed=0):
    rng = np.random.default_rng(seed)
    mean = (LO + HI) / 2; sigma = (HI - LO) * 0.3
    best = (-np.inf, None, None)
    with Pool(12) as pool:
        for g in range(gens):
            X = np.clip(mean + sigma * rng.standard_normal((pop, len(KEYS))), LO, HI)
            res = pool.map(evaluate, list(X))
            S = np.array([r[0] for r in res])
            order = np.argsort(-S)
            elite = X[order[:pop // 4]]
            mean = elite.mean(0); sigma = np.maximum(elite.std(0), (HI - LO) * 0.02)
            if S[order[0]] > best[0]:
                best = (S[order[0]], X[order[0]], res[order[0]][1])
            print(f'gen {g:2d}  best {S[order[0]]:6.2f}  {res[order[0]][1]}  overall {best[0]:6.2f}', flush=True)
    return best


if __name__ == '__main__':
    if sys.argv[1] == 'search':
        s, x, info = search(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
        Path('roll_best.json').write_text(json.dumps({'score': s, 'info': info, 'params': dict(zip(KEYS, map(float, x)))}, indent=1))
        print(json.dumps({'score': s, 'info': info}, indent=1))
    else:
        p = json.loads(Path(sys.argv[2]).read_text())['params']
        s, (info, rec) = rollout(np.array([p[k] for k in KEYS]), record=True)
        np.savez(sys.argv[3], qpos=rec, dt=DT * 10)
        print(json.dumps(info, indent=1))
