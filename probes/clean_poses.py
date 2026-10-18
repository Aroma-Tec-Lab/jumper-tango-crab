"""Arm poses that keep the claws out of the crab's own body.

The training envs collide a leg only with the ground and the OTHER legs (constants._collision_cfg):
arm vs shell is not simulated, so a reference pose that puts a claw into the shell is tracked
straight through it (measured in the battle: the overhead clap's opening claw 30 mm inside the
shell). This probe measures the clearance of a pose -- signed hull distance, mj_geomDistance,
between the arm's links from the forearm out and the torso (shell, display, ToF, camera) -- at
the HOME stance, and searches overhead-clap poses that stay clear both open and shut.

usage: clean_poses.py check | search [seed]
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

import roll_search as rs
from roll_search import HOME, STAND_Z

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'choreo'))
m = mujoco.MjModel.from_xml_path(str(rs.ROBOT / 'jumper/assets/jumper/jumper.xml'))
d = mujoco.MjData(m)
J = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
BID = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
TORSO = [g for g in range(m.ngeom) if m.geom_type[g] == 7 and m.geom_contype[g] and m.geom_bodyid[g] in
         {BID(n) for n in ('upper_shell_link', 'display_module_link', 'tof_sensor_link', 'camera_link')}]
BASE = [g for g in range(m.ngeom) if m.geom_type[g] == 7 and m.geom_contype[g] and m.geom_bodyid[g] == BID('base_link')]
SIGN = {0: -1, 1: 1, 2: -1, 3: -1, 4: -1}


def under(g, root):
    x = m.geom_bodyid[g]
    while x > 0 and x != BID(root):
        x = m.body_parentid[x]
    return x == BID(root)


ARM = {s: [g for g in range(m.ngeom) if m.geom_type[g] == 7 and m.geom_contype[g] and under(g, s + 'F_forearm_link')]
       for s in 'LR'}
FT = np.zeros(6)


def set_pose(larm, rarm=None):
    d.qpos[:] = 0; d.qpos[2] = STAND_Z; d.qpos[3] = 1
    for n, v in HOME.items():
        d.qpos[J(n)] = v
    for k in range(5):
        d.qpos[J(f'LF_J{k}_joint')] = larm[k]
        if rarm is not None:
            d.qpos[J(f'RF_J{k}_joint')] = rarm[k]
    mujoco.mj_kinematics(m, d)


def clearance(side='L', with_base=False):
    best, pair = 1.0, None
    for g1 in ARM[side]:
        for g2 in TORSO + (BASE if with_base else []):
            x = mujoco.mj_geomDistance(m, d, g1, g2, 0.2, FT)
            gap = float(np.linalg.norm(FT[3:] - FT[:3]))
            if x == 0.0 or (x < 0 and abs(gap - abs(x)) > 0.003):
                # measured: some mesh pairs return exactly 0, or a negative depth, with witness
                # points cm apart; a real overlap of depth p has them p apart
                x = gap
            if x < best:
                best, pair = x, (m.geom(g1).name, m.geom(g2).name)
    return best, pair


def claw_gap():
    """Signed distance between the two arms (the training sim does collide those)."""
    best = 1.0
    for g1 in ARM['L']:
        for g2 in ARM['R']:
            x = mujoco.mj_geomDistance(m, d, g1, g2, 0.2, FT)
            gap = float(np.linalg.norm(FT[3:] - FT[:3]))
            if x == 0.0 or (x < 0 and abs(gap - abs(x)) > 0.003):
                x = gap
            best = min(best, x)
    return best


def tip(side='L'):
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f'{side}F')].copy()


def mirror(arm):
    return [SIGN[k] * v for k, v in enumerate(arm)]


if __name__ == '__main__':
    import choreography_battle as cb
    if sys.argv[1] == 'check':
        for name in ('BAS', 'HAUT_OPEN', 'HAUT_CLAP', 'EYES_L', 'JAB_L', 'JAB_SHUT'):
            set_pose(getattr(cb, name))
            c, p = clearance('L'); cb_, pb = clearance('L', True)
            print(f'{name:10s} torso {c * 1000:6.1f} mm {p}   incl. base hull {cb_ * 1000:6.1f} mm')
    else:
        # overhead clap: claws high above the shell, J4 open and shut both clear by >= 10 mm
        rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        lo = np.array([-2.0, -3.8, -2.1, -2.0]); hi = np.array([0.6, -2.0, 1.5, 1.5])
        best = (-np.inf, None)
        for it in range(12000):
            q = rng.uniform(lo, hi) if it < 8000 or best[1] is None else \
                np.clip(best[1] + rng.normal(0, 0.08, 4), lo, hi)
            worst, gaps = 1.0, []
            for j4 in (-1.0, 0.0):
                set_pose(list(q) + [j4], mirror(list(q) + [j4]))
                c, _ = clearance('L'); worst = min(worst, c); gaps.append(claw_gap())
            # clear of the body open and shut; shut, the claws nearly touch (2-12 mm): the clap
            if worst < 0.010 or min(gaps) < 0.002 or gaps[1] > 0.012:
                continue
            set_pose(list(q) + [0.0], mirror(list(q) + [0.0]))
            tl = tip('L')
            score = tl[2] - 0.5 * max(0.0, tl[0] - 0.05)
            if score > best[0]:
                best = (score, q.copy(), worst, tl.copy())
        s, q, w, tl = best
        set_pose(list(q) + [0.0], mirror(list(q) + [0.0])); g0 = claw_gap()
        print('clap pose', np.round(q, 3).tolist(), f'clearance {w * 1000:.1f} mm  shut gap {g0 * 1000:.1f} mm  tip {np.round(tl, 3)}  '
              f'(tip height above base {tl[2] - STAND_Z:.3f} m)')


LEG_OF = {0: 'LF', 1: 'LM', 2: 'LR', 3: 'RR', 4: 'RM', 5: 'RF'}


def clip_qpos(path):
    """A dance clip (body_* / L{i}_j{k} channels, R = Rz Ry Rx) as qpos rows of this model."""
    z = np.load(path); n = len(z['time'])
    Q = np.zeros((n, m.nq))
    Q[:, 0], Q[:, 1], Q[:, 2] = z['body_x'], z['body_y'], z['body_z']
    for i in range(n):
        q = np.zeros(4); mat = np.zeros(9)
        r, p, y = z['body_roll'][i], z['body_pitch'][i], z['body_yaw'][i]
        cr, sr, cp_, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp_, 0, sp], [0, 1, 0], [-sp, 0, cp_]]) \
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        mujoco.mju_mat2Quat(q, R.flatten()); Q[i, 3:7] = q
    for L, leg in LEG_OF.items():
        for k in range(5):
            key = f'L{L}_j{k}'
            if key in z:
                Q[:, J(f'{leg}_J{k}_joint')] = z[key]
    return Q, float(z['dt'])


def clip_report(path, every=2):
    Q, dt = clip_qpos(path)
    bad = {}
    for i in range(0, len(Q), every):
        d.qpos[:] = Q[i]; mujoco.mj_kinematics(m, d)
        for s in 'LR':
            c, p = clearance(s)
            k = round(i * dt * 2) / 2
            if c < 0.005 and (k not in bad or c < bad[k][0]):
                bad[k] = (c, s, p)
        g = claw_gap()
        if g < 0.0:
            k = round(i * dt * 2) / 2
            bad.setdefault(k, (g, 'L-R', 'arms'))
    for k in sorted(bad):
        c, s, p = bad[k]
        print(f'  {k:5.1f} s {s} {c * 1000:6.1f} mm {p}')
    return bad
