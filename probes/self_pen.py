"""Do a crab's claws pass into its own body? Signed distance (mj_geomDistance) between every arm
collision geom and every body/shell geom, per frame of a qpos sequence.
usage: self_pen.py <states.npz> [key] [t0 t1]"""
import sys
import mujoco
import numpy as np
import roll_search as rs
z = np.load(sys.argv[1]); key = sys.argv[2] if len(sys.argv) > 2 else 'qpos'
Q = z[key]; dt = float(z['dt'])
a, b = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0, len(Q) * dt)
m = mujoco.MjModel.from_xml_path(str(rs.ROBOT / 'jumper/assets/jumper/jumper.xml')); d = mujoco.MjData(m)
B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
torso = {B('base_link'), B('upper_shell_link')} | {B(n) for n in ('display_module_link', 'tof_sensor_link', 'camera_link') if B(n) >= 0}
def under(g, root):
    x = m.geom_bodyid[g]
    while x > 0 and x != B(root):
        x = m.body_parentid[x]
    return x == B(root)
col = [g for g in range(m.ngeom) if m.geom_type[g] == 7 and m.geom_group[g] in (1, 3) or (m.geom_type[g] == 7 and m.geom_contype[g])]
body = [g for g in col if m.geom_bodyid[g] in torso]
arms = {s: [g for g in col if under(g, s + '_forearm_link')] for s in ('LF', 'RF')}
ft = np.zeros(6)
worst = []
for i in range(int(a / dt), min(int(b / dt), len(Q))):
    d.qpos[:] = Q[i]; mujoco.mj_kinematics(m, d)
    for s, gs in arms.items():
        mn, pair = 1.0, None
        for g1 in gs:
            for g2 in body:
                # distmax 0.05 measured to return spurious negatives for mesh pairs farther apart
                # than it (-48.9 mm for a pair 48.9 mm apart); 0.2 does not. An exact 0.0 comes
                # with witness points cm apart: use their distance then.
                dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.2, ft)
                gap = float(np.linalg.norm(ft[3:] - ft[:3]))
                if dist == 0.0:
                    dist = gap
                elif dist < 0 and abs(gap - abs(dist)) > 0.003:
                    # a real overlap of depth p has witness points p apart; seen: -48.9 mm with the
                    # witness points 110 mm apart, on a pair that is +48 mm a frame earlier
                    continue
                if dist < mn:
                    mn, pair = dist, (m.geom(g1).name, m.geom(g2).name)
        if mn < 0:
            worst.append((i * dt, s, mn, pair))
print(f'{len(worst)} arm-frames inside the body (deeper than 0 mm)')
bins = {}                                            # deepest per half second, either arm
for t, s, mn, p in worst:
    k = int(t * 2) / 2
    if k not in bins or mn < bins[k][0]:
        bins[k] = (mn, s, p)
for k in sorted(bins):
    mn, s, p = bins[k]
    if mn < -0.003:
        print(f'  {k:5.1f} s {s} deepest {mn * 1000:6.1f} mm {p}')
