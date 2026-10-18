"""Replay a trained jumper.dance_ballerina policy and measure, per control step, how high each limb
is above the floor (lowest collision-mesh vertex, by forward kinematics on the replayed state).

usage: eval_clearance.py <checkpoint.pt> [out.npz]
"""
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT / 'jumper'))
sys.path.insert(0, str(ROBOT / 'jumper/scripts'))

from mjrl.backend.resolve import resolve  # noqa: E402
from mjrl.backend.select import use_backend  # noqa: E402

res = resolve(backend='native', device='cpu', num_envs=1)
use_backend(res)
import tasks  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402

TASK = sys.argv[4] if len(sys.argv) > 4 else 'jumper.dance_ballerina'
cfg = tasks.load_env_cfg(TASK, asset=None, play=True)
cfg.scene.num_envs = 1
agent = tasks.load_agent_cfg(TASK)
env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = tasks.load_runner_cls(TASK)(wrapped, asdict(agent), device='cpu')
runner.load(sys.argv[1], load_cfg={'actor': True}, strict=True, map_location='cpu')
policy = runner.get_inference_policy(device='cpu')

robot = env.scene['robot']
names = list(robot.joint_names)
m = mujoco.MjModel.from_xml_path(str(ROBOT / 'jumper/assets/jumper/jumper.xml'))
d = mujoco.MjData(m)
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
LIMBS = {'LF': 'LF_shoulder_link', 'RF': 'RF_shoulder_link', 'LM': 'LM_hip_link',
         'RM': 'RM_hip_link', 'LR': 'LR_hip_link', 'RR': 'RR_hip_link', 'body': 'base_link'}
geoms = {}
for k, root in LIMBS.items():
    geoms[k] = []
    for gi in range(m.ngeom):
        if m.geom_type[gi] != 7 or not m.geom_contype[gi]:
            continue
        b = m.geom_bodyid[gi]
        if k == 'body':
            if b in (B('base_link'), B('upper_shell_link')):
                geoms[k].append(gi)
            continue
        while b > 0 and b != B(root):
            b = m.body_parentid[b]
        if b == B(root):
            geoms[k].append(gi)


def lowest(gis):
    z = np.inf
    for gi in gis:
        mid = m.geom_dataid[gi]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        z = min(z, float((v @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi])[:, 2].min()))
    return z


obs = wrapped.get_observations()
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1350
rec = {k: np.zeros(steps) for k in LIMBS}
qpos_log = np.zeros((steps, m.nq))
done_at = []
for s in range(steps):
    with torch.inference_mode():
        obs, _, dones, _ = wrapped.step(policy(obs))
    if bool(dones.any()):
        done_at.append(s)
    d.qpos[:3] = robot.data.root_link_pos_w[0].numpy()
    d.qpos[3:7] = robot.data.root_link_quat_w[0].numpy()
    d.qpos[qadr] = robot.data.joint_pos[0].numpy()
    mujoco.mj_forward(m, d)
    qpos_log[s] = d.qpos
    for k in LIMBS:
        rec[k][s] = lowest(geoms[k])
t = np.arange(steps) * env.step_dt
np.savez(sys.argv[2] if len(sys.argv) > 2 else 'clearance.npz', t=t, qpos=qpos_log, dt=env.step_dt, **rec)
print('episode ends at steps:', done_at[:10])
for a, b in [(15.5, 16.6), (16.6, 18.0), (18.0, 20.4), (20.4, 21.8)]:
    sel = (t >= a) & (t < b)
    print(f't {a:4.1f}-{b:4.1f} s  min height [mm]: ' + '  '.join(f'{k} {rec[k][sel].min()*1000:6.1f}' for k in LIMBS))
