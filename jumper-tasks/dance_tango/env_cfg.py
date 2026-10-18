"""Environment config for jumper.dance_tango: `media/tango.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every number.
The numbers follow the framework's keyframed dance tasks (the clip is built from keyframes by
`choreo/generate_tango.py`), plus arm-vs-torso contact so a claw cannot pass through the shell.

## This clip

`media/tango.npz`, 47.3 s at 100 Hz: 28.7 s of tango to "Cry for Me Again" (abrazo, head
snaps, caminata slow-slow-quick-quick-slow, ochos, a boleo with the left rear leg, a 360-degree
giro, golpecitos with the right claw, corte y quebrada held to the end), then back to standing
and a standing pad. Joint speeds peak at 14.5 rad/s (the boleo's whip); claws stay >= 10 mm
from the torso and clear of each other (probes/clean_poses.py).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab.managers.reward_manager import RewardTermCfg

from ..common.constants import _LEG_BIT
from ..common.dance.env import COMMAND, dance_env_cfg
from .mdp import lifted_limb_ground_contact, lifted_limb_sensors

#: This task's material, written by `tools/import_wbc_dances.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. The clip is 47.3 s, so an episode is a window into it
#: and the adaptive sampler chooses where it starts -- which is also what sends it
#: back to the arabesque when that is where it fails. 10 s, as jumper.dance.
EPISODE_S = 10.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    cfg = dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # jumper.dance's: fractions of this robot (10 mm is a tenth of its standing
        # height), not of the clip, so they carry over unchanged.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, as jumper.dance has them; not measured.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # Positive budget 6.5 a step, as jumper.dance: mjlab's six tracking terms
        # unchanged (5.0) plus the two joint-space terms below (1.5).
        #
        # Joint position: jumper.dance's rule is a std at 0.44 of the clip's RMS
        # amplitude. Here: 0.44 x 0.362 -> 0.16.
        joint_pos_std=0.16,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS. Here: 0.983 -> 0.66.
        # Half the position weight, so when they disagree the pose wins.
        joint_vel_std=0.66,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: kept at the dances' -0.02. At this clip's ~0.3 rad/s it costs a few
        # hundredths a step -- the term only bites if the policy jitters.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits: the reference stays inside every range by construction (the IK
        # clips to the limits); -10 as the dances, the soft limit keeps it off its stops.
        joint_limit_weight=-10.0,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip moves the body through 94.5-128.5 mm, a 34 mm band, so
        # 40 mm still exceeds all of it -- jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over". The choreography tilts at
        # most 6.9 degrees (the reverence).
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )

    # ── Lifted limbs stay off the floor ──────────────────────────────────
    # See `mdp/rewards.py` for the measurement that made this necessary: the first
    # run balanced the arabesque by tapping the floor with its right claw.
    # Four sensors, one per limb that the clip lifts at some point; maxmatch is a
    # per-world cap shared by every contact sensor, so it grows with them.
    cfg.scene.sensors = (cfg.scene.sensors or ()) + lifted_limb_sensors(cfg.decimation)
    cfg.sim.contact_sensor_maxmatch = 256
    # -0.75 per substep: one limb down for a whole policy step costs -3.0, about half
    # the positive tracking budget (6.5), and more than the ~6 cm position error that
    # made tapping pay. 40 mm: above any tip resting on the floor (pads sit at <10 mm)
    # and below every lifted pose in the clip (the arms never come lower than 50 mm
    # while lifted, the rear legs 68 mm once clear).
    cfg.rewards["lifted_limb_ground_contact"] = RewardTermCfg(
        func=lifted_limb_ground_contact,
        weight=-0.75,
        params={"command_name": COMMAND, "lift_height": 0.04, "force_threshold": 1.0},
    )

    # ── The claws collide with the crab's own torso ──────────────────────────────────
    # The family's masks collide a leg only with the ground and the other legs, and a
    # policy then tracks a claw straight through the shell.
    # The shell, display, ToF and camera hulls accept both arm bits; base_link stays out.
    robot = cfg.scene.entities["robot"]
    robot.collisions = tuple(_arm_torso_collision(c) for c in robot.collisions)
    return cfg


#: The torso hulls an arm can reach from above or in front.
_TORSO_RE = "^(upper_shell|display_module|tof_sensor|camera)_link_meshcol$"


def _arm_torso_collision(base: CollisionCfg) -> CollisionCfg:
    arm_bits = _LEG_BIT["LF"] | _LEG_BIT["RF"]

    def with_torso(d: dict[str, int], value: int) -> dict[str, int]:
        # Patterns match first-hit in insertion order: the torso entry goes before ".*".
        out = {k: v for k, v in d.items() if k != ".*"}
        out[_TORSO_RE] = value
        out[".*"] = d[".*"]
        return out

    return replace(
        base,
        contype=with_torso(base.contype, base.contype[".*"]),
        conaffinity=with_torso(base.conaffinity, arm_bits),
    )
