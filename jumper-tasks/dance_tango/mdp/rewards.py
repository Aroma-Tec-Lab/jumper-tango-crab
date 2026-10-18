"""Charging a limb for touching the floor while the choreography holds it in the air.

Why this task needs it, measured on the first run (`model_3999`, 4000 iterations, RTX 3090):
the rear legs did lift for the arabesque, 2.4 s at up to 96 mm, but the right claw -- whose
reference is 50-80 mm off the floor throughout -- came down and **tapped the floor for balance**
in 21% of the arabesque window, so the robot stood on two feet only in 0.2-0.4 s stretches
between taps. Tracking alone priced that at a ~6 cm position error on one of seven tracked
bodies, which the policy found worth paying for a third point of support.

The term is gated by the reference, not by the clock: a limb is charged only on frames where
its tracked tip sits above `lift_height` in the clip. The claws rest on the floor at the start
and at the end of the dance, and the rear legs everywhere but the arabesque, and none of that
is charged. Counting follows `self_collision_cost`: physics substeps (0..decimation per step)
in which the limb pushes on the ground harder than `force_threshold`.
"""

from __future__ import annotations

import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ...common.dance.env import TRACKED_BODIES

#: limb -> (subtree root whose geoms are watched, tracked tip body whose reference gates it)
LIFTED_LIMBS: dict[str, tuple[str, str]] = {
    "LF": ("LF_shoulder_link", "LF_palm_pad_b_link"),
    "RF": ("RF_shoulder_link", "RF_palm_pad_b_link"),
    "LR": ("LR_hip_link", "LR_foot_tip_link"),
    "RR": ("RR_hip_link", "RR_foot_tip_link"),
}


def sensor_name(limb: str) -> str:
    return f"{limb.lower()}_ground_contact"


def lifted_limb_sensors(history_length: int) -> tuple[ContactSensorCfg, ...]:
    """One limb-vs-floor contact sensor per entry of `LIFTED_LIMBS`."""
    return tuple(
        ContactSensorCfg(
            name=sensor_name(limb),
            primary=ContactMatch(mode="subtree", pattern=root, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=history_length,
        )
        for limb, (root, _) in LIFTED_LIMBS.items()
    )


def lifted_limb_ground_contact(
    env,
    command_name: str,
    lift_height: float,
    force_threshold: float,
) -> torch.Tensor:
    """Substeps in which a limb the reference has lifted pushes on the floor, summed over limbs."""
    command = env.command_manager.get_term(command_name)
    ref_pos = command.body_pos_w                                   # [B, bodies, 3]
    ground = env.scene.env_origins[:, 2]
    cost = torch.zeros(env.num_envs, device=env.device)
    for limb, (_, tip) in LIFTED_LIMBS.items():
        lifted = (ref_pos[:, TRACKED_BODIES.index(tip), 2] - ground) > lift_height
        data = env.scene[sensor_name(limb)].data
        force = torch.norm(data.force_history, dim=-1)              # [B, N, H]
        hits = (force > force_threshold).any(dim=1).sum(dim=-1).float()
        cost += hits * lifted.float()
    return cost
