"""Tango crab, to "Cry for Me Again" (Sight of Wonders), 28.7 s. Beat k at B0 + k * BEAT (measured
from the track's onsets: 65.9 BPM; a slow is a beat, a quick half of one).

Classic tango figures, solo, carried over to six legs and two claws:
  abrazo + postura   claws in a dance hold (left extended as the hand hold, right bent as the
                     embrace), body raised
  head snaps         staccato twists of the body on the accents (ballroom tango)
  caminata           tripod walk forward: slow, slow, quick, quick, slow
  parada             the stop, a snap of the head
  ochos              pivots left and right on the beat: +30, -60, +60, -30 degrees
  boleo              left rear leg lifts and whips out and back while the body counter-twists
  corrida            abrupt sideways runs: three quick steps left, dead stop, three quick
                     steps right, dead stop (the user's idea: more tango than a slow turn)
  golpecitos         the right claw taps the floor on the quicks
  corte y quebrada   on the climax (23.0 s): right claw down, left rear leg stretched back,
                     left claw high, body dipped and turned to the audience -- held to the end
Then (training only, after the music) back to standing and a long standing pad: the loader
judges the support feet on the median frame, and stepping or lifted-leg frames must stay the
minority.

The claws are front feet in the stepping figures (caminata, ochos, corrida), as in the
ballerina's pirouette, and in the hold everywhere else. All arm poses were checked with probes/clean_poses.py: >= 33 mm from the torso, 63 mm apart.
"""
import numpy as np
from posekit import HOME
from generate_tango import Step

B0, BEAT = 0.28, 0.910
Q = BEAT / 2
b = lambda k: B0 + k * BEAT

BAS = [HOME[f'LF_J{k}_joint'] for k in range(5)]
FRAME_L = [-0.9, -2.6, -0.3, -1.2, 0.0]       # the hand hold: out, forward and up, claw shut
FRAME_R = [-0.45, -1.9, -0.9, -1.3, 0.0]      # the embrace: in front of the chest
HIGH = [-1.813, -3.039, -1.103, -0.35, 0.0]   # overhead (the battle's clean clap pose, shut)
REAR_HOME = [HOME['LR_J0_joint'], HOME['LR_J1_joint'], HOME['LR_J2_joint']]
REAR_LIFT = [1.0, 1.5, -1.6]                  # boleo: leg up and back
REAR_WHIP = [2.2, 1.4, -1.4]                  # ... whipped out
REAR_QUEB = [1.3, 1.7, -1.1]                  # quebrada: stretched long behind
Z = [0, 0, 0, 0, 0, 0]
ALL = [1, 1, 1, 1, 1, 1]
NO_LR = [0, 0, 6, 6, 0, 1]                    # left rear lifted: middle feet + right rear
QUEB_SUP = [0, 2, 6, 6, 0, 1]                 # ... and the right claw planted in front
UP = [0, 0, 0.010, 0, 0, 0]                   # postura: tall
# The claws plant wider than HOME (+-0.071 m -> +-0.12 m): 42 degrees apart round the centre at
# HOME, so a 36-45 degree turning step carried one claw onto the other (39 mm overlap, measured)
FOOT_OFFSET = {'LF': (-0.015, 0.048), 'RF': (-0.015, -0.050)}


def base(dz=0.010, roll=0.0, pitch=0.0, yaw=0.0):
    return [0, 0, dz, roll, pitch, yaw]


STATE = dict(base=Z, larm=BAS, rarm=BAS, rear_l=REAR_HOME, rear_r=REAR_HOME, com=0.0,
             lift_l=0.0, lift_r=0.0, lf_plant=0.0, rf_plant=0.0, support=ALL)
KEYS = [dict(t=0.0, **STATE)]


def move(t0, dur, **ch):
    """Explicit hold: every changed channel keyed at its old value at t0 and its new one at t0+dur."""
    KEYS.append(dict(t=t0, **{k: STATE[k] for k in ch}))
    STATE.update(ch)
    KEYS.append(dict(t=t0 + dur, **ch))
    return t0 + dur


# abrazo: into the hold, rise
move(b(0), 2 * BEAT, larm=FRAME_L, rarm=FRAME_R, base=UP)
# head snaps on beats 2 and 3
move(b(2), 0.12, base=base(yaw=0.28))
move(b(3), 0.12, base=UP)

FIGS = []
# the claws come down as front feet for the stepping figures (tripods LF/RM/LR, RF/LM/RR)
move(b(3) + Q, Q, lf_plant=1.0, rf_plant=1.0)
# caminata, from beat 4: slow, slow, quick, quick, slow
t = b(4)
for half, dx in ((Q, 0.04), (Q, 0.04), (Q / 2, 0.03), (Q / 2, 0.03), (Q, 0.04)):
    FIGS.append(Step(t, 1, half, 0.03, dxy=(dx, 0.0))); t = FIGS[-1].t1
# parada on beat 8: a snap of the head the other way and back
move(b(8), 0.12, base=base(yaw=-0.28))
move(b(8) + Q, 0.12, base=UP)
# ochos from beat 9, drifting back a little (ochos atras)
t = b(9)
for dpsi, cyc in ((0.5, 1), (-1.0, 2), (1.0, 2), (-0.5, 1)):
    FIGS.append(Step(t, cyc, Q, 0.03, dpsi=dpsi, dxy=(-0.01 * cyc, 0.0))); t = FIGS[-1].t1
# boleo, beat 15: weight off the left rear leg, lift, whip out and back, down
t = b(15)
move(t, Q, com=1.0, support=NO_LR, base=base(yaw=0.15), lf_plant=0.0, rf_plant=0.0)
move(t + Q * 0.6, Q * 0.4, lift_l=1.0, rear_l=REAR_LIFT)
move(t + Q, Q / 2, rear_l=REAR_WHIP, base=base(yaw=-0.18))
move(t + 1.5 * Q, Q / 2, rear_l=REAR_LIFT, base=UP)
move(t + 2 * Q, Q, lift_l=0.0, rear_l=REAR_HOME)
move(t + 3 * Q, Q, com=0.0, support=ALL, lf_plant=1.0, rf_plant=1.0)
# corrida: from beat 17, abrupt sideways runs -- three quick steps to the left, a dead stop with a
# snap of the head, three quick steps back to the right, a dead stop, a snap -- the crab's own gait
t = b(17)
FIGS.append(Step(t, 3, Q / 2, 0.035, dxy=(0.0, 0.16)))          # left, 1.37 s
move(FIGS[-1].t1, 0.12, base=base(yaw=0.28))                     # stop: the head snaps left
move(FIGS[-1].t1 + Q - 0.12, 0.12, base=UP)
t = FIGS[-1].t1 + Q
FIGS.append(Step(t, 3, Q / 2, 0.035, dxy=(0.0, -0.16)))         # right, back to the centre
move(FIGS[-1].t1, 0.12, base=base(yaw=-0.28))                    # stop: the head snaps right
move(b(22) - 0.25, 0.12, base=UP)
# golpecitos, beat 22: the right claw taps the floor on the quicks
t = b(22)
for k in range(4):
    move(t + k * Q, Q / 2, rf_plant=1.0)
    move(t + k * Q + Q / 2, Q / 2, rf_plant=0.0)
# corte y quebrada: weight shifts on beat 24, the snap lands on beat 25 (the climax, 23.0 s)
t = b(24)
move(t, BEAT - 0.25, com=1.0, support=QUEB_SUP, rf_plant=1.0, lf_plant=0.0)
move(b(25) - 0.23, 0.23, lift_l=1.0, rear_l=REAR_QUEB, larm=HIGH,
     base=base(dz=-0.012, roll=-0.08, pitch=0.08, yaw=0.30))
# held to the end of the music (28.7 s), breathing
move(25.4, 2.2, base=base(dz=-0.010, roll=-0.08, pitch=0.07, yaw=0.30))
# training only: back to standing, then the standing pad
move(28.9, 1.6, lift_l=0.0, rear_l=REAR_HOME, larm=BAS, rarm=BAS, rf_plant=0.0, base=Z)
move(30.5, 0.8, com=0.0, support=ALL)
DURATION = 31.3 + 16.0
