# Tango Crab 🦀🌹

A tango solo for the **Tiao Tiao Jumper** crab robot
([KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper), 22 DoF), made for the
*Crab Robot Open-source Challenge 2026*. Everything in the video is **physics simulation in
MuJoCo** with the open-source Jumper model and the real servo's torque/speed curve, not animation.

**Video:** [on X](https://x.com/V_Racho/status/2111817516326535656) · music: "Cry for Me Again" by
Sight of Wonders (Epidemic Sound, not included here)

## From tango to six legs

Classic tango figures, carried over to a solo crab with six legs and two claws, on the track's beat
grid (65.9 BPM: a *slow* is one beat, 0.91 s, a *quick* half of one):

| Time | Figure | On the crab |
|---|---|---|
| 0-2 s | **abrazo + postura** | the claws go into a dance hold (left extended as the hand hold, right bent as the embrace), body raised |
| 2-3 s | **head snaps** (ballroom staccato) | sharp twists of the body on the beat |
| 3.9-7.6 s | **caminata** | tripod walk forward, slow, slow, quick, quick, slow, 18 cm |
| 7.6 s | **parada** | the stop, a snap of the head |
| 8.5-13.9 s | **ochos** | pivots on the beat: +30, -60, +60, -30 degrees, drifting back |
| 13.9-15.8 s | **boleo** | the left rear leg lifts and whips out and back while the body counter-twists |
| 15.8-20.3 s | **corrida** | abrupt sideways crab runs: three quick steps left, dead stop, three back, dead stop |
| 20.3-22.1 s | **golpecitos** | the right claw taps the floor on the quicks |
| 23.0 s (climax) | **corte y quebrada** | right claw down, left rear leg stretched long behind, left claw high, body dipped and turned to the audience, held to the end |

Partner figures (sacada, sentada, volcada) were left out: they need a second dancer.

## How it works

1. **Choreography as keyframes** (`choreo/choreography_tango.py`), turned into a kinematically
   consistent clip by `choreo/generate_tango.py`: feet planted by IK, the centre of mass pulled over
   the support feet, and tripod stepping figures that can walk and turn (`Step`) and lift one rear
   leg at a time. The claws are front feet while stepping and plant wider than at rest, so a
   turning step cannot carry one claw onto the other.
2. **Imitation policy**: `jumper.dance_tango` (in `jumper-tasks/`) is the framework's dance-tracking
   task with this clip; PPO learns to perform it in simulation with pushes, noise and the servo
   limits. The task adds **arm-vs-torso contact**: the framework's masks only collide a leg with the
   ground and the other legs, and without it a policy tracks a claw straight through the shell.
3. **Checks**: `probes/clean_poses.py` (claw clearance on the clip), `probes/self_pen.py` (claws
   inside the body on a replay: 0 frames for the checkpoint here).

**Measured on a replay of `policies/dance_tango_model_12949.pt`:** the base stays within about
1.5 cm of the reference through the walk, the ochos and the 14 cm sideways runs; the boleo lifts
the leg 82 mm; the quebrada holds a 62 mm leg extension steadily to the end of the music.

**The look** is paint only (`skins/tools/build_paint.py tango`): black shell, painted full red lips
under the display, crimson lashes above it, a red rose with two leaves on the crown, red claw tips.
The surface is the stock shell subdivided (max deviation 5e-6 mm), so the physics is the stock robot.

## Run it

```bash
git clone https://github.com/KingKongRobotics/jumper.git
git clone <this repo> jumper-tango-crab && cd jumper-tango-crab
ln -s ../jumper jumper          # the scripts expect jumper/ (and jumper-design/ for the skin) beside them
python jumper-tasks/install.py jumper
cd jumper && pip install -e ".[dev]" && cd ..

# replay the policy
cd probes && python eval_task.py ../policies/dance_tango_model_12949.pt tango_replay.npz 1450 jumper.dance_tango

# regenerate the clip / train further (GPU: --backend warp; Mac: --backend native --device cpu)
cd ../choreo && python generate_tango.py choreography_tango ../jumper/tasks/jumper/dance_tango/media/tango.npz
cd ../jumper && python scripts/train.py --task jumper.dance_tango --num_envs 4096 \
    --resume --checkpoint ../policies/dance_tango_model_12949.pt --headless

# render (needs the skin built with skins/tools + jumper-design)
cd ../probes && python tango_render.py tango_replay.npz <tango skin xml> 28.7 tango.mp4
```

## Honest notes

Simulation only: nothing here has run on a real Jumper yet. The servo model, PD gains and limits
are the framework's, but sim-to-real was not tested.

## License

Apache License 2.0, like the jumper framework it builds on (see `LICENSE` and `NOTICE`).

#Jumper #CrabRobot #OpenSourceRobot
