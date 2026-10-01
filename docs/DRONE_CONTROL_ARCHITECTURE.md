# Dune AI drone control architecture review

**Repository reviewed:** `zwethiha234-svg/dune-ai` at commit `5da87dc` (2026-09-27).
**Question:** why fixing the headless → head-forward (nose turns) behaviour broke
point A → point B navigation, where the structural bottleneck is, and what to fix
first so the fix-one-break-another cycle stops.

**Short answer.** The trained policy is a *headless* controller: it sees the
target in world axes and only learns to tilt the thrust vector given its own
yaw. Making the vehicle *headful* was done by streaming a yaw rate from a
virtual airframe that lives inside the adapter and whose heading is never
re-synchronised with the real vehicle. The moment the real vehicle starts
following that yaw rate, the adapter's picture of "which way is my nose" and
the vehicle's real nose diverge, and every subsequent tilt command is computed
for the wrong heading. Navigation does not fail because navigation is wrong.
It fails because the adapter is flying a ghost.

---

## 1. The pipeline as it exists today

```
MISSION (ENU waypoint A -> B)
  tools/fly_policy_mission.py / tools/fly_policy_px4.py / tools/airsim_hop_cycle_qa.py
        |
        |  VehicleState (ENU pos/vel, Attitude yaw/pitch/roll)   target Vector3 (ENU)
        v
ADAPTER   AiDroneFly/src/aidronefly/policies/ppo_policy.py  (PPOPolicyAdapter.step)
  1. ENU -> drone3d [E, Up, N] remap                          (ppo_policy.py:149-160)
  2. build 34-slot observation                                (ppo_policy.py:66-92)
       rel target (WORLD) | vel (WORLD) | sin/cos yaw, pitch, roll | body rates | alt | dist | config | rays
  3. policy forward  ->  a = [collective, pitch, roll, yaw]   (ppo_policy.py:520-526)
  4. action -> velocity, ONE of three translations:
       "velocity":  2-D yaw rotation of (fwd, right) using the REAL measured yaw   (ppo_policy.py:618-631)
       "physics":   step a persistent VIRTUAL Drone3D with a; command its velocity (ppo_policy.py:528-545)
                    + yaw_rate_dps = the VIRTUAL drone's yaw rate                  (ppo_policy.py:600-616)
       "physics_vzlead": physics + vertical lead law
  5. tracking compensation (K * (last cmd - achieved))         (ppo_policy.py:765-780)
  6. SafetySupervisor may override                            (ppo_policy.py:721-735)
        |
        |  set_velocity(velocity ENU, yaw_rate_dps)
        v
BACKEND   AiDroneFly/src/aidronefly/backends/*
  mock.py     integrates yaw rate into its heading                  (mock.py:121-126)
  airsim.py   ENU->NED, moveByVelocityAsync; yaw rate IGNORED;
              optional --yaw-follow = DrivetrainType.ForwardOnly    (airsim.py:162-190)
  px4.py      ENU->NED, VelocityNedYaw(…, yaw_deg);
              streamer ON : yaw rate integrated into yaw_deg, seeded from measured heading (px4.py:1568-1592, 2046-2048)
              streamer OFF: yaw_deg forced to 0 (nose North), yaw rate REFUSED        (px4.py:1433, 1469-1479)
        |
        v
AUTOPILOT / SIM   (AirSim or PX4 velocity + yaw controllers, then motors)
```

The policy was trained in `ppo_gpu.py` on the same `[collective, pitch, roll,
yaw]` action and the same observation. Training physics is `drone3d.py`
(Euler, `Ry(yaw)@Rx(pitch)@Rz(roll)`, `drone3d.py:590-607`) or, with
`--full-attitude`, a quaternion 6-DOF model in `ppo_gpu.py:943-967`.

### Frame conventions (these are actually consistent)

| Layer | Position frame | Yaw convention |
|---|---|---|
| Product types | ENU (E, N, Up) | compass: 0 = North, clockwise positive |
| Training sim (drone3d / ppo_gpu) | [E, Up, N], forward = [sin yaw, 0, cos yaw] | same (clockwise positive by construction) |
| AirSim / PX4 | NED | same |

So the raw axis conventions are **not** the bug. The adapter copies
`drone.yaw = state.attitude.yaw` directly and that is correct. The bug is
about *which yaw* is used, not how yaw is defined.

---

## 2. What "headless" and "headful" actually mean in this codebase

The code never uses those words for yaw. The equivalent concepts are:

| Your term | In dune-ai | Where |
|---|---|---|
| headless | policy's yaw channel `a[3]` is dropped; vehicle holds whatever heading it has; velocity is world ENU | `translation="velocity"`, or `physics` on AirSim (yaw ignored), or PX4 with streamer off (nose forced North) |
| headful | vehicle's nose is driven | `--yaw-follow` on AirSim (autopilot points nose along velocity), **or** `translation="physics"` + PX4 streamer on (adapter's virtual yaw rate is streamed) |
| head-forward reward | training bonus `heading_coef * cos(yaw - course)` | `ppo_gpu.py:1389-1396`, `--reward-heading`, used at 0.02–0.05 in some runs, 0 in others |

That is three different mechanisms, on three different layers, for one
behaviour. None of them share a definition of "the vehicle's heading".

---

## 3. Root cause of the A → B regression

### 3.1 The virtual airframe's heading is never re-synchronised (primary)

`_sync_vdrone` (`ppo_policy.py:370-383`) re-syncs **position and velocity**
from the vehicle every tick and deliberately keeps **attitude, body rates and
rotor thrusts internal**. The docstring calls that "hidden state the product
telemetry cannot provide".

That design was fine while the vehicle was headless: the virtual drone's
yaw could drift anywhere because (a) the observation told the policy the
virtual yaw, (b) the policy tilted relative to that yaw, (c) the virtual
physics rotated the tilt into a **world** velocity, and (d) only the world
velocity reached the vehicle. The real nose was irrelevant. A → B worked.

Switching to headful changed one thing: the vehicle now *also* receives the
virtual drone's yaw rate. Two problems follow immediately.

- **Two independent yaw integrators.** The virtual drone integrates
  `yaw += ang_vel[0] * dt` with the adapter's `dt` (0.1 s). The PX4 streamer
  integrates the same rate with its own `yaw_dt` and its own seed (the measured
  heading, or 0, or the previous link's heading, `px4.py:1575-1592`). The
  MockBackend integrates it a third way. Nothing ever compares them. Any
  difference in seed, period, clamping (the envelope clamps the rate to
  60 deg/s *after* the virtual drone already turned) or a dropped setpoint
  makes real heading ≠ virtual heading, and the error accumulates for the rest
  of the flight.
- **The policy's yaw channel was never trained to be flown.** With
  `--reward-heading 0` the policy has no reason to keep `a[3]` quiet; its
  virtual yaw rate is whatever the mixer happens to produce. Streaming that to
  a real yaw controller is the spin you saw. Clamping it (the "fix") stops the
  spin but does not restore agreement between the two headings.

Once the headings disagree, the velocity the adapter sends is still a world
vector, so why does A → B break? Because the next observation is built from
the **virtual** attitude, which now describes a vehicle that does not exist,
and the tracking compensation (`_compensate_velocity`) then fights the
difference between "what the ghost should have achieved" and "what the real
vehicle did". Under `velocity` translation the analogous failure is simpler:
`action_to_velocity` rotates by the *real* yaw while nothing in training
rewarded the policy for coping with a yaw it does not control.

### 3.2 Yaw behaviour depends on which backend and which flag (secondary)

| Backend | `translation=velocity` | `translation=physics` |
|---|---|---|
| Mock | heading never changes | heading follows virtual yaw rate |
| AirSim | heading holds (or follows velocity with `--yaw-follow`) | yaw rate silently ignored (same as left) |
| PX4, streamer off | nose forced to North | yaw rate refused (error) |
| PX4, streamer on | nose forced to North | heading follows integrated yaw rate |

Four physical behaviours for one policy. "It works in sim" and "it fails on
PX4" are not the same experiment, so every comparison between them is
confounded. `applies_yaw_rate` is declared `True` on Mock, `False` on AirSim
and undeclared on PX4 (`fly_policy_mission.py:218-226` reports "unknown").

### 3.3 The observation the policy sees on PX4 is not the one it trained on

Three parity breaks, independent of yaw, each large enough to break A → B
on its own. They explain why the problem "kind of works" in one place and not
another.

- **Box scale.** `policy_px4_attended_run.py:166-169` flies the frozen
  `field_quad_dr_r` candidate without `--city`, so `resolve_bounds`
  (`fly_policy_mission.py:315-319`) gives the open-space box ±15 m / 0–12 m.
  The contract's reference box for that run is ±60 m / 0–55 m. Observation
  slots 0–2 (relative target), 13 (altitude) and 14 (distance) are therefore
  scaled about 4× differently from training. The board already records a
  handoff caused by "an off-contract observation box"; nothing in
  `fly_policy_px4.py` compares its bounds to `observation_manifest.reference_box`.
- **Pitch sign.** In `drone3d.py:599-611` positive pitch tilts thrust toward
  +forward, i.e. nose *down*. PX4 and AirSim report positive pitch as nose
  *up*. `build_drone3d_from_state` (`ppo_policy.py:178`) copies the measured
  pitch without negating it, so slot 8 has the wrong sign under `velocity`
  translation and on the first tick of `physics` translation.
- **Time step.** Training decided every 4 × 1/60 s (66.7 ms); the adapter
  steps its virtual airframe once per 0.1 s and the PX4 streamer republishes
  at 20 Hz. The contract records all three numbers and resolves none.

### 3.4 Three physics models for one policy (tertiary)

- `drone3d.py` (CPU Euler, roll `Rz(+roll)`)
- `ppo_gpu.py` legacy branch (GPU Euler, inline matrix)
- `ppo_gpu.py --full-attitude` (quaternion, roll `Rz(-roll)`, no tilt clamp)

The adapter always flies the first one. If a checkpoint was trained on the
second or third, the "faithful physics translation" is faithful to a model the
policy never saw. The contract documents that `--full-attitude` is a different
regime, but nothing in the adapter refuses a mismatched checkpoint.

---

## 4. The bottleneck

> **There is no single owner of "the vehicle's heading".** The virtual
> airframe, the PX4 streamer, the Mock backend, AirSim's drivetrain and the
> training reward each hold their own, and the policy observes one of them
> while the motors obey another.

Every patch so far has been applied to one of those owners (clamp the rate,
rate-limit the turn, force yaw to North, add `--yaw-follow`). Each patch
makes the two headings agree for one test and disagree for the next.

---

## 5. Where to fix, in order

| # | Change | File(s) | What it removes |
|---|---|---|---|
| 1 | **Re-sync the virtual drone's yaw from the measured attitude every tick** (keep pitch/roll/rates internal if you must, but yaw is observable and must be the real one). Add a per-tick evidence field `yaw_virtual_minus_measured_deg` so divergence is visible in every flight record. | `ppo_policy.py:_sync_vdrone` | The ghost heading. This alone most likely restores A → B under headful. |
| 2 | **Stop integrating yaw rate in the backend.** Have the adapter emit an absolute **yaw setpoint** (its now-synced heading plus rate × dt) and have every backend send that angle. One integrator, in one place, with one dt. | `px4.py` streamer, `mock.py`, `core/backend.py` signature | The second and third integrators and their seed logic. |
| 3 | **Make heading behaviour a mission-level mode, declared once:** `hold`, `follow_course`, `policy`. The adapter computes the yaw setpoint for the mode; backends only execute. Delete `--yaw-follow`, `applies_yaw_rate` and the "refuse unless streaming" branch. | `fly_policy_mission.py`, `fly_policy_px4.py`, `airsim.py`, `px4.py` | The four-way behaviour table in 3.2. |
| 4 | **Make the runner refuse an off-contract observation box and negate pitch at the adapter boundary.** Compare the runner's bounds to `observation_manifest.reference_box` at `prepare()`; apply `pitch = -state.attitude.pitch` in `build_drone3d_from_state` with a test pinning the sign. | `fly_policy_px4.py:prepare`, `ppo_policy.py:163-182` | The 4× observation scale error and the wrong-sign pitch slot. |
| 5 | **Refuse checkpoints whose physics regime does not match the adapter's virtual airframe.** The contract sidecar already records the regime; make `load()` check it. | `ppo_policy.py:load`, `configs/policy_execution_contract_v1.json` | Silent physics-model mismatch. |
| 6 | **Only then** decide whether the *policy* should be headful: retrain with `--reward-heading > 0` and a yaw-error observation term, or keep it headless with mode `follow_course` handled by the autopilot. Do not tune gains or smoothing before 1–4 are in. | `ppo_gpu.py` | Re-tuning against a plant whose heading is undefined. |

If only one change is possible, do **1**. If two, do 1 and 4 (the box check is
a one-line guard and may by itself explain the PX4 A → B failure).

---

## 6. Confirm before changing anything

Run a headful flight on the MockBackend (no simulator needed) and log, per
tick: virtual yaw, backend-reported yaw, commanded yaw rate, commanded
velocity, achieved velocity. Expected result if the diagnosis is right: the
yaw difference grows monotonically after the first few ticks and the
velocity error grows with it. If the yaw difference stays near zero and A → B
still fails, the problem is elsewhere (most likely the tracking compensation
or the vertical lead law) and section 5 step 1 is unnecessary.

A second cheap check: run the same mission with `translation=velocity` and
with `translation=physics` on the same backend. If only `physics` fails, it
is the virtual airframe. If both fail, it is the backend yaw path.

A third, for the PX4 runs specifically: print the adapter's `bounds` next to
the contract's `reference_box` at `prepare()`. If they differ, fix that
before reading anything else out of the flight record.

---

## 7. Regression tests to lock it in

1. **Heading agreement:** after N ticks of a yaw-rate-commanding policy on
   MockBackend, `|virtual_yaw - backend_yaw| < 1 deg`.
2. **Mode isolation:** the same mission under heading modes `hold` and
   `follow_course` produces the same position track within tolerance.
3. **Backend parity:** Mock, AirSim and PX4 backends given the same
   `set_velocity(v, yaw_setpoint)` sequence record the same yaw setpoint
   sequence (unit test with fakes; all three already have fakes in `tests/`).
4. **Regime guard:** loading a `--full-attitude` checkpoint into the Euler
   adapter is refused with a clear message.

Test 1 would have caught this regression on the day the streamer landed.

---

## 8. Things noticed in passing (not the cause, worth a ticket each)

- `airsim_dqn_preview.py:40-64` feeds raw NED into an `[E, Up, N]`
  observation. Legacy script, but it will mislead anyone using it to compare.
- `backends/airsim.py:_quat_to_euler` and `tools/airsim_hop_cycle_qa.py:_rpy_deg`
  are duplicate quaternion decoders.
- `tools/sysid_fit.py` and `tools/px4_sih_sysid.py` map east → roll and
  north → pitch, which is only true at yaw 0. Any sysid flight with a
  non-zero heading is fitting the wrong axes.
- Roll sign differs between `drone3d.py` (`Rz(+roll)`) and the quaternion
  path in `ppo_gpu.py` (`Rz(-roll)`). The comment says they match; this was
  not verified here.

---

# Part 2: Options

## Option 1: Quick fix for the Raspberry Pi transfer (about 120 lines, 2–3 days)

**Principle.** Keep the policy headless. The autopilot owns the nose. The
nose is pointed along the commanded course, so the drone looks and moves
headful without any retraining and without the policy's untrained yaw channel
ever reaching the vehicle. In-repo precedent: the "smooth" yaw profile in
`tools/airsim_hop_cycle_qa.py:258-269` and `--yaw-follow` on AirSim.

| Step | File | Change | Size |
|---|---|---|---|
| A | new `AiDroneFly/src/aidronefly/control/heading.py` | `course_yaw_deg(velocity_enu) = atan2(east, north)`; `step_yaw_toward(current, target, dt, max_rate)` with wrap and a 0.5 m/s deadband | ~30 lines |
| B | `backends/px4.py` | ctor arg `yaw_mode = "hold" | "rate" | "follow_course"`. Under `follow_course`, `set_velocity` refuses a yaw rate; the streamer loop (`px4.py:2046-2048`) steps `_stream_yaw_deg` toward the course instead of integrating the rate. Expose `yaw_mode` in `setpoint_stream_status` | ~35 lines |
| C | `backends/mock.py:112-127` | same `yaw_mode`; `step` moves heading toward course with the same helper so the law is verified on Mock first | ~12 lines |
| D | `tools/fly_policy_px4.py`, `tools/fly_policy_mission.py` | `--yaw-mode` flag, default `hold` (today's behaviour). When not `rate`, null `yaw_rate_dps` in the command so evidence is honest. Record `yaw_mode` and measured yaw per tick. AirSim maps `follow_course` to `yaw_follow` | ~20 lines |
| E1 | `tools/fly_policy_px4.py:prepare()` after line 1250 | refuse unless runner bounds equal `observation_manifest.reference_box`; add `--bounds-from-contract` so the attended run can fly the ±60 / 0–55 box without OSM. **This will refuse today's attended configuration. That is the point.** | ~15 lines |
| E2 | `ppo_policy.py:178`, `core/types.py:74-81` | `drone.pitch = -state.attitude.pitch`; document nose-up positive; add a sign-pinning test (the current parity test copies pitch unsigned, so it cannot catch this) | ~10 lines |

**Why no yaw re-sync is needed for the Pi flight.** In open space
`raycast_obstacles` returns all ones (`drone3d.py:411-412`), so the only
yaw-dependent observation slots are sin/cos yaw, and those are
self-consistent with the virtual yaw. For `--city` flights add an opt-in
`sync_yaw=True` that copies measured yaw into the virtual drone each tick.

**Verification order.** Unit tests for the helper, Mock convergence, and the
fake-MAVSDK streamer test (extend `tests/test_px4_setpoint_streamer.py`).
Then Mock flight: `follow_course` and `hold` give the same position track and
heading error under 20 deg while moving. Then PX4 SIH via
`policy_px4_attended_run.py --bounds-from-contract --yaw-mode follow_course`.

**Do not touch.** `yaw_rate_dps()`, the envelope yaw clamp, `applies_yaw_rate`,
the contract JSON, `PARAM_ALLOWLIST`, the virtual pitch/roll internals,
`physics_vzlead`, any default flag value. PX4 `MPC_YAW_MODE` is irrelevant in
Offboard and is not in the param allowlist.

**Rejected for the quick fix.** Re-syncing virtual yaw and continuing to stream
the yaw rate. It cures the ghost but the rate is mixer noise from an untrained
channel: a slow random wander, not nose-along-course.

## Option 2: Structural fix (about 11.5 engineer-days, plus 3–5 days and GPU time if a truly headful policy is wanted)

**Target architecture.**

```
MISSION TOOL      --heading-mode {hold | follow_course | policy}   decided once, banked in flight.json
      |
HeadingController (control/heading.py)   in: measured yaw, commanded velocity, policy yaw rate
                                          out: absolute yaw setpoint, slew-limited
      |
PPOPolicyAdapter.step  ->  {velocity, yaw_setpoint}   virtual yaw == measured yaw, always
      |
EnvelopeMonitor        clamps setpoint slew
      |
Backend.set_velocity(velocity, yaw_setpoint)   absolute angle only, NO integrator in any backend
   Mock: slew to setpoint   AirSim: YawMode(False, deg)   PX4: VelocityNedYaw(.., deg) verbatim
      |
VehicleState: one contract, pitch nose-up positive, body_rates present or None, sim == real
```

| Step | Days | What | Key files |
|---|---|---|---|
| 0 | 0.5 | Evidence: log `yaw_virtual_minus_measured_deg` per tick; run the Mock flight from section 6 | `ppo_policy.py:716-720`, runners' tick records |
| 1 | 1 | Virtual drone yaw copies measured yaw every tick | `ppo_policy.py:370-383`; extend `AiDroneFly/tests/test_ppo_policy.py`, `tests/test_adapter_timing_yawrate.py` |
| 2 | 3 | Absolute yaw setpoint, one integrator in the adapter; delete PX4 streamer integration and seed logic, Mock integration; AirSim sends `YawMode(False, deg)`; envelope clamps slew | `core/backend.py:50-63`, `px4.py:400-407, 1469-1475, 1573-1593, 1972-1992, 2046-2048`, `mock.py:118-127`, `airsim.py:162-189`, `control/envelope.py:140-149, 353-358`; rewrite `tests/test_px4_setpoint_streamer.py` yaw tests; add backend-parity test |
| 3 | 2 | `HeadingController` with the three modes; `--heading-mode` on both runners; delete `--yaw-follow`, `applies_yaw_rate`, `yaw_rate_applied`; `policy` mode gated on a `headful` checkpoint flag | new `control/heading.py`, runners, `airsim.py:175-187` |
| 4 | 2 | One `VehicleState` contract: pitch sign pinned, `body_rates` field, PX4 subscribes `attitude_angular_velocity_body`, AirSim from IMU; adapter negates pitch and fills rates only behind `observe_body_rates=True` | `core/types.py:73-100`, `px4.py:3188-3197`, `ppo_policy.py:163-182`; extend `test_px4_backend.py::test_telemetry_is_converted_from_ned_to_enu` |
| 5 | 1.5 | Refuse mismatches at load: physics regime (`full_attitude`, `direct_motors`), box vs `reference_box`, dt vs `control_period`; `--allow-off-contract-box` must be explicit and banked | `ppo_policy.py:386-493`, `fly_policy_px4.py` after 1250, `fly_policy_mission.py` after 388, `policy_px4_attended_run.py:166-169` |
| 6 | 1.5 | One physics reference: parity test GPU legacy Euler vs `Drone3D.update` including roll sign; write `physics_model` into checkpoint meta and export | `ppo_gpu.py:968-974, 3376-3378`, `tools/export_policy_actor.py:494-531`, pattern in `tests/test_physics_v2.py:262` |
| 7 | 3–5 + GPU | Only if a headful *policy* is wanted: body-frame relative target in slots 0–2 (same dim), `--reward-heading 0.05–0.1`, yaw-rate penalty, train on the Euler reference at a 0.1 s decision period, warm-start from `field_quad_dr_r` with the heading coefficient ramped over the first 20% of iterations, gate on reach, crash and mean yaw-to-course error | `ppo_gpu.py:980-996, 1389-1396`, `train_dqn_headless.state_vector` |

**Contract JSON changes** (`configs/policy_execution_contract_v1.json`):
`checkpoint_contract.physics_model` and `refused_regimes`;
`observation_manifest.attitude_convention` (nose-up positive at the boundary,
negated into slot 8); `translation_modes.common.heading_modes`; replace
`envelope_limits.implemented.yaw_rate` with `yaw_setpoint`; a
`control_period.adapter_dt_must_equal` rule.

**How the two options relate.** Option 1 steps A–E are a strict subset of
Option 2 steps 1, 3, 4 and 5. Nothing in Option 1 is thrown away. Option 2
step 2 (absolute setpoint everywhere) is the one piece that changes the
backend contract and should wait until after the Pi transfer.
