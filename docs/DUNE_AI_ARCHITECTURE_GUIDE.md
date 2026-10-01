# dune-ai control stack: how it works today, where it is going, and how to get there

Reviewed at commit `5da87dc` of `zwethiha234-svg/dune-ai` (2026-09-27). All paths are
relative to the repo root. Every `file:line` was opened and checked by the reviewer;
"CONFIRMED" means the behaviour is literally in the quoted code, "INFERRED" means it
follows from the code but is not stated there, "UNVERIFIED" means it could not be
checked in this checkout.

**One-paragraph summary.** The trained PPO policy is *headless*: it sees the target in
world axes, is rewarded only for progress, and its fourth action (yaw) was never shaped
by the reward. The team wants *headful* flight (nose along the direction of travel, like
a real drone). The current attempt streams the yaw rate of a virtual airframe that lives
inside the policy adapter and whose heading is never re-synchronised with the vehicle.
As soon as the vehicle follows that rate, the adapter's idea of its own nose and the
real nose diverge, and A to B navigation collapses. Two unrelated parity breaks (an
observation box 4x smaller than training on the PX4 runs, and a pitch sign copied
without negation) make the PX4 flights fail even with yaw untouched. The fix direction
is "one owner per fact": measured yaw owns heading, one component computes the yaw
setpoint, backends only execute it, and the adapter refuses mismatched boxes,
regimes and sign conventions at load time.

---

# Part A. How the existing system works

## A1. Training

**Environment.** `ppo_gpu.py` runs N vectorised copies of a simplified quad
(`GPUVecDrone`), each in a box with one target point. The CPU twin is `ppo_trainer.py`
driving `drone3d.Drone3D`. Both use the same observation builder
(`train_dqn_headless.state_vector`, `train_dqn_headless.py:184-225`) and the same mixer.
Sim axes are `[x=East/right, y=Up, z=North/forward]` (`drone3d.py:11`); vertical is index 1.

**Box bounds and `--city`.** Without a city the box is ±15 m horizontal, 0-12 m vertical
(`ppo_gpu.py:2237`). With `--city` it becomes the terrain-radius square with a 55 m
ceiling (`ppo_gpu.py:2872-2873`):

```python
R = float(args.terrain_radius)
bounds = ((-R, R), (0.0, 55.0), (-R, R))
```

`--terrain-radius` defaults to 60 (`ppo_gpu.py:2365`), so a city run is ±60 m / 0-55 m.

**Observation (34 slots).** Built at `ppo_gpu.py:980-1016`; `state_vector` mirrors it;
the layout is written into every export manifest (`tools/export_policy_actor.py:94-105`).

| Slots | Content | Scaling |
|---|---|---|
| 0-2 | `rel = (target - pos) / span`, **world axes** | span = `[xmax-xmin, ymax, zmax-zmin]` |
| 3-5 | velocity, world axes | `/8`, clipped ±2 |
| 6-9 | `sin(yaw), cos(yaw), pitch, roll` | pitch/roll ÷ `MAX_TILT` = 70° (`ppo_gpu.py:39`) |
| 10-12 | body rates `[yaw, pitch, roll]` | `/6`, clipped ±2 |
| 13 | altitude | `/ ymax` |
| 14 | distance to target | `/ |span|` |
| 15-20 | airframe config | `drone3d.py:242-259`, present with `--airframes` |
| 21-32 | 12 raycasts, ray 0 along yaw | distance / 60 m; **all ones with no obstacles** (`drone3d.py:411-412`) |
| 33 | terrain clearance | `clip((up - floor)/60, 0, 1)` |

Slots 0-2, 13 and 14 are divided by the box span, so the same geometry produces about
4x different numbers in the ±15 box versus the ±60 box.

**Actions (4).** `a = [collective, desired_pitch, desired_roll, yaw]`, each in [-1, 1].

**Mixer.** `ppo_gpu.py:822-833`, identical in `drone3d.py:521-544`:

```python
desired_pitch = a[:, 1] * TILT_CMD          # 25 deg (ppo_gpu.py:41)
yaw_off = a[:, 3] * 0.03
pitch_mix = 0.23 * (desired_pitch - self.pitch) - 0.035 * self.ang_vel[:, 1]
yaw_mix = yaw_off - 0.02 * self.ang_vel[:, 0]
```

A PD loop turns desired tilt into four rotor thrust fractions
(`base ± pitch_mix ± roll_mix ± yaw_mix`). The yaw channel is a raw differential-thrust
offset, not a rate setpoint.

**Physics step.** Thrust acts along body-up after `Ry(yaw)@Rx(pitch)@Rz(roll)`
(`drone3d.py:590-607`); torques come from rotor differences (`drone3d.py:615-619`);
Euler integration updates velocity, position and attitude (`drone3d.py:649-659`).

```python
self.yaw += self.ang_vel[0] * dt                                  # drone3d.py:656
```

Pitch sign: the pitch matrix (`drone3d.py:599-602`) maps body-up `[0,T,0]` to
`[0, cp*T, sp*T]`, so **positive pitch pushes thrust forward, nose down**. CONFIRMED.

**Decision period.** Each policy step repeats physics `--action-repeat` (default 4)
times at `--dt` (default 1/60 s) (`ppo_gpu.py:1298-1299`, `2348-2349`): 66.7 ms.

**Reward.** `ppo_gpu.py:1337`:

```python
r = 4.0 * progress - 0.01 - 0.01 * speed - 0.015 * tilt
```

The heading bonus (`ppo_gpu.py:1389-1396`, flag `--reward-heading`, **default 0.0** at
`ppo_gpu.py:2476`):

```python
course = torch.atan2(vx, vz)
r = r + self.heading_coef * moving * torch.cos(self.yaw - course)
```

**What "headless" means here.** With `heading_coef = 0` nothing in the reward refers to
yaw, the target is in world axes, and velocity is world-frame. The policy only needs yaw
to know how its tilt maps to world motion (slots 6-7). It never has to point its nose
anywhere, and `a[3]` is unconstrained as far as the objective is concerned. CONFIRMED by
the reward; the consequence for `a[3]` is INFERRED.

**Spawn yaw.** Every reset draws a uniform heading (`ppo_gpu.py:549`):

```python
self.yaw[idx] = self._u(m, -math.pi, math.pi)
```

The policy learned to reach targets from any heading, but nothing about changing
heading on purpose.

## A2. Export

`tools/export_policy_actor.py` loads a `.pt` checkpoint, strips the actor MLP and saves
`W0,b0,W1,b1,...,log_std` as `actor.npz` (`export_policy_actor.py:472-477`). Beside it,
`manifest.json` carries `checkpoint_sha256` (515), the 34-slot layout (537) and
`files.npz.sha256` (545). The runtime loader recomputes the hash and refuses on mismatch
(`AiDroneFly/src/aidronefly/policies/numpy_actor.py:226-230`):

```python
npz_sha256 = sha256_file(npz_path)
if npz_sha256 != want_npz:
    raise ActorExportError(...)
```

It also checks geometry, tanh activation and both clamps (232-258). Forward pass:
`x = x @ w.T + b` with tanh between layers, then `clip(-10,10)` and `clip(-1,1)`
(`numpy_actor.py:115-125`). The attended PX4 run pins the parent checkpoint
(`tools/policy_px4_attended_run.py:155`, `712-718`).

## A3. Deployment adapter: `PPOPolicyAdapter.step()`

File: `AiDroneFly/src/aidronefly/policies/ppo_policy.py`.

**Frame remap.** Product `VehicleState` is ENU `(x=E, y=N, z=Up)`; drone3d is
`[E, Up, N]`. `drone3d_pos_from_state` (149-151) returns `[position.x, position.z, position.y]`.

**Observation reconstruction.** `build_drone3d_from_state` (163-182):

```python
drone.yaw = float(state.attitude.yaw)
drone.pitch = float(state.attitude.pitch)          # line 178, no sign change
drone.ang_vel[:] = 0.0  # product telemetry has no body-rate channel
```

Then `build_observation` (349-363) calls the same `state_vector` as training with the
adapter's `bounds`, airframe, buildings and ray settings. `load()` refuses a checkpoint
whose `obs_dim` differs from what this configuration builds (422, 477).

**Three translations** (`TRANSLATIONS`, line 140):

- `velocity` (618-631): drops `a[3]`, rotates `(fwd, right)` by the *measured* yaw into
  ENU and scales by `max_speed`:
  ```python
  east = fwd * math.sin(yaw) + right * math.cos(yaw)       # line 625
  ```
- `physics` (528-545): steps a persistent virtual `Drone3D` with the action and commands
  its resulting velocity:
  ```python
  vd.apply_continuous_action(np.asarray(action, dtype=np.float32), self.dt)   # 536
  vd.update(self.dt)                                                            # 537
  ```
- `physics_vzlead` (547-580): same horizontal; vertical is a leaky integrator of the
  virtual vertical acceleration.

**Virtual drone sync.** `_sync_vdrone` (370-383): on the first tick the virtual drone is
built from telemetry; on every later tick only position and velocity are overwritten:

```python
self._vdrone.pos[:] = drone3d_pos_from_state(state.position)      # 381
self._vdrone.vel[:] = _drone3d_vec_from_vector3(state.velocity)   # 382
```

Yaw, pitch, roll, `ang_vel` and `t1..t4` stay internal forever. CONFIRMED.

**Yaw-rate output.** Physics modes only (`yaw_rate_dps`, 600-616):

```python
return math.degrees(float(self._vdrone.ang_vel[0]))               # 616
```

This is the virtual airframe's body yaw rate after the mixer, not a function of `a[3]` alone.

**Tracking compensation** (`_compensate_velocity`, 765-780):
`command = proposed + K*(last_command - achieved)`, capped at `max_speed*(1+K)`. The
attended run uses K=0.5.

**Safety supervisor.** `step` (686-745) always calls `assess_state`; only
`PLAN_A` + `CONTINUE` lets the policy's `set_velocity` through (722-723), otherwise the
supervisor's `goto/land/hold` is sent. `_execute` (783-795) drops `yaw_rate_dps` when None.

## A4. Envelope and mission loops

**EnvelopeMonitor** (`AiDroneFly/src/aidronefly/control/envelope.py`). Defaults 7 m/s
horizontal, 3 m/s vertical, 60 deg/s yaw (101-103). Yaw clamp at 354-356:

```python
if yaw_in is not None and abs(yaw_in) > lim.yaw_rate_dps:
    yaw_out = math.copysign(lim.yaw_rate_dps, yaw_in)
```

The virtual drone has already turned by the unclamped rate before this runs
(`vd.update` at `ppo_policy.py:537` precedes the gate). INFERRED from ordering.

**`tools/fly_policy_mission.py`** (Mock/AirSim). Bounds mirror the trainer (315-319):
city gives ±R / 0-55, else ±15 / 0-12. Reach test
`st2.position.distance_to(wp) < args.reach_radius` (629), default 2.0 (763). With
`--envelope` the adapter is called with `execute=None` and `envelope_gate` (122-197)
sends the clamped command. `yaw_rate_applied` (218-227) reports Mock=True,
AirSim=False, PX4="unknown".

**`tools/fly_policy_px4.py`.** Reuses `fpm.resolve_bounds` (1250). Frame registration
at arm refuses unless a real position sample has arrived (1385-1396), then
`FrameRegistration.origin = st.position`. `BoxFrameBackend` (694-712) subtracts the
origin from positions and passes velocities and yaw rates through unchanged (702-706).
`--stream-hz` defaults to 20 (269) and becomes `PX4Backend(setpoint_stream_hz=...)`
(1267). The contract file is only hashed into evidence (4034); **nothing compares
bounds to it** (`reference_box` has zero hits in this file). CONFIRMED.

**`tools/policy_px4_attended_run.py`** builds the frozen argv (166-169, 617-626):
`--airframe small_quad --obstacle-sensing raycast --translation physics --max-speed 3.0
--velocity-gain 0.5 --alt 6 --dt 0.1 --stream-hz 20 --device cpu`, plus
`--checkpoint <export>/actor.npz --offline --launch-px4`. **No `--city`, so bounds
resolve to ±15 / 0-12.**

## A5. Backends

**Contract.** `VehicleState` (`AiDroneFly/src/aidronefly/core/types.py:84-95`): ENU
position/velocity, `Attitude(roll, pitch, yaw)` radians, yaw "0 = North, clockwise
positive" (73-81). `Backend.set_velocity(velocity, yaw_rate_dps=None)` (`core/backend.py:50`).

**PX4 (`backends/px4.py`).** NED to ENU on telemetry (3155-3164); attitude copied as-is,
pitch included (3190-3194). Sending (1697):

```python
setpoint = velocity_type(velocity.y, velocity.x, -velocity.z, yaw)   # VelocityNedYaw(N, E, D, yaw_deg)
```

`VelocityNedYaw` takes a yaw *angle*, so with the streamer on the rate is integrated on
the loop thread (2046-2048) with its own `yaw_dt` (1990-1991):

```python
self._stream_yaw_deg = self._wrap_deg(self._stream_yaw_deg + record.yaw_rate_dps * yaw_dt)
```

seeded from the last measured heading at stream start (1575-1580). Streamer off: yaw is
hard-wired 0° (nose North) and a non-None rate raises (1469-1473).

**AirSim (`backends/airsim.py`).** `_enu_to_ned` (38-40). `applies_yaw_rate = False`
(54). `set_velocity` (162-189) sends `moveByVelocityAsync(n, e, d, duration)` and
**ignores `yaw_rate_dps`**; with `yaw_follow` it uses `DrivetrainType.ForwardOnly` so
the autopilot points the nose along the velocity (183-187).

**Mock (`backends/mock.py`).** `applies_yaw_rate = True` (19); `step` integrates the
commanded rate kinematically (121-126).

## A6. Contract: `configs/policy_execution_contract_v1.json`

Pins `obs_dim 34` and the slot layout; action semantics (85-87); control period
(92-100: training 66.7 ms, adapter 0.1 s, "recorded, not resolved"); reference
checkpoint sha (101-103); `reference_box` ±60 / 0-55 (164-176); envelope targets (616);
yaw-rate backend behaviour (671-674). Provenance binds checkpoint sha, contract sha,
args and git HEAD (775-780). **It enforces nothing at runtime**: no code compares
bounds, dt or physics regime to it. CONFIRMED.

## A7. One control tick, today

```
waypoint (box ENU)      VehicleState (box ENU; PX4: px4_enu - origin)
      |                        |
      v                        v
 ppo_policy.step --ENU->[E,Up,N]--> _sync_vdrone (pos/vel ONLY) --> state_vector
                                    (34 slots: world-frame rel/vel + VIRTUAL yaw/pitch/roll)
                                                           |
                                                           v  obs
                                                   NumpyActor.act -> a[4] in [-1,1]
                                                           |
                      physics: vd.apply_continuous_action + vd.update   (drone3d [E,Up,N])
                                   |                              |
                   vel [E,Up,N] -> Vector3(E,N,Up)        yaw_rate = deg(vd.ang_vel[0])  (VIRTUAL)
                                   |                              |
            _compensate_velocity (ENU) ------------------------>  SafetySupervisor (PLAN_A?)
                                   |
                        EnvelopeMonitor (ENU; 7/3 m/s, 60 deg/s)
                                   |
           BoxFrameBackend.set_velocity (velocity ENU unchanged, yaw_rate unchanged)
                                   |
      PX4Backend: VelocityNedYaw(N=vel.y, E=vel.x, D=-vel.z, yaw_deg = integrated AGAIN on streamer)
                                   |
                  PX4 offboard velocity + yaw controllers -> motors
```

## A8. Where it breaks

1. **Virtual yaw never re-synced** (`ppo_policy.py:381-382`, CONFIRMED). After tick 1
   the observation's yaw (slots 6-7) and the raycast orientation are the ghost's, not
   the vehicle's.
2. **Streamed yaw rate is untrained mixer output** (`ppo_policy.py:616` +
   `ppo_gpu.py:2476` default 0, CONFIRMED). The vehicle's heading is driven by a channel
   the reward never shaped. This is the spin.
3. **Three independent yaw integrators** (virtual `drone3d.py:656` with adapter dt; PX4
   `px4.py:2046-2048` with loop `yaw_dt` and seed `px4.py:1580`; Mock `mock.py:124-126`),
   CONFIRMED. Nothing compares them.
4. **Envelope clamps after the virtual drone turned** (`envelope.py:354-356` runs after
   `ppo_policy.py:537`), CONFIRMED. The ghost turns at full rate, the vehicle at ≤60 deg/s.
5. **Backend-dependent heading** (`airsim.py:162-171` ignores; `px4.py:1469-1473`
   refuses or forces North; `mock.py:19` integrates), CONFIRMED. Same policy, different
   physical heading per backend.
6. **`velocity` translation rotates by measured yaw** (`ppo_policy.py:625`) while the
   vehicle's heading is autopilot-owned, CONFIRMED; the policy never trained with a yaw
   it does not control (INFERRED).
7. **Observation box parity.** Attended run passes no `--city`
   (`policy_px4_attended_run.py:166-169`) so `resolve_bounds`
   (`fly_policy_mission.py:315-319`) gives ±15 / 0-12 while the contract's
   `reference_box` is ±60 / 0-55 (contract 164-176); `fly_policy_px4.py` never checks.
   CONFIRMED. Slots 0-2, 13, 14 are scaled about 4x off training.
8. **Pitch sign parity.** Training positive pitch = nose down (`drone3d.py:599-602`);
   PX4 copies `pitch_deg` unchanged (`px4.py:3192`), AirSim decodes standard aerospace
   pitch (`airsim.py:380-381`); the adapter copies without negation
   (`ppo_policy.py:178`). That the autopilots report nose-up positive is INFERRED from
   their conventions. Slot 8 has the wrong sign under `velocity` translation and on the
   first physics tick.
9. **Decision period.** 66.7 ms training (`ppo_gpu.py:1298, 2348-2349`) vs 0.1 s adapter
   and 20 Hz republish (`fly_policy_px4.py:269`); recorded in the contract (92-100),
   not resolved. CONFIRMED.

---

# Part B. Where we are going and how to get there

## B1. Target architecture: one owner per fact

| Fact | One owner in the target |
|---|---|
| Vehicle heading | measured `VehicleState.attitude.yaw`; the virtual drone copies it every tick |
| Frame | ENU at the product boundary, `[E, Up, N]` only inside `ppo_policy.py` (`drone3d_pos_from_state`, 149-151) |
| Physics regime | the checkpoint's `collision_regime["dynamics"]` (`ppo_gpu.py:2089-2091`), checked at `load()` |
| Observation box | `observation_manifest.reference_box` in the contract, checked at `prepare()` |

Target control tick:

```
 MISSION TOOL  fly_policy_px4.py / fly_policy_mission.py
   --heading-mode {hold|follow_course|policy}   banked once in flight.json
        |  target [ENU]                       state [ENU, yaw compass CW+, pitch nose-up+]
        v
 +------------------------------+
 | PPOPolicyAdapter.step        |  ppo_policy.py 686-744
 |  _sync_vdrone: pos,vel,YAW   |  370-383  <- yaw copy added
 |  pitch = -state.pitch        |  178      <- sign flip added
 |  obs = state_vector [E,Up,N] |  66-92 (rel target WORLD frame, unchanged)
 |  a -> virtual Drone3D step   |  528-545
 |  velocity proposal [ENU]     |
 +------------------------------+
        |  velocity [ENU], policy yaw rate (evidence only unless mode=policy)
        v
 +------------------------------+
 | HeadingController (new       |  in: measured yaw, commanded velocity, mode
 |  control/heading.py)         |  out: ABSOLUTE yaw setpoint [compass deg], slew-limited
 +------------------------------+
        |
 +------------------------------+
 | EnvelopeMonitor              |  control/envelope.py; today clamps a RATE (353-358),
 |                              |  target clamps setpoint slew
 +------------------------------+
        |  set_velocity(velocity [ENU], yaw_setpoint_deg)
        v
 Backend (core/backend.py:50)  NO integrator anywhere
   Mock: slew to setpoint      AirSim: YawMode(False, deg)   PX4: VelocityNedYaw(n,e,d,deg) verbatim
        |
 VehicleState  core/types.py:74-96  (pitch nose-up+, body_rates present or None)
```

What each box becomes, and what is deleted:

- **Adapter** stays `PPOPolicyAdapter`. Deleted: the "attitude stays internal" rule for
  yaw in `_sync_vdrone` (370-383). `yaw_rate_dps()` (600-616) kept as evidence only.
- **HeadingController** is new. It absorbs the "smooth" yaw-profile idea in
  `tools/airsim_hop_cycle_qa.py:258-269` (move the setpoint, not the nose, at a max rate).
- **Envelope** keeps `EnvelopeLimits.yaw_rate_dps` (`envelope.py:103`, 60 deg/s) but
  applies it as a per-tick slew of the setpoint.
- **Backends.** Delete the PX4 streamer integrator (`px4.py:2046-2048`), its `yaw_dt`
  anchor (1988-1992), its seed logic (1573-1593), the `_StreamSetpoint.yaw_rate_dps`
  field (399-407); Mock's integration (`mock.py:121-126`); AirSim's `yaw_follow` branch
  (`airsim.py:175-187`); `applies_yaw_rate` (`mock.py:19`, `airsim.py:54`);
  `fpm.yaw_rate_applied` (`fly_policy_mission.py:218-226`).

## B2. Two heading strategies, and why we pick (A)

**(A) Headless policy + autopilot points the nose along course (`follow_course`).**
The network is untouched: it keeps seeing the target in world axes and keeps tilting
relative to the yaw it is told. The nose is driven by a yaw setpoint computed from the
*commanded velocity* (`atan2(east, north)`), slewed. `a[3]` never reaches the motors.
For the Pi this is a few lines of numpy trig per tick, no model change, no new hash.

**(B) Headful policy via retraining.** The network is retrained to see the target in
*body* frame and rewarded for nose-along-course (`ppo_gpu.py:1389-1396`,
`--reward-heading`, default 0 at 2476). The vehicle then flies the policy's yaw channel.
For the Pi this means a new checkpoint, export, parity evidence, and a yaw path that has
never flown on PX4.

**Recommendation: A now, B optional later.** The yaw channel was never trained to be
flown (`heading_coef` default 0, `ppo_gpu.py:220`), so streaming it is mixer noise. In
open space the only yaw-dependent observation slots are sin/cos yaw (rays are all ones
with no obstacles, `drone3d.py:410-412`), so a headless policy with an externally driven
nose loses nothing. (A) is about 120 lines and no GPU. (B) is only worth doing once (A)
flies cleanly; otherwise you are tuning a policy against a plant whose heading is undefined.

## B3. Migration path

Each step removes one duplicate owner before the next depends on it.

**Step 0. Evidence logging (0.5 day).** Add `yaw_virtual_minus_measured_deg` and
`yaw_measured_deg` beside `yaw_rate_dps` in the command dict (`ppo_policy.py:714-718`);
copy them into the per-tick record in `fly_policy_px4.py:2209-2241` (beside
`cmd_yaw_rate_dps`, 2230-2232) and `fly_policy_mission.py:576-603`. Neither tick record
carries measured attitude today. Test:
`tests/test_adapter_timing_yawrate.py::test_physics_flight_without_envelope_records_the_sent_yaw_and_the_mock_flies_it`
(576). Proof: a Mock flight with `--translation physics` shows the difference growing.

**Step 1. Virtual yaw re-sync (1 day).** `_sync_vdrone` (370-383) copies
`state.attitude.yaw` into `self._vdrone.yaw` each tick; pitch/roll/rates stay internal.
`AiDroneFly/tests/test_ppo_policy.py::test_physics_translation_attitude_persists_across_steps`
(365) asserts *pitch* survives, so it stays green. Extend
`tests/test_adapter_timing_yawrate.py::test_physics_yaw_rate_reaches_the_mock_and_turns_its_heading`
(209): after N ticks `|virtual - mock yaw| < 1 deg`. Proof: step-0 field reads ~0 on
Mock, then PX4 SIH via `tools/policy_px4_attended_run.py`.

**Step 2. `follow_course` mode + runner flag (2 days).** New `control/heading.py`
(`course_yaw_deg`, `step_yaw_toward` with wrap and a 0.5 m/s deadband). `PX4Backend`
ctor `yaw_mode`; under `follow_course`, `set_velocity` (1444-1494) refuses a rate and
the loop (2046-2048) steps `_stream_yaw_deg` toward the course; expose in
`setpoint_stream_status` (2142). Same in `MockBackend.step` (112-127). `--yaw-mode` on
both runners, **default `hold`**, nulling `yaw_rate_dps` when not `rate`. AirSim maps
`follow_course` to the existing `yaw_follow` branch (175-187). Tests: sibling of
`tests/test_px4_setpoint_streamer.py::test_yaw_rate_is_integrated_into_the_yaw_setpoint`
(528); sibling of
`tests/test_adapter_timing_yawrate.py::test_mock_integrates_the_yaw_rate_and_hold_clears_it`
(275). Proof: Mock `hold` vs `follow_course` give the same position track, heading
error under 20 deg while moving; then SIH.

**Step 3. Box guard, pitch sign, contract edits (1 day).** `fly_policy_px4.py::prepare`
after 1250 compares `bounds` to `observation_manifest.reference_box` and refuses; add
`--bounds-from-contract` so `policy_px4_attended_run.py` (166-169, no `--city`) can use
the contract box. `build_drone3d_from_state` line 178 becomes
`drone.pitch = -state.attitude.pitch` (PX4 fake telemetry reports `pitch_deg=-5.0`,
`AiDroneFly/tests/test_px4_backend.py:102`, and the test at 548 only asserts roll).
Document nose-up-positive on `Attitude` (`core/types.py:74`); contract gets
`observation_manifest.attitude_convention`. Tests:
`tests/test_fly_policy_px4.py::test_waypoint_outside_box_is_refused` (1133) for the
guard; `AiDroneFly/tests/test_ppo_policy.py::test_obs_parity_matches_real_state_vector`
(199, copies pitch unsigned at 219 so cannot catch this today) gets a sign-pinning case.
Proof: the attended run is refused until `--bounds-from-contract` is passed. **Intended.**

**Step 4. Absolute yaw setpoint replaces all integrators (3 days, after the Pi flight).**
`Backend.set_velocity` (`core/backend.py:50`) takes `yaw_setpoint_deg`; delete
`px4.py` 1573-1593 seed, 1988-1992 anchor, 2046-2048 integration, the `yaw=0.0` at
1475; delete `mock.py:121-126`; AirSim sends `YawMode(False, deg)`;
`VelocityCommand.yaw_rate_dps` (`envelope.py:139-149`) becomes a setpoint with a slew
clamp at 353-358. Tests: rewrite the yaw tests in `tests/test_px4_setpoint_streamer.py`
(418, 528, 624-647);
`tests/test_envelope_monitor.py::test_yaw_rate_clamp_keeps_sign_and_none_stays_none`
(284); new backend-parity test across the three fakes.

**Step 5. VehicleState contract with body rates (2 days).** `body_rates` on
`VehicleState` (`core/types.py:85-96`); PX4 subscribes `attitude_angular_velocity_body`
in the state stream list (3098-3111; today only in the optional recorder 492-499,
565-566, rate set via `set_rate_attitude_euler` 439-440); adapter fills `ang_vel`
(178-180) only behind `observe_body_rates=True` since the contract calls a live gyro a
regime change. Test: `AiDroneFly/tests/test_px4_backend.py::test_telemetry_is_converted_from_ned_to_enu` (548).

**Step 6. Load-time regime refusal + physics parity (1.5 days).** `load()` (386-454)
already reads `collision_regime` for `sensor_padding` (411-416); refuse when
`regime["dynamics"]` contains `full_attitude` or `direct_motors` (set in
`dynamics_regime_from_args`, `ppo_gpu.py:2028-2052`); same in `_load_actor_export` via
`ActorExport.collision_regime` (`numpy_actor.py:167`, written by
`export_policy_actor.py:531`). Parity test: GPU legacy Euler (`ppo_gpu.py:968-974`) vs
`Drone3D.update` including roll sign, patterned on
`tests/test_physics_v2.py::test_full_attitude_matches_euler_at_small_angles` (262) and
`test_legacy_and_v1_checkpoints_reject_armed_dynamics` (434).

**Step 7. Optional headful retraining (3-5 days + GPU).** Observation: body-frame
relative target in slots 0-2 (same dim; rotate `rel` by yaw in `ppo_gpu.observe`
980-996 and mirror in `_state_vector_port`). Reward: `--reward-heading` 0.05-0.1
(1389-1396) plus a yaw-rate penalty (none exists; `--action-smooth-coef` at 2479 is the
nearest hook). dt: train at a 0.1 s decision period (`--dt 1/60 --action-repeat 6`,
flags 2348-2349) so the adapter's `dt=0.1` matches. Warm start: UNVERIFIED. `ppo_gpu.py`
has only `--resume` (2432) and `runs/field_quad_dr_r` is not in the checkout; a fresh
run is the safe default. Gate on reach, crash (`tools/eval_gate.py:296-368`,
`--max-crash` 652) and mean yaw-to-course error.

## B4. What the Raspberry Pi deployment needs

- **Mandatory before the Pi flight:** steps 0, 1, 2, 3. Step 4 waits. Steps 5-7 not needed.
- **Torch-free runtime:** already exists. `PPOPolicyAdapter.load` dispatches `.npz` to
  `_load_actor_export` (456-494); `NumpyActor` (`numpy_actor.py:39`) runs the MLP in
  numpy; `drone3d.py` imports only numpy and math. Proof pattern:
  `tests/test_numpy_runner.py::test_torch_blocked_subprocess_px4_prepare_loads_the_npz`
  (303) and `tools/numpy_runner_check.py`. The Pi needs `AiDroneFly/`, `drone3d.py`,
  `tools/safety_supervisor.py`, the contract JSON, `actor.npz` + manifest; the frozen
  export is pinned by `PARENT_CHECKPOINT_SHA256` (`policy_px4_attended_run.py:155`).
- **MAVSDK streamer:** `PX4Backend(setpoint_stream_hz=20)` (`DEFAULT_STREAM_HZ`,
  `fly_policy_px4.py:269`); PX4 needs ≥2 Hz (`_STREAM_MIN_HZ`, `px4.py:261`);
  `hold_after_s` 0.5 and `supervisor_land_after_s` 2.0 (270-271) must satisfy the ctor
  rules at 775-794.
- **Timing budget:** contract `control_period` records training 66.7 ms, adapter 0.1 s,
  product setpoint 66.7 ms "not yet implemented". Runner refuses `dt < MIN_DT_S` 0.02
  (`fly_policy_mission.py:70`) and `dt >= hold_after` (1203-1206). On the Pi the whole
  observe → decide → send must fit inside 0.1 s with margin; `PolicyStepResult.timing`
  (186-206) and the tick `timing` block (2222-2228) already measure it.
- **Log on the Pi to prove parity with sim:** per tick `pos`, `vel`, `cmd_vel`,
  `cmd_yaw_rate_dps`, the step-0 yaw fields, `yaw_mode`, `timing.loop_period_s`,
  `timing.observe_to_command_s`; per flight `bounds` vs `reference_box`,
  `npz_sha256`/`manifest_sha256` (`numpy_actor.py:160-162`), `setpoint_stream_status()`
  (gaps over 0.25 s, `yaw_setpoint_deg`), and the ULog copy (`--ulog-copy`, 4140).
  Compare the observation vector itself against a Mock replay of the same states.

## B5. Before / after

| Fact | Today | Target |
|---|---|---|
| Heading owner | virtual Drone3D never re-synced (`ppo_policy.py:370-383`) | measured yaw, copied every tick |
| Yaw integrators | three: `drone3d.py:656`, `px4.py:2046-2048`, `mock.py:121-126` | one, in adapter/HeadingController |
| Obs target frame | world (`ppo_policy.py:72-73`; `ppo_gpu.py:982`) | unchanged under (A); body under (B) |
| Pitch sign | copied unsigned (`ppo_policy.py:178`); drone3d nose-down+ (`drone3d.py:599-611`) | negated at boundary, nose-up+ pinned on `Attitude` |
| Box check | none (`fly_policy_px4.py:1250` → `fly_policy_mission.py:315-319`) | refusal vs `reference_box` |
| Physics regime check | none; `load()` reads regime only for pad (411-416) | refuse `full_attitude` / `direct_motors` |
| dt | 66.7 ms training, 0.1 s adapter, 20 Hz stream (contract `control_period`) | contract rule `adapter_dt_must_equal`; (B) trains at 0.1 s |

## B6. Risks, and what we do not change

- Step 1 alone turns the virtual yaw into the real one but keeps streaming an untrained
  rate; that is why step 2 defaults to `hold` and nulls the rate outside `rate` mode.
- Step 3's box guard will refuse today's attended configuration. Intended.
- PX4 `MPC_YAW_MODE` is irrelevant in Offboard and is not in `PARAM_ALLOWLIST`
  (`px4.py:58`); do not add it.
- **Not changed:** training checkpoints and the frozen export hash; `PARAM_ALLOWLIST`;
  `physics_vzlead` and its lead law (547-576); `yaw_rate_dps()` (600-616); the envelope
  yaw clamp value 60; every default flag value (`--translation velocity` on the mission
  runner at 774, `--dt 0.1`, `--yaw-mode hold`); the contract JSON until step 3's
  additive edits.
- **UNVERIFIED:** the retraining warm-start path; `runs/field_quad_dr_r`'s actual
  `collision_regime` (not in the checkout); no test for AirSim `yaw_follow` exists in
  `tests/` or `AiDroneFly/tests/`.

---

## Fact-check record

Two researchers wrote Parts A and B independently and in parallel. The orchestrator
then opened a sample of their citations against the checkout. All sampled file paths,
line numbers and test function names were real, with one correction: the
`--terrain-radius` default is at `ppo_gpu.py:2365`, not 2364. Items the researchers
could not verify are marked UNVERIFIED above and were left as stated.
