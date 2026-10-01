# Drone Control Architecture Review

**Purpose:** understand why fixing the headless → headful spin behaviour broke
point‑A‑to‑point‑B navigation, find the structural bottleneck, and decide where
to fix so we stop chasing one bug into the next.

**Scope note:** the `zwe` repository currently has no drone source code in it,
so this review is based on the symptoms described (sim‑vs‑real yaw handling
fixed, navigation stopped working) and on the standard structure of a
quadrotor control stack. Section 6 is a checklist to confirm the diagnosis
against the real code before changing anything.

---

## 1. The symptom, translated

| What you saw | What it means structurally |
|---|---|
| Headless vs headful spinning differed between sim and real | Two parts of the system disagree about **which frame yaw is expressed in** (world/NED vs body) and/or **which direction is positive**. |
| "Fixing" the spin made it look right | The correction was applied *somewhere* in the pipeline so the output matched for that one test. |
| Navigation A → B stopped working afterwards | The place the correction was applied is **shared** with the navigation path, so navigation now receives a rotated/flipped command it did not expect. |

The second bug is not a new bug. It is the first bug moved to a different
layer. That is the signature of a missing **single frame‑conversion boundary**.

---

## 2. What a clean control stack looks like

Every working drone stack, sim or real, is the same five layers. Each layer
has **one** input frame and **one** output frame, and frames are only ever
converted at the boundaries, never inside a layer.

```
┌──────────────────────────────────────────────────────────────┐
│ 5. MISSION / NAVIGATION                                      │
│    "go from A to B"                                          │
│    Frame: WORLD (NED or ENU – pick one, write it down)       │
│    Output: desired position / velocity in WORLD              │
└───────────────┬──────────────────────────────────────────────┘
                │ world setpoint
┌───────────────▼──────────────────────────────────────────────┐
│ 4. OUTER LOOP (position → velocity → acceleration)           │
│    Frame: WORLD in, WORLD out                                │
│    Output: desired acceleration / thrust vector in WORLD     │
└───────────────┬──────────────────────────────────────────────┘
                │ ***  THE ONLY WORLD→BODY CONVERSION  ***
                │ uses current yaw (and roll/pitch) from state
┌───────────────▼──────────────────────────────────────────────┐
│ 3. ATTITUDE SETPOINT GENERATOR                               │
│    Frame: BODY                                               │
│    Output: desired roll, pitch, yaw(-rate), collective thrust│
└───────────────┬──────────────────────────────────────────────┘
                │ body setpoint
┌───────────────▼──────────────────────────────────────────────┐
│ 2. INNER LOOP (attitude / rate PID)                          │
│    Frame: BODY in, BODY out                                  │
│    Output: body torques + thrust                             │
└───────────────┬──────────────────────────────────────────────┘
                │ torques / thrust
┌───────────────▼──────────────────────────────────────────────┐
│ 1. MIXER + ACTUATORS                                         │
│    Frame: none (motor space)                                 │
│    Output: per‑motor commands                                │
└──────────────────────────────────────────────────────────────┘

                 ▲ STATE ESTIMATE flows back UP
                 │ position/velocity in WORLD, attitude as quaternion
                 │ produced by ONE adapter per platform (sim / real)
```

Two rules make this work:

1. **Frame conversion happens in exactly one place** (the 4→3 boundary).
   Nothing above it knows about body frame. Nothing below it knows about
   world frame.
2. **Platform differences are absorbed in exactly one place**: an adapter
   that turns the sim's or the real IMU/GPS's data into the *same* state
   estimate format, and turns the same motor command format into whatever
   the sim or the ESCs want.

Headless mode and headful mode are then **not** a control‑stack concept at
all. They are a *pilot input* interpretation that lives at layer 5 ("rotate
the stick vector by the heading, or don't") and nothing below layer 5 ever
sees the difference.

---

## 3. Where the current design is almost certainly broken

Given the symptom, one or more of these is true. They are listed in order of
how likely they are to be the root cause.

### 3.1 The yaw/frame fix was applied below the world→body boundary

Most likely. The spin problem was observed at the motor/attitude level, so the
natural "fix" is to flip a sign or add a rotation in the attitude controller,
the mixer, or the sim adapter. But those layers are **shared** with navigation.
Navigation then sends a world‑frame velocity, it gets rotated by the boundary
*and* again by the patch, and the drone flies the wrong direction or oscillates.

**Tell‑tale sign:** the fix lives in a file that is also on the navigation
code path (attitude controller, mixer, rate loop, sim bridge).

### 3.2 There is no single state‑estimate contract between sim and real

If the sim provides attitude as Euler ZYX in ENU and the real IMU gives a
quaternion in NED (or vice‑versa) and each consumer converts for itself, then
every controller carries its own copy of the conversion. Fix one copy, the
others are now inconsistent.

**Tell‑tale sign:** more than one place in the code does
`yaw = atan2(...)` or `if sim: ... else: ...` on orientation data.

### 3.3 Headless/headful logic is tangled into the controller

If "headless" is implemented as a flag checked inside the attitude or velocity
controller, then the controller behaves differently depending on a pilot‑UX
setting, and the autonomous navigation path (which should never care about
headless) inherits that behaviour.

**Tell‑tale sign:** a `headless` boolean appears anywhere below layer 5.

### 3.4 Sign and axis conventions are implicit

NED vs ENU, yaw positive clockwise vs counter‑clockwise, motor numbering
order, propeller spin direction: if any of these are not written down and
asserted in code, the sim and the real platform will drift apart and every fix
will be a guess.

**Tell‑tale sign:** no `conventions` document / module, and tuning constants
with unexplained negative signs.

### 3.5 Tuning is being done against an untrustworthy plant

"Model tuning" problems are usually not tuning problems. If the frame
plumbing is wrong, no set of PID gains will work in both environments, and
the gains that *appear* to work are compensating for a frame error. This is
why gains keep needing to be re‑tuned after each fix.

---

## 4. The bottleneck, stated plainly

> **There is no single, owned boundary where world‑frame intent becomes
> body‑frame command, and no single adapter that makes sim and real look
> identical above the mixer.**

Because that boundary is missing or duplicated, every layer has partial
knowledge of frames, every platform has partial knowledge of conventions, and
any local fix changes the global behaviour. That is the thing to fix. Not the
spin, not the navigation, not the gains.

---

## 5. Where to fix (in order), and what each step buys

| # | Fix | Why first | Debugging it removes |
|---|---|---|---|
| 1 | **Write the conventions down and assert them.** One file: world frame (NED or ENU), body frame, yaw sign, quaternion order (w,x,y,z or x,y,z,w), motor layout, prop directions. Add startup assertions / unit tests for each. | Everything else depends on it. Takes an hour. | "Is this sign right?" arguments. |
| 2 | **Build a single `StateEstimate` type and one adapter per platform** (`SimAdapter`, `RealAdapter`) that both produce it. Delete every other sim/real conditional above the mixer. | Makes sim and real indistinguishable to every controller. | "Works in sim, not in real" and vice‑versa. |
| 3 | **Create one `world_to_body()` function and call it in exactly one place** (outer loop → attitude setpoint). Grep for every other rotation / yaw usage in the control path and delete it. | This is the structural root cause of the A→B regression. | The fix‑one‑break‑another cycle. |
| 4 | **Move headless/headful entirely to the pilot‑input layer.** It becomes `if headless: stick = rotate(stick, -yaw)` and nothing else. Navigation never sees it. | Decouples UX from control. | Navigation mysteriously affected by RC mode. |
| 5 | **Re‑tune gains only after 1–4 are done**, inner loop first (rate → attitude), then outer loop (velocity → position), in sim, then confirm on real. | Gains tuned on a correct plant transfer; gains tuned on a frame‑broken plant do not. | Endless re‑tuning. |

If time allows only one step, do **step 3**. If time allows two, do 1 and 3.

---

## 6. Diagnosis checklist (run against the real code before changing it)

Answer each with a file/line reference. Any "more than one" or "not sure" is
the bottleneck.

- [ ] Where is the world frame defined (NED/ENU)? One place?
- [ ] How many functions convert world → body or body → world?
- [ ] Which file contains the headless/headful spin fix? Is it on the
      navigation code path?
- [ ] Does navigation output world‑frame velocity, body‑frame velocity, or
      attitude directly? (It should be world‑frame velocity or position.)
- [ ] How many places read raw sim orientation vs raw IMU orientation?
- [ ] Is `headless` referenced anywhere in controller / mixer code?
- [ ] Are there any negative‑sign tuning constants without a comment
      explaining the convention they compensate for?
- [ ] Can the attitude controller be unit‑tested with a hand‑made
      `StateEstimate` and no sim running? If not, layers are coupled.

---

## 7. Fast regression tests to keep the problem from coming back

Four tests, each < 50 lines, each runs with no sim and no hardware:

1. **Frame round‑trip:** `body_to_world(world_to_body(v, q), q) == v` for
   random vectors and quaternions.
2. **Yaw sign:** with yaw = +90°, a world "north" velocity command becomes a
   body "left" (or "right", per your convention) command. One assert.
3. **Adapter parity:** feed the sim adapter and the real adapter the same
   synthetic hover pose; both must emit an identical `StateEstimate`.
4. **Headless isolation:** with `headless=True` and `headless=False`, a
   navigation setpoint must produce byte‑identical attitude setpoints.

Test 4 would have caught the current regression immediately.

---

## 8. Summary

- The spin fix and the navigation breakage are the **same bug** seen from two
  layers, caused by frame conversion being spread across the stack instead of
  owned by one boundary.
- Fix the **structure** (one state type, one world→body conversion, headless
  only at the input layer) before touching gains again.
- Confirm with the checklist in section 6, then lock it in with the four
  tests in section 7.
