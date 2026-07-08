# 36 — Goalkeeper action space from Sabana Herons

Context: multiagent probe 4.2 currently keeps the goalkeeper as native B-Human.
For an 18-hour push toward a strong B-Human-compatible GK, the fastest path is
not to learn low-level keeper motions. The Sabana side already has solid keeper
primitives; RL should learn when to hold, reposition, intercept, clear, or
distribute.

## 1. What Sabana already gives us

### Keeper positioning

Source: `SabanaHerons2026/Src/Modules/BehaviorControl/GoaliePoseProvider/GoaliePoseProvider.cpp`

`GoaliePoseProvider` computes `goaliePoseField` from the ball-to-post opening.
It places the keeper on the bisector/BOB line, clips it near the goal area, and
sets post proximity flags (`isNearLeftPost`, `isNearRightPost`) used to avoid
unsafe/pointless jumps near a post.

`PositionRoles/Goalkeeper.cpp` then returns this pose unless the game state is
an own goal kick.

Immediate RL surface:

- `stand`: hold current keeper stance.
- `walk`: field-global walk target; can reproduce or bias `goaliePoseField`.
- `observe`: scan/look toward a point when the ball is lost.

### Catch / shot interception

Source: `SkillBehaviorControl/Options/HandleGoalkeeperCatchBall.h`

The native trigger is:

- own goalkeeper
- ball seen in the last 100 ms
- ball rolling toward own goal
- ball within 3000 mm
- own-y-axis intersection time roughly 0.3 to 3.0 s

After a short prepare state, it calls `InterceptBall` with allowed methods:

- `stand`
- `walk`
- `genuflectStandDefender` when stable enough
- `jumpLeft` / `jumpRight` inside own penalty area unless near the relevant post

Source: `SkillBehaviorControl/Skills/Ball/InterceptBall.cpp`

`InterceptBall` chooses the actual motor primitive from the predicted relative
intersection y:

- within `standRadius` -> stand and cover
- within `walkRadius` -> lateral walk to intersection
- within `genuflectStandRadius` -> genuflect
- within `jumpRadius` -> keeper jump left/right

Important config: in most full-game scenarios `keeperJumpingOn = false`; in
`4v4_RL2D` and `4v4_RL3D`, it is `true`.

Immediate RL surface:

- no Python-visible `intercept` request exists yet.
- If we do not edit C++, native B-Human can still run this behavior only when
  the GK is not fully replaced by the RL `SkillRequest` provider.
- If the GK is RL-controlled, add one narrow bridge skill: `keeper_intercept`.

### Clearing and distribution

Source: `SkillBehaviorControl/Skills/Ball/GoToBallAndKick.cpp`,
`StrategyBehaviorControl/ActiveRoles/PlayBall.cpp`

For ball-playing actions, Sabana already routes `shoot`, `pass`, and `dribble`
through `PlayBall` / `GoToBallAndKick`. Goalkeepers get special close-range
obstacle handling: below about 500 mm, obstacle avoidance is disabled; above
about 600 mm it returns.

Immediate RL surface:

- `shoot`: clear if no safe teammate is available.
- `pass`: distribute to an explicit teammate number.
- `dribble`: turn/drive ball out by target angle.

## 2. Current Python bridge action space

Source: `RL/action.py`,
`SabanaHerons2026/Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp`,
`SabanaHerons2026/Src/Representations/BehaviorControl/SkillRequest.h`

The Python enum has eight actions:

| id | action | GK use |
|---:|---|---|
| 0 | `stand` | hold line / freeze-safe fallback |
| 1 | `walk` | reposition on BOB line or step to interception lane |
| 2 | `shoot` | emergency clear |
| 3 | `pass` | keeper distribution |
| 4 | `dribble` | controlled carry/turn out of danger |
| 5 | `block` | field-player block; possible keeper lane-block proxy |
| 6 | `mark` | not a primary GK action |
| 7 | `observe` | ball search / look toward team-ball estimate |

The bridge does not expose `InterceptBall` or `Dive` directly. `Dive` is lower
level and should stay hidden; `InterceptBall` is the keeper abstraction worth
exposing if we need one new action.

## 3. Recommended GK action space for 18 hours

### Minimal, no C++ change

Train with a 6-action GK subset:

```text
stand, walk, observe, shoot, pass, dribble
```

Teacher labels:

- `walk`: target = Sabana-style `goaliePoseField` approximation or a clipped
  lateral interception point.
- `stand`: when already inside tolerance and no immediate shot.
- `observe`: ball stale; target = team-ball or expected danger corridor.
- `shoot`: ball in own box and no good pass outlet.
- `pass`: ball controlled and safe teammate/outlet exists.
- `dribble`: ball controlled but pass unavailable; drive away from own goal.

This is fastest because it uses the existing ONNX/output contract and the
already smoke-tested `SkillRequest` variants.

### Small bridge extension, higher ceiling

Add exactly one action:

```text
keeper_intercept
```

C++ behavior:

- add `SkillRequest::keeperIntercept`
- map Python string `keeper_intercept` in `RLSkillProvider`
- in `SkillBehaviorControl::executeRequest()`, call `theInterceptBallSkill`
  with the same method mask as `HandleGoalkeeperCatchBall`
- keep `Dive` hidden; let `InterceptBall` choose stand/walk/genuflect/jump

Suggested Python output shape if we want compatibility with current training:

- do not mutate the existing field-player `SKILL_ORDER`
- create a GK-specific action module/checkpoint with its own manifest
- or map `keeper_intercept` onto currently unused `block` only for the GK
  provider path, but document that overload clearly

Best 18-hour tradeoff: add the explicit action only if minimal-action probes
show the keeper failing to stop rolling shots that native `InterceptBall` would
handle.

## 4. Teacher-state features we need

The current 47-dim field-player observation is not enough for a specialized GK
unless we derive extra labels from backend/debug state while collecting. The
training obs can stay compact, but the teacher should compute:

- ball field position and velocity
- whether ball is rolling toward own goal
- predicted own-y-axis / goal-line intersection
- time to intersection
- whether ball was seen recently or only team-ball is available
- current keeper pose relative to Sabana `goaliePoseField`
- controlled-ball flag near keeper feet
- safe outlet teammate id / no-outlet flag

If we add a GK observation contract later, the first-class features should be:

- normalized ball position and velocity
- normalized predicted intersection y
- normalized time-to-intersection
- one-hot/sticky danger mode: hold, shift, intercept, clear, distribute, search
- teammate outlet summary
- post proximity / penalty-area flag

## 5. First probe scenarios

Use small, deterministic SimRobot worlds before full 4v4:

1. Rolling central shot: ball starts 2500-3500 mm out, velocity toward center of
   own goal. Expected label: `keeper_intercept` if available, else `walk`/`stand`.
2. Rolling near-post shots left/right. Expected label: lateral walk or intercept;
   never jump into the post.
3. Slow ball in own box. Expected label: `shoot` clear or `pass` outlet.
4. Ball lost but team-ball valid. Expected label: `observe`/`walk` to line, not
   dead stand-freeze.
5. Back-pass / keeper distribution. Expected label: `pass` to safest field
   player when available.

Promotion metrics:

- save rate on rolling shots
- own-goal rate
- time out of danger after controlled ball
- illegal/unsafe dive count
- stand-freeze count when team-ball is valid
- successful distribution count

## 6. Practical next step

Start with a `teacher_goalkeeper_dataset.py` that generates the five probes
above and emits the minimal 6-action labels. In parallel, run one native
B-Human-GK baseline on the same worlds. If native saves shots that the minimal
RL action space cannot reproduce, implement `keeper_intercept` as the only C++
bridge addition.
