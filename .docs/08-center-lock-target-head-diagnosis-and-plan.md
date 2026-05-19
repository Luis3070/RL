# Center-Lock Diagnosis and Action Plan

## Purpose

This note documents a second major failure mode discovered during visible
checkpoint inspection:

- the policy can choose the correct discrete skill, especially `walk`,
- but still move poorly because the continuous parameter head emits targets near
  the field center,
- which makes B-Human walk toward `(0, 0)` instead of toward the ball.

This is separate from the earlier dribble-collapse issue. Both matter, but they
 affect different parts of the hybrid policy:

- dribble collapse: discrete skill-selection problem,
- center lock: continuous action-parameter problem.

The goal of this document is to describe:

1. what was observed,
2. why it is happening,
3. the practical solution for tonight's push to Stage 3,
4. the cleaner solution for future training.

## Short Diagnosis

The current hybrid policy has:

- a categorical head for skill identity,
- a Gaussian continuous head for B-Human parameters:
  - `target_x`
  - `target_y`
  - `target_theta`
  - `pass_target`

The visible evaluator showed that the policy often selects `walk` correctly, but
its `target_x` and `target_y` are very close to zero while the ball is far away.

Example pattern seen during live evaluation:

```text
policy=walk
executed=walk
target_x ~= -1 mm
target_y ~=  0 mm
ball_x   ~= 735 mm
ball_y   ~= 115 mm
robot_x  ~= -750 mm
robot_y  ~=  -50 mm
```

Operationally this means:

- the discrete choice is "walk",
- but the walk destination is near the origin,
- so B-Human walks toward the center instead of approaching the ball.

This matches the visual behavior:

- if the ball is not centered, the robot still drifts or commits toward the
  center,
- even in cases where the skill choice itself is no longer the main problem.

## Evidence

The model evaluator outputs step-by-step telemetry with:

- chosen skill,
- executed skill,
- policy target values,
- ball distance,
- ball position,
- robot position,
- per-skill probabilities.

Recent live evaluator CSVs showed:

- `policy_skill_name` was often `walk`,
- `executed_skill_name` was also `walk`,
- `target_x` and `target_y` stayed roughly in the range `[-2, 1]`,
- while `ball_x` stayed near `700-800 mm`,
- and the robot started hundreds of millimeters away from the ball.

That confirms the continuous target head is not producing meaningful field-space
destinations for `walk`.

## Root Cause

### 1. The continuous head is emitting raw field values with tiny effective exploration

The current policy head outputs parameters directly in physical action units.
For `target_x` and `target_y`, that means millimeters over a field scale of
thousands.

However, the head is initialized near zero and its standard deviation is also
small relative to the field scale. In practice this yields something like:

```text
target_x = 0 +/- very small noise
target_y = 0 +/- very small noise
```

So the policy barely explores useful destinations.

### 2. The discrete skill head learns faster than the continuous target head

The skill-selection problem is easier:

- reward can clearly reinforce `walk` vs `dribble` vs `shoot`,
- the categorical head quickly picks up those signals.

The continuous target head is harder:

- it only gets indirect reward through downstream execution quality,
- and because its exploration is tiny, it rarely discovers useful ball-directed
  targets,
- so it stays near the origin.

### 3. Dribble hid the issue for a while

`dribbleTo(theta)` depends mostly on heading, not on `target_x` and `target_y`.
While the policy was overusing `dribble`, the broken `walk` target generation was
less visible.

Once the Stage 0 reset distribution was corrected to force more real walk
approach, the target-head problem became obvious.

### 4. The current reward does not supervise target geometry directly

The reward shapes skill compatibility and approach quality, but it does not
directly teach:

- "when you choose `walk`, your `target_x/target_y` should be near the ball or
  near a useful handoff point behind the ball."

That leaves too much burden on pure PPO discovery for the continuous head.

## Why This Matters Tonight

If we do nothing, the training run can still fail even after fixing Stage 0
distance sampling:

- Stage 0 may now encourage `walk`,
- but the `walk` parameter head still points near center,
- so the agent can choose the right skill and still fail the task.

That means:

- reward and gates will interpret it as poor approach behavior,
- the agent may oscillate or stall,
- promotion to later stages becomes unlikely within tonight's time budget.

For tonight's goal, we need a fix that improves execution immediately without
throwing away the new hybrid policy work.

## Solution for Tonight

### Principle

Do not rely on the policy to discover good continuous `walk` and `dribble`
targets from scratch tonight.

Instead:

- let the policy continue learning the discrete skill identity,
- repair or replace the continuous parameters at execution time with B-Human-safe
  targets derived from current ball geometry,
- keep telemetry that distinguishes:
  - what the policy proposed,
  - what was actually executed.

This gives us a path to Stage 3 tonight because the discrete policy can still
learn the right skill chain while execution becomes geometrically sane.

### Tonight's execution policy

For `walk`:

- executed `target_x` should be the ball x-coordinate or a handoff approach
  target behind the ball,
- executed `target_y` should be the ball y-coordinate,
- executed `target_theta` should face the ball or the handoff line.

Recommended version:

```text
walk target = approach point behind the ball
```

That is more useful than simply walking to the ball center because it aligns the
robot for takeover.

For `dribble`:

- executed `target_theta` should be goal-directed,
- executed `target_x/target_y` are not the main control signal for B-Human
  dribble and can be replaced with safe fallback values.

For `shoot`:

- no meaningful `target_x/target_y` learning is required tonight,
- only the discrete decision and timing matter.

### Implementation strategy for tonight

Add an execution-side target repair layer in the environment:

1. Keep the policy discrete skill output untouched.
2. For active skills, detect whether continuous targets are implausible or
   should be overridden for tonight.
3. Replace executed targets with heuristic safe targets derived from:
   - ball position,
   - robot position,
   - goal heading,
   - handoff offset.
4. Log both:
   - raw policy targets,
   - executed targets after repair.

This should be treated as a temporary production bridge, not as the final
learning design.

### Telemetry required tonight

To keep the run diagnosable, telemetry should clearly expose:

- `policy_target_x`
- `policy_target_y`
- `policy_target_theta`
- `executed_target_x`
- `executed_target_y`
- `executed_target_theta`
- whether target repair was applied
- reason for repair if available

That lets us distinguish:

- skill-choice failures,
- target-head failures,
- execution-bridge dependence.

### What success looks like tonight

A successful overnight run does not require the continuous head to be solved.
It requires:

- `walk` chosen when far from ball,
- ball-directed approach instead of center-lock,
- `dribble` chosen near takeover,
- `shoot` appearing in Stage 1+ when geometrically valid,
- stable progression to Stages 1, 2, and ideally 3.

If this requires target repair, that is acceptable for tonight.

## Future Training Solution

Tonight's repair is pragmatic. It should not become the final architecture.

The long-term fix is to make the continuous head learnable on the correct scale
and around the correct reference geometry.

### Option A: Normalized continuous outputs

Instead of predicting raw millimeters directly, predict normalized values such as:

- `target_x_norm in [-1, 1]`
- `target_y_norm in [-1, 1]`
- `target_theta_norm in [-1, 1]`

Then map them to physical units before execution.

Benefits:

- exploration scale becomes meaningful,
- PPO gradients are better conditioned,
- initialization near zero is no longer catastrophic.

### Option B: Residual targets around a heuristic anchor

This is the preferred long-term solution.

For each skill, define a heuristic anchor target:

- `walk` anchor:
  - approach point behind the ball,
  - orientation toward the ball or takeover line.
- `dribble` anchor:
  - heading toward the goal.
- `shoot` anchor:
  - goal-facing shot context, mostly timing-driven.

Then train the policy to output residuals:

```text
executed_target = heuristic_anchor + learned_residual
```

Benefits:

- exploration starts from useful behavior,
- PPO only learns corrections,
- continuous learning becomes much easier,
- center-lock disappears because the origin is no longer the reference point.

### Option C: Skill-specific parameter heads

The current parameter head is shared across all skills. That means:

- `walk`, `dribble`, and `shoot` all push on the same continuous outputs,
- but they do not use parameters in the same way.

A cleaner design is:

- skill-specific parameterization,
- or at least skill-conditioned parameter decoding.

Examples:

- `walk`:
  - `(target_x, target_y, target_theta)`
- `dribble`:
  - `(target_theta)` primarily
- `shoot`:
  - no real spatial parameter needed for now
- `pass`:
  - `(pass_target)`

This avoids wasting learning capacity on parameters a skill does not actually
use.

### Option D: Reward shaping for target quality

Even with normalized or residual targets, the reward should explicitly reinforce
good target geometry.

For `walk`, reward should increase when:

- the target lies near the ball approach corridor,
- the target is behind the ball rather than at the field center,
- the target reduces approach error.

This can be done as auxiliary shaping, not necessarily as a hard gate.

## Recommended Roadmap

### Tonight

1. Keep the new stage reset curriculum.
2. Add execution-side target repair for `walk` and `dribble`.
3. Log target repair telemetry.
4. Use the evaluator to confirm:
   - walk is ball-directed,
   - dribble begins near takeover,
   - the center-lock disappears.
5. Run the overnight curriculum push toward Stage 3.

### Next training cycle

1. Replace raw field-space parameter prediction with normalized or residual
   prediction.
2. Prefer heuristic-anchor plus residuals.
3. Add explicit target-quality shaping for `walk`.
4. Consider skill-specific parameter heads.
5. Reduce or remove execution-side repair only after telemetry shows the policy
   produces sane targets on its own.

## Final Recommendation

For tonight, the correct posture is pragmatic:

- discrete skill learning remains the real policy problem,
- continuous target generation is currently under-trained and poorly scaled,
- execution-side repair is the fastest safe way to unblock progress to Stage 3.

For future training, the correct posture is architectural:

- do not keep asking PPO to discover raw millimeter targets from near-zero
  Gaussian outputs,
- move to normalized or residual target prediction anchored in B-Human geometry.

That gives us a clean separation:

- tonight: make the run succeed,
- later: make the policy truly own the continuous parameters.

## Additional Tonight Work: Discrete Stability

After the center-lock bridge was introduced, a second practical issue became
clear in the evaluator:

- the agent does not walk persistently,
- it often alternates between `walk`, `stand`, and premature `shoot`,
- `takeover_success_rate` and `clean_takeover_rate` become volatile because the
  discrete policy is still unstable even when execution targets are repaired.

The practical interpretation is:

- the continuous target problem is bridged well enough for tonight,
- the next bottleneck is the categorical policy behavior in walk-zone.

### Symptoms

Observed in evaluator and eval telemetry:

- the agent walks a few steps, then stops,
- `stand` appears in walk-zone often enough to break production continuity,
- `shoot` appears before `shoot_zone`,
- `clean_takeover_rate` drops even when takeover is technically reached,
- short eval windows make volatility appear as jumps between `0.0`, `0.5`, and
  `1.0`.

### Tonight's discrete-policy solution

For tonight, add shaping and defaults that push the policy toward a more
production-like sequence:

1. Penalize idle `stand` in walk-zone.
2. Reward persistent `walk` while the ball is trusted and still far.
3. Penalize `shoot` more strongly outside `shoot_zone`, especially in Stage 1.
4. Keep target repair active for `walk` and `dribble`.
5. Increase evaluation episodes above `2` so volatility is less misleading.

### Future solution

For future training, we should not depend only on reward shaping for continuity.
The cleaner path is to combine:

- better temporal persistence incentives,
- stronger stage-specific skill priors,
- possibly explicit action smoothing or persistence-aware policy design.

That is a later improvement. For tonight, reward shaping plus better eval
stability is the right move.
