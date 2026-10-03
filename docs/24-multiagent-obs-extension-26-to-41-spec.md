# 24 — Multi-agent observation extension: 26 → 41 (spec + migration + transplant)

## § 0 Purpose, audience, status

This document specifies the extension of the RL agent's observation space from the current 26 dimensions to 41 dimensions, in preparation for multi-agent coordination work. It is split into three audiences:

1. **The policy-side migration owner.** Responsible for taking the current 26-dim production PPO checkpoint and the training stack (`train_production.py`, `behavioral_evaluator.py`, `production_curriculum.py`, BC dataset) and making them functional under `OBS_SIZE = 41` while preserving the validated single-agent behavior bit-for-bit (within seed noise). This is a **behavior-preservation migration**, not a learning migration. See § 7 and § 8.
2. **The BHuman / pybh transplant owner.** Responsible for surfacing the new BHuman fields (`TeammateData`, `GlobalTeamBallModel`, `FieldDimensions`) through the pybh exporter into the `BackendObservation` payload. See § 6.
3. **The multi-agent work owners.** Responsible for the downstream coordination workstreams that depend on this substrate (role-arbitration wrapper, multi-agent retrain, graph-coordination follow-ups). See § 11 for what becomes possible after this migration lands.

**Status:** spec frozen pending the policy-side migration. The shape of the new obs (15 new dims, all BHuman-sourced, zero-pad safe) is finalized. The BHuman wiring on the C++ side is mechanical (no new percept providers). The migration on the Python side is well-scoped (~3 file edits + a sanity test pass).

This document deliberately does not contain a calendar or a deadline. The migration is gated on the test plan in § 8 passing; the multi-agent work that consumes the substrate is gated on the migration landing.

---

## § 1 Scope and non-goals

### In scope

- Specifying the 41-dim obs layout, with index, name, BHuman source, normalization, and zero-value semantics for every new dimension.
- Specifying the BHuman → pybh → Python data flow for each new dimension.
- Specifying the migration path for the existing 26-dim PPO checkpoint to function under `OBS_SIZE = 41` without behavioral regression.
- Specifying the sanity test plan that must pass before any multi-agent retrain is started.

### Out of scope

- Multi-agent training itself (reward shaping, role emergence, BC dataset re-collection on the 41-dim obs, curriculum design for coordination). These follow this migration.
- Opponent enumeration. BHuman has no opponent-broadcast channel; opponents stay as local-perception nearest-K via the existing indices 17–22.
- Role-conditioning the policy. Roles are an output-side emergent property, not an input-side teacher signal. See § 5.
- Set-play behavior, score-state awareness, time-remaining strategy. Separate workstream.
- Changes to `gate.py`, `gate_skill_mask_from_obs`, `GateConfig`, or any code path that touches the trained corridor. The Design A invariant (mask reads exclusively from indices 23/24/25) is preserved unchanged.

### What this document is NOT

- It is not a multi-agent training plan. It is the substrate that a multi-agent training plan would consume.
- It is not a coordination-algorithm spec. Coordination is intended to emerge from reward + the substrate; no role bits, no precomputed coordination signals.
- It is not a transplant plan for the policy onto NAOv6. That work is covered in `docs/23-plan-transplante-ppo-naov6-hsl2026.md`. The obs extension is compatible with that transplant (the new fields are all already present in BHuman state), but the transplant decisions are independent of this document.

---

## § 2 Design philosophy: four hard constraints

Every dimension in the extension satisfies all four:

1. **The 26-dim corridor is preserved exactly.** Indices 0–25 do not change. Normalization constants for the original 26 dims do not change. The mask still reads exclusively from indices 23/24/25. The trained policy's input distribution on those indices is untouched.

2. **All new dimensions are zero-pad safe.** The existing `_load_state_dict_obs_padded` utility (in `train.py`) initializes input-layer weights to zero for new input columns. Day-0 outputs of the policy must be identical to the current 26-dim policy when the new dimensions are zero. For this to hold, every new dimension's zero value must semantically match the implicit single-agent default ("no teammate," "no team ball," "team disagrees"). See the per-block tables in § 3.

3. **All new data comes from BHuman state that already exists.** No new percept providers, no new C++ representations, no new SPL message fields. The pybh exporter forwards fields that BHuman already computes (`TeammateData[]`, `GlobalTeamBallModel`, `FieldDimensions`); the Python side reads them.

4. **No role tags, no opponent enumeration, no precomputed coordination answers.** Roles are an emergent output property; opponents are not broadcast on the team channel; coordination signals (which teammate to support, when to pass, when to back off) must be derivable by the policy from positions + behavior + reward, not handed to it as input bits. This is the same principle as the existing single-agent obs: the policy gets the geometric and perceptual substrate, not the answer.

These four are not negotiable. Any future extension proposal that violates one of them — adding a role one-hot, adding opponent identities, adding precomputed "i am the striker" bits — needs to be evaluated against this constraint set before landing.

---

## § 3 The 41-dim specification

### § 3.1 Block A — Original single-agent obs (indices 0–25, 26 dims)

Unchanged. Source for each field is `BackendObservation` (built in `pybh_backend.py:258`), encoded by `_encode_obs` in `environment.py:697-733`. The mask reads exclusively from indices 23, 24, 25.

| Idx | Name | Block role |
|---|---|---|
| 0–2 | `robot_x`, `robot_y`, `robot_theta` | Self pose, field-normalized |
| 3–4 | `ball_rel_x`, `ball_rel_y` | Own ball belief, robot frame |
| 5–6 | `ball_end_rel_x`, `ball_end_rel_y` | Predicted ball stop position |
| 7–8 | `ball_vel_x`, `ball_vel_y` | Own ball velocity |
| 9 | `time_since_ball_seen` | Own perception trust |
| 10 | `time_since_ball_disappeared` | Own perception trust |
| 11 | `ball_seen_percentage` | Own perception trust |
| 12 | `ball_consistent_with_game_state` | Game-state cross-check bit |
| 13 | `can_score_now` | Discrete shoot signal |
| 14 | `shot_quality_no_obstacles` | Geometric shoot quality |
| 15 | `shot_opening_with_obstacles` | Obstacle-aware shot opening (gate reads this) |
| 16 | `pass_options_count` | Pass-availability count (vestigial today) |
| 17 | `nearest_teammate_dist` | Nearest teammate, omnidirectional |
| 18 | `nearest_opponent_dist` | Nearest opponent, omnidirectional |
| 19 | `nearest_uncertain_obstacle_dist` | Nearest unclassified obstacle |
| 20 | `nearest_teammate_front_dist` | Nearest teammate, front cone |
| 21 | `nearest_opponent_front_dist` | Nearest opponent, front cone |
| 22 | `nearest_uncertain_front_dist` | Nearest unclassified obstacle, front cone |
| 23 | `shoot_armed` | **Mask reads this** — gate state, Design A |
| 24 | `dribble_armed` | **Mask reads this** — gate state, Design A |
| 25 | `shoot_arm_progress` | Debounce fill toward arming |

### § 3.2 Block B — Teammate spatial state (indices 26–31, 6 dims)

Teammate positions in the agent's own robot frame, sorted by distance to own robot, stale-zeroed. The sort is by `‖teammate.position − own.position‖`, ascending. Teammates with message age > 1000 ms are excluded from the sort; their slot zeros out.

| Idx | Name | BHuman source | Normalization | Zero-value semantics |
|---|---|---|---|---|
| 26 | `nearest_teammate_x_rel` | `TeammateData[sort[0]].theRobotPose.translation.x` transformed to own robot frame | `/ACTION.x_range`, clip to [-1, 1] | "No fresh teammate in slot 0" — matches single-agent (no teammates) |
| 27 | `nearest_teammate_y_rel` | same, y component | `/ACTION.y_range`, clip to [-1, 1] | same |
| 28 | `second_teammate_x_rel` | `TeammateData[sort[1]]`, x in own frame | `/ACTION.x_range`, clip to [-1, 1] | "No fresh teammate in slot 1" |
| 29 | `second_teammate_y_rel` | same, y | `/ACTION.y_range`, clip to [-1, 1] | same |
| 30 | `third_teammate_x_rel` | `TeammateData[sort[2]]`, x in own frame (typically the GK) | `/ACTION.x_range`, clip to [-1, 1] | "No fresh teammate in slot 2" |
| 31 | `third_teammate_y_rel` | same, y | `/ACTION.y_range`, clip to [-1, 1] | same |

**Why sort by distance to me (not by player number or by ball distance):** distance-to-me is stable per step (small position changes do not swap slots), preserves the intuition "closest teammate matters most," and composes with the existing `nearest_teammate_dist` at idx 17 which is already distance-sorted. Sorting by player number ties identity to slot, which means a different teammate in the same slot teaches the policy contradictory things; sorting by ball-distance creates frequent slot swaps when teammates are nearly equidistant to the ball, which is a common geometry in play.

**Stale handling:** stale → excluded from sort, slot zeroed. This matches the BHuman convention of trusting only fresh team data. The zero-pad single-agent default ("no teammates present") is preserved when the team channel is dead.

### § 3.3 Block C — Team-merged ball state (indices 32–34, 3 dims)

Where the *team* thinks the ball is, separately from the agent's own belief at indices 3–4. Gated by Block E (index 38) so that when the team has no consensus, this block falls into the policy's noise floor.

| Idx | Name | BHuman source | Normalization | Zero-value semantics |
|---|---|---|---|---|
| 32 | `global_team_ball_x_rel` | `GlobalTeamBallModel.position.x` transformed to own robot frame | `/ACTION.x_range`, clip to [-1, 1] | "Team has no ball belief" — fall back to own belief |
| 33 | `global_team_ball_y_rel` | same, y | `/ACTION.y_range`, clip to [-1, 1] | same |
| 34 | `time_since_team_ball_update` | now − `GlobalTeamBallModel.timeWhenLastSeen` (ms) | `/TIME_NORM_MS`, clip to [0, 1] | "Just now / no team ball" — the policy ignores via bit 38 |

### § 3.4 Block D — Per-teammate behavioral state (indices 35–37, 3 dims)

Per-teammate "engaging the ball" bit, slot-aligned with Block B. Reads what each teammate is **doing**, not just where they are. This is the highest-leverage coordination signal — the policy can distinguish "nearest teammate is on it, I should back off" from "no teammate is engaging, the ball is contested" — and it is **not derivable from positions alone**.

| Idx | Name | BHuman source | Normalization | Zero-value semantics |
|---|---|---|---|---|
| 35 | `nearest_teammate_engaging_ball` | `TeammateData[sort[0]].theBehaviorStatus.activity` ∈ {`dribbling`, `kicking`, `goingToBall` within takeover distance} | bool → {0, 1} | "Nearest teammate isn't on the ball" — single-agent default |
| 36 | `second_teammate_engaging_ball` | same, slot 1 | bool → {0, 1} | same |
| 37 | `third_teammate_engaging_ball` | same, slot 2 | bool → {0, 1} | same |

**Note on "engaging" semantics:** the mapping from `BehaviorStatus.activity` to the bit is a Python-side derivation in the obs encoder. The mapping for the migration should be:

```
engaging = activity in {dribbling, kicking}
          or (activity == goingToBall
              and ‖teammate.position − team_ball.position‖ < ENGAGING_DISTANCE_MM)
```

with `ENGAGING_DISTANCE_MM = 700.0` as a starting value (close-enough that the teammate is committed to taking the ball). The threshold is a single Python constant in the encoder; it can be tuned later.

**Stale handling:** stale → bit is 0 (matches "not engaging" default). This composes with the stale-zero in Block B: stale teammates have zero position AND zero engagement bit, which the policy reads as "this slot is empty."

### § 3.5 Block E — Team-ball consensus (index 38, 1 dim)

A single bit that gates whether the policy should trust Block C (the team-merged ball coords). When this bit is 0, the policy treats Block C as noise; when it is 1, the policy uses the team ball coords as a second opinion to its own belief.

| Idx | Name | BHuman source | Normalization | Zero-value semantics |
|---|---|---|---|---|
| 38 | `team_ball_agrees_with_mine` | `‖own.BallModel.position − GlobalTeamBallModel.position‖ < 800 mm` AND `GlobalTeamBallModel.isValid` | bool → {0, 1} | "Team disagrees / no team ball" — fall back to own belief |

**Critical zero-pad property:** Block C (32–34) becomes meaningful only when Block E (38) is 1. At zero-pad warm-start, bit 38 = 0 forces Block C into the noise floor for the new policy. This is what makes adding Block C zero-pad safe: even if 32–34 are populated immediately by the transplant work, the policy ignores them until bit 38 says otherwise.

The 800 mm distance threshold is consistent with the existing gate's distance bands (e.g., `dribble_enter_ball_mm = 450 mm`, `dribble_exit_ball_mm = 600 mm`). It is the smallest distance at which "the team agrees with my ball belief" is a useful signal; below that, own belief and team belief are interchangeable.

### § 3.6 Block F — Opponent goal geometry (indices 39–40, 2 dims)

Explicit vector from the agent's robot frame to the center of the opponent goal. The current 26-dim policy can implicitly infer this from `robot_x` + `robot_theta` + the static `GOAL_X` constant, but making it explicit is a strict simplification of the policy's reasoning path and is especially valuable for multi-agent (teammate-to-goal comparisons become trivial geometry instead of multi-layer inference).

| Idx | Name | BHuman source | Normalization | Zero-value semantics |
|---|---|---|---|---|
| 39 | `opponent_goal_x_rel` | `theFieldDimensions.xPosOpponentGroundLine` (constant, ≈ +4500 mm) transformed to own robot frame | `/ACTION.x_range`, clip to [-1, 1] | "Goal at my position" — degenerate but never literal in practice |
| 40 | `opponent_goal_y_rel` | center of opponent goal: `(theFieldDimensions.yPosLeftGoal + theFieldDimensions.yPosRightGoal) / 2` (≈ 0), transformed to own robot frame | `/ACTION.y_range`, clip to [-1, 1] | same |

**Always populated.** Goal coordinates are field constants; this block has no stale-zero semantics. The zero-pad warm-start still works because the input weights at indices 39–40 start at zero, so the policy's day-0 behavior is unchanged.

### § 3.7 Summary

| Block | Indices | Dims | Purpose | BHuman source |
|---|---|---|---|---|
| A — Single-agent obs | 0–25 | 26 | Unchanged corridor | (existing) |
| B — Teammate positions | 26–31 | 6 | Where teammates are | `TeammateData[].theRobotPose` |
| C — Team-merged ball | 32–34 | 3 | Where team thinks ball is | `GlobalTeamBallModel` |
| D — Teammate engagement | 35–37 | 3 | What teammates are doing | `TeammateData[].theBehaviorStatus` |
| E — Ball consensus | 38 | 1 | When to trust team ball | derived from A's idx 3–4 + C's idx 32–33 |
| F — Opponent goal | 39–40 | 2 | Field reference | `theFieldDimensions` (constants) |
| **Total** | **0–40** | **41** | | |

---

## § 4 What the policy can reason about under the 41-dim obs

The substrate enables the policy to answer the following coordination questions from inputs alone (no precomputed answers):

| Question | Answered by indices |
|---|---|
| Where are my teammates? | Block B (26–31) |
| Is any teammate actively on the ball? | Block D (35–37) — OR of the three bits |
| Which teammate is closest to me? | Slot 0 of Block B (26–27) |
| Does my team agree on the ball location? | Block E (38) |
| If the team agrees, where does the team think the ball is? | Block C (32–34), gated by E |
| Am I positioned to score? | Block F (39–40) + own pose (0–2) + own ball (3–4) |
| Should I pass instead of shoot? | Teammate positions (B) + opponent goal (F) — policy computes teammate-to-goal vectors and compares |
| Is a teammate closer to the ball than I am? | Block B + own ball (3–4) — policy computes distances |
| Is my own ball belief reliable? | Block E (38) — disagreement with team signals own-perception fragility |

What the policy cannot do (and is not expected to):

- Identify specific teammates (no player numbers in obs, sorted by distance).
- Enumerate opponents (only nearest-K via 17–22).
- Read teammate intent beyond engagement (no "they plan to pass to me" bit).
- React to set plays as a category (the policy sees the field state, not the rule context).

These are deliberate scope limits, not missing features. They follow from the four design constraints in § 2.

---

## § 5 What is explicitly NOT in the obs (and why)

Each excluded candidate was considered and rejected for the reason listed.

| Candidate | Why excluded |
|---|---|
| Own `PlayerRole.role` one-hot | Hardcodes the answer the multi-agent RL is supposed to discover. The policy would shortcut into "if striker bit on, use shoot residual" and never learn emergent role assignment. |
| Per-teammate `PlayerRole.role` one-hot | Same reason. Lets the policy defer to BHuman's role assignment instead of learning to coordinate. |
| Per-teammate ball belief position (`TeammateData[].theBallModel.estimate.position`) | Marginal over Block E (38). Six dims for one extra fact (which teammate disagrees most) is poor value-per-dim. Block E summarizes the consensus question into a single bit. |
| Per-teammate message age (3 dims) | Already implicit: stale teammates have their position and engagement zeroed via stale-zero handling, so freshness is encoded by "this slot is zero." Adding explicit age dims is double-coverage. |
| Aggregate `teammates_engaging_count` (1 dim) | Strictly less informative than per-teammate bits 35–37. If only one dim were available, this would be it — but the per-teammate version preserves which slot is engaging, which composes with positions in Block B. |
| `GameInfo.secondaryState` (set-play one-hot, 4–5 dims) | Set-play behavior is a separate workstream. Adding now puts dead inputs in the obs. Current single-agent training does not expose these. |
| `GlobalTeamBallModel.velocity` (2 dims) | Policy already has own ball velocity at 7–8. Team velocity rarely differs unless own perception is dead, in which case Block E (38) already gates trust. Marginal value at two dims. |
| Score differential (`OwnTeamInfo.score − OpponentTeamInfo.score`) | Doesn't change frame-to-frame behavior in the policy's regime. Match-strategic, not coordination-tactical. |
| Time remaining | Same reason. |
| Opponent enumeration | Not broadcast on the SPL team channel. BHuman only knows opponents via local perception (already in obs as 18, 21). Adding more would require a percept-level workstream. |
| Pass-target positions (`pass_target_x_rel`, `pass_target_y_rel`) | Precomputed coordination answer. The policy can derive a pass target from teammate positions (B) + opponent goal (F) once retrained. Giving it the answer is teacher-supervision. |
| RL gate state of teammates (`teammate.shoot_armed`, `teammate.dribble_armed`) | Would require adding new SPL message fields and broadcasting Python-side state. `BehaviorStatus` already proxies this acceptably. Defer until empirically warranted. |
| Per-teammate intended target position | Lossy and high-bandwidth. Same emergent-vs-supervised concern. |
| Vector to own goal | Field-relative; the policy can compute from own pose + the `GOAL_X` constant on the opponent side and symmetry. If empirically needed, can be added as v2. |
| Frame history / temporal stack | Bigger architectural change. The current obs encodes temporal context via `time_since_ball_seen`, `ball_vel`, `shoot_arm_progress`. Stacking frames is a separate research direction. |
| Proprioception (joint angles, foot contacts) | BHuman handles this internally; the RL stack operates above it. Adding would require a different policy architecture. |

---

## § 6 BHuman → Python data flow (transplant guidance)

This section is for the BHuman / pybh transplant owner. The work is mechanical: forward existing BHuman state into the obs payload that `pybh_backend.py` reads.

### § 6.1 BHuman representations to read

All of these are already part of the BHuman 2026 tree. No new providers required.

| Representation | Fields used | Notes |
|---|---|---|
| `TeammateData` | `[i].theRobotPose.translation.x/y`, `[i].theRobotPose.rotation`, `[i].theFrameInfo.time`, `[i].theBehaviorStatus.activity`, `[i].thePlayerRole` (if needed for filtering) | Array indexed by teammate slot; iterate present slots |
| `GlobalTeamBallModel` | `position.x`, `position.y`, `timeWhenLastSeen`, `isValid` | Team-merged ball estimate; updated by BHuman's team fusion code |
| `FieldDimensions` | `xPosOpponentGroundLine`, `yPosLeftGoal`, `yPosRightGoal` | Static; can be read once at agent boot |
| `FrameInfo` | `time` | For computing message age deltas |

### § 6.2 pybh exporter changes

The pybh exporter (the C++ side that builds the obs payload that `pybh.rl_get_obs()` returns) needs to add the following keys to the payload dictionary. The Python side (`BackendObservation.from_payload`) will read them.

```
teammate_x_mm:        list[float]   # length matches present teammates, in field frame
teammate_y_mm:        list[float]   # same length
teammate_theta:       list[float]   # rotation, optional (kept for future)
teammate_age_ms:      list[float]   # message age per teammate
teammate_activity:    list[int]     # BehaviorStatus.activity enum value per teammate
global_team_ball_x_mm:    float    # team-merged ball, field frame
global_team_ball_y_mm:    float
global_team_ball_age_ms:  float    # = frame.time - GlobalTeamBallModel.timeWhenLastSeen
global_team_ball_valid:   bool     # GlobalTeamBallModel.isValid
goal_opponent_x_mm:       float    # = FieldDimensions.xPosOpponentGroundLine
goal_opponent_y_mm:       float    # = (FieldDimensions.yPosLeftGoal + yPosRightGoal) / 2
```

All distances in millimeters, all angles in radians, all times in milliseconds. This matches the convention of the existing `BackendObservation` fields. The Python side handles transforms to robot frame and normalization.

### § 6.3 Frame-to-robot transform

The Python encoder transforms `teammate_x_mm, teammate_y_mm` from the field frame to the agent's own robot frame using its own pose at indices 0–2. The same transform applies to `global_team_ball_x_mm, global_team_ball_y_mm` and to the opponent goal coordinates. The transform is:

```
dx = field_x - own_robot_x
dy = field_y - own_robot_y
rel_x = dx * cos(-own_robot_theta) - dy * sin(-own_robot_theta)
rel_y = dx * sin(-own_robot_theta) + dy * cos(-own_robot_theta)
```

This is the same convention as the existing `ball_rel_x, ball_rel_y` (indices 3–4) in `BackendObservation`. The encoder should reuse the existing transform helper if one is present.

### § 6.4 Sort and stale handling (Python-side)

In the obs encoder (`_encode_obs` in `environment.py:697`):

1. Read `teammate_*` lists from `BackendObservation`.
2. Drop any teammate with `teammate_age_ms[i] > TEAMMATE_STALE_THRESHOLD_MS` (suggested: `1000.0`).
3. Compute distance to own robot for each surviving teammate.
4. Sort by ascending distance; pad to length 3 with zero-position sentinels.
5. For each slot, compute the engagement bit:
   - `bit = 1` if `teammate_activity` is in the engaging set ({dribbling, kicking}, or {goingToBall} with `‖teammate − global_team_ball‖ < ENGAGING_DISTANCE_MM`).
   - `bit = 0` otherwise (including the stale-padded slots).
6. Compute Block E (38) from own ball (3–4) + team ball (32–33 before normalization) + `global_team_ball_valid`.

### § 6.5 What does NOT need to change in BHuman C++

- No new representation providers.
- No new SPL TeamMessage fields.
- No new behavior modules.
- No changes to `theBehaviorStatus` enum values; the engagement mapping happens entirely Python-side.
- No changes to `theBallModel` propagation, the BHuman gate, or any skill executor.
- No interop concerns with the BHuman GK: it broadcasts `BehaviorStatus` already, so it shows up correctly in the RL agents' `TeammateData[]`.

---

## § 7 Migration: 26-dim PPO → 41-dim, behavior-preserving

This section is for the policy-side migration owner. The goal is **functional preservation**: under `OBS_SIZE = 41`, the existing 26-dim production checkpoint must behave identically (within seed noise) to the 26-dim baseline on `behavioral_evaluator`. This is the gate that lets multi-agent work start; if this migration introduces any regression on the validated single-agent metrics, the substrate is not safe for multi-agent retrain.

### § 7.1 Migration principle

This is a **substrate migration, not a learning migration.** The policy weights are not retrained. Behavior must be byte-for-byte identical on day 0 (modulo seed noise), and this is achievable because:

- Indices 0–25 are unchanged in semantics, source, and normalization.
- The mask reads exclusively from 23/24/25, which are unchanged.
- The new indices 26–40 are loaded into a freshly-zero input column (via `_load_state_dict_obs_padded` in `train.py`), so the new dims contribute nothing to policy outputs until retrained.
- The new indices' zero values match the implicit single-agent assumptions ("no teammates," "no team ball," etc.) so even if they are populated immediately, the policy still ignores them on day 0 because the input weights are zero.

### § 7.2 What the migration owner needs to change

The migration is contained to the Python stack. The BHuman/pybh side is a parallel workstream (§ 6) and does not need to land before this migration is testable — the new dims can initially be populated with zeros and switched to real BHuman values once the C++ side is ready.

**Files to edit:**

1. **`observation.py`** — add 15 new `ObsIndex` entries (26–40) with the names from § 3. The new `OBS_SIZE` becomes 41 automatically since it's derived from the enum.

2. **`pybh_backend.py:258`** (`BackendObservation`) — add fields for the new payload:
   ```
   teammate_poses: tuple[tuple[float, float, float], ...]   # (x_mm, y_mm, theta_rad) per fresh teammate, sort applied Python-side
   teammate_ages_ms: tuple[float, ...]
   teammate_activities: tuple[int, ...]
   global_team_ball_x_mm: float
   global_team_ball_y_mm: float
   global_team_ball_age_ms: float
   global_team_ball_valid: bool
   goal_opponent_x_mm: float
   goal_opponent_y_mm: float
   ```
   For the migration's first pass these can default to zeros / empty tuples / `False`; § 6 lands the real values from BHuman later.

3. **`pybh_backend.py`** (`BackendObservation.from_payload`) — parse the new keys from the payload dict. During the migration period, missing keys should default to the same zero values, so older C++ exporters that haven't been updated yet still produce a working obs (just with all new dims at zero).

4. **`environment.py:697`** (`_encode_obs`) — append the 15 normalized features in order:

   ```python
   # Block B — teammate positions (sorted, stale-zeroed)
   fresh = [
       (px, py)
       for (px, py, _theta), age in zip(obs.teammate_poses, obs.teammate_ages_ms)
       if age <= TEAMMATE_STALE_THRESHOLD_MS
   ]
   fresh.sort(key=lambda p: (p[0] - obs.robot_x) ** 2 + (p[1] - obs.robot_y) ** 2)
   slot_positions = [fresh[i] if i < len(fresh) else None for i in range(3)]
   teammate_features = []
   for slot in slot_positions:
       if slot is None:
           teammate_features.extend([0.0, 0.0])
       else:
           rx, ry = transform_to_robot_frame(slot, obs.robot_x, obs.robot_y, obs.robot_theta)
           teammate_features.extend([
               float(np.clip(rx / ACTION.x_range, -1.0, 1.0)),
               float(np.clip(ry / ACTION.y_range, -1.0, 1.0)),
           ])

   # Block C — team-merged ball
   tbx, tby = transform_to_robot_frame(
       (obs.global_team_ball_x_mm, obs.global_team_ball_y_mm),
       obs.robot_x, obs.robot_y, obs.robot_theta,
   )
   team_ball_features = [
       float(np.clip(tbx / ACTION.x_range, -1.0, 1.0)),
       float(np.clip(tby / ACTION.y_range, -1.0, 1.0)),
       float(np.clip(obs.global_team_ball_age_ms / TIME_NORM_MS, 0.0, 1.0)),
   ]

   # Block D — per-teammate engagement, slot-aligned with Block B
   engagement_features = []
   for i in range(3):
       if slot_positions[i] is None:
           engagement_features.append(0.0)
       else:
           # Look up the activity for the teammate whose pose ended up in this sorted slot
           activity = lookup_activity_for_slot(slot_positions[i], obs)
           engaging = activity_is_engaging(activity, slot_positions[i], obs.global_team_ball_x_mm, obs.global_team_ball_y_mm)
           engagement_features.append(1.0 if engaging else 0.0)

   # Block E — team-ball consensus
   own_ball_field = (obs.robot_x + obs.ball_rel_x, obs.robot_y + obs.ball_rel_y)
   team_ball_field = (obs.global_team_ball_x_mm, obs.global_team_ball_y_mm)
   distance_to_team_ball = math.hypot(own_ball_field[0] - team_ball_field[0], own_ball_field[1] - team_ball_field[1])
   consensus = 1.0 if (obs.global_team_ball_valid and distance_to_team_ball < 800.0) else 0.0

   # Block F — opponent goal (always populated)
   gx, gy = transform_to_robot_frame(
       (obs.goal_opponent_x_mm, obs.goal_opponent_y_mm),
       obs.robot_x, obs.robot_y, obs.robot_theta,
   )
   goal_features = [
       float(np.clip(gx / ACTION.x_range, -1.0, 1.0)),
       float(np.clip(gy / ACTION.y_range, -1.0, 1.0)),
   ]
   ```

   The pseudocode above shows the structure; the exact implementation should match the project's existing coding style (helper functions, etc.).

5. **`train.py`** — no changes to `gate_skill_mask_from_obs`. The function continues to read indices 23, 24, 25 only. The function should pass a unit test that verifies it does not index past 25.

6. **`train.py:_load_state_dict_obs_padded`** — already handles obs growth via zero-padding of new input columns. No changes needed, but verify the function is invoked when loading the 26-dim checkpoint under `OBS_SIZE = 41`.

### § 7.3 What about the BC dataset?

The existing BC dataset (`runs/fine_tuning_production_dataset/dataset_...`) is shape-locked at `observations[N, 26]`. The `BCReplayRegularizer` (`train.py:1340`) loads it raw into the policy. For the behavior-preservation migration phase:

- **If BC replay is used during the migration smoke**, the dataset must be either (a) re-collected on the 41-dim obs (long, requires the BHuman wiring to land first), or (b) zero-padded at load time so the dataset's 26 columns map to the first 26 indices and the new 26–40 are filled with zeros for those samples.
- **Recommended for the migration phase**: option (b). It is a 5-line change in the dataset loader and matches the policy-side zero-pad guarantee. Train BC replay on zero-padded dataset → policy learns nothing about the new dims (correct, since we want behavior preservation, not learning).
- **For the eventual multi-agent retrain**: the dataset must be re-collected with the new 15 dims actually populated. That is downstream of this migration, not part of it.

### § 7.4 What about the KL anchor?

The frozen KL reference clone in `train.py:clone_frozen_policy_reference` is loaded from the same checkpoint as the warm-start policy. Under the migration, both share the same `_load_state_dict_obs_padded` path, so the KL reference is also 41-dim with zero new-column weights. This means:

- Day-0 KL between policy and reference is zero on the new dims (both ignore them).
- During eventual retrain, KL pulls policy back toward the reference's behavior on the original 26 dims, which is the desired anchoring.

No code changes needed for the KL reference under this migration.

### § 7.5 What about `behavioral_evaluator` and `train_production`?

These do not need code changes for the migration phase, beyond the changes in `observation.py`, `pybh_backend.py`, and `environment.py` that propagate transparently. After the changes:

- `behavioral_evaluator` runs against `OBS_SIZE = 41`. The checkpoint is loaded via the same path (`_load_checkpoint_metadata` + `_policy_payloads`). The zero-pad happens at policy load time.
- `train_production` similarly loads the 26-dim checkpoint into a 41-dim policy via warm-start. Behavior-preservation tests (§ 8) should run on `behavioral_evaluator` outputs.

### § 7.6 Scope discipline for the migration owner

Things the migration owner **should** do:

- Add the 15 new `ObsIndex` entries.
- Extend `BackendObservation` with the new fields (zero-defaultable during the C++ wiring period).
- Append the 15 normalized features in `_encode_obs` in the order specified.
- Verify `_load_state_dict_obs_padded` correctly grows the input layer from 26 to 41.
- Run the sanity test plan in § 8.

Things the migration owner **should NOT** do:

- Change `gate.py` or any field of `GateConfig`.
- Change `gate_skill_mask_from_obs`.
- Change the normalization of indices 0–25.
- Change any reward weight in `curriculum.py` or `reward.py`.
- Train a new policy or fine-tune the existing one. This is a behavior-preservation migration, not a learning migration.
- Add any role-conditioning, opponent enumeration, or precomputed coordination signal.
- Touch `train_production.py` config defaults (`bc_beta_max`, half-life, replay weight, etc.).

---

## § 8 Sanity test plan

Every test below must pass before the migration is considered complete and multi-agent retrain is allowed to start.

### § 8.1 Static checks

**T1 — `OBS_SIZE` propagation**

```bash
PYTHONPATH=$PWD python -c "from observation import OBS_SIZE; assert OBS_SIZE == 41, OBS_SIZE; print('OBS_SIZE=', OBS_SIZE)"
```

Expected: `OBS_SIZE= 41`.

**T2 — Mask reads only indices 23/24/25**

A unit test that constructs a synthetic obs vector of shape `(1, 41)` with random values at every index, calls `gate_skill_mask_from_obs`, then verifies that perturbing indices 0–22 and 26–40 does not change the mask output.

```python
import torch
from train import gate_skill_mask_from_obs

base_obs = torch.zeros((1, 41), dtype=torch.float32)
base_obs[0, 23] = 1.0  # shoot_armed
base_mask = gate_skill_mask_from_obs(base_obs).clone()

for perturb_idx in list(range(0, 23)) + list(range(26, 41)):
    obs = base_obs.clone()
    obs[0, perturb_idx] = 0.5
    mask = gate_skill_mask_from_obs(obs)
    assert torch.equal(mask, base_mask), f"mask depends on idx {perturb_idx}"
print("T2 passed: mask reads only 23/24/25")
```

**T3 — Zero-pad load preserves output**

A unit test that:
1. Loads the 26-dim production checkpoint via `_load_state_dict_obs_padded` into a 41-dim policy.
2. Constructs a synthetic obs of shape `(B, 41)` where indices 0–25 hold the original 26-dim test values and indices 26–40 are zero.
3. Calls the 41-dim policy forward pass.
4. Loads the same checkpoint into a 26-dim policy.
5. Calls the 26-dim policy forward pass on the same indices 0–25.
6. Asserts the outputs (skill logits, residual mean, residual log-std) are within `1e-5` of each other.

If this fails, `_load_state_dict_obs_padded` is not correctly zero-initializing the new input columns. Fix that before proceeding.

### § 8.2 Behavioral checks (the headline gate)

**T4 — Zero-padded behavioral evaluation matches baseline**

Run `behavioral_evaluator` with `OBS_SIZE = 41` and the new indices 26–40 forced to zero (i.e., the BHuman wiring is not yet returning real values). The single-agent production checkpoint should produce metrics within seed noise of the validated 26-dim baseline.

```bash
PYTHONPATH=$PWD python behavioral_evaluator.py \
  --checkpoint runs/production_readiness_v2/checkpoint_stage4_step10000.pt \
  --episodes 10 \
  --seed 4200 \
  --max-episode-steps 1000
```

Expected: every per-scenario metric (clean_takeover_rate, goal_rate, clean_finish_rate, shoot_finish_rate, collision_rate, clean_episode_rate, etc.) within ±5% absolute of the validated 26-dim baseline on the same checkpoint, same seed.

**If T4 fails:** the migration has perturbed the corridor. Most likely cause: a normalization constant changed, an index was misnumbered, or the policy was loaded without `_load_state_dict_obs_padded`. Investigate before any further work.

**T5 — Populated single-agent behavioral evaluation also matches baseline**

Same as T4, but with the BHuman wiring active (indices 26–40 populated from real `TeammateData[]` and `FieldDimensions`). In single-agent scenarios there are no teammates on the field, so:
- 26–31 (Block B): all zero (no fresh teammates).
- 32–34 (Block C): zero or stale (no team ball consensus).
- 35–37 (Block D): all zero.
- 38 (Block E): zero (no team ball).
- 39–40 (Block F): non-zero (always populated from `FieldDimensions`).

Block F is the only block that is non-zero in single-agent scenarios. Since the input weights at indices 39–40 are zero in the warm-started policy, the policy's behavior should still match the 26-dim baseline.

Expected: per-scenario metrics within ±5% absolute of the validated 26-dim baseline. Same checkpoint, same seed.

**If T5 fails but T4 passes:** the BHuman wiring is producing values at indices 26–40 that the policy is reading non-trivially, which contradicts the zero-pad guarantee. Investigate the warm-start path.

### § 8.3 Robustness checks

**T6 — Stale teammate degrades cleanly**

In a sim scenario with one teammate present, simulate a comms drop by forcing `teammate_ages_ms[0]` to `5000.0` (well past the 1000 ms stale threshold). Verify:
- Block B slot 0 zeros out.
- Block D bit 35 = 0.
- The policy's outputs in this state match the single-agent baseline (no teammates) within `1e-5`.

This validates the stale-zero handling path.

**T7 — Slot stability**

In a sim scenario with two teammates roughly equidistant from the agent, run for 100 steps. Count slot swaps (slot 0 ↔ slot 1). Expected: fewer than 5 swaps over 100 steps in steady state, demonstrating the sort is stable when teammates are not actively crossing.

If the count is much higher, consider adding a small hysteresis to the sort (e.g., require a 50 mm distance margin to swap slots) — but only after this migration's behavior-preservation tests pass.

### § 8.4 Acceptance criteria

The migration is complete when:

- T1, T2, T3 pass deterministically.
- T4 passes (zero-padded behavioral_evaluator within ±5% of baseline).
- T5 passes (populated single-agent behavioral_evaluator within ±5% of baseline, when the BHuman wiring is active).
- T6 passes (stale handling produces single-agent baseline behavior).
- T7 passes (slot stability ≥ 95% under near-equidistant teammates).

Multi-agent retrain may start once all of the above pass. Until then, the multi-agent work is gated.

---

## § 9 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Normalization drift on indices 0–25 during refactor | Medium | High (breaks the corridor) | T4 catches this; the existing normalization constants are kept exactly as in `_encode_obs:707-729` |
| Off-by-one in `ObsIndex` enum | Low | High (mask reads wrong column) | T2 catches this directly |
| `_load_state_dict_obs_padded` not invoked for the value/critic head | Medium | Medium (training instability, not day-0 behavior) | Verify the function is called in both `load_bc_warm_start` and `clone_frozen_policy_reference` |
| BC dataset shape mismatch breaks `BCReplayRegularizer` | High | Medium (BC replay throws) | Use § 7.3 option (b): zero-pad dataset at load time. 5-line change in the loader |
| Block F input weights non-zero at warm-start (would perturb single-agent behavior) | Low | Medium | T5 catches this; standard `_load_state_dict_obs_padded` zero-initializes new columns |
| Engagement bit threshold (`ENGAGING_DISTANCE_MM`) too tight or loose | Low | Low (affects only multi-agent retrain, not migration) | Defer tuning to multi-agent work; 700 mm is a reasonable starting value |
| Slot thrash when teammates are near-equidistant | Low | Low (affects retrain stability) | T7 catches this; add sort hysteresis if needed, post-migration |
| BHuman C++ wiring lags Python migration | High | Low | Python side defaults to zero values when BHuman fields are absent (§ 7.2); the migration is testable with T4 before the C++ side lands |
| BHuman pybh exporter changes break the existing single-agent build | Medium | High | Test the pybh build with the existing `behavioral_evaluator` before integrating with the new obs encoder |
| Production C++ build inconsistency between the RL workstation and competition robots | Medium | High | Standard transplant testing covers this; out of scope for the obs extension itself |

---

## § 10 File-by-file change checklist

For the migration owner. Each line is a discrete edit; tick as you go.

### Python (RL repo)

- [ ] `observation.py`: add `ObsIndex` entries 26–40 with names from § 3. Update `OBS_FEATURE_NAMES` (automatic via the enum).
- [ ] `pybh_backend.py:258` (`BackendObservation` dataclass): add fields per § 7.2.
- [ ] `pybh_backend.py` (`BackendObservation.from_payload`): parse new payload keys with zero defaults for missing keys.
- [ ] `environment.py:697` (`_encode_obs`): append 15 new features in the order specified in § 3. Add helper `transform_to_robot_frame` if not present, and constants `TEAMMATE_STALE_THRESHOLD_MS = 1000.0`, `TEAM_BALL_CONSENSUS_MM = 800.0`, `ENGAGING_DISTANCE_MM = 700.0`.
- [ ] Activity-to-engagement helper: introduce `activity_is_engaging(activity_enum_value, slot_position, team_ball_xy) -> bool`. The enum values can come from a small Python-side mapping that mirrors BHuman's `BehaviorStatus.activity` enum.
- [ ] Verify `train.py:_load_state_dict_obs_padded` (line ~1206) is invoked when loading the 26-dim checkpoint at `OBS_SIZE = 41`. Add an explicit log line at warm-start showing `prev_in_features=26, new_in_features=41`.
- [ ] Verify the BC dataset loader zero-pads observations when loaded under `OBS_SIZE = 41`. The current dataset is shape-locked at 26; the loader should reshape to `(N, 41)` with zeros at indices 26–40.

### C++ (pybh exporter, SabanaHerons2026 tree)

- [ ] In the obs payload builder, add the keys from § 6.2. Read from the existing BHuman representations (no new providers).
- [ ] Verify the build succeeds and the existing single-agent `behavioral_evaluator` still passes before integrating with the Python side (this validates the C++ change is isolated).

### Tests

- [ ] Add unit test for T2 (mask reads only 23/24/25) in the project's test suite.
- [ ] Add unit test for T3 (zero-pad load equivalence).
- [ ] Document the manual T4–T7 procedure in a runbook (e.g., `docs/24-migration-runbook.md` if useful).

### Sign-off

Migration is complete when § 8 acceptance criteria pass and the above checklist is fully ticked. Multi-agent work is unblocked at that point.

---

## § 11 What this unlocks for multi-agent work

Once the migration lands and the sanity tests pass, the following workstreams become well-defined:

### § 11.1 Wrapper-coordinated multi-agent (the 5-day sprint)

The role-arbitration wrapper described in `selfplay-4v4-wrapper-strategy` becomes implementable on the 41-dim substrate without needing to touch the policy itself. Each agent runs the 41-dim policy with zero-padded behavior (so it behaves like the validated single-agent). The wrapper handles:

- Role assignment (closest-to-ball = striker, others = positional supports).
- Support state machines that bypass the policy and run BHuman's existing walk-to-target skill.

This work does not retrain the policy. It uses the substrate to enable team coordination via the wrapper, while the policy itself continues to behave as in single-agent.

### § 11.2 Multi-agent BC + retrain

With the substrate in place, a multi-agent BC dataset can be collected (each agent's obs now includes teammate state and team ball), and BC replay + PPO continuation training can teach the policy to use indices 26–40. This is multi-agent learning proper, gated on:

- BC dataset re-collection at 41-dim obs.
- Multi-agent curriculum design (separate doc).
- KL anchor strategy (anchor to current production checkpoint via zero-pad, lower `bc_beta_max`, longer `bc_competence_half_life_evals` — already documented for the graph-coordination plan in `docs/notes/10-graph-coordination-post-production-roadmap.md`).

### § 11.3 What is still NOT enabled by this substrate

For completeness, the substrate alone does not provide:

- **Learned coordination behavior.** That comes from retrain on the new dims with multi-agent reward and dataset. The substrate is the *input* the multi-agent retrain consumes.
- **Opponent enumeration.** Would need a separate percept-level workstream.
- **Set-play awareness.** Separate workstream.
- **Robustness improvements to the single-agent corridor.** The brittleness of the current production checkpoint is a separate concern; the substrate preserves it as-is.

---

## § 12 Glossary

- **Block A–F**: the six groups of indices in the 41-dim obs, defined in § 3.
- **Corridor**: the narrow region of the obs-action distribution the production policy was trained in. The four design constraints in § 2 are about preserving this corridor.
- **Design A**: the invariant that the per-step skill mask is a pure function of the stored observation (read from indices 23/24/25), so MAPPO's update-time log-prob recompute reproduces it exactly. See `gate.py` docstring.
- **Engagement bit**: a per-teammate boolean in Block D indicating whether that teammate is actively committed to the ball, derived from `theBehaviorStatus.activity`.
- **Sort by distance to me**: the rule used for slot assignment in Block B and Block D — teammates are ordered by ascending distance from the agent's own robot position.
- **Stale-zero**: a teammate whose message age exceeds the threshold (1000 ms) has its position slot and engagement bit zeroed.
- **Substrate**: the obs space, viewed as the *input* a future multi-agent retrain will consume. This document specifies the substrate; learning on it is downstream.
- **Team-ball consensus**: the agreement between an agent's own ball belief and the team-merged `GlobalTeamBallModel`. Block E (38) is the gate for this consensus.
- **Zero-pad warm-start**: the existing mechanism (`_load_state_dict_obs_padded` in `train.py`) that loads a smaller policy into a larger policy by zero-initializing new input columns, preserving day-0 outputs.

---

*End of document.*
