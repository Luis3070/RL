# MAPPO Wire Findings In SabanaHerons

This note records the actual 3D `SabanaHerons2026 -> pybh -> RL` semantics behind MAPPO blocks `D` and `E`.

## 1. No native `BehaviorStatus.activity` was found

Branches checked:

- `origin/master`
- `origin/Play2026`
- `origin/rl-simrobot3d`
- `origin/REINFORCEMENT-LEARNING-BRIDGE-TO-PYTHON`

Result:

- No checked branch exposes `BehaviorStatus.activity`.
- `Play2026` adds `lastKickTimestamp` and `lastKickWasOutsideCenterCircle`, but still no `activity`.

Therefore the current RL bridge uses a derived proxy in
[SkillBehaviorControl.cpp](/Src/Modules/BehaviorControl/SkillBehaviorControl/SkillBehaviorControl.cpp):

- `kicking`: `shootingTo` is set or `passTarget >= 0`
- `dribbling`: teammate has a fresh ball and the ball is within close-control distance
- `goingToBall`: `playBall` role, or fresh ball nearby, or walk target near the fresh ball
- `idle`: otherwise

## 2. What RL block `E` actually means

In RL, block `E` is the single bit at index `38`:

- `team_ball_agrees_with_mine = 1` iff
  - `global_team_ball_valid == true`
  - distance between own ball belief and team ball belief is `< 800 mm`

The current encoder does **not** use `global_team_ball_age_ms` to decide `E`.
Age is exported for block `C` context, but the consensus gate is strictly `valid + distance`.

## 3. Where the team ball comes from in 3D

The exported team ball is backed by `TeammatesBallModel`, produced by
`TeammatesBallModelProvider`, not by a representation that stores its own timestamp.

Relevant code:

- [TeammatesBallModel.h](/Src/Representations/Modeling/TeammatesBallModel.h)
- [TeammatesBallModelProvider.cpp](/Src/Modules/Modeling/TeammatesBallModelProvider/TeammatesBallModelProvider.cpp)

Important properties:

- `TeammatesBallModel` stores `position`, `velocity`, `isValid`, `newerThanOwnBall`
- It does **not** store `timeWhenLastSeen`
- The RL bridge currently exports `global_team_ball_age_ms` using the newest teammate `BallModel.timeWhenLastSeen` seen during export

That age is a practical proxy, not a literal field from the team-ball representation.

## 4. Why `global_team_ball_valid` can flicker in visible 3D probes

`TeammatesBallModelProvider` invalidates or suppresses the team ball in several cases:

- no active teammate balls survive filtering
- teammate observation disappeared too long ago
- teammate is penalized or not upright
- teammate pose jumped recently
- game state is not `PLAYING` or `SET`
- a new goal kick or corner kick starts

Key thresholds in `4v4_RL3D`:

- `inactivityInvalidationTimeSpan = 1000`
- `ballLastSeenTimeout = 15000`
- `ballDisappearedTimeout = 100`
- `timestampTolerance = 25`

Ball clustering also merges only compatible teammate balls within `777 mm`.

## 5. How team communication affects the availability of a fresh team ball

`TeamMessageHandler` decides when ball-related changes justify sending another team message.
This matters because the team ball can only stay fresh if teammates keep communicating useful ball updates.

Relevant parameters in `4v4_RL3D`:

- `minSendInterval = 1000`
- `teamBallThresholdBase = 9000`
- `teamBallThresholdFactor = 500`
- `newBallThreshold = 500`
- `disappearedThreshold = 3000`

`teamBallOld()` returns true only when:

- our own local ball is still considered new (`<= 500 ms` old), and
- the newest communicated teammate ball has become older than
  `9000 + playerNumber * 500 ms`

This does not define RL block `E` directly, but it affects whether the upstream team-ball estimate stays current and valid.

## 6. Practical consequences for MAPPO validation

- `D` depends on the exported teammate-activity proxy, not a native B-Human activity enum
- `E` depends on team-ball validity and geometric agreement, not on team-ball age
- block `34` (`time_since_team_ball_update`) is a proxy age in the current 3D wire
- visible 3D probes can legitimately show intermittent `E=0` when the team ball goes invalid upstream
