# Using B-Human Skills for Natural SimRobot 3D Motion

> Nota: este documento queda como historico de investigacion. El estado actual
> mantenido esta en [docs/README.md](./docs/README.md), especialmente en
> [docs/01-flujo-natural-bhuman.md](./docs/01-flujo-natural-bhuman.md).

## Implementation Status

Implemented in the current working tree.

The 3D profile now uses:

- `Config/Scenes/RL1v0_3D.ros2`
- `Config/Scenes/RL1v0_3D.con`
- `Config/Scenarios/4v4_RL3D`

`RLSkillProvider` no longer provides `MotionRequest`. The normal B-Human chain is restored:

```text
RLSkillProvider -> SkillRequest
SkillBehaviorControl -> MotionRequest
MotionEngine / WalkingEngine -> JointRequest
SimulatedRobot3D -> joint actuators
```

Python now sends field-global `walkTo` targets for `simrobot` and `simrobot_visible`.

## Current Problem

The current RL SimRobot 3D visible path does not use the normal B-Human skill-to-motion pipeline.

Right now `RLSkillProvider` provides both:

- `SkillRequest`
- `MotionRequest`

and `Config/Scenarios/4v4_Scaled/threads.cfg` maps:

```cfg
{representation = MotionRequest; provider = RLSkillProvider;},
```

That bypasses `SkillBehaviorControl`, so B-Human skills such as `WalkToPointObstacle`, `GoToBallAndKick`, `DribbleToGoal`, and the normal head/arm behavior are not what drives the robot. The robot receives a low-level `MotionRequest` directly from the RL bridge.

In normal B-Human scenes, the chain is:

```text
Strategy layer
  -> SkillRequest
SkillBehaviorControl
  -> MotionRequest, HeadMotionRequest, ArmMotionRequest
MotionEngine / WalkingEngine / KickEngine
  -> JointRequest
SimulatedRobot3D
  -> joint actuators
```

The RL integration should replace only the strategy layer.

## Correct Architecture

`RLSkillProvider` should provide:

- `SkillRequest`
- `StrategyStatus`

`SkillBehaviorControl` should provide:

- `MotionRequest`
- `HeadMotionRequest`
- `ArmMotionRequest`
- related behavior outputs

Then the normal motion stack produces `JointRequest`, and `SimulatedRobot3D::getAndSetJointData(...)` applies joint actuator commands naturally.

## Required SabanaHerons2026 Changes

### 1. Restore MotionRequest provider in 4v4_Scaled

File:

```text
SabanaHerons2026/Config/Scenarios/4v4_Scaled/threads.cfg
```

Change:

```cfg
{representation = MotionRequest; provider = RLSkillProvider;},
```

to:

```cfg
{representation = MotionRequest; provider = SkillBehaviorControl;},
```

Keep:

```cfg
{representation = SkillRequest; provider = RLSkillProvider;},
{representation = StrategyStatus; provider = RLSkillProvider;},
```

### 2. Remove MotionRequest from RLSkillProvider

File:

```text
SabanaHerons2026/Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.h
```

Remove:

```cpp
#include "Representations/MotionControl/MotionRequest.h"
PROVIDES(MotionRequest),
void update(MotionRequest& motionRequest) override;
```

Also remove `REQUIRES(FieldDimensions)` if only used for direct `MotionRequest` generation.

File:

```text
SabanaHerons2026/Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp
```

Remove:

- `buildMotionRequest(...)`
- `RLSkillProvider::update(MotionRequest&)`
- the direct low-level motion mapping from RL skill strings to `MotionRequest`

Keep `update(SkillRequest&)`, because that is the correct injection point for RL.

### 3. Keep observations in RLSkillProvider

`RLSkillProvider::update(SkillRequest&)` should continue writing:

- ball position
- robot position
- robot theta
- frame
- `obsReady`

That part is still useful because it bridges SimRobot state back to Python.

## Required RL Changes

### 1. Stop converting walk targets to robot-relative for natural skills

`SkillRequest::Builder::walkTo(...)` expects field coordinates.

Current RL code often calls `adapt_action_to_backend(...)`, which converts walk targets into robot-relative coordinates for non-`cpp` modes. That was compatible with the direct `MotionRequest` shortcut, but it is wrong for the B-Human skill pipeline.

File:

```text
RL/pybh_backend.py
```

For the natural B-Human skill path, `simrobot` and `simrobot_visible` should send field-global walk targets unchanged.

The adaptation rule should become:

```text
cpp / simrobot / simrobot_visible: keep field-global target
legacy direct-motion path only: convert to robot-relative target
```

### 2. Fix live_simrobot_probe.py

File:

```text
RL/live_simrobot_probe.py
```

The heuristic policy currently computes a field-global ball target and then calls:

```python
adapt_action_to_backend(global_action, obs, mode="python")
```

For natural B-Human skills, it should send the field-global target directly:

```python
pybh.rl_set_action(player_number, "walkTo", ball_x, ball_y, target_theta)
```

The printed action should say `walkToField(...)`, not `walkToRel(...)`, to avoid confusion.

### 3. Disable abstract 3D motion for natural mode

The env var:

```text
PYBH_SIMROBOT_RL_ABSTRACT_MOTION=1
```

activates the temporary root-body motion emulation in `SimulatedRobot3D::getAndSetMotionData(...)`.

For natural motion, do not set it. Natural movement should come from:

```text
MotionEngine -> JointRequest -> SimulatedRobot3D::getAndSetJointData(...)
```

## Why The Previous Attempt Failed

The failed attempt likely changed only part of the graph. Both sides must match:

- `threads.cfg` must map `MotionRequest` to `SkillBehaviorControl`
- `RLSkillProvider` module metadata must no longer claim `PROVIDES(MotionRequest)`
- Python must send field-global `SkillRequest` targets

If any one of those remains inconsistent, B-Human either complains about a missing provider or receives semantically wrong targets.

## Validation Plan

1. Build:

```bash
cd /home/rafael/Documentos/Robocup2026/SabanaHerons2026
cmake --build Build/Linux/CMake/Develop --target SimulatedNao PythonController B-Human
```

2. Copy Python module locally:

```bash
cp Build/Linux/Python/Develop/controller.cpython-312-x86_64-linux-gnu.so Make/Python/pybh/controller.cpython-312-x86_64-linux-gnu.so
```

3. Run visible probe without abstract motion:

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
.venv/bin/python live_simrobot_probe.py --launch --policy heuristic --simrobot-profile 3d --steps 1000
```

4. In SimRobot, press `F5`.

Expected result:

- `MotionRequest` comes from `SkillBehaviorControl`
- `JointRequest` changes over time
- the robot walks using the normal 3D walking engine, not root-body teleport/emulation

Useful debug views in SimRobot:

```text
dr representation:SkillRequest
dr representation:MotionRequest
dr representation:JointRequest
```
