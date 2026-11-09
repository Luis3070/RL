# Flujo Natural de B-Human

## Regla principal

RL solo reemplaza la decision de alto nivel. No reemplaza el motor de caminata.

La frontera correcta es `SkillRequest`, no `MotionRequest` directo.

Flujo correcto:

```text
Python/RL
  -> pybh.rl_set_action(...)
  -> RLSharedState
  -> RLSkillProvider::update(SkillRequest&)
  -> SkillBehaviorControl::executeRequest()
  -> MotionRequest
  -> MotionEngine
  -> WalkingEngine / KickEngine / otros motores
  -> JointRequest
  -> SimRobot
```

## Por que no se debe bypassear `SkillBehaviorControl`

El bypass directo a `MotionRequest` parecia tentador porque el robot recibia comandos mas rapido, pero rompe partes importantes de B-Human:

- seleccion de skills reales;
- control de cabeza;
- control de brazos;
- comportamiento de caminar alrededor de obstaculos;
- go-to-ball-and-kick;
- dribble;
- consistencia con escenas normales de SimRobot;
- trazabilidad del comportamiento.

En 3D, el bypass o el root-assist puede producir deslizamiento: la pose cambia sin que la caminata fisica sea la causa real del movimiento. Eso no sirve para entrenar ni para inspeccionar comportamiento.

## `HandleRLRequest`

`HandleRLRequest` es la opcion minima que conecta el skill externo de RL con el sistema normal de skills:

```text
option(HandleRLRequest)
{
  initial_state(start)
  {
    transition
    {
      goto execute;
    }
  }

  state(execute)
  {
    action
    {
      executeRequest();
    }
  }
}
```

Esto hace que `SkillBehaviorControl` ejecute exactamente el `SkillRequest` que llega desde `RLSkillProvider`.

## Mapping actual de skills

Desde Python se usan nombres simples:

```text
stand
walk / walkTo
shoot
pass
dribble
block
mark
observe
```

En C++ `RLSkillProvider::update(SkillRequest&)` los convierte a:

```text
stand   -> SkillRequest::Builder::stand()
walkTo  -> SkillRequest::Builder::walkTo(Pose2f(theta, x, y))
shoot   -> SkillRequest::Builder::shoot()
pass    -> SkillRequest::Builder::passTo(passTarget)
dribble -> SkillRequest::Builder::dribbleTo(theta)
block   -> SkillRequest::Builder::block(Vector2f(x, y))
mark    -> SkillRequest::Builder::mark(Vector2f(x, y))
observe -> SkillRequest::Builder::observe(Vector2f(x, y))
```

`walkTo` usa coordenadas de campo, no coordenadas relativas al robot.

## Trazas que confirman el flujo

Las trazas importantes en `rl_get_obs()` son:

```text
requested_skill
skill_behavior_skill_request
skill_behavior_motion_request
motion_engine_input_request
motion_engine_effective_request
motion_engine_phase
joint_l_hip_pitch
joint_l_knee_pitch
joint_r_hip_pitch
joint_r_knee_pitch
```

Para caminata natural se espera:

```text
requested_skill = walkTo
skill_behavior_skill_request = 2
skill_behavior_motion_request = 4
motion_engine_input_request = 4
motion_engine_effective_request = 4
```

El `4` de `MotionRequest` corresponde a `walkToPose`.

Si `provider_motion_request` cambia pero `skill_behavior_motion_request` no cambia, probablemente se volvio a usar un bypass.

## 2D tambien respeta el flujo

El perfil `2d` sigue exactamente la misma frontera de arquitectura:

```text
RLSkillProvider::update(SkillRequest&)
  -> solo produce SkillRequest
SkillBehaviorControl
  -> produce MotionRequest
```

La diferencia con `3d` no es el wire, sino el motor físico que ejecuta el
request.
