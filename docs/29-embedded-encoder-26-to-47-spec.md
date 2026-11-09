# 29 — Embedded PPO encoder extension: obs26 → obs47 (spec for Rafael)

Fecha: 2026-06-12. Owner de implementación: Rafael (C++, `hsl2026-ppo-naov6-integration`).
Autor del contrato: línea multi-agente (Luis + Claude). Referencia Python canónica:
`RL/environment.py:_encode_obs` (la implementación C++ debe ser numéricamente equivalente).

## Por qué

El ONNX multi-agente (deliverable del sábado, `tools/export_ppo_policy_team.py`) tiene
entrada de **47 floats**. El encoder embebido actual produce 26. Las 21 dimensiones
nuevas son: bloque de teammates (6), team ball (3), engagement (3), consensus (1),
goal vector (2), gate bits nuevos (3), role one-hot (3). Todos los campos wire ya
existen en `RLSharedState` / `io.*` (commit `1d9fa649`) — esto es plomería + dos
funciones nuevas, no campos C++ nuevos.

## Constantes de normalización (idénticas a Python)

```
X_RANGE        = 4500.0   // ACTION.x_range
Y_RANGE        = 3000.0   // ACTION.y_range
TIME_NORM_MS   = 5000.0
TEAMMATE_STALE_THRESHOLD_MS = 1000.0
ENGAGING_DISTANCE_MM        = 700.0
TEAM_BALL_CONSENSUS_MM      = 800.0
```

`norm_dist(v) = clip(v / X_RANGE, 0, 1)`
`norm_time(v) = clip(v / TIME_NORM_MS, 0, 1)`
`to_robot_frame(fx, fy)`: rotar (fx - robot_x, fy - robot_y) por -robot_theta.

## Dims 0–25: sin cambios

Idénticas al contrato obs26 actual (`ppo_obs_v26_hsl2026`). No tocar.

## Dims 26–31 — Block B: teammates (3 slots × [x_rel, y_rel])

Fuente: `theTeamData.teammates` (lo que hoy alimenta `io.teammateX/Y/AgeMs`).

1. Filtrar teammates con `age_ms <= TEAMMATE_STALE_THRESHOLD_MS`.
2. Ordenar los frescos por distancia euclidiana al robot (campo, ascendente).
3. Tomar hasta 3. Por slot: `to_robot_frame(x, y)` → `clip(rx/X_RANGE, -1, 1)`,
   `clip(ry/Y_RANGE, -1, 1)`.
4. Slots vacíos → `0.0, 0.0`.

⚠️ El orden filtrado+ordenado debe ser EXACTO (el mismo del Python) — el mapping
slot→player de pass_target depende de él.

## Dims 32–34 — team ball

Fuente: `theTeammatesBallModel` (hoy `io.globalTeamBallX/Y/AgeMs/Valid`).

- 32: `clip(to_robot_frame(tb_x, tb_y).x / X_RANGE, -1, 1)`
- 33: ídem con `.y / Y_RANGE`
- 34: `norm_time(team_ball_age_ms)`

🚨 **HAZARD (encontrado 2026-06-12, crasheó el training):** con team comm caída,
`theTeammatesBallModel.position` puede ser **NaN**. Un NaN aquí envenena los logits
de la política EN SILENCIO (en el robot no crashea — juega basura). Guard
obligatorio: si `!std::isfinite(tb_x) || !std::isfinite(tb_y)` → tratar como
team ball en (0,0) con valid=false y age=∞ (dim 34 = 1.0, dim 38 = 0).

## Dims 35–37 — teammate engaging flags (por slot, mismo orden que 26–31)

`1.0` si el activity del teammate es `dribbling` o `kicking`; o si es
`goingToBall` Y team ball válido Y `dist(teammate, team_ball) < ENGAGING_DISTANCE_MM`.
Si no → `0.0`. Activity = `encodeTeammateActivity()` (ya existe,
SkillBehaviorControl.cpp:61-85).

## Dim 38 — team ball consensus

`1.0` si `team_ball_valid && dist(own_ball_field, team_ball_field) < TEAM_BALL_CONSENSUS_MM`
(own_ball_field = robot pose + ball rel). Si no → `0.0`.

## Dims 39–40 — opponent goal vector

`to_robot_frame(xPosOpponentGroundLine, 0)` → `clip(/X_RANGE, -1, 1)`, `clip(/Y_RANGE, -1, 1)`.
(Ya exportado como `io.goalOpponentX/Y`.)

## Dims 41–43 — gate bits nuevos (Design A, paralelos a 23–25)

El gate C++ que hoy computa `shoot_armed/dribble_armed/shoot_arm_progress` (23–25)
se extiende con:

- **41 `pass_armed`**: rol == striker AND (shoot_armed OR dribble_armed) AND
  smashOrPass propone pass. smashOrPass ya existe en PlayBall.cpp:33-118 — reusar
  la decisión real de BHuman es PREFERIBLE a re-portarla. Si el robot no es
  striker → 0.
- **42 `observe_armed`**: 0.0 fijo este sprint (observe está deferred).
- **43 `pass_arm_progress`**: debounce fill 0→1 análogo a shoot_arm_progress;
  si no se implementa el debounce, usar `pass_armed` como valor (0/1). El
  policy tolera esta simplificación (el bit 41 es el que gobierna el mask).

🚨 **Calibración de gate**: los umbrales con los que se entrenó están en el
manifest del ONNX (`gate_config`): `shoot_open_enter=0.30`, `shoot_open_exit=0.20`
sobre `theExpectedGoals.getRating(ballOnField)`. El gate embebido DEBE usar estos
valores, no los 0.6/0.45 históricos (calibrados a un binario viejo con xG inflado).

## Dims 44–46 — role one-hot

Mapping desde la elección de rol de BHuman (la misma que hoy gobierna
`embeddedPPODynamicPlayBall`):

- `is_striker` (44): rol estrategia == `playBall`.
- `is_open_support` (45): no-playBall con posición de rol más adelantada
  (mayor x de los soportes; con Tactic position roles: el Forward).
- `is_off_ball_support` (46): el resto de field players RL.
- GK: nunca corre PPO (exclusión existente).

Exactamente UN bit en 1.0 por robot RL por frame. La histéresis del rol la pone
BHuman (TTRB) — no añadir histéresis extra aquí.

## Validación (antes del sábado noche)

1. **Paridad numérica**: log de los 47 floats C++ vs `environment.py:_encode_obs`
   sobre el mismo estado (tolerancia 1e-4). El probe de paridad de obs26 que ya
   existe (`a733c39 Add MAPPO observation wire probes and tests`) es el patrón.
2. **NaN guard test**: forzar team comm off → dims 32-34,38 deben ser
   (0,0,1.0,0) — nunca NaN.
3. **Carga ONNX**: `ppo_team_hsl2026.onnx` (47-dim) carga e infiere < 30 ms
   (mismo harness del modelo actual).
4. Los logs existentes `params[]/targetAbs/targetRel` siguen siendo obligatorios.

## Qué NO hacer

- No tocar dims 0–25 (romper el contrato obs26 rompe el fallback Plan A).
- No computar el rol con lógica nueva — SOLO mapear la elección BHuman existente.
- No usar los umbrales de gate históricos (0.6/0.45) con el ONNX team.
- No interpolar/extrapolar teammates stale — slot vacío es (0,0), igual que Python.
