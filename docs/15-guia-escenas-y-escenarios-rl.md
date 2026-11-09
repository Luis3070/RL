# Guia practica para armar escenas y escenarios RL

## Objetivo

Este documento junta lo aprendido al depurar escenas RL visibles en SimRobot 3D:

- como se compone una escena nueva,
- que debe vivir en `Scene` y que debe vivir en `Scenario`,
- cuando una pose se cambia en el `.ros2`,
- cuando la escena inicial es pisada por Python o por el game state,
- como crear obstaculos fijos,
- y que revisar cuando "se ve bien en SimRobot" pero el pipeline RL/B-Human no se comporta como esperas.

No es teoria general de SimRobot. Es una guia operativa para este stack:

- repo Python: `RL/`
- repo B-Human/SimRobot: `SabanaHerons2026/`
- equipo RL: `team 24`
- robot controlado por Python: `player 1`, `robot1`

## Mapa mental: que archivo decide que

### 1. Escena `.ros2` o `.ros2d`

La escena define el mundo cargado por SimRobot:

- que field se usa,
- que teams existen,
- que scenario se activa,
- que robots se instancian,
- que extras se dibujan,
- donde aparece la pelota,
- y la pose inicial literal de cada `Body`.

Ejemplo base:

- [RL1v0_3D.ros2](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenes/RL1v0_3D.ros2)

### 2. Archivo `.con`

El `.con` no define geometria. Define comandos de consola al abrir la escena:

- que include de consola se carga,
- si se fuerza `gc playing`,
- y otras ayudas de debugging.

Ejemplo base:

- [RL1v0_3D.con](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenes/RL1v0_3D.con)

En este proyecto, para escenas RL naturales, el `.con` debe arrancar el juego en estado activo:

```txt
call Includes/RLNatural
gc playing
```

Esto importa mucho. Si el game state no esta en una fase activa, partes del pipeline normal de B-Human no se comportan como en juego, y eso ya rompió `ObstacleModel` durante el diagnostico.

### 3. `Config/Scenarios/<scenario>/...`

El scenario define el pipeline B-Human:

- providers,
- threads,
- skill behavior,
- setup poses,
- configuraciones de perception/modeling/motion.

El escenario RL 3D actual es:

- `SabanaHerons2026/Config/Scenarios/4v4_RL3D/`

Piezas especialmente importantes:

- [threads.cfg](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenarios/4v4_RL3D/threads.cfg)
- [setupPosesProvider.cfg](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenarios/4v4_RL3D/setupPosesProvider.cfg)

### 4. Python `live_simrobot_probe.py`

El probe visible no solo observa. Tambien puede pisar la pose inicial:

- hace `rl_reset_world(...)` por default,
- solo respeta la pose literal de la escena cuando se usa `--no-reset`.

Archivo:

- [live_simrobot_probe.py](/home/rafael/Documentos/Robocup2026/RL/live_simrobot_probe.py)

## Minimo que debe tener una escena nueva

Una escena RL 3D nueva debe tener, como minimo:

1. `Include` de `NaoV6H25`, `Ball2016SPL` y `Field2020SPL`.
2. `Scene name="RoboCup" controller="SimulatedNao"`.
3. `Compound name="teams"`.
4. `Compound name="location"`.
5. `Compound name="scenarios"`.
6. `Compound name="robots"`.
7. `Compound name="extras"`.
8. `Compound name="balls"`.
9. `Compound ref="field"`.

Estructura de referencia:

```xml
<Scene name="RoboCup" controller="SimulatedNao" ...>
  <Compound name="teams">...</Compound>
  <Compound name="location">...</Compound>
  <Compound name="scenarios">...</Compound>
  <Compound name="robots">...</Compound>
  <Compound name="extras">...</Compound>
  <Compound name="balls">...</Compound>
  <Compound ref="field"/>
</Scene>
```

Si falta alguno de esos bloques, normalmente estas mezclando responsabilidades o cargando una escena pensada para otro flujo.

## Que valores no debes cambiar a ciegas

### Teams

En nuestras escenas RL, el equipo controlado por Python es:

- `team 24`

Eso no es decorativo. Hoy el bridge y varios bypass de debug estan pensados para ese equipo.

Referencias:

- [config.py](/home/rafael/Documentos/Robocup2026/RL/config.py)
- `TEAM_NUMBER = 24`

### Robot controlado por Python

Por default, Python controla:

- `player 1`
- `robot1`

Referencias:

- [config.py](/home/rafael/Documentos/Robocup2026/RL/config.py)
- `PLAYER_NUMBER = 1`

Si cambias el robot visible o el player number, documentalo tambien en Python o vas a terminar mandando acciones al robot equivocado.

## `robots` vs `extras`

Esta es la distincion mas importante al armar escenas nuevas.

### `robots`

Todo lo que pongas dentro de `RoboCup.robots` pasa a ser un jugador controlado por `SimulatedNao`.

Consecuencias:

- entra al flujo de player numbers,
- puede ser reubicado por `SetupPoses`,
- el game state lo trata como robot de partido,
- y moverlo en el `.ros2` no garantiza que se quede ahi.

Esto explica por que un `robot21` dentro de `robots` terminaba pegado a una linea aunque lo movieras a mano en la escena: el sistema lo reinterpretaba como jugador.

### `extras`

Todo lo que pongas dentro de `RoboCup.extras` es un objeto de escena.

Consecuencias:

- queda fijo donde lo pongas,
- no participa como jugador del partido,
- no recibe recolocacion por `SetupPoses`,
- sirve para obstaculos fijos, referencias visuales o diagnosticos.

Para obstaculos fijos, la regla practica es:

- si debe quedarse donde lo pones, va en `extras`;
- si debe comportarse como jugador del partido, va en `robots`.

## Como crear un obstaculo fijo

La forma mas estable que usamos fue:

```xml
<Compound name="extras">
  <Body ref="NaoDummy" name="robot21">
    <Set name="NaoColor" value="red"/>
    <Translation x="1.219" y="-0.9375" z="320mm"/>
    <Rotation z="180degree"/>
  </Body>
</Compound>
```

Referencia:

- [RL1v0_3D_extra_aligned.ros2](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenes/RL1v0_3D_extra_aligned.ros2)

Notas practicas:

- `NaoDummy` fue suficiente para percepción/modelado de obstaculos.
- `Nao` completo en `extras` tambien se puede usar para diagnostico visual, pero no hace falta para un obstaculo fijo basico.
- si el obstaculo tiene que quedar exactamente sobre una trayectoria, calcula su `Translation` en el mismo sistema del campo que usa la escena.

## Como cambiar la posicion de un jugador

La respuesta corta es:

- si quieres cambiar la pose inicial visible en SimRobot, cambiala en la escena;
- si quieres que Python tambien arranque desde ahi, usa `--no-reset` o cambia el reset del probe;
- si el robot vive en `robots`, recuerda que `SetupPoses` o el game state pueden pisarla.

### Caso A: cambiar la pose en la escena

Se cambia en el `.ros2`, por ejemplo:

```xml
<Body ref="Nao" name="robot1">
  <Translation x="2.438" y="-1.875" z="320mm"/>
  <Rotation z="180degree"/>
</Body>
```

Eso sirve para:

- abrir la escena manualmente en SimRobot,
- pruebas con `--no-reset`,
- y diagnostico visual rapido.

### Caso B: cambiar la pose desde Python

Por default, `live_simrobot_probe.py` hace:

```python
pybh.rl_reset_world(
    player_number,
    LIVE_RESET_BALL_POS[0],
    LIVE_RESET_BALL_POS[1],
    LIVE_RESET_ROBOT_POS[0],
    LIVE_RESET_ROBOT_POS[1],
    LIVE_RESET_ROBOT_THETA,
)
```

Constantes actuales:

- `LIVE_RESET_BALL_POS = (0, 0)`
- `LIVE_RESET_ROBOT_POS = (-2438, 1875)`
- `LIVE_RESET_ROBOT_THETA = -pi/2`

Archivo:

- [live_simrobot_probe.py](/home/rafael/Documentos/Robocup2026/RL/live_simrobot_probe.py)

Entonces:

- si corres el probe normal, la escena inicial puede ser pisada;
- si quieres respetar exactamente la pose del `.ros2`, usa `--no-reset`;
- si quieres otra pose controlada desde Python, cambia esas constantes o la llamada a `rl_reset_world(...)`.

### Caso C: cambiar pose y que NO la pise nada

Usa las dos cosas:

1. cambia la pose en el `.ros2`
2. corre con `--no-reset`

Si ademas necesitas un obstaculo fijo, ponlo en `extras`, no en `robots`.

## `--no-reset` no significa "todo queda como en la escena"

`--no-reset` solo evita que Python haga `rl_reset_world`.

Todavia hay otras dos fuentes que pueden mover cosas:

1. `SetupPosesProvider`
2. game state / flujo normal del scenario

El caso mas comun:

- `robot1` en `robots`
- scene lo pone en una pose
- scenario `4v4_RL3D` vuelve a colocarlo segun `playerNumber`

Referencia:

- [setupPosesProvider.cfg](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenarios/4v4_RL3D/setupPosesProvider.cfg)

Ahí se ve, por ejemplo, que `player 1` arranca en:

- `x = -2438`
- `y = 1875`

### `StaticInitialPoseProvider`

No fue el culpable en nuestras pruebas porque en `field_e` esta apagado:

- [staticInitialPoseProvider.cfg](/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Locations/field_e/staticInitialPoseProvider.cfg)
- `isActive = false`

## Regla operativa para poses

Usa esta tabla:

| Lo que quieres | Donde cambiarlo |
| --- | --- |
| Pose visual inicial del robot | `.ros2` |
| Pose de reset usada por probe | `live_simrobot_probe.py` |
| Pose del jugador en un scenario de partido | `setupPosesProvider.cfg` |
| Obstaculo fijo que no sea recolocado | `extras` en `.ros2` |

## Que debe tener un scenario RL nuevo

Si clonas `4v4_RL3D` para crear un scenario nuevo, no basta con cambiar el nombre de la carpeta. Debes conservar el pipeline RL/B-Human.

### Requisitos minimos

1. `SkillRequest` debe venir de `RLSkillProvider`.
2. `StrategyStatus` debe venir de `RLSkillProvider`.
3. `RobotPose`, `BallModel`, `ObstacleModel` y representaciones principales no deben volver a oracle si el objetivo es entrenamiento/percepción natural.
4. `PathPlanner` debe seguir activo si quieres avoidance real por trayectoria.
5. `SetupPosesProvider` debe estar alineado con el tipo de experimento que quieres correr.
6. `StaticInitialPoseProvider` debe estar entendido, no solo copiado.

En `4v4_RL3D/threads.cfg` ya existe una referencia correcta.

## Obstaculos: lo que tiene que pasar para que haya avoidance

Durante esta depuración cerramos la cadena completa:

1. escena con obstaculo visible,
2. percepción lo detecta,
3. `ObstacleModel` publica obstáculos,
4. avoidance deja de estar deshabilitado,
5. `PathPlanner` mete segmentos de evasión.

### Lo que se comprobó

- `ObstacleModel` ya funciona en el scenario RL cuando el game state esta activo.
- el robot RL era tratado como goalkeeper (`player 1`) y por eso el skill apagaba `obstacleAvoidance`.
- se parcheó `SkillBehaviorControl` para que el equipo RL no desactive avoidance solo por ser goalkeeper.

Consecuencia práctica:

- ver `libwalk disabled=0` significa que el bypass ya esta funcionando;
- ver `avoid ... reqp=1` o `walkp=1` con segmentos reales significa que el planner ya está evadiendo.

No necesitas ver `libwalk sel != 0` para concluir que hay evasión:

- `LibWalk` es avoidance local,
- `PathPlanner` puede generar evasión de trayectoria aunque `LibWalk` deje `sel=0`.

## Como leer los logs del probe

### `walkToField(...)`

Es la acción heurística hacia el target actual.

### `searchBallAt(...)`

Aparece cuando la pelota deja de ser confiable y el probe vuelve al modo de búsqueda.

Importante:

- el objetivo de búsqueda usa `LIVE_RESET_BALL_POS`
- hoy ese punto por default es `(0, 0)`

Entonces, si pones la pelota inicial en otro sitio pero el robot luego la pierde, el probe va a empezar a buscar hacia `(0, 0)`.

Eso explica varias trayectorias "raras" que no eran fallas de escena sino del modo de búsqueda.

### `obstacles=(...)`

Campos más útiles:

- `n`: obstáculos publicados en `ObstacleModel`
- `front`: cuántos están delante
- `near`: distancia y ángulo del más cercano
- `opp`, `unc`: tipo aproximado

### `avoid=(...)`

Si ves:

- `reqp=1` o más
- `walkp=1` o más
- segmentos reales en `reqseg` o `walkseg`

entonces ya hay avoidance/path planning activo.

### `libwalk=(...)`

Si ves:

- `disabled=1`: avoidance estaba apagado por lógica superior
- `disabled=0`: ya entra al pipeline normal
- `obs=1, active=1`: `LibWalk` ya considera obstáculos

Que `sel=0.00` no implica automáticamente "no evade". Solo implica que `LibWalk` no está metiendo un offset angular extra en ese frame.

## Checklist para una escena nueva

Antes de perder tiempo depurando:

1. La escena apunta al scenario correcto en `Compound name="scenarios"`.
2. El `.con` llama `Includes/RLNatural` y hace `gc playing`.
3. El robot controlado por Python es `robot1` y corresponde al `player-number` esperado.
4. Si quieres respetar la pose del `.ros2`, corres con `--no-reset`.
5. Si quieres un obstáculo fijo, va en `extras`.
6. Si el obstáculo está en `robots`, asume que puede ser recolocado.
7. Si el probe pierde la pelota, recuerda que `searchBallAt` vuelve a `(0, 0)` por default.
8. Si el log muestra `ObstacleModel` pero no `avoid`, revisa si el problema es de game state, de goalkeeper o de geometría.

## Checklist para un scenario nuevo

1. Clona desde `4v4_RL3D` o `4v4_RL2D`, no desde un scenario normal cualquiera.
2. Verifica `threads.cfg` antes de tocar providers.
3. Mantén `SkillRequest -> RLSkillProvider`.
4. No reintroduzcas oracle en `RobotPose`, `BallModel` u `ObstacleModel` salvo que ese sea el objetivo experimental.
5. Verifica `setupPosesProvider.cfg` porque puede pisar lo que creías controlar desde la escena.
6. Si el robot RL sigue siendo `player 1`, recuerda que ya existe un parche para no apagar obstacle avoidance por goalkeeper en el equipo RL.

## Comandos utiles

### Probe visible 3D respetando escena inicial

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 live_simrobot_probe.py --launch --policy heuristic --simrobot-profile 3d --no-reset
```

### Probe visible con escena específica

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 live_simrobot_probe.py --launch --policy heuristic --simrobot-profile 3d \
  --scene /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenes/RL1v0_3D_extra_aligned.ros2 \
  --no-reset
```

### Wrapper que usa escena extra por default

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 live_simrobot_probe_extra.py --launch --policy heuristic --simrobot-profile 3d --no-reset
```

## Resumen corto

- La escena define el mundo visible.
- El scenario define el pipeline B-Human.
- El probe puede pisar la pose inicial si no usas `--no-reset`.
- `robots` no es para props fijos; `extras` sí.
- Si cambias la pose de un jugador, decide si eso debe vivir en la escena, en `SetupPoses`, o en el reset Python.
- Para scenes RL nuevas, el minimo viable no es solo "que abra SimRobot": tiene que arrancar en `gc playing`, con providers correctos y con el robot controlado por el bridge correcto.
