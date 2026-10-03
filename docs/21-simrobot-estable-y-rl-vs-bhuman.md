# SimRobot estable y partido hibrido RL vs B-Human

## Objetivo

Dejar por escrito que fue lo que se tuvo que respetar para que `simrobot`
volviera a ser util para RL, y como correr el partido mixto `RL vs B-Human`
para observar activaciones de `Zweikampf`.

## Que se corrigio para que SimRobot dejara de quedarse freeze

El problema que habia que documentar no era un bug del backend embebido. Era el
freeze de escenas comunes al separarse del comportamiento que si seguia vivo en
`master`.

La referencia correcta para depurar esa clase de regresiones es:

1. comparar contra `master`
2. verificar si la escena comun entra de verdad a `playing`
3. revisar si el escenario sigue exportando el flujo de juego que esperan
   `GameStateProvider`, `SkillBehaviorControl` y `StrategyBehaviorControl`

Los cambios que quedaron ligados a esa recuperacion fueron:

- no asumir `F5` en `simrobot_visible`; la escena visible arranca sola y los
  scripts solo deben esperar a que cargue
- permitir que las escenas comunes salgan de `standby` y entren a juego tambien
  cuando el flujo RL esta activo
- evitar configuraciones de escenario que congelan o contaminan la escena
  comun, como providers de `GroundTruthRobotPose` donde no corresponden
- mantener vivo el camino normal de `SkillRequest -> MotionRequest -> SimRobot`
  en vez de arreglar sintomas desde Python
- para cualquier regresion parecida, comparar primero contra `master` antes de
  inventar workarounds

## Flujo que no se debe romper

Si este flujo se rompe, lo normal es volver a caer en escenas que cargan pero
no se comportan igual entre visible y embebido:

```text
Python RL
  -> RLSharedState
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine / WalkingEngine
  -> JointRequest
  -> SimRobot
```

Reglas practicas:

- `simrobot_visible` se usa para comparar contra el comportamiento real de la
  escena y del robot en pantalla
- `simrobot` embebido se usa para episodios y stepping programatico
- si una escena comun se queda freeze, primero se revisa el flujo de game state,
  refereeing y scenario config antes que la politica RL

## Partido hibrido RL vs B-Human

Se agrego el runner:

```text
RL/run_rl_vs_bhuman_3v3.py
```

Y la escena del lado B-Human:

```text
SabanaHerons2026/Config/Scenes/RLvsBH3v3_3D.ros2
SabanaHerons2026/Config/Scenes/RLvsBH3v3_3D.con
```

La prueba monta:

- equipo `24`
- `player 1`: B-Human puro
- `player 2` y `player 3`: controlados por RL
- equipo rival completo: B-Human puro

Esto se activa con:

- `PYBH_RL_OVERRIDE_TEAM`
- `PYBH_RL_ACTIVE_PLAYERS`

Uso:

```bash
cd 
PYTHONPATH=/Make/Python \
python3 run_rl_vs_bhuman_3v3.py --launch --seconds 60
```

## Como leer la salida de Zweikampf

El runner imprime:

- eventos `[zweikampf]` cuando el flag sube de `0` a `1`
- trazas por jugador con `zk=activo/calls`

Ejemplo:

```text
zk=0/23
```

Significa:

- `0`: `zweikampf_active` esta apagado en esa observacion
- `23`: `zweikampf_call_count` ya acumulo 23 entradas a la skill en esa corrida

## Recomendacion para futuras depuraciones

Si algo se rompe otra vez, el orden correcto de depuracion es:

1. comparar `simrobot_visible` contra `simrobot` embebido con la misma escena
2. confirmar reset, pose y telemetria antes de tocar la politica
3. revisar `SabanaHerons2026` si la diferencia aparece solo en `headless`
