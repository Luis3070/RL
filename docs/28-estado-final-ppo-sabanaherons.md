# 28 - Estado final PPO / SabanaHerons

Fecha de corte: 2026-06-11.

Este documento resume como se llego al estado actual de la integracion PPO entre `RL` y `SabanaHerons2026`, que quedo implementado, que se valido, que queda como cambio local pendiente y cuales son los siguientes puntos tecnicos.

## Repositorios y ramas

`RL`

- Rama: `rl-simrobot3d`.
- HEAD documentado: `26858d4 Track PPO 5of5 checkpoint and repair tooling`.
- Estado al momento de este documento: limpio contra `origin/rl-simrobot3d`.

`SabanaHerons2026`

- Rama: `hsl2026-ppo-naov6-integration`.
- HEAD documentado: `490cd276 Fix full scenario configs and Zweikampf edge case`.
- Estado al momento de este documento: hay cambios locales no commiteados en:
  - `Config/Scenarios/4v4_Full/strategyBehaviorControl.cfg`
  - `Config/teams.cfg`
  - `Make/Linux/.idea/runConfigurations/SimRobot.xml`
  - `Make/Linux/.idea/runConfigurations/Tests.xml`

## Decisiones base

La integracion actual mantiene el contrato de observaciones de 26 entradas. No se uso la expansion MAPPO de 41 observaciones porque el lado de SimRobot/SabanaHerons que estamos probando no acepta todavia ese contrato. La investigacion partio de la linea obs26 que ya se habia identificado como compatible:

- `SabanaHerons2026`: referencia historica `00ba473` en `origin/rl-simrobot3d`.
- `RL`: referencia historica `9a97634`, antes de depender de la expansion a 41 observaciones.

La rama de SabanaHerons no usa bridge Python contra `RL`. La politica se ejecuta embebida desde ONNX con codigo C++ y archivos locales bajo `Config/NeuralNets/RLPolicy/`. Esto mantiene la regla de que solo la rama `rl-simrobot3d` debe tener puente directo Python/RL.

El watchdog contra `stand` quedo disponible como mecanismo local de emergencia, pero el baseline de prueba final lo deja apagado con:

```cfg
embeddedPPOStandWatchdogMs = 0;
embeddedPPOStandWatchdogCooldownMs = 3000;
embeddedPPOStandWatchdogForceWalk = false;
```

Eso significa que la correccion principal buscada no es el fallback en C++, sino una politica que no colapse a `stand` en los estados problematicos.

## Que se hizo en RL

Se investigo el comportamiento de la politica PPO usada por SabanaHerons, especialmente el caso donde la red elegia `stand` repetidamente aunque `walk` estaba disponible. En los logs originales se veia que `dribble` y `shoot` estaban enmascarados por gates, y que la decision real quedaba entre `stand` y `walk` con logits muy cercanos. Eso explicaba por que el partido podia trabarse sin que hubiera un error de runtime.

Para atacar esto desde la raiz se hicieron reparaciones offline sobre checkpoints existentes, en vez de reiniciar un entrenamiento largo online con SimRobot. La idea fue usar el comportamiento ya aprendido y ajustar casos concretos donde el evaluador o los logs mostraban fallas: colapso a `stand`, targets de walk degenerados y casos de preshoot/collision.

Archivos y artefactos relevantes:

- `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`
- `runs/production_stability_repair_v1/checkpoint_stage4_step2500.pt`
- `runs/production_stability_repair_v1/checkpoint_stage4_step7500.pt`
- `runs/production_stability_repair_v1/checkpoint_stage4_step10000.pt`
- `runs/production_preshoot_repair_v1/checkpoint_stage4_step12000.pt`
- `ppo_striker_hsl2026.onnx`
- `ppo_striker_hsl2026.onnx.data`
- `train_stability_repair.py`
- `train_preshoot_repair.py`
- `train_preshoot_blocked_stability_repair.py`
- `train_preshoot_blocked_gate_repair.py`
- `behavioral_evaluator.py`
- `behavioral_stability.py`
- `behavioral_stability_probe.py`
- `test_behavioral_stability.py`
- `.deps/obs26-base/`

El checkpoint que se transplanto inicialmente a SabanaHerons fue:

```text
runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt
```

Ese checkpoint fue convertido a ONNX como:

```text
ppo_striker_hsl2026.onnx
ppo_striker_hsl2026.onnx.data
```

El evaluador de comportamiento ya habia mostrado una mejora historica importante, pero el caso de preshoot seguia siendo el punto delicado. En la documentacion previa quedo registrado que el behavioral evaluator llego a `4/5` con el caso `preshoot_bad_angle_boundary` fallando. Despues se agregaron herramientas y checkpoints para reparar especificamente ese frente, incluyendo el tracking del checkpoint 5/5, pero la recomendacion practica sigue siendo revalidar con el evaluador y con pruebas multi-seed antes de declarar un checkpoint como baseline final de competencia.

## Que se hizo en SabanaHerons2026

Se agrego la ejecucion embebida de PPO desde ONNX dentro de `StrategyBehaviorControl`, usando el modelo en:

```text
Config/NeuralNets/RLPolicy/ppo_striker_hsl2026.onnx
Config/NeuralNets/RLPolicy/ppo_striker_hsl2026.json
```

La configuracion permite activar PPO por escenario, limitarlo a un equipo o dejarlo dinamico para quien tenga `playBall`. Los campos principales son:

```cfg
enableEmbeddedPPO = true;
embeddedPPOModelPath = "Config/NeuralNets/RLPolicy/ppo_striker_hsl2026.onnx";
embeddedPPOTeamNumber = -1;
embeddedPPODynamicPlayBall = true;
embeddedPPOPlayers = [];
embeddedPPOStandWatchdogMs = 0;
embeddedPPOStandWatchdogCooldownMs = 3000;
embeddedPPOStandWatchdogForceWalk = false;
```

`embeddedPPOTeamNumber = -1` significa que se usa el equipo propio del robot en ese escenario. Para forzar PPO solo en un equipo especifico, se puede poner el numero de equipo. Para una prueba PPO vs no PPO, se deja PPO activo solo en el escenario/equipo deseado y el otro equipo corre con comportamiento B-Human normal.

Tambien se excluyeron los arqueros de la ejecucion RL: si el jugador es goalkeeper, no entra a Embedded PPO.

Se agregaron logs de diagnostico para confirmar carga, modo y accion:

```text
Warning: [RL] Embedded PPO model loaded
Warning: [RL] mode=EmbeddedPPOActive ...
Warning: [RL] Embedded PPO action ... logits[...] params[...] targetAbs[...] targetRel[...]
Warning: [RL] Embedded PPO inference failed
Warning: [RL] Embedded PPO model load failed
```

Estos logs fueron claves para distinguir dos problemas distintos:

- La politica anterior podia escoger `stand` demasiadas veces.
- La politica reparada podia escoger `walk`, pero si los parametros generaban targets relativos casi nulos, el robot visualmente podia quedarse quieto aunque el skill fuera `walk`.

Por eso se agrego logging de parametros (`params[x,y,th,p]`) y targets (`targetAbs`, `targetRel`) para depurar si el problema era decision de skill o geometria/parametrizacion del target.

## Sincronizacion con ramas de SabanaHerons

La rama `hsl2026-ppo-naov6-integration` fue revisada contra `StableGame`, `Play 2026` y `master`.

Resultado:

- `StableGame` ya estaba cubierto como ancestro o equivalente.
- `Play 2026` no tenia cambios patch-equivalent faltantes para esta rama.
- `master` si tenia cambios relevantes y se integraron por cherry-pick de commits no merge para no borrar la integracion PPO.

Antes de sincronizar se dejo backup en:

```text
backup/hsl2026-ppo-naov6-integration-before-master-sync
```

Tambien quedaron integrados cambios recientes de Trionda/YOLO/Kalman, whistle baseline, README y ajustes de estrategia.

## Fixes de compilacion y runtime en SabanaHerons

Se arreglo un problema de `Nao` causado por PCH obsoleto. El error era:

```text
DebugDrawings.h has been modified since the precompiled header ... was built
```

La solucion fue limpiar solo el build de `Nao` y recompilar:

```bash
cmake --build Build/Linux/CMake/Develop/Nao-prefix/src/Nao-build --target clean
Make/Linux/compile Develop Nao
```

Despues de eso `Nao` compilo correctamente.

Tambien se normalizaron los `yoloBallDetector.cfg` de escenarios para que tengan los atributos que espera el codigo actual, incluyendo:

```cfg
modelName = "NeuralNets/BallDetector/yolo_ball_best1.onnx";
inferenceIntervalMs = 120;
enableKalman = true;
publishPredictedPercepts = true;
kalmanInitConf = 0.4;
kalmanActiveConf = 0.15;
kalmanInstantInitConf = 0.65;
kalmanInitConfirmFrames = 2;
kalmanInitGatePx = 100;
kalmanFarGatePx = 260;
kalmanNearGateScale = 0.14;
kalmanMaxMissed = 8;
kalmanStrongPredictionMissed = 3;
kalmanMaxSpeedPx = 140;
kalmanMissedVelocityDecay = 0.82;
kalmanProcessNoise = 18;
kalmanMeasurementNoise = 24;
```

Se agregaron los campos faltantes de PPO watchdog a `3v3_Full`, `4v4_Full` y `4v4_RL3D` para evitar errores de streaming como:

```text
attribute 'embeddedPPOStandWatchdogMs' not found
```

Finalmente se corrigio un `VERIFY` en `Zweikampf.cpp` que podia abortar SimRobot en geometria degenerada:

```text
VERIFY(Geometry::getIntersectionPointsOfLineAndRectangle(...)) failed
```

El fix evita abortar cuando la interseccion no existe o los valores no son finitos. En vez de caer por `VERIFY`, descarta ese candidato o sale de la rama correspondiente.

Validacion hecha despues de esos cambios:

```bash
Make/Linux/compile Develop SimRobot
```

## Estado de configuracion de escenarios

En el commit `490cd276`, `3v3_Full` y `4v4_Full` quedaron preparados con campos PPO completos.

Estado actual observado en el working tree:

`3v3_Full`

```cfg
enableEmbeddedPPO = true;
embeddedPPOTeamNumber = -1;
embeddedPPODynamicPlayBall = true;
embeddedPPOStandWatchdogMs = 0;
```

`4v4_Full`

```cfg
enableEmbeddedPPO = false;
embeddedPPOTeamNumber = -1;
embeddedPPODynamicPlayBall = false;
embeddedPPOStandWatchdogMs = 0;
```

Importante: esa desactivacion de `4v4_Full` esta como cambio local no commiteado al momento de este documento. El commit `490cd276` tenia `4v4_Full` con PPO activo. Antes de una prueba o push final hay que decidir si `4v4_Full` debe quedar con PPO encendido o apagado.

`Default`

```cfg
enableEmbeddedPPO = true;
embeddedPPOTeamNumber = 5;
embeddedPPODynamicPlayBall = true;
embeddedPPOStandWatchdogMs = 0;
```

## Como probar

Compilar SimRobot:

```bash
cd 
Make/Linux/compile Develop SimRobot
```

Si aparece el error de PCH en `Nao`, limpiar solo el build de `Nao`:

```bash
cd 
cmake --build Build/Linux/CMake/Develop/Nao-prefix/src/Nao-build --target clean
Make/Linux/compile Develop Nao
```

Para confirmar que PPO esta corriendo, buscar estos logs:

```text
[RL] Embedded PPO model loaded
[RL] mode=EmbeddedPPOActive
[RL] Embedded PPO action
```

Si solo aparecen logs como `mode=BHuman`, entonces ese robot no esta usando PPO. Las razones normales son:

- PPO esta desactivado en el escenario.
- El equipo no coincide con `embeddedPPOTeamNumber`.
- El jugador no tiene rol `playBall` cuando `embeddedPPODynamicPlayBall = true`.
- El jugador es arquero.
- El modelo ONNX no cargo.

Para evaluar desde `RL`, usar el behavioral evaluator y los probes de estabilidad sobre el checkpoint candidato antes de transplantar otro ONNX. El candidato principal transplantado fue:

```text
runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt
```

El candidato posterior de reparacion preshoot es:

```text
runs/production_preshoot_repair_v1/checkpoint_stage4_step12000.pt
```

Antes de reemplazar el ONNX de SabanaHerons por otro candidato, conviene correr:

```bash
cd 
python3 behavioral_evaluator.py --checkpoint runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt
python3 behavioral_stability_probe.py --checkpoint runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt
```

Si el CLI local cambia, revisar `python3 behavioral_evaluator.py --help` y `python3 behavioral_stability_probe.py --help`.

## Pendientes y riesgos

El riesgo principal ya no es solo que la politica elija `stand`. Ahora tambien hay que vigilar que `walk` no genere targets degenerados o demasiado pequenos. Para eso los logs con `params`, `targetAbs` y `targetRel` son obligatorios durante pruebas.

El caso preshoot/collision sigue siendo el frente que requiere mas validacion. Aunque hay tooling y checkpoint de reparacion, la documentacion historica indica que el behavioral evaluator llego a `4/5` con fallo en `preshoot_bad_angle_boundary`. No se debe declarar una politica final solo por un run aislado: hay que repetir evaluator y stability probe multi-seed.

Antes de cerrar una version final de competencia:

- Decidir si `4v4_Full` queda con PPO activo o desactivado.
- Resolver los cambios locales en `Config/teams.cfg`.
- Decidir si se conservan o restauran las eliminaciones locales en `.idea/runConfigurations`.
- Revalidar `3v3_Full` y `4v4_Full` con SimRobot.
- Revalidar behavioral evaluator en `RL`.
- Si se usa `checkpoint_stage4_step12000.pt`, convertirlo a ONNX y transplantarlo explicitamente a SabanaHerons.

## Commits relevantes

`RL`

- `26858d4 Track PPO 5of5 checkpoint and repair tooling`
- `b2f62ae Create onnx from production_stability_repair_v1/checkpoint_stage4_step5000.pt`
- `a733c39 Add MAPPO observation wire probes and tests`
- `9a97634 Track PPO stability repair checkpoints`
- `91f3885 Stabilize PPO baseline and update transplant docs`

`SabanaHerons2026`

- `490cd276 Fix full scenario configs and Zweikampf edge case`
- `69e2fa48 Mention new whistle baseline in README`
- `7af82ac9 Update README for current HSL and Trionda baseline`
- `2dd7c148 Preserve local Kalman scenario configs`
- `ca20c84d Restore baseline Kalman far-ball detector`
- `712b3a41 Integrate Trionda ball detection with ScaledDimensions`
- `9b3f4e98 Update embedded PPO repaired policy`
- `5863bd50 new onnx`
- `9b8b919c Add embedded PPO stand watchdog`

