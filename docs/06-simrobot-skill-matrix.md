# Validacion de skills RL SimRobot y comparativo 2D vs 3D

Fecha: 2026-05-01 16:47:49

## Configuracion

- perfiles validados: `2d`, `3d`
- backend usado: `PyBHBackend(mode="simrobot")`
- frames por skill: `20`
- skill set validado: `stand, walk, shoot, pass, dribble, block, mark, observe`

## Resumen cuantitativo

| profile | skills_ok | reset_ms_avg | step_ms_avg | cpu_ms_avg | rss_kb_avg | abs_speed_xy_avg |
| --- | --- | --- | --- | --- | --- | --- |
| 2d | 8/8 | 117.7 | 326.9 | 1138.9 | 179997 | 112.4 |
| 3d | 8/8 | 48.4 | 235.6 | 160.5 | 191310 | 0.0 |

## Matriz por skill

| profile | skill | pass | requested | skill_req | motion | engine_in | phase | fall | reset_ms | step_ms | cpu_ms | rss_kb | speed_xyr | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2d | stand | yes | stand | stand | stand | stand | 1 | 0 | 794.0 | 283.4 | 1144.2 | 179964 | 0.0/0.0/0.00 | ok |
| 2d | walk | yes | walkTo | walk | walkToPose | walkToPose | 1 | 0 | 15.6 | 333.6 | 1145.0 | 179980 | 145.8/0.0/0.00 | ok |
| 2d | shoot | yes | shoot | shoot | walkToBallAndKick | walkToBallAndKick | 1 | 0 | 33.0 | 332.8 | 1136.5 | 179996 | -30.9/0.0/0.04 | ok |
| 2d | pass | yes | pass | pass | walkToBallAndKick | walkToBallAndKick | 1 | 0 | 16.6 | 332.9 | 1139.9 | 180004 | -109.0/0.0/-0.29 | ok |
| 2d | dribble | yes | dribble | dribble | dribble | dribble | 1 | 0 | 16.6 | 333.2 | 1128.8 | 180008 | 203.6/16.1/0.31 | ok |
| 2d | block | yes | block | block | walkToPose | walkToPose | 1 | 0 | 33.9 | 332.3 | 1130.5 | 180008 | 142.4/0.0/0.04 | ok |
| 2d | mark | yes | mark | mark | walkToPose | walkToPose | 1 | 0 | 16.6 | 333.6 | 1151.1 | 180008 | 94.1/14.1/0.59 | ok |
| 2d | observe | yes | observe | observe | walkToPose | walkToPose | 1 | 0 | 15.7 | 333.8 | 1135.2 | 180008 | 142.9/0.0/-0.03 | ok |
| 3d | stand | yes | stand | stand | stand | stand | 1 | 1 | 275.0 | 211.6 | 392.8 | 191300 | 0.0/0.0/0.00 | ok |
| 3d | walk | yes | walkTo | walk | walkToPose | walkToPose | 1 | 1 | 11.6 | 239.7 | 134.0 | 191308 | 0.0/0.0/0.00 | ok |
| 3d | shoot | yes | shoot | shoot | walkToBallAndKick | walkToBallAndKick | 2 | 1 | 26.5 | 237.2 | 127.3 | 191312 | 0.0/0.0/0.00 | ok |
| 3d | pass | yes | pass | pass | walkToBallAndKick | walkToBallAndKick | 2 | 1 | 11.0 | 240.6 | 129.5 | 191312 | 0.0/0.0/0.00 | ok |
| 3d | dribble | yes | dribble | dribble | dribble | dribble | 2 | 2 | 11.4 | 239.7 | 122.7 | 191312 | 0.0/0.0/0.00 | ok |
| 3d | block | yes | block | block | walkToPose | walkToPose | 2 | 2 | 25.6 | 238.3 | 127.3 | 191312 | 0.0/0.0/0.00 | ok |
| 3d | mark | yes | mark | mark | walkToPose | walkToPose | 4 | 3 | 13.9 | 237.3 | 122.2 | 191312 | 0.0/0.0/0.00 | ok |
| 3d | observe | yes | observe | observe | walkToPose | walkToPose | 4 | 3 | 12.3 | 240.1 | 128.0 | 191316 | 0.0/0.0/0.00 | ok |

## Lectura tecnica

### 2D

- Usa escena `RL1v0.ros2d` y escenario `4v4_RL2D`.
- Tras el fix de pose oracular, `walk`, `block`, `mark` y `observe` dejaron de orbitar la pelota y pasaron a producir `walkToPose` coherente.
- El reset frio fue caro (`794.0 ms` en `stand`), pero el reset caliente promedio bajo a `21.1 ms`.
- El RSS promedio fue `179997 KB` y el maximo observado `180008 KB`.
- Las 8 corridas de 2D reportaron velocidad no nula en `7/8` casos, asi que `motion_speed_*` es una señal util en este perfil.
- `engine_phase` se mantuvo en `1` y `fall_state` promedio fue `0.0`.

### 3D

- Usa escena `RL1v0_3D.ros2` y escenario `4v4_RL3D`.
- Valida el flujo natural real de B-Human: `SkillBehaviorControl -> MotionEngine -> JointRequest`.
- El reset frio fue `275.0 ms`, y el reset caliente promedio `16.0 ms`.
- El RSS promedio fue `191310 KB`, unos `11314 KB` por encima de 2D.
- `engine_phase` recorrió `1,2,4` y `fall_state` promedio fue `1.8`.
- `motion_speed_*` salio nulo en `8/8` corridas. Eso no invalida la caminata: en 3D embebido la validacion fuerte sigue siendo `motion_request`, fases y joints.

### Diferencias que si importan

- En esta muestra, `3d` no fue mas lento en `step_wall_ms_avg`; de hecho quedo `91.4 ms` por debajo de `2d`. Eso no significa que el motor 3D sea mas barato: significa que este benchmark mide el proceso completo Python + bridge + stepping, no un microbenchmark aislado del motor fisico.
- 2D expone mejor la velocidad planar efectiva en `motion_speed_*`; 3D expone mejor el estado interno del motor a traves de `engine_phase`, `fall_state` y joints, aunque la traslacion global visible pueda tardar mas en aparecer.
- En skills de pelota (`shoot`, `pass`, `dribble`) ambos perfiles entran al request esperado, pero 3D necesita mas frames para manifestar el comportamiento fisico completo. Por eso esta matriz valida pipeline y activacion, no exito final de la patada.
- El costo de memoria adicional de 3D es persistente pero no explosivo en esta muestra: alrededor de 12 MB sobre 2D. El costo cualitativo fuerte de 3D no es solo RSS, sino la dependencia en joints, contacto y fases del motor.
- La mayor fragilidad operativa estuvo del lado 2D antes del fix de `RobotPose` y `BallModel`: cuando la pose era estimada y no oracular, `SkillBehaviorControl` construia targets relativos absurdos y el robot orbitaba la pelota.

### Limites de esta comparacion

- `cpu_ms` puede superar `wall_ms` porque `time.process_time()` suma CPU consumido por varios threads internos de B-Human.
- El primer reset de cada perfil mezcla costo de carga de escena con costo de reset. Por eso en la lectura comparativa hay que separar reset frio de reset caliente.
- `motion_speed_*` no es simetrico entre perfiles en esta instrumentacion. Es comparable dentro de 2D, pero no debe usarse como la señal principal para decir que 3D no se mueve.
- Este benchmark usa `20` frames por skill. Es suficiente para validar enrutamiento y activacion, pero no para evaluar calidad completa de locomocion, alineacion fina o remates largos.

## Criterio de validacion

- `obs_ready=True`.
- `requested_skill` coincide con la skill enviada, aceptando `walk -> walkTo`.
- `skill_behavior_motion_request` cae en el conjunto esperado para la skill.
- `motion_engine_input_request` coincide con la salida de `SkillBehaviorControl`.

## Notas

- Los tiempos reportados son del proceso Python actual. Incluyen bridge, stepping embebido y espera de observacion.
- `rss_kb` es la memoria residente del proceso en `/proc/self/status` despues de cada skill.
- Las skills `shoot` y `pass` validan que el pipeline entre a `walkToBallAndKick`; no garantizan que el remate completo termine en un solo burst de frames.
- En 3D la caminata puede mostrar poca traslacion global en ventanas cortas aun con joints y fases correctas; por eso la validacion no depende solo de `robot_x`.
