# Plan De Transplante PPO A NAO v6 Para RoboCup HSL 2026

Fecha de corte: 2026-06-08  
Repositorio RL: `rl-simrobot3d` en commit `ed8ffdc`  
Repositorio base B-Human/Sabana esperado: `../SabanaHerons2026`

## 1. Alcance exacto

Este documento describe el transplante **unicamente del policy PPO single-agent** de este proyecto a un robot NAO v6 para jugar en RoboCup Humanoid Soccer League 2026.

El objetivo no es portar todo el sistema de entrenamiento. El objetivo es ejecutar en el NAO un actor PPO ya entrenado, de forma local, determinista, segura y compatible con B-Human, para que produzca decisiones de alto nivel tipo `SkillRequest`.

Dentro del alcance:

- Exportar y ejecutar el actor PPO entrenado.
- Reproducir en robot real el mismo contrato de observacion de 26 dimensiones.
- Reproducir el mismo contrato de accion estructurada de 5 valores.
- Reproducir las mismas gates de `shoot` y `dribble`.
- Emitir `SkillRequest` hacia el stack B-Human/Sabana.
- Mantener a B-Human como responsable de locomocion, kick, balance, get-up, path planning, motion safety y cumplimiento del `GameController`.
- Validar el PPO primero en SimRobot visible 3D y luego en NAO v6 real.
- Cumplir las reglas HSL 2026, especialmente estados de juego, comunicacion, seguridad, pelota, contacto y comportamiento en stop/set.

Fuera del alcance por ahora:

- MAPPO.
- Multi-agent RL aprendido.
- Graph learning.
- Reentrenar con observaciones nuevas.
- Cambiar `OBS_SIZE`.
- Cambiar `SKILL_ORDER`.
- Cambiar el dataset BC productivo.
- Entrenar en el robot.
- Usar Python como loop principal de inferencia en partido oficial.
- Controlar joints, velocidades articulares o motion primitives directamente desde PPO.
- Usar observaciones sinteticas, corregidas por SimRobot o root-assist en robot real.
- Reemplazar el comportamiento completo de B-Human para goalkeeper, set plays o estados no abiertos.

## 2. Principio arquitectonico

La integracion correcta es:

```text
Representaciones B-Human naturales
  -> encoder PPO compatible con RL
  -> actor PPO
  -> decoder/gate/action repair
  -> SkillRequest
  -> SkillBehaviorControl
  -> MotionRequest
  -> MotionEngine / WalkingEngine / KickEngine
  -> NAO v6
```

La integracion incorrecta, y prohibida para este transplante, es:

```text
PPO -> joints / motores / LoLA directo
PPO -> MotionRequest directo saltando SkillBehaviorControl
PPO -> walkTo relativo cuando B-Human espera coordenadas de campo
PPO -> observacion corregida/sintetica en robot real
PPO -> decisiones activas durante set/stopped/penalized
```

La regla base del proyecto sigue vigente: RL decide la intencion de alto nivel; B-Human decide como ejecutarla fisicamente.

## 3. Estado actual relevante

### 3.1 Proyecto RL

El repo RL ya esta actualizado a `ed8ffdc`.

El contrato actual del PPO esta en:

- `observation.py`: `OBS_SIZE = 26`.
- `action.py`: `SkillType`, `SKILL_ORDER`, `ACTION_PARAM_MASKS`.
- `gate.py`: `SkillGate`, `GateConfig`, armado de `shoot` y `dribble`.
- `train.py`: arquitectura del actor PPO, mascara de skills y carga de checkpoints.
- `environment.py`: decode de accion, repair de target y stepping contra B-Human/SimRobot.
- `reward.py`: anchors de walk/dribble/shoot y condiciones de shoot-zone.
- `FOR_TRAIN_PRODUCTION.md` [removed from this branch; available on rl-v4.2-multiagent-gk]: contexto historico del primer intento productivo.
- `docs/notes/cambios_para_produccion.md`: resumen historico del salto original a 4/5; ya no debe leerse como baseline vigente.
- `docs/25-guia-transplante-ppo-bhuman.md`: referencia exacta del pipeline de inferencia que se debe replicar en robot.

Punto critico actualizado:

- El checkpoint historico `runs/production_finetune_v1/checkpoint_stage4_step35000.pt` ya no es la referencia correcta para transplante.
- La referencia vigente del transplante es ahora:
  - checkpoint candidato: `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`
  - evaluacion canonica actual: `runs/behavioral_evaluator/behavior_20260608_181747/summary.csv`
  - estado canonico validado: **3/5** escenarios aprobados antes de cortar la corrida (`finish_band_2p5_3m_clean`, `finish_band_2p5_3m_static_lane`, `dynamic_full_field_pressure`)
  - evidencia extra de estabilidad parcial: `runs/behavioral_stability/stability_20260608_175828/per_rep_summary.csv`
- El `step10000` de `production_readiness_v2` queda como referencia historica importante, pero no como baseline vigente para transplante.
- El objetivo del transplante no es “mejorar el PPO” en esta fase, sino **replicar exactamente** en B-Human el comportamiento de inferencia del mejor checkpoint hoy revalidado. Cualquier optimizacion posterior debe partir de una paridad funcional ya validada.

### 3.2 SabanaHerons2026

La rama inspeccionada de `../SabanaHerons2026` es `test-old-yolo-model-with-ball-fixes`.

Esa rama tiene cambios locales relevantes:

- `Config/Locations/*/ballSpecification.cfg` modificados.
- `Config/NeuralNets/BallDetector/yolo_ball.onnx` modificado.
- submodulos `Util/CompiledNN` y `Util/SimRobot` modificados.
- `BHuman2023_Architecture_Review.md` sin trackear.

Esa rama no contiene todavia el bridge RL activo:

- No esta `Src/Libs/Python/Controller/RLSharedState.h`.
- No esta `Src/Modules/Infrastructure/RLSkillProvider/RLSkillProvider.cpp`.
- No estan `Config/Scenarios/4v4_RL2D` ni `Config/Scenarios/4v4_RL3D`.
- `SkillBehaviorControl.cpp` no exporta observaciones RL.
- `Module.cpp` tiene bindings `pybh` basicos, no el host SimRobot/RL completo.

Pero el repo si tiene ramas y commits con el bridge:

- `rl-simrobot3d`
- `origin/rl-simrobot3d`
- commits como `4a9029f4 Sync B-Human RL stack`, `e6003730 Stabilize RL observation export and embedded SimRobot host`, `53b09605 Fix RL ball export telemetry fields`.

Conclusion: el transplante necesita una rama de integracion que combine la base HSL/YOLO actual con el bridge RL probado, sin pisar los cambios de balon y percepcion.

### 3.3 Reglas HSL 2026 relevantes

La fuente local es `../Rules-HSL-2026.pdf`, fecha 2026-05-26.

Restricciones principales para NAO v6:

- NAO v5 y NAO v6 son plataformas standard aprobadas.
- Small Division: campo 9.0 m x 6.0 m.
- Small Division: FIFA Mini Ball.
- Small Foundation: 4 robots.
- Small Advanced: 7 robots.
- Debe haber maximo un goalkeeper por equipo y, a full strength, debe haber goalkeeper.
- No se permite procesamiento remoto, remote brain, teleoperacion ni sensing externo.
- La comunicacion robot-robot debe usar WiFi del evento, UDP broadcast, payload maximo 512 B, puerto `10000 + team number`.
- El presupuesto de comunicacion por equipo es 12000 mensajes por partido, con incrementos definidos para tiempo agregado/extra time.
- Debug externo: maximo un paquete UDP por segundo hacia un unico equipo por cable.
- El robot debe obedecer `GameController`.
- En `initial`: puede mover cabeza/calibrar, no locomocion.
- En `ready`: puede moverse a posiciones legales; duracion normal 45 s.
- En `set`: debe quedarse quieto; no locomocion salvo get-up minimo si cayo.
- En `playing`: aplican reglas normales.
- En `penalized`: debe permanecer quieto fuera del campo hasta fin de penalizacion.
- En `finished`: debe cesar actividad de juego.
- En brief stop: debe parar inmediatamente, quedarse quieto y no hacer get-up.
- Movimiento en `set` implica penalizacion in-place.
- Movimiento en stop implica standard penalty.
- Ball holding: 5 s para field player, 10 s para goalkeeper dentro de su penalty area.
- Fallen/incapable: debe intentar get-up antes de 20 s y no fallar mas de 3 intentos, o 4 si fue perturbado.
- Leaving field, arms/hands, pushing y goal-area overcrowding son penalizables.
- En set/playing, maximo tres jugadores del mismo equipo dentro de su propia goal area.
- Kick-off tiene restricciones de posicion, no permite gol directo, y con 3+ robots exige dos robots tocando antes de anotar.
- Free kicks duran 45 s y usan avoidance region de radio igual al center circle, salvo goal kick donde la avoidance region es toda el penalty area.
- Kick-in/throw-in es indirect free kick.
- Corner, goal kick y pushing free kick son direct free kicks.
- Penalty kick tiene ready de 30 s y ejecucion de 60 s.

El PPO no debe intentar resolver estas reglas por aprendizaje. Deben implementarse como guardrails deterministas alrededor del policy.

## 4. Contrato PPO que no se debe romper

### 4.1 Observacion de 26 dimensiones

La observacion productiva es:

```text
0  robot_x
1  robot_y
2  robot_theta
3  ball_rel_x
4  ball_rel_y
5  ball_end_rel_x
6  ball_end_rel_y
7  ball_vel_x
8  ball_vel_y
9  time_since_ball_seen
10 time_since_ball_disappeared
11 ball_seen_percentage
12 ball_consistent
13 can_score_now
14 shot_quality_no_obstacles
15 shot_opening_with_obstacles
16 pass_options_count
17 nearest_teammate_dist
18 nearest_opponent_dist
19 nearest_uncertain_obstacle_dist
20 nearest_teammate_front_dist
21 nearest_opponent_front_dist
22 nearest_uncertain_front_dist
23 shoot_armed
24 dribble_armed
25 shoot_arm_progress
```

Estas 26 features deben producirse en el NAO a partir de representaciones B-Human reales. No se agregan features de rol, grafo, teammate pose ni informacion global no disponible naturalmente.

### 4.2 Normalizacion

La normalizacion usada por el actor debe ser bit-a-bit compatible a nivel semantico con la que se usa durante entrenamiento:

- Posiciones de campo escaladas por `FIELD_X_HALF = 4500`.
- Eje Y escalado por `FIELD_Y_HALF = 3000`.
- Angulos normalizados a rango compatible con `[-pi, pi]`.
- Distancias de ball/obstacle escaladas con los mismos factores que el encoder de `environment.py`.
- Tiempos de percepcion saturados igual que en Python.
- Flags booleanos como `0.0` o `1.0`.

Se debe crear una prueba de paridad con logs: dado un snapshot de representaciones B-Human, el encoder C++ y el encoder Python deben producir la misma observacion dentro de tolerancias documentadas.

### 4.3 Accion de 5 valores

La accion PPO es:

```text
[skill_id, target_x, target_y, target_theta, pass_target]
```

La interpretacion actual es:

- `skill_id`: skill discreta.
- `target_x`: residual/target segun skill, actualmente walk tiene X pinneado por mascara.
- `target_y`: residual lateral.
- `target_theta`: residual angular.
- `pass_target`: indice o parametro de pase, no prioritario en este transplante PPO single-agent.

Skills:

```text
0 stand
1 walk
2 shoot
3 pass
4 dribble
5 block
6 mark
7 observe
```

Para el primer transplante PPO real, las skills activas deben limitarse a:

```text
stand, walk, dribble, shoot
```

`pass`, `block`, `mark` y `observe` pueden permanecer en el enum, pero deben estar mascaradas o delegadas a B-Human salvo que exista una razon de prueba especifica.

### 4.4 Mascaras de parametros

Las mascaras actuales son:

```text
stand   -> no parametros
walk    -> target_y, target_theta
shoot   -> no parametros
pass    -> pass_target
dribble -> target_y
block   -> target_y
mark    -> target_y
observe -> no parametros
```

El decoder en NAO debe aplicar la misma mascara antes de construir `SkillRequest`.

### 4.5 Gates

`SkillGate` es parte del contrato productivo, no una mejora opcional.

Condiciones base:

- `shoot` solo se permite si `shoot_armed`.
- `dribble` solo se permite si `dribble_armed`.
- cuando `finish_armed = shoot_armed or dribble_armed`, la politica **no** puede seguir usando `walk` ni `stand`;
- el gate y la mascara sobre logits trabajan juntos: el gate arma/desarma, y la mascara fuerza el compromiso con la skill correcta cuando existe oportunidad de finish;
- `shoot` requiere pelota natural fresca.
- `dribble` permite una gracia temporal mayor para evitar deadlock de mirar/acercarse.

Valores relevantes actuales:

- Shoot enter ball distance: 380 mm.
- Shoot enter ball angle: 25 grados.
- Shoot enter goal distance: 3000 mm.
- Shoot hold goal distance: 3200 mm.
- Shoot arm debounce: 3 frames consecutivos.
- Dribble enter ball distance: 450 mm.
- Dribble enter ball angle: 35 grados.
- Dribble hold distance: 600 mm.
- Dribble enter freshness grace: 1500 ms.
- Dribble hold freshness grace: 2500 ms.

La implementacion C++ debe reproducir:

- estado con hysteresis por episodio/robot;
- hold/exit conditions;
- progreso de armado `shoot_arm_progress`;
- `allowed_skills`;
- reinicio de estado al cambio de episodio, penalizacion o reset operacional.

### 4.6 Mascara de logits en inferencia

`docs/25-guia-transplante-ppo-bhuman.md` convierte esta pieza en requisito de transplante, no en detalle opcional.

La red no decide skills directamente desde los logits crudos. El contrato correcto es:

1. construir `obs[23]`, `obs[24]`, `obs[25]` desde `SkillGate`;
2. correr el actor;
3. aplicar `gate_skill_mask_from_obs(obs)` sobre los logits;
4. ejecutar `argmax` sobre los logits ya enmascarados.

Reglas exactas del mask:

- `shoot` se lleva a `-inf` si `shoot_armed` es falso;
- `dribble` se lleva a `-inf` si `dribble_armed` es falso;
- `stand` y `walk` se llevan a `-inf` cuando `finish_armed` es verdadero;
- el stage mask de produccion sigue dejando activas solo las skills de Stage 4.

Consecuencia de diseño:

- fuera de oportunidad de finish, `walk` y `stand` son fallback;
- dentro de oportunidad de finish, el robot debe comprometerse con `shoot` o `dribble`;
- si esta mascara no se replica, el transplante no va a reproducir el `clean_takeover_rate` del baseline candidato vigente.

## 5. Arquitectura final en robot

### 5.1 Modulos propuestos

La ruta limpia en `SabanaHerons2026` debe introducir o adaptar estos componentes:

```text
Src/Modules/Infrastructure/RLPolicyProvider/
  RLPolicyProvider.h
  RLPolicyProvider.cpp
  RLPolicyProvider.cfg

Src/Libs/RL/
  PPOPolicyModel.h
  PPOPolicyModel.cpp
  PPOObservationEncoder.h
  PPOObservationEncoder.cpp
  PPOActionDecoder.h
  PPOActionDecoder.cpp
  PPOSkillGate.h
  PPOSkillGate.cpp
  PPOFailSafe.h
  PPOFailSafe.cpp
```

El nombre exacto puede cambiar para ajustarse al estilo B-Human, pero la separacion conceptual debe mantenerse:

- encoder de observacion;
- inferencia del actor;
- `SkillGate` stateful;
- mascara de logits pre-`argmax`;
- decoder de accion;
- fail-safe;
- provider B-Human.

Adicionalmente, el provider final debe respetar el mismo orden operacional del repo RL actual:

```text
percepcion B-Human
  -> SkillGate.step(raw_obs)
  -> obs[23:25] baked
  -> actor PPO
  -> gate_skill_mask_from_obs(obs)
  -> argmax(masked_logits)
  -> parameter masking
  -> anchor + residual decode
  -> target repair / projection
  -> SkillRequest
```

### 5.2 Representaciones requeridas

El provider PPO debe leer solo representaciones disponibles en el robot:

- `RobotPose`
- `BallModel` o `FieldBall`
- `ObstacleModel`
- `ExpectedGoals`
- `TeamData`
- `GameState`
- `RobotInfo`
- `FrameInfo`
- `MotionInfo`
- `FallDownState`
- `RobotHealth`
- representaciones de field dimensions/location

No debe requerir:

- pose SimRobot;
- pelota SimRobot;
- reset buffers;
- oracle world state;
- ground truth;
- labels del entrenamiento;
- estado Python remoto;
- informacion de rivales no percibida ni comunicada legalmente.

### 5.3 Representaciones producidas

La salida primaria debe ser:

- `SkillRequest`

La salida secundaria, si se requiere para compatibilidad del grafo de modulos:

- `StrategyStatus`

Para debug interno:

- representacion propia `RLPolicyStatus` o debug drawings/logs de bajo costo.

### 5.4 Frecuencia

El provider debe correr en el thread de Cognition. El actor PPO no debe bloquear el ciclo.

Presupuesto inicial recomendado:

- promedio de inferencia menor a 5 ms;
- p95 menor a 10 ms;
- timeout duro configurable, por ejemplo 15 ms;
- si excede timeout, reusar ultima accion segura o delegar a fallback.

La accion puede mantenerse por varios frames si el modelo no se evalua en cada frame. Si se usa hold, debe documentarse y probarse con stop/set para que no haya movimiento ilegal.

### 5.5 Inference runtime

Opciones aceptables:

1. ONNX Runtime C++ embebido en Nao/Sabana.
2. TorchScript/LibTorch si el toolchain ya lo soporta bien.
3. Implementacion C++ manual del MLP solo si la arquitectura del actor queda congelada y se exportan pesos de forma verificable.

Recomendacion: ONNX Runtime si esta disponible, porque separa entrenamiento Python de inferencia robot y facilita tests de equivalencia.

La exportacion debe incluir:

- trunk MLP del actor;
- head de logits de skill;
- head de parametros continuos;
- no sampling en robot;
- inferencia deterministica: `argmax` para skill y media para parametros;
- aplicacion externa de skill masks y parameter masks.

No se debe exportar:

- value network;
- optimizer state;
- replay buffer;
- reward;
- curriculum;
- BC regularizer;
- log_std para muestreo estocastico, salvo que se use solo para compatibilidad del grafo.

## 6. Plan de integracion en SabanaHerons2026

### 6.1 Preparacion de rama

1. En `../SabanaHerons2026`, crear rama:

```bash
git switch -c hsl2026-ppo-naov6-integration
```

2. Antes de editar, guardar:

```bash
git status -sb
git submodule status
git rev-parse HEAD
sha256sum Config/NeuralNets/BallDetector/yolo_ball.onnx
```

3. No limpiar el worktree sin autorizacion. Hay cambios de balon/YOLO que pueden ser parte del esfuerzo HSL.

### 6.2 Traer bridge RL de la rama `rl-simrobot3d`

El bridge de simulacion sigue siendo util para validar PPO contra B-Human antes del robot. Se debe traer de forma selectiva desde `rl-simrobot3d`:

- `Src/Libs/Python/Controller/RLSharedState.h`
- `Src/Libs/Python/Controller/RLSharedState.cpp` si existe en la rama fuente
- extensiones en `Src/Libs/Python/Controller/Module.cpp`
- `Src/Modules/Infrastructure/RLSkillProvider/`
- cambios en `SkillBehaviorControl.cpp` para exportar observaciones y preservar obstacle avoidance
- escenarios `Config/Scenarios/4v4_RL2D/`
- escenarios `Config/Scenarios/4v4_RL3D/`
- escenas `Config/Scenes/RL*.ros2d`
- escenas `Config/Scenes/RL*.ros2`
- includes `Config/Scenes/Includes/RLNatural.con`
- cambios de build necesarios para `pybh`, `PythonController` y `SimulatedNao`

Cada archivo debe revisarse contra la rama actual HSL antes de aceptar el merge.

### 6.3 Separar bridge de entrenamiento y provider de robot

El bridge `RLSharedState` sirve para Python/SimRobot. No debe ser la unica arquitectura del robot real.

La integracion final debe quedar asi:

```text
SimRobot/train:
Python PPO -> pybh -> RLSharedState -> RLSkillProvider -> B-Human

Robot/match:
B-Human representations -> PPO C++ inference -> SkillRequest -> B-Human
```

Esto evita depender de Python, shared memory y semaforos para partido oficial.

### 6.4 Configuracion por scenario

Crear o adaptar scenarios separados:

```text
Config/Scenarios/4v4_PPO_HSL2026/
Config/Scenarios/4v4_PPO_HSL2026_Safe/
```

El scenario `Safe` debe permitir apagar PPO y usar B-Human normal sin recompilar.

Provider esperado en el scenario PPO:

```text
SkillRequest = RLPolicyProvider
StrategyStatus = RLPolicyProvider o StrategyBehaviorControl segun compatibilidad
```

Si `StrategyStatus` de B-Human es necesario para set plays o team behavior, preferir que PPO solo reemplace `SkillRequest` en open play y que B-Human conserve estrategia para estados especiales.

### 6.5 Config por robot

Agregar config por robot o location:

```text
rlPolicyProvider.cfg
```

Parametros minimos:

```text
enabled = true;
modelPath = "/home/nao/Config/Models/ppo_striker.onnx";
fallbackToBHuman = true;
activeOnlyInPlaying = true;
activeOnlyForOpenPlay = true;
activePlayerNumbers = [2];
allowShoot = true;
allowDribble = true;
maxInferenceMs = 15;
minLocalizationQuality = ...;
maxBallAgeMs = ...;
debugLevel = 0;
```

El primer despliegue recomendado es solo un field player PPO, no goalkeeper.

## 7. Exportacion del policy PPO

### 7.1 Checkpoint de entrada

No usar un checkpoint solo porque carga. Debe cumplir:

- paridad con el baseline documentado en `docs/25-guia-transplante-ppo-bhuman.md`;
- `wire_audit_status` PASS en simulacion;
- baseline candidato actual: `behavioral_evaluator >= 3/5` sobre `behavior_20260608_181747`;
- `policy_shoot_rate` consistente con el summary de referencia;
- tasa de shoot dentro de shoot-zone no cero.
- sin colisiones excesivas.
- sin target repair excesivo.
- sin dependencia de obs corregidas/sinteticas.

La prioridad del transplante es esta:

1. primero igualar en robot/sim el checkpoint candidato `production_stability_repair_v1/checkpoint_stage4_step5000.pt`;
2. luego, si hace falta, promover a un checkpoint mejor solo despues de revalidar toda la rubrica canonica y multi-seed.

### 7.2 Script de exportacion

Crear:

```text
tools/export/export_ppo_policy.py
```

Responsabilidades:

- cargar checkpoint `.pt`;
- construir `PolicyNetwork` con los mismos `active_skill_names`;
- cargar pesos con la misma logica de `train.py`;
- fijar modo eval;
- exportar actor a ONNX con input `[1, 26]`;
- producir outputs:
  - `skill_logits`
  - `param_mean`
- guardar manifiesto JSON.

Ejemplo de manifiesto:

```json
{
  "format": "onnx",
  "obs_size": 26,
  "skill_order": ["stand", "walk", "shoot", "pass", "dribble", "block", "mark", "observe"],
  "active_skills": ["stand", "walk", "dribble", "shoot"],
  "checkpoint": "runs/.../checkpoint.pt",
  "checkpoint_sha256": "...",
  "export_commit": "ed8ffdc",
  "normalization_contract": "ppo_obs_v26_hsl2026",
  "deterministic": true
}
```

### 7.3 Pruebas de equivalencia Python vs export

Antes de mover al NAO:

- tomar 1000 observaciones de logs SimRobot;
- correr actor PyTorch;
- correr actor ONNX;
- comparar logits y medias;
- comparar skill final despues de mascara;
- comparar accion final despues de decoder/gate;
- fallar si hay mismatch de skill mayor a 0.5% en casos no empatados;
- fallar si error absoluto de parametros supera tolerancia definida, por ejemplo `1e-4` antes de escalado.

## 8. Encoder C++ de observacion

### 8.1 Fuente de verdad

El encoder C++ debe implementarse contra una especificacion escrita, no "a ojo" desde Python.

Crear:

```text
docs/ppo_obs_v26_contract.md
```

Contenido requerido:

- definicion de cada indice;
- unidad fisica original;
- representacion B-Human fuente;
- transformacion a coordenadas relativas/globales;
- normalizacion;
- saturacion;
- valor fallback;
- condicion de stale;
- test unitario asociado.

La referencia escrita principal para transplante es `docs/25-guia-transplante-ppo-bhuman.md`. Si ese documento y el plan difieren, para la ruta de inferencia manda el `25`.

### 8.2 Mapeo de features

Mapeo esperado:

- `robot_x/y/theta`: desde `RobotPose`, en coordenadas de campo.
- `ball_rel_x/y`: pelota relativa al robot desde `FieldBall`/`BallModel`.
- `ball_end_rel_x/y`: estimacion de end position de pelota, si existe; fallback a pelota actual.
- `ball_vel_x/y`: velocidad estimada de pelota en marco robot o campo segun contrato Python; debe verificarse.
- `time_since_ball_seen`: `FrameInfo.time - ballModel.timeWhenLastSeen`.
- `time_since_ball_disappeared`: equivalente de desaparicion si existe; si no, especificar aproximacion.
- `ball_seen_percentage`: fuente de percepcion B-Human o aproximacion documentada.
- `ball_consistent`: consistencia natural de pelota, no oracle.
- `can_score_now`: desde `ExpectedGoals` o helper equivalente.
- `shot_quality_no_obstacles`: estimacion geometrica sin obstaculos.
- `shot_opening_with_obstacles`: estimacion con `ObstacleModel`.
- `pass_options_count`: inicialmente 0 o derivado de teammates visibles/legal team data; no debe inventar informacion.
- `nearest_*`: desde `ObstacleModel`, clasificando teammate/opponent/unknown.
- `shoot_armed`, `dribble_armed`, `shoot_arm_progress`: desde `PPOSkillGate`.

### 8.3 Paridad

Crear tests:

```text
Tests/RL/PPOObservationEncoderTest.cpp
Tests/RL/PPOSkillGateTest.cpp
Tests/RL/PPOActionDecoderTest.cpp
```

Cada test debe usar fixtures generados desde Python con inputs y outputs esperados.

## 9. Decoder C++ a SkillRequest

### 9.1 Skill mapping

Mapping obligatorio:

```text
stand   -> SkillRequest::Builder::stand()
walk    -> SkillRequest::Builder::walkTo(Pose2f(theta, x, y))
shoot   -> SkillRequest::Builder::shoot()
dribble -> SkillRequest::Builder::dribble(...)
pass    -> deshabilitado inicialmente o B-Human pass si se valida
block   -> deshabilitado inicialmente
mark    -> deshabilitado inicialmente
observe -> observe/search si existe y es seguro
```

Para `walkTo`, las coordenadas son **globales de campo**, no relativas. Esta regla es critica y ya esta documentada en el contrato RL.

### 9.2 Anchors y repair

El policy PPO no debe emitir targets libres sin reparacion.

Se debe portar la logica equivalente a:

- fallback target de walk hacia pelota;
- fallback target de dribble hacia pelota/goal;
- target theta hacia pelota o arco;
- repair si el target queda demasiado lejos del anchor;
- clamp a campo legal;
- avoidance de penalty areas y set play zones cuando aplica;
- fallback a `walk` o `stand` si el skill es invalido.

El decoder final tambien debe conservar el comportamiento actual del repo:

- projection de skills fuera de `active_skill_names` a `walk`;
- `target_repair` para `walk` y `dribble` cuando el residual cae fuera del radio permitido;
- `target_x` pinned por `residual_x_range = 0.0`;
- `target_y` y `target_theta` como residuales sobre anchors heuristicas.

### 9.3 Determinismo

En robot real:

- skill = `argmax(masked_logits)`;
- parametros = media del actor;
- no sampling;
- no exploracion;
- no temperatura;
- no ruido gaussiano;
- no epsilon-greedy.

## 10. Fail-safes de competencia

### 10.1 GameController

PPO solo puede controlar open play.

Desactivar PPO si:

- `GameState` no es `playing`;
- `GameState` esta en `stopped`;
- robot esta `penalized`;
- substate indica penalty kick y el robot no es striker asignado;
- free kick/kickoff/dropped ball aun no esta "Ball Free";
- set play requiere posicionamiento que no esta validado para PPO;
- finished/initial/set/ready salvo que se haya implementado un modo de posicionamiento seguro separado.

Para el primer transplante, en `ready` y set plays debe mandar B-Human normal.

### 10.2 Seguridad fisica

Desactivar PPO si:

- `FallDownState` indica caida o get-up requerido;
- `MotionInfo` indica kick/get-up en progreso que no debe interrumpirse;
- temperatura o corriente exceden umbral;
- bateria critica;
- el robot esta en contacto peligroso o empujando;
- localizacion esta mala;
- pelota stale mas alla del maximo configurado;
- inferencia excedio timeout;
- output contiene NaN/Inf;
- modelo no cargo o manifiesto no coincide.

### 10.3 Reglas HSL

Guardrails deterministas:

- no locomocion en `set`;
- no locomocion en `stopped`;
- no get-up durante brief stop;
- no jugar pelota con brazos/manos;
- evitar ball holding mayor a 5 s como field player;
- evitar pushes sostenidos mas de 5 s;
- evitar salir del carpet/field;
- evitar entrar en goal/net;
- respetar avoidance region en free kicks;
- respetar penalty area avoidance en goal kick;
- no tiro directo desde kickoff;
- no segundo toque ilegal si aplica;
- no invadir center circle en dropped ball set.

Estas reglas deben estar fuera del PPO y encima del decoder.

## 11. Estrategia de despliegue

### 11.1 Fase 0: congelar baseline

Entregables:

- commit exacto RL;
- checkpoint candidato;
- manifest del checkpoint;
- estado de `SabanaHerons2026`;
- checksum de modelo ONNX;
- config del robot;
- reglas HSL usadas.

No avanzar si:

- checkpoint no dispara;
- no hay `behavioral_evaluator >= 3/5`;
- no existe rollback a B-Human;
- Sabana no compila.

### 11.2 Fase 1: simulacion con bridge existente

Objetivo: confirmar que PPO sigue funcionando a traves del stack B-Human natural.

Pruebas:

- `simrobot_visible` 3D.
- `4v4_RL3D`.
- `wire_audit_status = PASS`.
- smoke de `SkillRequest`.
- action sweep de `stand/walk/dribble/shoot`.
- reset natural sin oracle en observacion.
- episodio 1v0 stage 4.
- escenarios productivos de finishing.
- `behavioral_evaluator`.

Gates:

- sin crash;
- obs_ready estable;
- synthetic pose 0 en modo natural;
- ball stale dentro de limites;
- `obs[23]`, `obs[24]`, `obs[25]` no son ceros fijos;
- `gate_skill_mask_from_obs` bloquea `walk`/`stand` cuando `finish_armed=True`;
- shoot ocurre en shoot-zone;
- no shoot spam fuera de shoot-zone;
- target repair bajo;
- collision bajo.

### 11.3 Fase 2: export ONNX y equivalencia

Objetivo: que el actor exportado sea igual al actor PyTorch.

Pruebas:

- 1000 observaciones reales de SimRobot;
- comparacion PyTorch vs ONNX;
- comparacion de logits;
- comparacion de accion final post-gate;
- comparacion de `masked_logits`, no solo de logits crudos;
- comparacion del `argmax` final despues del mask;
- benchmark CPU equivalente al NAO si es posible;
- test de carga de manifiesto.

Gates:

- mismatch de skill casi cero;
- latencia aceptable;
- archivo de modelo versionado;
- manifest completo;
- fallback si modelo no carga.

### 11.4 Fase 3: provider C++ en SimRobot

Objetivo: correr el mismo ONNX desde C++ dentro de B-Human/SimRobot.

Pruebas:

- compilar `SimulatedNao`;
- scenario `4v4_PPO_HSL2026`;
- comparar C++ provider vs Python wrapper usando mismos snapshots;
- revisar `SkillRequest` en debug;
- validar `walkTo` global;
- validar gates;
- validar fallback.

Gates:

- mismo comportamiento que Python en escenarios congelados;
- no bloquea Cognition;
- no rompe Motion;
- no rompe set plays B-Human.

### 11.5 Fase 4: NAO v6 sin pelota

Objetivo: validar deploy, carga, latencia y seguridad sin riesgo de juego.

Pruebas:

- deploy binario y configs a `/home/nao/Config`;
- cargar ONNX;
- correr provider con `enabled=false`;
- correr provider con `enabled=true` pero salida forzada a `stand`;
- medir inferencia;
- probar GameController states;
- probar emergency stop;
- probar fallback por modelo ausente;
- probar fallback por NaN artificial si existe test hook.

Gates:

- robot no camina en set/stopped;
- debug no excede 1 Hz si esta conectado;
- CPU no satura;
- no caidas;
- rollback a B-Human funciona.

### 11.6 Fase 5: NAO v6 con pelota en laboratorio

Objetivo: probar habilidades aisladas.

Pruebas minimas por robot:

- 20 aproximaciones a pelota estatica.
- 20 dribbles cortos.
- 20 situaciones de shoot-zone en banda 2.5-3.0 m.
- 10 casos de pelota stale.
- 10 casos con obstaculo humano/robot pasivo.
- 10 stop/set transitions durante accion.
- 10 recoveries despues de caida.

Metricas:

- porcentaje de aproximacion limpia;
- porcentaje de dribble estable;
- porcentaje de shoot cuando `shoot_armed`;
- porcentaje de no-shoot cuando no armado;
- caidas por minuto;
- tiempo de get-up;
- ball holding;
- pushing/contactos;
- salida de campo;
- latencia p50/p95;
- temperatura.

Gates:

- cero movimiento ilegal en stop/set;
- shoot solo cuando gate lo permite;
- no ball holding >5 s;
- no arms/hands intencional;
- caidas bajo umbral definido por el equipo;
- emergency stop validado.

### 11.7 Fase 6: scrimmage 1 PPO + B-Human

Objetivo: usar PPO como striker unico.

Configuracion recomendada:

- Player 1: B-Human goalkeeper.
- Player 2: PPO striker.
- Players 3-4: B-Human field players.

PPO no controla:

- goalkeeper;
- set plays;
- kickoff avanzado;
- penalty shootout;
- multi-agent coordination.

Pruebas:

- 10 minutos contra robots pasivos.
- 10 minutos contra B-Human defensivo.
- 10 minutos full match controlado.
- restart matrix: kickoff, dropped ball, kick-in, goal kick, corner, direct free kick, indirect free kick, penalty kick, timeout, brief stop.

Gates:

- cero penalties por estado de juego;
- cero comunicaciones ilegales;
- no degrada goalkeeper/set plays;
- PPO aporta avance o tiro sin romper estructura.

### 11.8 Fase 7: competencia

Antes de competir:

- congelar modelo;
- congelar configs;
- imprimir manifest;
- dejar rollback probado;
- documentar robot/player que usa PPO;
- revisar con capitan/handler los casos de apagado.

Durante competencia:

- no cambiar modelo durante partido;
- no usar procesamiento remoto;
- no aumentar debug wireless;
- monitorear temperatura y penalizaciones;
- apagar PPO si aparecen penalties por movimiento/reglas.

## 12. Matriz de pruebas obligatoria

### 12.1 Unitarias

- Encoder de 26 features.
- Normalizacion.
- Gate shoot/dribble.
- Mascara de logits `finish_armed`.
- Skill mask.
- Action decoder.
- Target repair.
- GameState fail-safe.
- HSL set/stopped fail-safe.
- Manifest/model compatibility.

### 12.2 Integracion

- PyTorch vs ONNX.
- Python env vs C++ provider.
- `SkillRequest` generado vs esperado.
- `SkillBehaviorControl` consume request.
- `MotionRequest` resultante no es vacio cuando debe moverse.
- fallback a B-Human.

### 12.3 SimRobot

- 1v0 finishing.
- Obstaculos estaticos.
- Obstaculos dinamicos.
- Bad angle boundary.
- Blocked corridor.
- Full field.
- 4v4 hybrid.
- Restart HSL.

### 12.4 Robot real

- carga de modelo;
- latencia;
- stand/fallback;
- walk to ball;
- dribble;
- shoot;
- stop/set;
- penalized;
- fall/get-up;
- emergency stop;
- thermal stress;
- comm budget.

## 13. Criterios de salida a campo

El PPO puede ir a campo de competencia solo si se cumplen todos:

- Sabana compila para NAO y SimRobot.
- Modelo exportado tiene manifest valido.
- `OBS_SIZE=26` confirmado.
- `SKILL_ORDER` confirmado.
- PyTorch vs ONNX pasa.
- Encoder C++ vs Python pasa.
- primero iguala el baseline candidato `3/5` ya revalidado del `behavioral_evaluator`;
- cualquier checkpoint nuevo debe igualar o superar ese baseline bajo la misma rubrica y, de preferencia, mejorar `blocked_corridor_repair`;
- `wire_audit_status = PASS`.
- `policy_shoot_rate >= 0.05`.
- `shoot` ocurre en shoot-zone.
- No hay shoot spam fuera de shoot-zone.
- Stop/set/penalized/finished probados en robot real.
- Emergency stop probado.
- Rollback a B-Human probado.
- No hay penalties HSL recurrentes en scrimmage.
- El handler sabe como apagar PPO.

## 14. Criterios de bloqueo

No desplegar PPO si:

- el checkpoint sigue sin disparar;
- `behavioral_evaluator < 3/5`;
- ONNX no coincide con PyTorch;
- encoder C++ no coincide con Python;
- el provider bloquea Cognition;
- el robot se mueve en `set` o `stopped`;
- no hay fallback a B-Human;
- el modelo requiere observaciones sinteticas;
- el robot usa procesamiento remoto;
- las configs HSL de ball/field no estan verificadas;
- el equipo no ha decidido Foundation 4v4 vs Advanced 7v7;
- no se probo con GameController v20 real;
- hay caidas o pushes frecuentes en pruebas;
- no se puede parar fisicamente el robot de inmediato.

## 15. Riesgos principales

### 15.1 Sim-to-real gap

El PPO fue entrenado en SimRobot visible con percepcion natural simulada. En robot real cambian:

- iluminacion;
- turf;
- FIFA Mini Ball real;
- friccion;
- calibracion de camaras;
- kick distance;
- odometria;
- latencia;
- estabilidad del NAO.

Mitigacion:

- usar gates conservadoras;
- probar por robot;
- no activar PPO para todos los jugadores;
- mantener fallback B-Human;
- ajustar ball specs y percepcion antes de culpar al policy.

### 15.2 Checkpoint no finalizador

El estado documentado indica que un checkpoint previo no dispara. Si el PPO no finaliza en simulacion, el transplante solo llevara al robot un mal comportamiento.

Mitigacion:

- terminar fine-tune productivo;
- exigir `behavioral_evaluator >= 3/5`;
- exigir shoot rate minimo;
- no compensar con hacks de robot real antes de arreglar policy.

### 15.3 Reglas HSL

Un PPO puede aprender a acercarse y empujar, sostener pelota o moverse cuando no debe si no se encapsula.

Mitigacion:

- PPO solo open play;
- guardrails deterministas;
- tests de GameController;
- rollback inmediato si hay penalty por motion/set/stop.

### 15.4 Integracion con Sabana actual

La rama actual tiene cambios de HSL/YOLO/ball. El bridge RL vive en otra rama.

Mitigacion:

- merge selectivo;
- revisar archivo por archivo;
- no pisar ball detector ni field configs;
- compilar despues de cada bloque;
- mantener branch de rollback.

### 15.5 Latencia en NAO

El NAO v6 tiene recursos limitados. Un runtime pesado puede afectar Cognition/Motion.

Mitigacion:

- actor pequeno;
- ONNX optimizado;
- determinismo sin value net;
- timeout duro;
- benchmark en robot;
- opcion de evaluar cada N frames.

## 16. Entregables finales

Codigo:

- provider PPO C++.
- encoder C++ v26.
- skill gate C++.
- decoder C++.
- fail-safe C++.
- scenario/config PPO HSL.
- script de export ONNX.
- tests de paridad.

Modelos:

- checkpoint `.pt` fuente.
- modelo `.onnx` exportado.
- manifest JSON.
- checksum.

Docs:

- contrato `ppo_obs_v26`.
- contrato `ppo_gate_and_mask_v1`.
- instrucciones de deploy NAO.
- matriz de pruebas.
- reporte de behavioral evaluator.
- reporte de scrimmage.
- checklist HSL.
- procedimiento de rollback.

Operacion:

- config para activar/desactivar PPO por robot.
- lista de handlers y procedimiento de stop.
- decision Foundation 4v4 o Advanced 7v7.
- player number que usara PPO.
- plan de fallback para partido.

## 17. Orden recomendado de trabajo

1. Terminar o seleccionar checkpoint PPO que pase gates productivas.
2. Exportar ONNX y probar equivalencia.
3. Crear rama de integracion Sabana desde la rama HSL/YOLO actual.
4. Traer bridge RL de `rl-simrobot3d` solo para validacion SimRobot.
5. Compilar Sabana.
6. Crear provider PPO C++ separado del bridge Python.
7. Implementar encoder v26 C++.
8. Implementar gates C++.
9. Implementar decoder a `SkillRequest`.
10. Implementar fail-safe GameController/HSL.
11. Correr unit tests.
12. Correr SimRobot provider C++.
13. Correr matriz HSL en SimRobot.
14. Deploy en NAO con PPO desactivado.
15. Activar PPO en modo stand-only.
16. Activar PPO walk/dribble sin shoot.
17. Activar shoot en laboratorio.
18. Scrimmage con un PPO striker.
19. Congelar modelo/config.
20. Competir con rollback listo.

## 18. Decision recomendada para el primer partido

Para minimizar riesgo:

- usar PPO solo como striker field player;
- mantener goalkeeper B-Human;
- mantener set plays B-Human;
- mantener supports B-Human;
- activar PPO solo en `playing` open play;
- desactivar PPO automaticamente ante cualquier substate no probado;
- no usar passing PPO;
- no usar multi-agent PPO;
- no usar graph coordination;
- no usar entrenamiento online.

La primera version exitosa no es "todo el equipo con RL". Es un PPO striker seguro, legal, local y reversible que no rompe el stack B-Human.

## 19. Fuentes revisadas

Locales:

- `README.md`
- `FOR_TRAIN_PRODUCTION.md` [removed from this branch; available on rl-v4.2-multiagent-gk]
- `HANDOFF_VISIBLE_3D_STATUS.md` [removed from this branch; available on rl-v4.2-multiagent-gk]
- `docs/03-contrato-rl-env.md`
- `docs/05-premisas-irrompibles.md`
- `docs/12-wire-rl-realista-final.md`
- `docs/15-guia-escenas-y-escenarios-rl.md`
- `docs/20-validacion-n6-2026-05-15.md`
- `docs/22-handoff-optimizacion-training-32cores.md`
- `docs/notes/09-production-selfplay-and-multiagent-handoff.md`
- `docs/notes/10-graph-coordination-post-production-roadmap.md`
- `../SabanaHerons2026/HSL2026_Migration_Checklist.md`
- `../SabanaHerons2026/BHuman2023_Architecture_Review.md`
- `../Rules-HSL-2026.pdf`

Oficiales B-Human:

- https://docs.b-human.de/coderelease2023/
- https://docs.b-human.de/coderelease2023/architecture/overview/
- https://docs.b-human.de/coderelease2023/architecture/modules-representations/
- https://docs.b-human.de/coderelease2023/motion/motion-framework/
- https://docs.b-human.de/coderelease2023/simrobot/
- https://docs.b-human.de/coderelease2023/python-bindings/
