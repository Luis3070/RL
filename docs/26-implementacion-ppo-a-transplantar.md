# Implementación del PPO a Transplantar

Fecha de corte: 2026-06-08  
Commit de referencia del repo RL: `ed8ffdc`  
Baseline de transplante vigente: `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`  
Rubrica canónica vigente: `runs/behavioral_evaluator/behavior_20260608_181747/summary.csv`  
Estado canónico vigente del baseline: `3/5` escenarios aprobados al menos (`finish_band_2p5_3m_clean`, `finish_band_2p5_3m_static_lane`, `dynamic_full_field_pressure`)  
Evidencia adicional de estabilidad parcial: `runs/behavioral_stability/stability_20260608_175828/per_rep_summary.csv`

## 1. Propósito de este documento

Este documento explica, con foco técnico, qué es exactamente el PPO que se quiere transplantar a B-Human/NAO v6, cómo está implementado en este repositorio, qué partes lo componen, qué hace en cada paso de inferencia y qué piezas son obligatorias para conservar el comportamiento actual.

El objetivo aquí no es describir el plan de integración con B-Human. Ese plan vive en `docs/23-plan-transplante-ppo-naov6-hsl2026.md`. Este documento describe el sistema PPO mismo:

- qué entrada consume;
- qué salida produce;
- cómo decide la skill;
- cómo usa las gates;
- cómo se decodifican los parámetros continuos;
- qué aprendió durante entrenamiento;
- qué se debe portar al robot y qué no.

## 2. Resumen ejecutivo

El PPO a transplantar es un policy **single-agent** entrenado sobre un entorno PettingZoo/SimRobot/B-Human, pero ejecutado en práctica como un atacante monoagente que aprende una cadena:

```text
walk -> dribble -> shoot
```

No es un policy de control articular. No mueve joints. No produce `MotionRequest` directo. Produce una decisión híbrida de alto nivel:

```text
skill discreta + parámetros continuos residuales
```

Esa decisión luego se traduce a `SkillRequest` de B-Human.

La inferencia correcta no es solo “correr una red”. El comportamiento productivo depende de cuatro capas:

1. observación de 26 floats;
2. `SkillGate` stateful que arma `shoot` y `dribble`;
3. máscara de logits `gate_skill_mask_from_obs`;
4. decoder con anchors heurísticos, projection y `target_repair`.

Si se porta solo la red y se omiten 2, 3 o 4, el robot no reproducirá el comportamiento actual aunque los pesos sean correctos.

## 3. Mapa de archivos relevantes

### Núcleo del policy

- `train.py`
- `observation.py`
- `action.py`
- `gate.py`
- `environment.py`
- `reward.py`
- `curriculum.py`
- `production_curriculum.py`
- `behavioral_evaluator.py`

### Documentación operativa relevante

- `docs/25-guia-transplante-ppo-bhuman.md`
- `cambios_para_produccion.md`
- `docs/23-plan-transplante-ppo-naov6-hsl2026.md`
- `FOR_TRAIN_PRODUCTION.md` [removed from this branch; available on rl-v4.2-multiagent-gk]

### Artefactos de referencia

- `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`
- `runs/behavioral_evaluator/behavior_20260608_181747/summary.csv`

## 4. Qué problema resuelve este PPO

El policy fue diseñado para resolver un problema muy concreto:

- el robot percibe su pose, la pelota, obstáculos y una estimación de la calidad de tiro;
- debe acercarse a la pelota de forma limpia;
- debe comprometerse con `dribble` cuando entra al takeover envelope;
- debe disparar cuando se cumplen condiciones geométricas y perceptuales suficientemente fuertes;
- debe evitar quedarse caminando o parado encima de la pelota;
- debe sobrevivir a percepción imperfecta y a rivales/obstáculos en Stage 4.

No es un policy de juego completo de equipo. No resuelve:

- coordinación multiagente aprendida;
- set plays de HSL;
- penalty shootout;
- goalkeeper;
- decisiones tácticas globales.

Su nicho actual es “striker single-agent productivo con finish gating”.

## 5. Formulación de entrenamiento

### 5.1 Entorno

El entorno principal es `RealSimRobotEnv` en `environment.py`.

Características:

- es un `ParallelEnv` tipo PettingZoo;
- usa backend `pybh` para hablar con B-Human/SimRobot;
- puede correr con `simrobot` o `simrobot_visible`;
- construye la observación normalizada de 26 dimensiones;
- aplica `SkillGate`;
- decodifica la acción híbrida;
- proyecta skills inválidas;
- repara targets erráticos;
- evalúa reward;
- acumula telemetría rica para evaluación.

Aunque la infraestructura de entrenamiento usa `MAPPO`, el caso de uso real del policy a transplantar es **single-agent**. La parte multi-slot sirve para throughput y para correr varios jugadores/slots en paralelo, no porque el actor deployado sea un policy multiagente aprendido.

### 5.2 Algoritmo

El entrenamiento usa `MAPPO` de `skrl` en `train.py`.

En la práctica, para este policy:

- hay un actor compartido;
- hay un crítico compartido;
- el actor es el componente que se transplantará;
- el crítico solo sirve durante entrenamiento;
- la normalización de valor usa `RunningStandardScaler`.

### 5.3 Stages

El curriculum está en `curriculum.py`.

Stages relevantes:

- `-1`: solo aproximación `walk`
- `0`: handoff limpio `walk -> dribble`
- `1`: cadena `walk -> dribble -> shoot` sin obstáculos
- `2`: obstáculos estáticos
- `3`: full field con obstáculos dinámicos
- `4`: validación productiva

El stage de transplante es **Stage 4**.

`Stage 4` usa:

- `active_skill_names = ("stand", "walk", "dribble", "shoot")`
- campo completo;
- obstáculos dinámicos;
- reward enfocada en finish real, no en exploración abstracta.

Pesos Stage 4 más relevantes:

- `good_walk = 0.05`
- `good_dribble = 1.2`
- `good_shoot = 2.0`
- `bad_shoot = -2.0`
- `collision = -2.0`
- `corridor_keep = 1.0`
- `ball_progress_post_takeover = 1.8`
- `goal = 7.0`
- `timeout = -3.0`
- `missed_dribble_handoff = -0.80`
- `shoot_skill_intent = 0.70`

Lectura práctica:

- caminar sirve para aproximarse, pero ya no es la skill valiosa;
- dribble debe comprometer la toma de pelota;
- shoot es la skill más premiada cuando la geometría lo justifica;
- parquearse encima de la pelota o perder la oportunidad de dribble/shoot es costoso.

### 5.4 Escenarios productivos

`production_curriculum.py` define los escenarios que importan para deploy.

Los escenarios base del evaluator son:

- `finish_band_2p5_3m_clean`
- `finish_band_2p5_3m_static_lane`
- `preshoot_bad_angle_boundary`
- `dynamic_full_field_pressure`
- `blocked_corridor_repair`

Estos escenarios son importantes por dos razones:

1. definen lo que el equipo hoy considera “production-ready”;
2. están alineados con la banda de tiro del gate y con el evaluator.

## 6. Arquitectura del actor PPO

### 6.1 Entrada

La entrada del actor es un tensor con forma:

```text
[batch, 26]
```

donde las 26 columnas son la observación normalizada definida en `observation.py` y producida por `_encode_obs()` en `environment.py`.

### 6.2 Salida

La red no produce una única acción plana final. Produce dos salidas:

1. `skill_logits`: logits discretos para 8 skills
2. `param_mean`: 4 parámetros continuos en `[-1, 1]` después de `tanh`

La acción lógica completa es:

```text
[skill_id, target_x, target_y, target_theta, pass_target]
```

### 6.3 Estructura interna

La clase está en `train.py` como `PolicyNetwork`.

Arquitectura:

```text
Linear(26 -> 128)
ELU
Linear(128 -> 128)
ELU
├── skill_head: Linear(128 -> 8)
└── param_head: Linear(128 -> 4) + tanh
```

Notas:

- `hidden_size = 128`
- activación = `ELU`
- dos heads separadas:
  - una discreta para skill;
  - una continua para parámetros.

### 6.4 Distribuciones durante entrenamiento

Durante entrenamiento:

- la cabeza discreta usa `Categorical(logits=masked_skill_logits)`
- la cabeza continua usa `Normal(param_mean, exp(log_std))`

`log_std_parameter`:

- es un parámetro entrenable global por dimensión continua;
- se inicializa en `-0.9`;
- se clampa a `[-2.0, -0.7]` durante `act()`.

Esto controla el nivel de ruido de exploración de los residuales continuos.

### 6.5 Qué se usa en deploy

En el robot no hace falta el comportamiento estocástico completo de PPO.

Para transplante, lo correcto es:

- usar el trunk del actor;
- usar `skill_head` y `param_head`;
- aplicar el mismo masking;
- usar `argmax(masked_logits)` para la skill;
- usar la media `param_mean` como parámetros continuos;
- aplicar las máscaras de parámetros y el decoder de acción.

No hace falta:

- value network;
- sampling estocástico;
- log-prob;
- entropy;
- optimizador;
- PPO update.

## 7. Espacio de acción

### 7.1 Orden de skills

`action.py` define:

```text
SKILL_ORDER = ("stand", "walk", "shoot", "pass", "dribble", "block", "mark", "observe")
```

Índices:

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

### 7.2 Skills activas en producción

En Stage 4, el stage mask deja activas:

```text
("stand", "walk", "dribble", "shoot")
```

Las demás (`pass`, `block`, `mark`, `observe`) quedan fuera por mask de stage.

### 7.3 Parámetros continuos por skill

El actor siempre produce 4 parámetros continuos, pero no todos aplican a todas las skills.

`ACTION_PARAM_MASKS`:

```text
stand   -> (0, 0, 0, 0)
walk    -> (0, 1, 1, 0)
shoot   -> (0, 0, 0, 0)
pass    -> (0, 0, 0, 1)
dribble -> (0, 1, 0, 0)
block   -> (0, 1, 0, 0)
mark    -> (0, 1, 0, 0)
observe -> (0, 0, 0, 0)
```

En el policy real a transplantar, los casos importantes son:

- `walk`: usa `target_y`, `target_theta`
- `dribble`: usa `target_y`
- `shoot`: no usa parámetros
- `stand`: no usa parámetros

### 7.4 Rangos

`config.py` fija:

```text
residual_x_range = 0.0
residual_y_range = 450.0 mm
residual_theta_range = 0.75 rad
flat_size = 5
```

Implicación crítica:

- `target_x` está **pinneado** al anchor, porque `residual_x_range = 0.0`
- la política solo desplaza lateralmente y rota alrededor del anchor

Esto fue una decisión deliberada para matar una dimensión muerta que degradaba la aproximación.

## 8. Espacio de observación

### 8.1 Estructura

`observation.py` define `ObsIndex` con 26 columnas.

Orden:

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
12 ball_consistent_with_game_state
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

### 8.2 Significado

Las primeras 23 columnas son percepción/estado derivado del backend.

Las últimas 3 no son percepción directa. Son el estado del gate:

- `obs[23] = shoot_armed`
- `obs[24] = dribble_armed`
- `obs[25] = shoot_arm_progress`

Esto es central. El mask de inferencia depende de estas columnas.

### 8.3 Normalización

`environment.py:_encode_obs()` normaliza así:

- `robot_x / ACTION.x_range`
- `robot_y / ACTION.y_range`
- `robot_theta / pi`
- `ball_rel_x / ACTION.x_range`
- `ball_rel_y / ACTION.y_range`
- `ball_end_rel_x / ACTION.x_range`
- `ball_end_rel_y / ACTION.y_range`
- `ball_vel_x / BALL_VEL_NORM`
- `ball_vel_y / BALL_VEL_NORM`
- tiempos saturados por `TIME_NORM_MS`
- `ball_seen_percentage / 100`
- flags como `0.0/1.0`
- conteo de pass options saturado
- distancias de obstáculos saturadas a `[0, 1]`
- gate features inyectadas al final

El actor no ve los valores crudos del backend. Ve esta versión normalizada.

### 8.4 Fuente semántica

El policy aprendió sobre este espacio exacto.

Si en B-Human se cambia cualquiera de estas cosas:

- orden de columnas;
- unidad de tiempo;
- signo de `robot_theta`;
- rango de normalización;
- columnas 23-25;

la política se rompe aunque los pesos sean correctos.

## 9. SkillGate

### 9.1 Qué es

`gate.py` implementa un gate stateful que decide si `shoot` y `dribble` están “armados”.

No decide por sí solo la acción final. Decide qué finish skills son legales en este paso.

### 9.2 Estado interno

`SkillGate` mantiene:

```text
shoot_armed: bool
dribble_armed: bool
shoot_streak: int
dribble_streak: int
```

Este estado vive a lo largo del episodio y se resetea al inicio de cada episodio.

### 9.3 Parámetros actuales de producción

#### Shoot

- enter ball distance: `380 mm`
- exit ball distance: `460 mm`
- enter ball angle: `25°`
- exit ball angle: `35°`
- goal enter distance: `3000 mm`
- goal exit distance: `3200 mm`
- open enter: `0.6`
- open exit: `0.45`
- align enter: `30°`
- align exit: `45°`
- debounce: `3` frames

#### Dribble

- enter distance: `450 mm`
- exit distance: `600 mm`
- enter angle: `35°`
- exit angle: `55°`
- debounce: `1` frame
- grace enter age: `1500 ms`
- grace hold age: `2500 ms`

#### Percepción natural confiable

- `trusted_max_age_ms = 600`

### 9.4 Dos canales de pelota

Este es uno de los detalles más importantes del sistema.

El gate no usa un único concepto de “pelota”.

Usa dos canales:

1. **canal natural**
   - timestamp del último avistamiento visual real;
   - usado para decidir si el balón está fresco;
   - crítico para `shoot`;
2. **canal exportado/estimado**
   - `BallModel` ya propagado/corregido;
   - usado para distancia, ángulo, apertura y resto de la observación.

Esto existe porque:

- `shoot` no debe armarse sobre una pelota meramente propagada;
- `dribble` necesita tolerar pequeños dropouts de percepción mientras el robot se compromete al takeover.

### 9.5 Resultado del gate

`SkillGate.step()` devuelve:

- `shoot_armed`
- `dribble_armed`
- `shoot_arm_progress`
- `allowed_skills`

Además, esas salidas se “bakean” en la observación como `obs[23:25]`.

## 10. Máscara de logits

### 10.1 Qué hace

`train.py:gate_skill_mask_from_obs()` transforma la observación en una máscara multiplicativa por skill.

Reglas:

- `shoot` permitido solo si `obs[23] >= 0.5`
- `dribble` permitido solo si `obs[24] >= 0.5`
- `stand` prohibido si `finish_armed`
- `walk` prohibido si `finish_armed`

donde:

```text
finish_armed = shoot_armed or dribble_armed
```

### 10.2 Qué cambió recientemente

La palanca histórica que llevó el baseline anterior a `4/5` fue precisamente esta:

- antes `walk` y `stand` seguían disponibles aunque hubiera takeover opportunity;
- ahora se bloquean cuando `finish_armed=True`.

Eso elevó `clean_takeover_rate` porque el actor dejó de “caminar encima de la pelota” cuando ya debía dribblear o disparar.

### 10.3 Orden correcto

El orden no es negociable:

```text
obs -> gate baked -> actor -> gate mask -> argmax
```

No es correcto:

```text
obs -> actor -> argmax -> luego filtrar skill
```

porque entonces la política ya tomó una decisión sobre logits no restringidos y la paridad con entrenamiento se pierde.

### 10.4 Invariante

La máscara no debe bloquear todas las skills:

- si `finish_armed=False`, quedan `stand/walk`;
- si `finish_armed=True`, queda al menos una finish skill.

## 11. Decoder de acción

### 11.1 Acción estructurada

El output crudo del actor se convierte en `StructuredSkillAction`.

Esa estructura contiene:

```text
skill_type
target_x
target_y
target_theta
pass_target
```

### 11.2 Decodificación residual

`environment.py:_decode_residual_policy_action()` hace lo siguiente:

1. clampa residuales a `[-1, 1]`;
2. aplica `ACTION_PARAM_MASK` por skill;
3. obtiene anchors heurísticos con `build_fallback_action_targets()`;
4. transforma residuales a targets absolutos:
   - `target_x = anchor_x + residual_x * residual_x_range`
   - `target_y = anchor_y + residual_y * residual_y_range`
   - `target_theta = anchor_theta + residual_theta * residual_theta_range`
5. clampa a límites de campo;
6. devuelve `StructuredSkillAction`.

### 11.3 Anchors

`reward.py:build_fallback_action_targets()` define anchors por skill.

La más importante:

#### Walk

El target de walk no apunta al balón directo. Apunta a un offset ideal detrás del balón:

```text
target_x = ball_x - 210 mm
target_y = ball_y
target_theta = heading hacia el balón
```

Esa es la razón por la que la aproximación aprendida no es “chocar de frente” sino entrar por una postura más favorable para takeover.

#### Dribble

El anchor de dribble está más pegado al balón y orientado al arco.

#### Shoot

No usa target continuo. Depende de que la geometría previa ya haya alineado al robot.

### 11.4 Projection

`environment.py:project_action()` fuerza a `walk` cualquier skill fuera de `active_skill_names`.

Esto evita ejecutar skills no activas en el stage actual.

### 11.5 Target repair

`environment.py:_repair_execution_targets()` repara `walk` y `dribble` si el residual se desvía demasiado del anchor esperado.

Lógica:

- calcular error del target respecto al anchor;
- si cae fuera del radio residual permitido o del theta permitido;
- reemplazar por el anchor heurístico puro.

Esto mata targets absurdos sin esconder la telemetría:

- el policy sigue “aprendiendo” lo que emitió;
- la ejecución no se va a una posición físicamente absurda.

## 12. Qué hace el policy en términos conductuales

### 12.1 Antes de takeover

Si el balón no está en takeover envelope:

- el stage mask permite `stand`, `walk`, `dribble`, `shoot`;
- el gate normalmente desarma `dribble` y `shoot`;
- la máscara deja esencialmente `stand/walk`;
- el policy aprende a caminar hacia el balón usando el anchor de aproximación.

### 12.2 En takeover envelope

Cuando el balón entra al envelope y la percepción natural lo respalda:

- `dribble_armed` pasa a verdadero;
- `obs[24]` cambia;
- `gate_skill_mask_from_obs()` bloquea `walk` y `stand`;
- el actor ya no puede “titubear” en esas skills;
- debe comprometerse con `dribble` o, si también se arma `shoot`, con la skill de finish.

### 12.3 En shoot zone

Si además:

- la distancia al balón es suficientemente corta;
- la alineación al arco es suficientemente buena;
- la apertura de tiro o `can_score_now` lo justifican;
- la distancia balón-arco está dentro del rango del gate;
- la visión natural está fresca;

entonces `shoot_armed=True`.

Eso alimenta:

- `obs[23] = 1`
- `obs[25] = shoot_arm_progress`

y la máscara habilita `shoot`.

### 12.4 Por qué no dispara siempre

El policy no fue entrenado para “tirar cuando pueda, pase lo que pase”.

Fue entrenado para:

- no disparar fuera de la geometry;
- no disparar si la visión natural del balón está degradada;
- no disparar cuando el ángulo es malo;
- no disparar si hay señales de que la mejor conducta es terminar de dribblear.

Por eso el gate es más estricto para `shoot` que para `dribble`.

## 13. Qué aprendió el PPO y por qué se comporta así

### 13.1 Aprendizaje buscado

El reward no buscó “maximizar goles” en abstracto solamente. Buscó una cadena estable:

1. ver el balón;
2. acercarse sin pararse;
3. tomarlo limpiamente;
4. mantenerlo al frente;
5. progresar hacia el arco;
6. disparar solo cuando la geometría y la percepción sean buenas.

### 13.2 Ejemplos de sesgos inducidos por reward

- `bad_walk` evita caminar cuando ya se tiene takeover;
- `bad_dribble` evita dribblear lejos del balón;
- `bad_shoot` evita tiros malos;
- `missed_dribble_handoff` castiga no comprometerse con `dribble` o `shoot` en takeover estable;
- `missed_shoot` castiga no disparar en una oportunidad genuina;
- `ball_progress_post_takeover` hace que dribblear hacia el arco valga más que dribblear en el sitio;
- `collision` y `corridor_keep` introducen disciplina frente a obstáculos.

### 13.3 Lo que no aprendió

No aprendió:

- coordinación de equipo con roles;
- passing aprendido de producción;
- estrategia general de HSL;
- comportamiento de goalkeeper;
- compliance completo de set plays por aprendizaje.

Eso es deliberado. Muchas de esas reglas deben estar fuera del PPO y encima de él.

## 14. Evaluación actual del baseline

### 14.1 Baseline

La referencia de transplante vigente es:

```text
runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt
```

### 14.2 Resultado

Según `runs/behavioral_evaluator/behavior_20260608_181747/summary.csv`, el baseline candidato vigente ya revalidó al menos 3 escenarios:

- `finish_band_2p5_3m_clean` -> pasa
- `finish_band_2p5_3m_static_lane` -> pasa
- `dynamic_full_field_pressure` -> pasa
- `preshoot_bad_angle_boundary` -> falla

### 14.3 Falla restante

La falla restante es:

- `collision_rate > 0.12` en `preshoot_bad_angle_boundary`
- `blocked_corridor_repair` todavía no quedó completamente estable en todas las seeds revalidadas

Esto no es un problema de exportación del policy. Es un límite real del comportamiento actual frente a obstáculos dinámicos y ángulos malos.

### 14.4 Qué sí demuestra el baseline

El baseline demuestra que:

- el actor ya sí dispara;
- el handoff limpio sí está ocurriendo;
- la cadena `walk -> dribble -> shoot` ya es reproducible;
- el gate y la máscara actual son parte esencial del éxito.

## 15. Qué se debe transplantar al robot

### Obligatorio

1. observación de 26 dimensiones con la misma semántica;
2. normalización exacta;
3. `SkillGate` con el mismo estado y parámetros;
4. baking de `obs[23:25]`;
5. actor PPO;
6. stage mask de producción;
7. `gate_skill_mask_from_obs`;
8. `argmax` sobre logits ya enmascarados;
9. `ACTION_PARAM_MASKS`;
10. decoder residual con anchors;
11. projection y `target_repair`;
12. traducción final a `SkillRequest`.

### No obligatorio

- value network;
- reward;
- `behavioral_evaluator`;
- curriculum;
- BC replay;
- telemetría de entrenamiento;
- escenarios productivos;
- sampling estocástico;
- optimizador.

## 16. Qué no debe correr en el robot

Las siguientes piezas son de entrenamiento/evaluación y deben quedar fuera del runtime final:

- `reward.py` como función de reward;
- `behavioral_evaluator.py`;
- `train.py` como loop PPO;
- `build_mappo`;
- `MAPPO_CFG`;
- memory buffers;
- advantage estimation;
- stage promotion;
- dataset BC;
- scripts de producción;
- plots y CSV de telemetría.

El robot necesita solo inferencia, no entrenamiento.

## 17. Riesgos de mala reimplementación

### 17.1 Portar solo la red

Si se porta solo el `.pt` o un export ONNX del actor:

- el robot puede quedarse caminando con el balón;
- no respetará el compromiso de takeover;
- `shoot` no aparecerá con la frecuencia correcta;
- se perderá la paridad con el evaluator.

### 17.2 Romper la observación

Errores típicos:

- usar pelota absoluta donde iba relativa;
- usar segundos donde iban milisegundos;
- cambiar el signo de theta;
- dejar `obs[23:25]` en cero;
- no saturar/normalizar igual que Python.

### 17.3 Omitir el mask de logits

Si el mask se aplica después del `argmax`, o no se aplica:

- la red puede elegir `walk` o `stand` dentro de takeover;
- reaparece exactamente el problema que bloqueaba `clean_takeover_rate`.

### 17.4 Omitir anchors y repair

Sin anchors:

- el actor queda demasiado sensible al drift de los parámetros continuos;
- `walk` y `dribble` se vuelven físicamente inconsistentes.

Sin `target_repair`:

- un residual anómalo puede mandar al robot a un target sin sentido.

## 18. Checklist mínimo de paridad

Antes de decir “el PPO está transplantado”, deben cumplirse estas igualdades contra el pipeline Python:

- mismo vector normalizado de 26 floats;
- mismo estado del gate;
- mismos valores `obs[23:25]`;
- mismo mask por skill;
- mismos `masked_logits`;
- mismo `argmax` final;
- mismos parámetros enmascarados por skill;
- mismos anchors;
- mismo `target_repair`;
- mismo `SkillRequest` final.

La unidad de verificación correcta no es “la red cargó”. La unidad correcta es:

```text
BackendObservation fijo -> acción final idéntica
```

## 19. Resumen final

El PPO a transplantar no es simplemente una red neuronal entrenada para “jugar fútbol”. Es una política híbrida, muy estructurada, cuyo comportamiento emerge de la interacción entre:

- observación normalizada;
- gate stateful;
- máscara de logits;
- decoder residual;
- repair heurístico;
- skill execution de B-Human.

La red sola no explica el baseline vigente. El pipeline completo sí.

Por eso, el transplante correcto consiste en portar **el actor y su contrato completo de inferencia**, no solo sus pesos.
