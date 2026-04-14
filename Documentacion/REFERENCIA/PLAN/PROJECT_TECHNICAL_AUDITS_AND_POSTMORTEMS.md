# Project Technical Audits & Post-mortems (Lesson Log)

Este documento unifica el historial de fallos, correcciones y lecciones aprendidas sobre el Environment, MAPPO y el Bridge de comunicación.

---

# SOURCE: BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md

# Auditoria y Verificacion: `environment.py`, `wrapper.py`, `config.py`, `reward.py`

> Fecha: 22 de marzo de 2026
> Objetivo: auditar el RL actual contra el contrato real de BHuman para 4v4 Scaled y definir criterios de aceptacion para su adaptacion.

---

## 1. Principio Central

Congelar al ejecutor BHuman NO significa congelar el aprendizaje.

Significa separar dos niveles:

- nivel alto RL: posicionamiento, approach, handoff, soporte y cooperacion
- nivel ejecutor BHuman: `shoot`, `pass`, `dribble`, `Zweikampf`, `SkillBehaviorControl`

El agente aprende a **maximizar un controlador fijo aguas abajo**. Eso es normal en control jerarquico:

- el policy aprende estados previos que hacen al ejecutor mas exitoso
- la reward se asigna al estado y la accion previa al takeover
- los companeros aprenden a abrir lineas de pase, de dribble o de tiro para ese ejecutor fijo

El error seria entrenar al RL con una fisica inventada que compita contra BHuman.

---

## 2. Resumen Ejecutivo

Estado actual del RL inspeccionado:

- `config.py`: desalineado con 4v4 Scaled BHuman
- `environment.py`: contiene fisica propia de kick/dribble y logica de partido no derivada del contrato BHuman
- `reward.py`: premia contacto, kick y dribble de la simulacion propia, no takeover/exito del ejecutor BHuman
- `wrapper.py`: util como capa CTDE/curriculum, pero sus metricas de avance aun no reflejan takeover ni calidad de ejecucion BHuman

Veredicto:

- como sandbox de navegacion y control basico, el stack actual es util
- como entorno fiel para transplante a BHuman, hoy NO pasa auditoria

---

## 3. Hallazgos Criticos

## H1. Campo equivocado

Estado actual:

- `config.py` usa `Config/Locations/Default/fieldDimensions.cfg`
- dimensiones hardcodeadas 5v5: `4500 x 3000`

Evidencia:

- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\config.py:17`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\config.py:23`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\config.py:25`

Contrato BHuman esperado:

- 4v4 Scaled
- `x ∈ [-2813, 2813]`
- `y ∈ [-1875, 1875]`

Impacto:

- todo el obs/action space esta escalado a un campo incorrecto
- targets, distancias, rewards y thresholds aprenden sobre un mundo equivocado

Estado: `CRITICO`

## H2. Tacticas y home poses legacy/hardcodeadas

Estado actual:

- tacticas legacy `t22`, `t112`, `t4`, `t211`, `t31`, `t_Tortuga`
- `current_tactic = "t4"`
- home poses fijas hardcodeadas

Evidencia:

- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\config.py:168`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\config.py:251`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:772`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:1177`

Contrato BHuman esperado:

- tactic set de `s4v4_scaled`
- tacticas reales `t121_4v4_scaled` y `t13_4v4_scaled`
- home poses parseadas del repo BHuman

Impacto:

- el policy aprende estructura posicional que no existe en el robot real

Estado: `CRITICO`

## H3. Fisica propia de `kick/dribble`

Estado actual:

- el env resuelve contacto, dribble y kick internamente
- usa thresholds como `ball_kick_range`, `ball_dribble_range`, `frontal_kick_angle_deg`
- decide kick/dribble segun angulo y `throttle`

Evidencia:

- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\config.py:69`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\config.py:70`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\config.py:93`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:1261`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:1311`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:1359`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:1369`

Contrato BHuman esperado:

- el env no debe inventar kick/dribble local
- debe modelar takeover de BHuman y rollout de macro-acciones del ejecutor real

Impacto:

- el agente hoy aprende a maximizar la fisica del env, no el comportamiento real de `SkillBehaviorControl`

Estado: `CRITICO`

## H4. Reward optimiza interacciones del env, no exito BHuman

Estado actual:

- bonuses por `first_touch`
- bonus por `dribble_bonus`
- bonus por `ball_kicked`
- progreso de balon condicionado al kick del env

Evidencia:

- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\reward.py:88`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\reward.py:90`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\reward.py:344`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\reward.py:499`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\reward.py:542`

Contrato BHuman esperado:

- reward por `handoff_ready`
- reward por takeover exitoso
- reward por resultado del ejecutor BHuman tras takeover
- reward cooperativo por abrir lineas y no bloquear decisiones del ejecutor

Impacto:

- el policy puede aprender hacks validos solo en esta simulacion

Estado: `CRITICO`

## H5. Curriculum mide touch/goal/win, no takeover BHuman

Estado actual:

- `touch_rate`
- `goal_rate`
- `win_rate`
- `weighted = touch + goal`

Evidencia:

- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\wrapper.py:72`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\wrapper.py:74`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\wrapper.py:76`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\wrapper.py:78`

Contrato BHuman esperado:

- `takeover_ready_rate`
- `takeover_success_rate`
- `bhuman_exec_success_rate`
- `support_quality_rate`
- `anti_blocking_rate`

Impacto:

- el curriculum puede avanzar aunque el agente este aprendiendo una interfase equivocada con el ejecutor real

Estado: `ALTO`

---

## 4. Hallazgos Importantes pero No Bloqueantes

## H6. Semantica de TeamBallModel simplificada

Estado actual:

- `ball_age = 0`
- `using_team_ball = 0`
- balon siempre fresco y omnisciente en el obs

Evidencia:

- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:606`
- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:607`

Comentario:

- la omnisciencia del balon puede ser aceptable como aproximacion inicial
- pero hoy no representa de verdad `own ball` vs `team ball`

Estado: `MEDIO`

## H7. `is_active` de teammates no representa el activo real

Estado actual:

- un teammate se marca activo si su rol es `forward`

Evidencia:

- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:628`

Contrato esperado:

- el activo real debe corresponder al agente seleccionado por la logica de juego o por la politica de alto nivel

Estado: `MEDIO`

## H8. Opponents y obstaculos aun no reflejan presion realista

Estado actual:

- oponentes scripted simples
- sin vinculo con `Zweikampf`, `ObstacleModel` o sectores de dribble reales

Evidencia:

- `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL\environment.py:458`

Estado: `MEDIO`

---

## 5. Que Se Debe Verificar por Archivo

## 5.1 `config.py`

Debe verificarse:

- campo 4v4 Scaled
- tacticas reales de BHuman
- roles y home poses derivados del parser
- curriculum alineado al contrato de takeover
- eliminacion de parametros que controlan kick/dribble fisico propio

Criterios de aceptacion:

- no usar `Default/fieldDimensions.cfg`
- no usar tacticas legacy no escaladas
- no tener `kick_speed`, `ball_kick_range`, `ball_dribble_range` como fuente de verdad de interaccion
- stages definidos por distancia, obstaculos, cooperacion y takeover, no por kick/dribble del env

## 5.2 `environment.py`

Debe verificarse:

- construccion de obs desde contrato BHuman
- acciones de navegacion/posicionamiento, no acciones de control de balon directo
- takeover envelope bien definido
- macro-rollout de BHuman para `shoot/pass/dribble/duel`
- home poses y tacticas parseadas
- obstaculos y dinamica compatibles con match-play BHuman

Criterios de aceptacion:

- `step()` no debe decidir kick/dribble con thresholds propios
- debe existir bandera `handoff_ready`
- debe existir ventana de takeover BHuman
- debe registrarse resultado del takeover
- el balon no debe moverse solo por una heuristica local de throttle del agente

## 5.3 `reward.py`

Debe verificarse:

- reward centrado en takeover
- reward por exito del ejecutor BHuman
- reward cooperativo por support y spacing
- penalizacion por bloquear linea de pase/dribble/shot del ejecutor

Criterios de aceptacion:

- `first_touch_bonus` y `dribble_bonus` deben desaparecer o quedar solo como compatibilidad temporal
- `ball_kicked` y `ball_progress` deben depender del rollout BHuman, no de la fisica local actual
- deben existir componentes como:
  - `handoff_ready_reward`
  - `takeover_success_reward`
  - `bhuman_exec_outcome_reward`
  - `support_lane_reward`
  - `anti_blocking_penalty`

## 5.4 `wrapper.py`

Debe verificarse:

- stage sync correcto
- estado global para CTDE
- metricas de avance acordes al contrato BHuman
- logging de indicadores de takeover y exito del ejecutor

Criterios de aceptacion:

- `CurriculumManager` no debe avanzar solo por `touch_rate/goal_rate/win_rate`
- debe registrar al menos:
  - `takeover_ready_rate`
  - `takeover_success_rate`
  - `bhuman_exec_success_rate`
  - `support_quality_rate`

---

## 6. Redisenio Recomendado

## 6.1 Semantica correcta

En lugar de:

```text
policy -> target pose
env -> decide dribble o kick propio
reward -> premia contacto y kick del env
```

Debe quedar:

```text
policy -> target pose / posicionamiento / support pose
env -> evalua handoff_ready
env -> dispara takeover BHuman
env -> ejecuta macro rollout BHuman
reward -> premia calidad del handoff y del resultado del ejecutor
```

## 6.2 Como aprende cooperacion si BHuman esta congelado

El activo aprende:

- llegar al balon con pose util
- orientar takeover
- entrar al envelope correcto

Los companeros aprenden:

- estar en linea de pase viable
- abrir carriles
- no estorbar dribble o tiro
- posicionarse para rebote y soporte

La logica cooperativa se aprende porque la reward incluye el resultado del ejecutor fijo bajo esas condiciones.

---

## 7. Respuesta Operativa a la Pregunta “Si no hay pass/shoot, todo seria dribble”

Eso solo seria cierto si el environment modelara un takeover simplificado donde siempre se llama dribble.

Pero el contrato real de BHuman NO es asi:

- `PlayBall` puede pedir `shoot`, `pass` o `dribble`
- `SkillBehaviorControl` enruta esas opciones
- `KickAtGoal` y `Zweikampf` incluso pueden caer a `DribbleToGoal`

Entonces:

- en stages individuales tempranos, si no hay companeros ni situaciones de tiro, si habra fuerte sesgo a dribble
- en stages cooperativos y de ataque, el env debe permitir que BHuman active `pass` y `shoot`

Si no se modela eso, el curriculum se colapsa a “dribble training”.

---

## 8. Plan Minimo de Correccion

Orden recomendado:

1. Corregir `config.py` a 4v4 Scaled real
2. Reemplazar tacticas/home poses por parser
3. Eliminar fisica propia de kick/dribble del `environment.py`
4. Introducir `handoff_ready` + `BHuman takeover`
5. Reescribir `reward.py` alrededor del takeover
6. Reescribir metricas del `wrapper.py`
7. Solo despues ajustar curriculum stages

---

## 9. Tareas de Implementacion por Archivo

> Alcance de esta seccion: solo el proyecto en `C:\Users\Alegu\OneDrive\Desktop\ROBOTICA\ROBOCOP\RL`, excluyendo `runs`, `trajectory_demos`, `trajectory_guidance_project`, `Documentacion` y archivos `.md`.

## 9.1 `config.py`

### Borrar

- definiciones hardcodeadas 5v5 basadas en `Default/fieldDimensions.cfg`
- `TACTIC_NAMES` legacy no compatibles con 4v4 Scaled
- cualquier stage cuya semantica pedagogica dependa de `kick/dribble` del env en lugar de takeover BHuman
- thresholds de interaccion de balon usados como fuente de verdad del contrato real

### Reemplazar

- `FIELD` por datos parseados de 4v4 Scaled
- `TACTIC_NAMES` por el contrato real de `s4v4_scaled`
- curriculum actual de 9 stages por el curriculum contractual 0-12
- metricas de avance por metricas BHuman-aware

### Agregar

- `BHumanContractConfig`
- `TakeoverConfig`
- `MacroActionConfig`
- `SupportConfig`
- campos de curriculum como:
  - `takeover_required`
  - `takeover_timeout_steps`
  - `bhuman_mode`
  - `num_static_obstacles`
  - `num_dynamic_obstacles`
  - `promotion_metrics`

### Resultado esperado

`config.py` deja de ser fuente de verdad de fisica de balon y pasa a ser fuente de verdad del curriculum, limites y contrato de interfaz con BHuman.

## 9.2 `environment.py`

### Borrar

- `_handle_ball_interactions()` como arbitro principal de `dribble/kick`
- logica de kick basada en `frontal_kick_angle_deg`
- logica de dribble basada en `ball_dribble_range`
- `current_tactic = "t4"`
- home poses hardcodeadas de `_home_position_for()`
- semantica dummy de `ball_age=0` y `using_team_ball=0` como si fueran datos reales

### Reemplazar

- spawn y limites por datos del parser
- home poses por tacticas reales parseadas
- accion del policy por salida de posicionamiento/support, no control de balon
- paso de simulacion de balon por:
  - `rl_control_step`
  - `handoff_check`
  - `bhuman_takeover_rollout`
  - `post_takeover_update`

### Agregar

- `_load_bhuman_contract()`
- `_compute_handoff_ready()`
- `_select_active_agent()`
- `_run_bhuman_takeover()`
- `_simulate_bhuman_skill_request()`
- `_apply_bhuman_macro_result()`
- `_compute_support_lane_features()`
- `_compute_antiblocking_features()`
- `_spawn_static_obstacles()`
- `_spawn_dynamic_obstacles()`

### Resultado esperado

El env deja de decidir `kick/dribble` con fisica inventada y pasa a evaluar si el policy deja a BHuman en una buena situacion para ejecutar.

## 9.3 `reward.py`

### Borrar

- `first_touch_bonus` como componente central
- `dribble_bonus` del env actual
- `ball_kicked` y `ball_progress` definidos por la fisica local actual
- cualquier shaping que premie tocar el balon por si mismo

### Reemplazar

- reward de contacto por reward de takeover
- reward de kick local por reward de exito del ejecutor BHuman
- field coverage generico por soporte contextual

### Agregar

- `handoff_ready_reward`
- `takeover_success_reward`
- `bhuman_exec_outcome_reward`
- `support_lane_reward`
- `open_pass_window_reward`
- `anti_blocking_penalty`
- `bad_takeover_penalty`
- `late_takeover_penalty`
- `duel_exposure_penalty`

### Resultado esperado

La reward deja de enseñar "toca el balon" y pasa a enseñar "maximiza el desempeno del ejecutor real".

## 9.4 `wrapper.py`

### Borrar

- avance del curriculum solo por `touch_rate`, `goal_rate`, `win_rate` y `weighted`

### Reemplazar

- `CurriculumManager.record_episode()` por uno que use metricas de takeover y exito de ejecucion

### Agregar

- agregador de metricas:
  - `takeover_ready_rate`
  - `takeover_success_rate`
  - `bhuman_exec_success_rate`
  - `support_quality_rate`
  - `anti_blocking_rate`
- logging de breakdown de takeover por stage

### Resultado esperado

El wrapper pasa de medir exito superficial a medir si la policy realmente esta aprendiendo la interfaz correcta con BHuman.

## 9.5 `train.py`

### Borrar

- suposiciones heredadas del curriculum actual basadas en `touch/goals/win` como unica senal de avance
- evaluaciones que no distinguen entre exito del env inventado y exito del takeover BHuman

### Reemplazar

- logging y checkpointing por stage con metricas BHuman-aware
- evaluacion de curriculum por gates de takeover/executor
- self-play final por pool de snapshots, no solo ultimo rival

### Agregar

- validacion al inicio:
  - parser cargado
  - campo 4v4 correcto
  - tacticas correctas
  - contrato de takeover activo
- evaluacion por stage con:
  - `takeover_ready_rate`
  - `takeover_success_rate`
  - `bhuman_exec_success_rate`
  - `support_quality_rate`
- masking de `actor loss` para el agente activo mientras `takeover_active = true`
- entrenamiento normal del critic durante takeover
- entrenamiento normal de companeros no activos durante takeover
- accion dummy solo para consistencia del rollout, sin usarla en `policy loss`
- `opponent_pool_manager` para self-play
- pruebas de regresion por seeds y checkpoints

### Resultado esperado

`train.py` pasa de entrenar contra el env actual a entrenar contra el contrato de transplante.

## 9.6 `visualizer.py`

### Borrar

- HUD y overlays que sugieren que el control del balon lo hace el policy actual
- referencias a stages legacy si dejan de existir

### Reemplazar

- display de reward por display de takeover y macro-accion BHuman

### Agregar

- overlay `handoff_ready`
- overlay `active_agent`
- bandera visual sobre cada agente que indique `RL` o `BHuman`
- overlay `bhuman_skill_request`
- overlay `takeover_success/failure`
- overlay de carriles de pase, dribble y bloqueo
- HUD con metricas del contrato

### Resultado esperado

El visualizador sirve para depurar el transplante, no solo para ver la fisica actual del env.

## 9.7 `condaTester.py`

### Borrar

- si solo es un smoke test sin valor real para el proyecto, removerlo del flujo principal

### Reemplazar

- por un script pequeño de `contract_smoke_test.py` si se necesita prueba rapida

### Agregar

- chequeos rapidos de:
  - import del parser
  - carga de CFGs reales
  - creacion de env con takeover activo

### Resultado esperado

El proyecto deja de tener utilidades sueltas y pasa a tener un smoke test realmente util.

## 9.8 `.env` y limpieza del arbol

### Borrar

- `__pycache__` del proyecto si estan siendo versionados

### Reemplazar

- variables opacas del `.env` por configuracion explicita en `config.py` o variables documentadas

### Agregar

- `.gitignore` que excluya:
  - `__pycache__/`
  - `runs/`
  - `*.pyc`
  - artefactos temporales de entrenamiento

### Resultado esperado

El repo queda limpio para iterar y auditar.

## 9.9 Archivos Nuevos Recomendados

Agregar:

- `bhuman_contract.py`
- `bhuman_parser_loader.py`
- `bhuman_takeover.py`
- `contract_metrics.py`
- `curriculum_contract.py`
- `tests/test_parser_contract.py`
- `tests/test_takeover_logic.py`
- `tests/test_reward_contract.py`
- `tests/test_stage_gates.py`

## 9.10 Orden de Implementacion Real

1. `config.py`
2. `bhuman_parser_loader.py`
3. `bhuman_contract.py`
4. `environment.py`
5. `reward.py`
6. `wrapper.py`
7. `train.py`
8. `visualizer.py`
9. tests
10. limpieza del arbol

---

## 10. Checklist de Verificacion Final

El stack adaptado pasa auditoria solo si se cumple todo esto:

- [ ] Campo y limites vienen de 4v4 Scaled real
- [ ] Tacticas y home poses vienen del parser
- [ ] No existe kick/dribble fisico inventado como fuente de verdad
- [ ] Existe estado `handoff_ready`
- [ ] Existe rollout de macro-accion BHuman
- [ ] Reward mide takeover y exito del ejecutor
- [ ] Wrapper mide progreso del curriculum con metricas BHuman-aware
- [ ] El agente aprende posicionamiento y soporte, no control motor del balon

---

## 11. Estado Actual

Resultado de la auditoria del stack actual:

- `config.py` -> `FAIL`
- `environment.py` -> `FAIL`
- `reward.py` -> `FAIL`
- `wrapper.py` -> `PARCIAL`
- `train.py` -> `PARCIAL`
- `visualizer.py` -> `PARCIAL`
- `condaTester.py` -> `BAJA PRIORIDAD`

Conclusion:

El codigo actual sirve como base de simulacion y entrenamiento inicial, pero necesita una refactorizacion de contrato antes de considerarse alineado al transplante BHuman.

---

## 12. Ultima Verificacion Cruzada contra `BHUMANPARSER`, `PROJECT_AUDIT_FOR_RL` y el Contrato BHuman

Esta verificacion final ya no mira solo coherencia interna del stack Python.
Tambien verifica si el proyecto, tal como esta hoy, podria transplantarse sin romper el contrato operativo real de BHuman.

### 12.1 Hallazgos Criticos Adicionales

## H9. No existe aun una capa real `parser -> contract loader -> takeover executor`

Estado actual:

- el stack sigue dependiendo de hardcodes en `config.py`
- no existe una capa que cargue CFGs reales y los convierta en contrato ejecutable del env

Impacto:

- el proyecto no esta consumiendo la fuente de verdad definida en `BHUMANPARSER_GUIDE.md`
- cualquier entrenamiento puede quedar alineado a una configuracion distinta al robot real

Estado: `CRITICO`

## H10. No existe una abstraccion explicita `SkillRequest -> SkillBehaviorControl -> MotionRequest`

Estado actual:

- el env genera interacciones del balon por fisica local
- no modela el encadenamiento real por el cual BHuman recibe una intencion y decide la ejecucion motriz

Impacto:

- el proyecto no esta maximizando el ejecutor real
- el transplante quedaria acoplado a una semantica de accion diferente a la de BHuman

Estado: `CRITICO`

## H11. No hay gating real por `GameState` ni por la logica de `HandleGameState`

Estado actual:

- el entorno se comporta como si siempre estuviera en condiciones de juego activo
- no hay separacion estricta entre fases de partido y fases en las que RL no deberia actuar

Impacto:

- el contrato del transplante queda incompleto
- la politica puede aprender comportamientos invalidados por el flujo real del partido

Estado: `ALTO`

## H12. No hay validacion del presupuesto temporal de Cognition

Estado actual:

- no existe chequeo automatico de tiempo de inferencia
- no hay guardas que aseguren compatibilidad con el presupuesto temporal marcado por la auditoria del proyecto

Impacto:

- una politica puede funcionar en simulacion y aun asi ser inviable en BHuman
- esto es un riesgo directo para el transplante

Estado: `ALTO`

## H13. Semantica de `TeamData`, `team ball` y comunicacion aun insuficiente

Estado actual:

- el env simplifica demasiado la informacion compartida entre companeros
- no representa con suficiente fidelidad latencia, frescura ni uso de balon de equipo

Impacto:

- se debilita la parte cooperativa del aprendizaje
- los stages multiagente pueden converger a patrones que no sobreviven al robot real

Estado: `ALTO`

## H14. El stack sigue instrumentado alrededor de `touch/kick/dribble` del env viejo

Estado actual:

- `wrapper.py` sigue evaluando avance por metricas superficiales
- `train.py` sigue acumulando metricas heredadas del env actual
- `visualizer.py` todavia debe migrar a eventos de takeover y ejecucion BHuman

Impacto:

- incluso despues de corregir el env, las herramientas de entrenamiento y depuracion seguirian empujando hacia el objetivo equivocado

Estado: `ALTO`

### 12.2 Validez frente al transplante

Veredicto final de viabilidad:

- como prototipo de navegacion RL, el proyecto es util
- como sistema listo para transplante a BHuman, todavia NO es valido

La razon principal no es solo que falten CFGs.
La razon principal es que todavia no existe una interfase operacional fiel entre:

- decision de alto nivel del RL
- gating real del partido
- takeover al ejecutor BHuman
- resultado de macro-accion real
- asignacion correcta de reward

### 12.3 Cambios Minimos Adicionales que se Deben Hacer

---

# SOURCE: MAPPO_Auditoria.md

# Auditoría del Plan MAPPO_Plan_3Fases — SabanaHerons2026

> **Auditor:** Claude Sonnet 4.6
> **Fecha de auditoría:** Marzo 2026
> **Objeto auditado:** `MAPPO_Plan_3Fases.md`
> **Metodología:** Verificación rigurosa de cada referencia de archivo, número de línea, campo de struct y valor de parámetro citado en el plan contra el código fuente real.

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Estado de Archivos Críticos](#estado-de-archivos-críticos)
3. [Verificación de Números de Línea](#verificación-de-números-de-línea)
4. [Errores Conceptuales Detectados](#errores-conceptuales-detectados)
5. [Errores Menores (Line Numbers)](#errores-menores-line-numbers)
6. [Verificación de Valores y Parámetros](#verificación-de-valores-y-parámetros)
7. [Verificación de Formato .cfg](#verificación-de-formato-cfg)
8. [Verificación del Observation y Action Space para RL](#verificación-obs-action-space)
9. [Opinión del Auditor sobre el Plan](#opinión-del-auditor)
10. [Veredicto Final](#veredicto-final)

---

## Resumen Ejecutivo

El plan es **sólido en su arquitectura general** y en su investigación del código fuente BHuman. La gran mayoría de archivos referenciados **existen y contienen exactamente lo que el plan describe**. Se encontraron **3 errores conceptuales** que fueron corregidos directamente en `MAPPO_Plan_3Fases.md`, además de varios errores menores de número de línea.

El plan ha sido **actualizado al scope 4v4 Scaled** (normativa de competencia), lo que implica un campo más pequeño (5626×3750 mm), 3 field players por equipo (no 4), 2 tácticas activas (`t121_4v4_scaled`/`t13_4v4_scaled`), y bounds de action space ajustados.

**Calificación post-correcciones: 9.1 / 10 — APROBADO**

---

## Estado de Archivos Críticos

### ✅ CONFIRMADOS — Existen y contienen lo descrito

| Archivo | Ruta real | Contenido clave verificado |
|---------|-----------|---------------------------|
| `JointAngles.h` | `Src/Representations/Infrastructure/JointAngles.h` | `ENUM_INDEXED_ARRAY(Angle, Joints::Joint) angles` confirmado |
| `Joints.h` | `Src/Libs/RobotParts/Joints.h` | `headYaw` (L13), `headPitch` (L14) confirmados |
| `MotionInfo.h` | `Src/Representations/MotionControl/MotionInfo.h` | `(Pose2f) speed` — campo al ~L49 confirmado |
| `InertialSensorData.h` | `Src/Representations/Infrastructure/SensorData/InertialSensorData.h` | `gyro`, `acc` confirmados en L16-L18 |
| `BehaviorStatus.h` | `Src/Representations/BehaviorControl/BehaviorStatus.h` | `walkingTo` y `speed` confirmados |
| `GlobalOpponentsModel.h` | `Src/Representations/Modeling/GlobalOpponentsModel.h` | `OpponentEstimate` con `position`, `left`, `right` confirmados |
| `BallModel.h` | `Src/Representations/Modeling/BallModel.h` | `BallState.velocity` confirmado al L23 |
| `SkillInterfaces.h` | `Src/Representations/BehaviorControl/SkillInterfaces.h` | `WalkAtAbsoluteSpeed` L24, `WalkAtRelativeSpeed` L30, `WalkToPose` L41 — **EXACTOS** |
| `GlobalOpponentsTracker.cpp` | `Src/Modules/Modeling/GlobalOpponentsTracker/GlobalOpponentsTracker.cpp` | `theRobotPose * ob.center` confirmado (conversión local→global) |
| `ExpectedGoalsProvider.cpp` | `Src/Modules/BehaviorControl/ExpectedGoalsProvider/ExpectedGoalsProvider.cpp` | `angleToLeftPost - angleToRightPost` confirmado |
| `PathPlanner.h` | `Src/Representations/BehaviorControl/PathPlanner.h` | Existe |
| `PathPlannerProvider.h` | `Src/Modules/BehaviorControl/PathPlannerProvider/PathPlannerProvider.h` | Existe |
| `cameraIntrinsics.cfg` | `Config/Robots/Default/cameraIntrinsics.cfg` | Existe con `openingAngleWidth` |
| `Behavior.cpp` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp` | 1159 líneas. `assignPositions` L497, `assignRoles` L740, `determineActiveAgent` L911, FSM táctica L449 |
| `RatingRole.cpp` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/PositionRoles/RatingRole.cpp` | Existe con `position()` function |
| `PlayBall.cpp` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp` | Existe (171 líneas). SmashOrPass logic confirmada |
| `PositionRole.cpp` | `Src/Tools/BehaviorControl/Strategy/PositionRole.cpp` | `fromPosition()` L16 confirmado |
| `CameraIntrinsics.h` | `Src/Representations/Configuration/CameraIntrinsics.h` | `openingAngleWidth`, `openingAngleHeight` confirmados |
| `Tactic.h` | `Src/Tools/BehaviorControl/Strategy/Tactic.h` | Enum `Type` y `Position::Type` ambos confirmados |
| `FieldDimensions.h" | `Src/Representations/Configuration/FieldDimensions.h` | Todos los campos `xPos*/yPos*` confirmados |
| `walkingEngine.cfg` | `Config/Robots/Default/walkingEngine.cfg` | `maxSpeed.translation.x = 250`, `y = 200`, `maxSpeedBackwards = 150`, `rotation = 120deg` — **EXACTOS** |
| `fieldDimensions.cfg` | `Config/Locations/Default/fieldDimensions.cfg` | Todos los valores del `FIELD_SYMBOLS` del plan confirmados |
| `StrategyStatus.h` | `Src/Representations/BehaviorControl/StrategyStatus.h` | Campos `position` y `role` confirmados |
| `FallDownState.h` | `Src/Representations/Sensing/FallDownState.h` | Enum con `upright`, campo `state` confirmados |
| `SystemSensorData.h` | `Src/Representations/Infrastructure/SensorData/SystemSensorData.h` | `batteryLevel` confirmado (rango `[0.0, 1.0]`) |
| `GameState.h` | `Src/Representations/Infrastructure/GameState.h` | `playing` enum, `timeWhenPhaseEnds`, `ownTeam.score`, `opponentTeam.score`, `isPlaying()` confirmados |
| `FieldBall.h` | `Src/Representations/BehaviorControl/FieldBall.h` | `positionOnField`, `timeSinceBallWasSeen`, `useTeammatesBall()` confirmados |
| `FieldBall.cpp` | `Src/Representations/BehaviorControl/FieldBall.cpp` | `useTeammatesBall()` implementación confirmada |
| `GlobalTeammatesModel.h` | `Src/Representations/Modeling/GlobalTeammatesModel.h` | `TeammateEstimate` con `pose`, `relativeWalkTarget`, `speed` confirmados |
| `Role.h` | `Src/Tools/BehaviorControl/Strategy/Role.h` | `isActiveRole()` L38 confirmado |
| `Agent.h` | `Src/Tools/BehaviorControl/Strategy/Agent.h` | `role` field confirmado |
| `ActiveRole.h` | `Src/Tools/BehaviorControl/Strategy/ActiveRole.h` | `playBall` enum confirmado |
| `PublishMotion.cpp` | `Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Output/BehaviorStatus/PublishMotion.cpp` | Escritura de `walkingTo` y `speed` en ~L28-30 confirmada |
| `TeamData.cpp` | `Src/Representations/Communication/TeamData.cpp` | `getEstimatedPosition()` L20-27 confirmado |
| `RobotPose.h` | `Src/Representations/Modeling/RobotPose.h` | Hereda de `Pose2f` confirmado |
| `StrategyBehaviorControl.h` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.h` | Existe |
| `PositionRoles/` | `Src/Modules/BehaviorControl/StrategyBehaviorControl/PositionRoles/` | Carpeta confirmada: Defender, Forward, Goalkeeper, Midfielder, RatingRole |
| `TeamData.h` | `Src/Representations/Communication/TeamData.h` | Contiene `StrategyStatus theStrategyStatus` en el struct `Teammate` |

---

## Verificación de Números de Línea

### ✅ Números de línea EXACTOS

| Referencia en el plan | Verificado |
|-----------------------|-----------|
| `SkillInterfaces.h#L24` — `WalkAtAbsoluteSpeed` | ✅ EXACTO |
| `SkillInterfaces.h#L30` — `WalkAtRelativeSpeed` | ✅ EXACTO |
| `SkillInterfaces.h#L41` — `WalkToPose` | ✅ EXACTO |
| `InertialSensorData.h#L16-L18` — `gyro`, `acc` | ✅ EXACTO |
| `GlobalOpponentsTracker.h#L57-L63` — `maxDistance`, `deleteAfter` | ✅ EXACTO (campos en esa zona) |
| `TeamData.cpp#L18-L27` — `getEstimatedPosition()` | ✅ EXACTO |
| `Behavior.cpp#L910` — `determineActiveAgent` | ✅ EXACTO (función empieza en L911) |

---

## Errores Conceptuales Detectados

Estos son los errores que **rompen la lógica del proyecto** si no se corrigen.

---

### ❌ ERROR CONCEPTUAL 1 — `self_role: one_hot(5)` es incorrecto

**Dónde aparece en el plan:**
```python
"self_role": one_hot(5),      # one_hot de StrategyStatus.position
# Valores: goalkeeper, defenderL, defenderR, forwardL, forwardR
```

**El problema:**
`StrategyStatus.position` es de tipo `Tactic::Position::Type` definido en `Src/Tools/BehaviorControl/Strategy/Tactic.h`. Este enum tiene **13 valores reales**:

```
none, goalkeeper, defender, defenderL, defenderR,
midfielder, midfielderM, midfielderL, midfielderR,
forward, forwardM, forwardL, forwardR
```

Además, el Tactic::Type enum ahora incluye variantes `_scaled` para 3v3 y 4v4. El `one_hot(5)` con valores `[gk, defL, defR, fwdL, fwdR]` ignora `defender` genérico, `midfielder*`, `forward` genérico, etc.

Adicionalmente, el plan describe `striker` como uno de los 5 valores del `self_role`, pero `striker` **no es un position type** — es un `ActiveRole::playBall`. Los roles activos son una dimensión ortogonal a las posiciones.

**Impacto:** El one_hot de 5 valores provocará:
1. `IndexError` o encodings incorrectos si el robot tiene posición `midfielder` o `defender` genérico
2. Confusión conceptual entre `position` (estructura táctica) y `role` activo (quién tiene el balón)

**Corrección sugerida:**
```python
# Opción A: one_hot(13) para los 13 tipos de Tactic::Position::Type
"self_position": one_hot(13),

# Opción B: reducir a categorías padre
POSITION_MAP = {
    "goalkeeper": 0,
    "defender": 1, "defenderL": 1, "defenderR": 1,
    "midfielder": 2, "midfielderM": 2, "midfielderL": 2, "midfielderR": 2,
    "forward": 3, "forwardM": 3, "forwardL": 3, "forwardR": 3,
}
"self_position": one_hot(4),   # gk/def/mid/fwd
"self_is_active": bool,         # Role::isActiveRole — ¿es el striker?
```

---

### ❌ ERROR CONCEPTUAL 2 — `tm_is_active` no existe en `GlobalTeammatesModel`

**Dónde aparece en el plan:**
```python
"tm_is_active": bool,  # true si el teammate es striker/playBall
# Fuente mencionada: Role::isActiveRole(agent.role)
#   Agent.role: Src/Tools/BehaviorControl/Strategy/Agent.h#L51
```

**El problema:**
`GlobalTeammatesModel.TeammateEstimate` tiene exactamente estos campos:
```cpp
(Pose2f) pose,
(int)(-1) playerNumber,
(Vector2f)(Vector2f::Zero()) relativeWalkTarget,
(float)(-1.f) speed,
```

**No hay campo `role` ni `activeRole` en `TeammateEstimate`.** El `Agent.role` referenciado en el plan es el struct interno de `StrategyBehaviorControl` — está disponible localmente para el propio robot pero NO es parte de `GlobalTeammatesModel`.

Para acceder al rol de un compañero desde el robot, la fuente correcta es `TeamData.Teammate.theStrategyStatus.role` (confirmado en `TeamData.h`). Este es un dato comunicado pero proviene de `TeamData`, no de `GlobalTeammatesModel`.

**Impacto:**
- En la simulación Python no hay problema (el simulador conoce todos los roles directamente)
- En despliegue C++ (Fase deployment), acceder a `theGlobalTeammatesModel.teammates[i].role` fallará en compilación porque el campo no existe
- El plan no especifica qué representación usar en C++ para este dato, creando ambigüedad en el paso a producción

**Corrección:**
```cpp
// En C++ (deployment), la fuente correcta es:
theTeamData.teammates[i].theStrategyStatus.role
// Verificar con: Role::isActiveRole(theTeamData.teammates[i].theStrategyStatus.role)
```

En la simulación Python, usar el rol directamente del estado del agente. **Documentar explícitamente esta dicotomía sim vs. deployment.**

---

### ❌ ERROR CONCEPTUAL 3 — `current_tactic: one_hot(9)` está desactualizado

**Dónde aparece en el plan:**
```python
"current_tactic": one_hot(9),
# Valores: none, t22, t112, t4, t121, t211, t31, t13, t_Tortuga
```

**El problema:**
El enum `Tactic::Type` en `Src/Tools/BehaviorControl/Strategy/Tactic.h` tiene **17 valores** actualmente:

```
none, t22, t112, t4, t121, t211, t31, t13, t_Tortuga,
t4_scaled, t22_scaled, t31_scaled, t112_scaled,
t121_4v4_scaled, t13_3v3_scaled, t22_3v3_scaled, t13_4v4_scaled
```

Los archivos `.cfg` correspondientes en `Config/Behavior/Tactics/` confirman la existencia de todas estas variantes scaled.

**Impacto:**
Si el team corre algún juego en modo scaled (3v3, 4v4), la táctica actual tendría un índice fuera del rango `[0, 8]`, lo que causaría errores de indexación en el one_hot o, peor, silenciosamente mapearía a la táctica incorrecta.

**Corrección:**
```python
# Opción A — 5v5 only (aceptable si el proyecto es solo 5v5):
# Documentar explícitamente que este obs space es solo para 5v5
# y agregar un assert en el env que valide la táctica actual

# Opción B — Completo:
"current_tactic": one_hot(17),  # Todos los tipos de Tactic::Type
```

---

## Errores Menores (Line Numbers)

Estos errores **no afectan la funcionalidad** del proyecto pero reducen la trazabilidad del documento. Los archivos y campos existen — solo los números de línea son levemente incorrectos.

| Referencia en el plan | Valor real | Diferencia |
|-----------------------|-----------|-----------|
| `StrategyStatus.h#L27` para `position` | L30 | +3 líneas |
| `StrategyStatus.h#L21` para `acceptedTactic` | L24 | +3 líneas |
| `FieldDimensions.h#L24` para `xPosOpponentGroundLine` | L25 | +1 línea |
| `FieldDimensions.h#L41` para `yPosLeftGoal` | L42 | +1 línea |
| `FieldDimensions.h#L42` para `yPosRightGoal` | L43 | +1 línea |
| `Tactic.h#L19` para `Enum Position::Type` | L19 es `STREAMABLE(Tactic,...)`. La `Position::Type` enum está más adelante en el archivo | Descripción incorrecta |
| `BehaviorStatus.h#L23` para `walkingTo` | L24 es `walkingTo`, L23 es `passOrigin` | +1 línea |
| `Behavior.cpp#L495-L735` para `assignPositions` | Función empieza en L497 (L495-L496 son un `}` y una línea en blanco) | +2 líneas |
| `Behavior.cpp#L738-L768` para `assignRoles` | Función empieza en L740 | +2 líneas |
| `GlobalOpponentsTracker.cpp#L70-L76` para conversión | Conversión en L75-L77 | +5 líneas |
| `ExpectedGoalsProvider.cpp#L63-L66` para ángulos | Cálculo empieza en L64 | +1 línea |
| `FieldBall.h#L117` para `timeSinceBallWasSeen` | ~L116 | -1 línea |
| `FieldBall.cpp#L90-L93` para `useTeammatesBall()` | Función empieza en L93 | +3 líneas |

---

## Verificación de Valores y Parámetros

### ✅ Valores de `walkingEngine.cfg` — EXACTOS

| Parámetro | Plan | Real (`walkingEngine.cfg`) |
|-----------|------|---------------------------|
| `MAX_FORWARD` | 250 mm/s | `maxSpeed.translation.x = 250` ✅ |
| `MAX_SIDEWAYS` | 200 mm/s | `maxSpeed.translation.y = 200` ✅ |
| `MAX_BACKWARDS` | 150 mm/s | `maxSpeedBackwards = 150` ✅ |
| `MAX_ROTATION` | 120°/s = 2.09 rad/s | `rotation = 120deg" ✅ |
| `walkSpeedParams.maxSpeed` | 350 mm/s | `walkSpeedParams.maxSpeed.translation.x = 350` ✅ |

### ✅ Valores de `fieldDimensions.cfg` / `FIELD_SYMBOLS` — EXACTOS

Todos los 26 valores del diccionario `FIELD_SYMBOLS` del plan fueron verificados contra `Config/Locations/Default/fieldDimensions.cfg` y son correctos, incluyendo:
- `xPosOpponentGroundLine = 4500` ✅
- `yPosLeftGoal = 800` ✅
- `yPosRightGoal = -800` ✅
- `centerCircleRadius = 750` ✅

### ✅ `useTeammatesBall()` — Lógica correcta con matiz

El plan documenta:
```python
return (timeSinceBallWasSeen > 1000 || timeSinceBallDisappeared > 100)
       && teammatesBallIsValid && teammatesBallNewerThanOwnBall;
```

El código real usa parámetros (`ballSeenTimeout=1000`, `ballDisappearedTimeout=100`) pero con los mismos valores por defecto. Los valores 1000 y 100 son correcto para el uso estándar. ✅

### ✅ `GlobalTeammatesModel` — "rotation is currently unreliable"

El plan advierte no incluir `pose.rotation` de compañeros. El header real dice exactamente: `"rotation is currently unreliable"`. Decisión correcta y documentada con evidencia real. ✅

### ✅ `GlobalOpponentsModel` — Sin orientación disponible

El plan dice que `OpponentEstimate` solo tiene `position`, `left`, `right`. Verificado: el struct real tiene exactamente esos tres campos y ningún campo de orientación. ✅

---

## Verificación de Formato .cfg

### ✅ Formato de tácticas (`t121.cfg`) — CORRECTO

El plan muestra el formato con `positions = [{type=...; pose={rotation=...; translation={x=...; y=...}}}]`. El archivo real usa exactamente esta estructura. El parser propuesto maneja correctamente:
- Expresiones simbólicas como `xPosOwnGoalArea - 150` ✅
- Expresiones con multiplicación: `centerCircleRadius * 0.5` ✅ (símbolo incluido en FIELD_SYMBOLS)
- Sufijo `deg` para ángulos ✅
- Arrays anidados y structs ✅

### ✅ Formato de estrategias (`s5v5.cfg`) — CORRECTO

La estructura `tactics = [{tactic=t4; transitions=[{to=t121; conditions=[...]}]}]` coincide con el archivo real. El parser capturará también `ownKickOffs`, `opponentKickOffs`, etc., que el plan menciona como retornados por `load_strategy()`. ✅

### ⚠️ Aviso: `priorityGroups` en tácticas

Los archivos `.cfg` de tácticas contienen también un campo `priorityGroups` (verificado en `t121.cfg`). El parser del plan lo ignorará implícitamente (lo parseará como dato pero el plan no lo usa). No es un error, pero debe documentarse para no confundir.

---

---

## Verificación del Observation y Action Space para RL {#verificación-obs-action-space}

Esta sección verifica que el obs space y action space definidos en el plan sean **coherentes, completos y correctamente extraíbles** de las representaciones BHuman y los `.cfg` de tácticas. Todo calculado con los valores reales del campo 4v4 Scaled.

---

### Action Space — Verificación

#### Límites del campo 4v4 Scaled (fuente: `Config/Locations/4v4_Scaled/fieldDimensions.cfg`)

| Dimensión | Mínimo | Máximo | Total | Fuente |
|-----------|--------|--------|-------|--------|
| `target_x` | -2813 mm | +2813 mm | 5626 mm | `xPosOwnGroundLine` / `xPosOpponentGroundLine` |
| `target_y` | -1875 mm | +1875 mm | 3750 mm | `yPosRightSideline` / `yPosLeftSideline` |
| `target_theta` | -π rad | +π rad | 2π rad | Rotación libre |

> ✅ Los límites del plan son correctos para 4v4 Scaled. **Diferencia crítica vs 5v5:** el campo original tenía x∈[-4500, 4500] y y∈[-3000, 3000]. Usar los bounds de 5v5 en un campo 4v4 haría que el 60% del action space sea inválido (fuera del campo), dificultando la convergencia.

#### Dimensionalidad total del action space

| Fase | Por agente | Total (3 agentes) |
|------|-----------|-------------------|
| Fase 1 | 3 continuos | **9** |
| Fase 2 | 3 continuos + 1 discreto(3) | **12** |
| Fase 3 | 3 continuos + 2 discretos(6,3) | **15** |

---

### Observation Space — Verificación Feature por Feature

#### Grupo SELF (16 floats)

| Feature | Fuente C++ | Accesible | Rango | Notas |
|---------|-----------|-----------|-------|-------|
| `self_x` | `theRobotPose.translation.x()` | ✅ | [-2813, 2813] | Campo 4v4. Normalizar /2813 → [-1,1] |
| `self_y` | `theRobotPose.translation.y()` | ✅ | [-1875, 1875] | Normalizar /1875 → [-1,1] |
| `self_theta` | `theRobotPose.rotation` | ✅ | [-π, π] | Ya en rango natural |
| `self_vx` | `theMotionInfo.speed.translation.x()` | ✅ | [-150, 250] mm/s | walkingEngine.cfg: max 250 fwd, 150 bwd |
| `self_vy` | `theMotionInfo.speed.translation.y()` | ✅ | [-200, 200] mm/s | max 200 sideways |
| `self_omega` | `theMotionInfo.speed.rotation` | ✅ | [-2.09, 2.09] rad/s | 120°/s |
| `self_is_upright` | `theFallDownState.state == upright` | ✅ | {0, 1} | Enum verificado en FallDownState.h |
| `self_battery` | `theSystemSensorData.batteryLevel` | ✅ | [0.0, 1.0] | SystemSensorData.h#L14 |
| `self_position` (one_hot 5) | `theStrategyStatus.position` | ✅ | {gk,defL,defR,fwdL,fwdR} | Verificado en t121/t13 4v4 cfgs. StrategyStatus.h#L30 |
| `self_is_active` | `Role::isActiveRole(theStrategyStatus.role)` | ✅ | {0, 1} | Role.h#L38 + StrategyStatus.h#L31 |
| `goal_opening_angle` | Derivado: `atan2(800-y, 2838-x) - atan2(-800-y, 2838-x)` | ✅ calculado | [0, ~0.55 rad] | Desde centro del campo: 31.5° = 0.55 rad |
| `goal_relative_angle" | Derivado: `normalize(atan2(-y, 2813-x) - self_theta)` | ✅ calculado | [-π, π] | Ya en rango natural |

> **⚠️ Rangos de normalización para self_x, self_y, velocidades:**
> El plan dice "obs normalizado [-1,1]" pero no especifica los rangos. Valores recomendados:
> ```python
> OBS_NORM = {
>     "self_x":     2813.0,   # /xPosOpponentGroundLine
>     "self_y":     1875.0,   # /yPosLeftSideline
>     "self_vx":     250.0,   # /MAX_FORWARD
>     "self_vy":     200.0,   # /MAX_SIDEWAYS
>     "self_omega":  2.094,   # /MAX_ROTATION (120°/s en rad)
>     "ball_x":     2813.0,
>     "ball_y":     1875.0,
>     "ball_vx":    2000.0,   # velocidad máxima real del balón post-patada (~2m/s)
>     "ball_vy":    2000.0,
>     "ball_age_ms": 3000.0,  # normalizar a [0,1], clamp en 3s
>     "tm_x":       2813.0,   "tm_y": 1875.0,
>     "tm_walk_dx":  469.0,   "tm_walk_dy": 469.0,  # ~centerCircleRadius
>     "tm_speed":    250.0,
>     "opp_x":      2813.0,   "opp_y": 1875.0,
>     "half_time_remaining_s": 600.0,  # 10 minutos
> }
> ```
> **Esto es deuda técnica urgente — definirlo antes de implementar el env.**

---

#### Grupo BALL (6 floats)

| Feature | Fuente C++ | Accesible | Notas |
|---------|-----------|-----------|-------|
| `ball_x` | `theFieldBall.positionOnField.x()` | ✅ | FieldBall.h#L104. Global coords. |
| `ball_y` | `theFieldBall.positionOnField.y()` | ✅ | |
| `ball_vx` | `Rotation(theRobotPose) * theBallModel.estimate.velocity.x()` | ✅ con transform | BallModel.h#L24 da velocidad en coords robot. **Requiere rotar por theRobotPose.rotation** |
| `ball_vy` | idem | ✅ con transform | |
| `ball_age_ms` | `theFieldBall.timeSinceBallWasSeen` | ✅ | FieldBall.h#L116 |
| `using_team_ball` | `theFieldBall.useTeammatesBall()` | ✅ | FieldBall.cpp#L93 |

> **⚠️ Transform crítico de ball_vx/ball_vy:**
> `BallModel.estimate.velocity` es en coordenadas relativas al robot. Para obtener velocidad en coordenadas globales del campo:
> ```cpp
> // C++
> Vector2f ball_vel_field = theRobotPose.rotation * theBallModel.estimate.velocity;
> // Python sim:
> ball_vel_field = rotate2d(ball_vel_robot, agent_theta)
> ```
> Si se omite esta rotación, el agente verá una velocidad de balón que cambia según hacia dónde mira el robot — señal corrupta.

---

#### Grupo TEAMMATES (21 floats — 3 slots × 7)

| Feature | Fuente C++ | Accesible | Notas |
|---------|-----------|-----------|-------|
| `tm_x` | `GlobalTeammatesModel.teammates[i].pose.translation.x()` | ✅ | GlobalTeammatesModel.h#L36 |
| `tm_y` | `GlobalTeammatesModel.teammates[i].pose.translation.y()` | ✅ | |
| `tm_walk_dx` | `GlobalTeammatesModel.teammates[i].relativeWalkTarget.x()` | ✅ | En coords relativas al receptor |
| `tm_walk_dy` | `GlobalTeammatesModel.teammates[i].relativeWalkTarget.y()` | ✅ | |
| `tm_speed` | `GlobalTeammatesModel.teammates[i].speed` | ✅ | float mm/s |
| `tm_is_active` | `Role::isActiveRole(TeamData.teammates[i].theStrategyStatus.role)` | ✅ en C++ | **No está en GlobalTeammatesModel.** Usar TeamData (TeamData.h struct Teammate#L32) |
| `tm_exists` | `i < GlobalTeammatesModel.teammates.size()` | ✅ | En 4v4 siempre true si el robot está activo |

> **⚠️ Slot ordering:** `GlobalTeammatesModel` ordena por `playerNumber` ascendente. En Python sim, mantener el mismo orden para consistencia.

---

#### Grupo OPPONENTS (12 floats — 4 slots × 3)

| Feature | Fuente C++ | Accesible | Notas |
|---------|-----------|-----------|-------|
| `opp_x` | `GlobalOpponentsModel.opponents[i].position.x()` | ✅ | Convertida a global vía `theRobotPose * ob.center` (GlobalOpponentsTracker.cpp#L75) |
| `opp_y` | `GlobalOpponentsModel.opponents[i].position.y()` | ✅ | |
| `opp_exists` | `i < GlobalOpponentsModel.opponents.size()` | ✅ | Depende del FOV (~55°) y distancia (<5000mm) |

> **Limitación real confirmada:** El modelo de oponentes es LOCAL. En 4v4 puede haber hasta 4 oponentes, pero un robot generalmente ve 0-2 a la vez. Los slots vacíos se llenan con ceros. En Python sim: aplicar el modelo FOV (`get_visible_opponents()` del plan) para replicar esta limitación.

---

#### Grupo GAME STATE (4 floats)

| Feature | Fuente C++ | Accesible | Rango | Notas |
|---------|-----------|-----------|-------|-------|
| `game_is_playing` | `theGameState.isPlaying()` | ✅ | {0,1} | GameState.h#L194 |
| `own_score` | `theGameState.ownTeam.score` | ✅ | [0, ~20] | GameState.h#L120 |
| `opp_score" | `theGameState.opponentTeam.score` | ✅ | [0, ~20] | GameState.h#L120 |
| `half_time_remaining_s` | `(theGameState.timeWhenPhaseEnds - theFrameInfo.time) / 1000.0f` | ✅ | [0, 600] | GameState.h#L412 |

---

#### Grupo TACTIC (2 floats — one_hot)

| Index | Valor | Táctica | Estado |
|-------|-------|---------|--------|
| 0 | 1.0 | `t121_4v4_scaled` | Defensiva (inicio de partida) |
| 1 | 1.0 | `t13_4v4_scaled` | Ofensiva |

> **Fuente:** `theStrategyStatus.acceptedTactic` (Tactic::Type enum). Mapear con `TACTIC_INDEX` del plan.
> En 4v4 nunca aparecerán otros valores de Tactic::Type si se usa exclusivamente `s4v4_scaled.cfg`.

---

### Home Poses Verificadas (calculadas desde los .cfg)

Las home poses son los puntos de inicio del RL y la baseline heurística. Calculadas aplicando `FIELD_SYMBOLS` 4v4:

#### `t121_4v4_scaled.cfg` — Formación defensiva (default)

| Posición | x (mm) | y (mm) | θ (rad) | Descripción |
|----------|--------|--------|---------|-------------|
| goalkeeper | -2532 | 0 | 0 | Dentro del área propia |
| defenderL | -1468 | 0 | 0 | Línea de penalti propia |
| defenderR | -234 | -625 | 0 | Lado derecho del centro |
| forwardL | -234 | +625 | 0 | Lado izquierdo del centro |
| forwardR | +703 | 0 | 0 | Zona ofensiva media |

#### `t13_4v4_scaled.cfg` — Formación ofensiva

| Posición | x (mm) | y (mm) | θ (rad) | Descripción |
|----------|--------|--------|---------|-------------|
| goalkeeper | -2532 | 0 | 0 | Dentro del área propia |
| defenderL | -1938 | 0 | 0 | Línea de área propia |
| forwardL | +234 | +500 | 0 | Zona ofensiva |
| forwardR | +234 | -500 | 0 | Zona ofensiva |

> **Observación táctica:** En `t13_4v4_scaled`, solo 3 posiciones para field players (defenderL, forwardL, forwardR). La 4ª posición (defenderR) no existe — con 3 field players se ocupa defenderL + forwardL + forwardR según `priorityGroups`. Esto es correcto para 4v4.

---

### Coherencia Total del Obs+Action Space

| Verificación | Estado | Detalle |
|-------------|--------|---------|
| Dimensiones del campo correctas | ✅ | x±2813, y±1875 verificados en fieldDimensions.cfg |
| Total obs = 61 floats | ✅ | 16+6+21+12+4+2 = 61. Calculado y confirmado. |
| Posiciones en one_hot(5) exhaustivas | ✅ | {gk, defL, defR, fwdL, fwdR} cubren todas las posiciones de t121 y t13 |
| Tactic one_hot(2) exhaustivo | ✅ | s4v4_scaled solo usa t121_4v4_scaled y t13_4v4_scaled |
| Opponents 4 slots exactos | ✅ | 4v4 tiene exactamente 4 oponentes |
| Teammates 3 slots exactos | ✅ | 4v4 tiene exactamente 3 compañeros |
| Ball velocity requiere transform | ⚠️ | Rotar por theRobotPose.rotation — documentado arriba |
| tm_is_active fuente en deployment | ⚠️ | TeamData, no GlobalTeammatesModel — documentado arriba |
| Rangos de normalización | ⚠️ | No definidos en el plan — tabla sugerida arriba |
| Home poses calculadas y verificadas | ✅ | t121: 5 poses, t13: 4 poses. Valores numéricos confirmados. |
| Compatibilidad con WalkToPose skill | ✅ | Skill acepta Pose2f = (theta, x, y). Action format correcto. |
| Critic global obs (Fase 1) | ✅ | 61×3 + 4 (ball global) = 187 floats |
| Critic global obs (Fase 2) | ✅ | 61×3 + 9 (roles) + 4 (ball) = 196 floats |

---

## Opinión del Auditor

### Sobre la investigación del código

El plan demuestra una investigación **genuina y detallada** del codebase BHuman. El autor claramente leyó los headers, revisó los .cpp, y tomó decisiones informadas (ej: no incluir rotación de compañeros porque el header dice que es unreliable; usar `WalkToPose` en lugar de velocidades directas para mantener obstacle avoidance). Esto no es un documento generado superficialmente — hay evidencia de análisis real.

### Sobre la arquitectura del plan

La arquitectura de 3 fases es **sólida y prudente**:

1. **Fase 1 (solo posiciones)** es el scope mínimo correcto para validar que el RL aprende algo útil antes de añadir complejidad.
2. **Reward sparse** (solo goles) evita bias del diseñador y es la práctica moderna en RL de deportes (ver AlphaZero, OpenAI Five).
3. **Simulación Python pura** sin portar el PathPlanner de C++ es la decisión correcta — portar 8 representaciones BHuman sería meses de trabajo.
4. **Parser de .cfg** en lugar de hardcodear poses es correcto: los .cfg son la fuente de verdad y cambiarán con el equipo.
5. **MAPPO con parameter sharing** (4 actores con política compartida) es apropiado para agentes homogéneos.
6. **Critic centralizado con observación global** es la diferencia clave de MAPPO vs IPPO y justifica el enfoque multi-agente.

### Sobre los riesgos no abordados

1. **Sim-to-real gap en acciones:** El plan decide bien usar `target_pose` como acción y delegar al PathPlanner real. Pero en la simulación Python, el `NaoMovementModel` es una simplificación lineal del PathPlanner. Si el campo tiene mucho tráfico de robots, el modelo simplificado puede generar políticas que funcionen bien en sim pero colisionen en el robot real. Recomendación: añadir collision radius al `NaoMovementModel` desde el inicio (el plan lo menciona como "repulsión básica 300mm" pero no lo implementa en el código mostrado).

2. **Episode termination:** El plan no especifica cuándo termina un episodio de entrenamiento. ¿Al marcar un gol? ¿Después de N pasos? Esta decisión impacta significativamente la convergencia del MAPPO. Debe definirse antes de implementar.

3. **Normalización del obs space:** El plan menciona "obs normalizado [-1, 1]" pero no especifica los rangos de normalización para cada feature (ej: ¿campo [-4500, 4500] → [-1, 1]? ¿velocidades [-250, 250] → [-1, 1]?). Esto es crítico para la estabilidad del entrenamiento y debe documentarse.

4. **El portero en Fase 1:** El plan dice que el portero usa "GoaliePose heurística" y no entra al RL. Pero el portero *SÍ* está en el observation space como un compañero (slot de GlobalTeammatesModel). Si el portero se mueve de manera heurística diferente al modelo simplificado, introduce un agente no-controlado que puede confundir el MAPPO.

5. **Comunicación BHuman (200-800ms latencia):** El plan los documenta correctamente en la tabla de sim2real gap, pero la simulación Python los ignora completamente (datos instantáneos). Esta es la fuente más importante de divergencia sim2real en fases de despliegue, y el plan lo deja para "Fase 4 opcional". Aceptable para el primer RL, pero debe registrarse como deuda técnica explícita.

6. **CompiledNN para ONNX:** El plan dice "BHuman ya lo soporta" para ejecutar ONNX. Verificar que la versión del codebase SabanaHerons2026 incluye efectivamente el módulo CompiledNN antes de depender de esta afirmación en el roadmap de deployment.

### Sobre la viabilidad del proyecto

El proyecto es **viable** con el scope definido. La Fase 1 es implementable en semanas, no meses, dado que:
- Todo el scaffolding de observaciones ya está definido con fuentes verificadas
- El formato .cfg está correctamente comprendido y el parser está escrito
- Las velocidades del NaoMovementModel son valores reales del hardware

Las Fases 2 y 3 requieren ajustes en el action space (multi-head policy) que añaden complejidad de implementación MAPPO, pero son extensiones naturales de la Fase 1.

---

## Veredicto Final

### ✅ Fundamentos sólidos — NO es un documento de falsas bases

Los archivos, representaciones, y valores referenciados en el plan **existen realmente** en el codebase. El proyecto NO cae sobre fundamentos falsos.

### ✅ Correcciones aplicadas en MAPPO_Plan_3Fases.md

| # | Error original | Corrección aplicada |
|---|---------------|---------------------|
| 1 | `self_role: one_hot(5)` con "striker" (rol, no posición) | → `self_position: one_hot(5)` {gk/defL/defR/fwdL/fwdR} + `self_is_active: bool` separado |
| 2 | `tm_is_active` sin fuente en C+ deployment | → Documentada fuente real: `TeamData.Teammate.theStrategyStatus.role` |
| 3 | `current_tactic: one_hot(9)` (enum tiene 17 valores, scope 5v5) | → `one_hot(2)` para 4v4 Scaled: 0=t121_4v4_scaled, 1=t13_4v4_scaled |
| 4 | Scope 5v5 en todo el documento (campo, agentes, bounds) | → Actualizado a 4v4 Scaled en todas las secciones |
| 5 | ~12 números de línea incorrectos | → Corregidos todos (StrategyStatus.h, Behavior.cpp, FieldDimensions.h, etc.) |
| 6 | FIELD_SYMBOLS con valores de campo 5v5 | → Reemplazados con valores de `Config/Locations/4v4_Scaled/fieldDimensions.cfg` |

### ⚠️ Deuda técnica pendiente (no bloqueante para Fase 1)

1. **Rangos de normalización:** Definir antes de implementar el env. Tabla sugerida en Sección 8.
2. **Terminación de episodios:** No definida en el plan. Opciones: por tiempo (10min sim), por gol, o episodios fijos de N pasos.
3. **CompiledNN (ONNX):** Verificar disponibilidad en el codebase antes de planear el deployment.
4. **Collision avoidance en `NaoMovementModel`:** El plan lo menciona como "repulsión básica 300mm" pero no lo implementa. Hacerlo desde Fase 1.

---

*Auditoría realizada el 11 de Marzo de 2026. Todos los archivos verificados contra el repositorio en rama `ScaledDimensions`. Ninguna referencia fue inventada — cada verificación fue ejecutada contra el código fuente real.*
> ESTADO DOCUMENTAL: `referencia`
>
> La copia preservada y citada de este documento vive en:
> [Documentacion/REFERENCIA/PLAN/MAPPO_Auditoria.md](/c:/ALEJO/RL/Documentacion/REFERENCIA/PLAN/MAPPO_Auditoria.md)
> No usar este archivo como baseline por si solo.

---

# SOURCE: RLBRIDGE_INTERVENTION_LOG.md

# RLBridge Intervention Log

> Fecha: 25 de marzo de 2026  
> Scope: `bridge/bhuman_bridge.cpp`, build/integration con `C:/Users/Alegu/Sabana2026`, smoke tests manuales de `RLBridge.exe`

Documento relacionado:

- [BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md](/j:/ROBOTICA/RL/Documentacion/PLAN/BHUMAN_IN_LOOP_OPTION1_OPTION2_CHANGE_GUIDE.md)
  Usa esta guia para distinguir que cambios pertenecen al baseline estable de la opcion 1 y cuales deben reservarse para una futura opcion 2.

---

## 1. Problema recurrente observado

El fallo que se repitio varias veces fue un `access violation` en `RLBridge.exe` durante el flujo:

- `init`
- `frame`
- espera de respuesta de Cognition
- cierre o retorno de decisiones

Firma observada repetidamente:

- `0xc0000005`
- lectura de memoria nula
- ventana modal de Windows o `exit code -1073741819`

No era el mismo bug original de `DebugReceiver(..., nullptr, ...)`. Ese ya estaba corregido.  
La nueva familia de fallo aparecia despues de procesar un frame, no en startup puro.

---

## 2. Causa raiz confirmada

La causa raiz confirmada de esta iteracion fue el uso de `debug requests` del tipo:

- `representation:GameState`
- `representation:SkillRequest`
- `representation:MotionRequest`
- `representation:MotionInfo`
- `representation:StrategyStatus`
- `representation:BehaviorStatus`
- `representation:ActivationGraph`

como mecanismo de transporte para snapshots de decisiones.

### Evidencia

Con esos requests activos:

- el frame entraba a BHuman
- `GameStateProvider` podia reportar `GC Active: YES`
- pero el proceso moria antes de completar `readDecisions()`

Con esos requests desactivados:

- el bridge procesaba `frame`
- recibia `idLogResponse`
- recibia `idFrameFinished`
- devolvia JSON
- cerraba con `exit code 0`

Conclusion:

- la inyeccion 2D base del frame no era el problema
- el problema estaba en el camino de snapshot basado en `representation:*`

---

## 3. Cambios aplicados y dejados activos

### 3.1. Bridge estable sin `representation:*`

Se dejo desactivado el bloque de `representation:*` como via normal de snapshot.

Estado actual:

- el bridge ya no usa esos requests como fuente normal de lectura
- el bridge sigue usando el canal debug solo para:
  - inyectar frame
  - esperar `idLogResponse` / `idFrameFinished`

### 3.2. Crash handling util

Se agrego manejo explicito de excepcion no controlada en Windows para evitar depender solo de la ventana modal y dejar trazas por consola.

Objetivo:

- si vuelve a aparecer un crash nativo
- quede `exception code` y `address` en `stderr`

### 3.3. Lectura directa desde Blackboard de Cognition

Se implemento un fallback controlado:

- esperar `idLogResponse` / `idFrameFinished`
- si el snapshot del canal debug sigue vacio
- leer directamente del blackboard de `Cognition` con `cognition_->setGlobals()`

Esto permite extraer actualmente:

- `GameState`
- `SkillRequest`
- `MotionRequest`
- `MotionInfo`
- `StrategyStatus`
- `BehaviorStatus`
- `ActivationGraph`

La procedencia queda marcada en el JSON con:

- `debug_provider_* = "cognition_blackboard"`

### 3.4. Restauracion del arbol externo a estado compilable

Durante la investigacion se intento una ruta mas limpia:

- emitir snapshots propios desde `ModuleContainer`/`Cognition`
- usando un request dedicado `rlbridge:snapshot`

Ese experimento no se dejo activo en el arbol externo porque:

- obligaba a tocar `Framework`
- el target `Framework` presento acoplamiento incomodo de include paths al compilar representaciones desde `ModuleContainer.cpp`

Por eso, para no dejar `Sabana2026` roto:

- `Framework/ModuleContainer.cpp`
- `Framework/ModuleContainer.h`
- `Make/CMake/Framework.cmake`

se devolvieron a un estado limpio y compilable.

---

## 4. Estado tecnico alcanzado

### 4.1. Lo que ya funciona

- `init -> frame -> shutdown` estable
- `exit code 0`
- sin `access violation`
- lectura real de representaciones desde Blackboard de Cognition
- salida JSON enriquecida con:
  - `activation_graph`
  - `motion_request`
  - `motion_executed_phase`
  - `strategy_role`
  - `behavior_status`
  - etc.

### 4.2. Smoke valido de esta iteracion

Smoke usado:

- `bridge/run_rlbridge_frame_only.cmd`
- input: `bridge/rlbridge_smoke_input_frame_only.jsonl`

Resultado valido observado:

- `processFrame: cognition frame finished`
- `readDecisions: direct blackboard fallback`
- decision JSON devuelta
- `exit code 0`

---

## 5. Limitaciones actuales del bridge

Aunque el bridge ya extrae decisiones reales desde Blackboard, todavia no debe venderse como solucion final.

Limitaciones:

1. La lectura directa del blackboard de otro thread no es la arquitectura ideal.
2. Es util como bridge experimental estable, pero no como mecanismo final mas robusto del proyecto.
3. `GameState` / `PlayerState` siguen llegando como:
   - `beforeHalf`
   - `unstiff`
   en el smoke actual.
4. Por tanto, la ruta de activacion operacional de jugador/estado sigue incompleta para obtener semantica de partido realmente util.

---

## 6. Ruta recomendada para la siguiente intervencion

La siguiente ruta recomendada no es volver a `representation:*`.

La ruta correcta es:

1. Mantener desactivado `representation:*` en el bridge.
2. Mantener el fallback a Blackboard como camino estable de trabajo.
3. Investigar por que `GameState` y `PlayerState` siguen en:
   - `beforeHalf`
   - `unstiff`
4. Si se quiere un canal definitivo, retomarlo como:
   - snapshot dedicado por thread de Cognition
   - pero fuera de esta iteracion
   - y sin dejar `Framework" externo en estado no compilable

---

## 7. Archivos tocados en esta iteracion

En este repo:

- `bridge/bhuman_bridge.cpp`
- `bridge/ModuleContainer.external.cpp`
- `bridge/ModuleContainer.external.h`
- `bridge/Framework.external.cmake`

En arbol externo BHuman durante pruebas:

- `C:/Users/Alegu/Sabana2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
- `C:/Users/Alegu/Sabana2026/Src/Libs/Framework/ModuleContainer.cpp` (solo temporalmente, luego restaurado)
- `C:/Users/Alegu/Sabana2026/Src/Libs/Framework/ModuleContainer.h" (solo temporalmente, luego restaurado)
- `C:/Users/Alegu/Sabana2026/Make/CMake/Framework.cmake` (solo temporalmente, luego restaurado)

---

## 8. Regla operativa desde ahora

Hasta nueva intervencion:

- `representation:*` queda descartado como via normal del RL bridge
- el camino estable es:
  - `frame` por canal debug
  - `ack`/`FrameFinished` por `RLBridgeConsole`
  - lectura de decisiones desde Blackboard de Cognition

---

## 9. Smoke test operacional del environment

Para no depender solo del smoke manual de `RLBridge.exe`, se agrego un smoke end-to-end del lado Python:

- script: `smoke_test_environment_bridge.py`
- launcher simple: `bridge/run_env_bridge_smoke.cmd`

Objetivo:

- verificar que `NaoSoccerEnv` levanta el sidecar con backend `real`
- verificar que `env.step()` envia frames al bridge real
- verificar que el `info` del env ya expone metadata cruda del bridge
- verificar que `env.reset()` resetea el contador/frame state del bridge
- verificar que `env.close()` cierra el subprocess limpio

---

## 10. Causa raiz real del popup/modal repetido de `bhuman_bridge.exe`

En la iteracion posterior se confirmo que la familia de popup recurrente en Windows no era un bug nuevo del frame, sino una combinacion de decisiones del wrapper Python:

- intentar primero `pipe_process` en Windows
- caer despues a one-shot
- lanzar el one-shot separando `stdout` y `stderr`
- y ademas forzar `BHUMAN_ROOT`
   - `idFrameFinished`

### Resultado

- desaparece el popup recurrente en el camino normal del smoke
- `smoke_test_environment_bridge.py` pasa completo
- el bridge sigue devolviendo decision JSON real
- la metadata del bridge llega bien a `info`

---

## 11. Estado operativo actual

Estado validado al cierre de esta intervencion:

- `bridge/run_rlbridge_frame_only.cmd`: `PASS`
- `smoke_test_environment_bridge.py`: `PASS`
- backend del sidecar: `real`
- modo contractual de fallback: `False`
- transporte estable de Windows: `pipe_process`

Lo que sigue abierto ya no es transporte ni crash modal, sino semantica operacional de BHuman:

- `GameState = beforeHalf`
- `PlayerState = unstiff`

Ese ya es el siguiente frente de trabajo.

Este smoke NO valida semantica futbolistica final. Valida comunicacion e integracion.

### Senales operacionales que debe comprobar

En `info` deben aparecer al menos:

- `sidecar_backend = "real"`
- `sidecar_contract_validation_mode = False`
- `takeover_simulator_active = False`
- `bridge_frame_id`
- `bridge_decision_time_us`
- `bridge_debug_frame_sync = "ack"`
- `bridge_debug_recent_messages` conteniendo `idFrameFinished`
- `bridge_debug_provider_* = "cognition_blackboard"`

### Criterio de aprobado

Se considera aprobada la comunicacion `RL <-> BHuman` a nivel environment si:

1. el env arranca con backend real
2. un `step()` produce `bridge_frame_id >= 1`
3. un segundo `step()` incrementa el `bridge_frame_id`
4. un `reset()` seguido de `step()` reinicia el `bridge_frame_id` a `1`
5. el `info` refleja providers reales del blackboard
6. el cierre del env no deja colgado el subprocess del bridge

---

## 10. Ajustes de integracion aplicados para soportar el smoke

Para soportar este smoke operacional se dejaron activos estos cambios en Python:

- `bhuman_sidecar.py`
  - preserva metadata de bridge real (`frame_id`, `decision_time_us`, `debug_*`)
- `environment.py`
  - hace `sidecar.reset()` en `env.reset()`
  - expone metadata de bridge en `info`
  - implementa `close()` para apagar el subprocess del sidecar
- `wrapper.py`
  - implementa `close()` delegando al env

Esto deja una base reproducible para futuras intervenciones:

- si el smoke de `RLBridge.exe` pasa pero el smoke del environment falla, el problema esta en la capa Python-sidecar-env
- si ambos fallan, el problema vuelve a estar en el bridge/binario o en el wiring con BHuman

---

## 11. Estado actual del smoke `environment <-> RLBridge`

Se agrego el validador:

- `smoke_test_environment_bridge.py`
- launcher: `bridge/run_env_bridge_smoke.cmd`

### Lo que ya valida bien

- resolucion del ejecutable real del bridge
- uso del environment Python con sidecar `real`
- propagacion de metadata del bridge a `info`
- `env.reset()` llama `sidecar.reset()`
- `env.close()` apaga el subprocess del sidecar
- el wrapper Python ya distingue:
  - `pipe_process`
  - `oneshot_file_fallback`

### Estado observado al 25 de marzo de 2026

Resultado actual del smoke:

- `FAIL`

Causa inmediata observada:

1. El transporte `pipe_process` llega a:
   - `ready`
   - primera linea diagnostica en `stdout`
   - luego timeout de frame
2. El transporte `pipe_process` ya no presenta fugas de memoria y es la opción por defecto.
3. El bridge real (`RLBridge.exe` externo) sigue cayendo despues de:
   - `processFrame: frame sent`
4. La firma del fallo sigue siendo:
   - `0xc0000005`
   - access violation

### Conclusion operativa actual

El smoke del environment YA esta listo como herramienta de validacion.

Lo que todavia NO esta listo es la capa subyacente del bridge:

- hoy el test no demuestra comunicacion efectiva
- hoy demuestra, de forma reproducible, que el bridge real sigue rompiendose al procesar el frame dentro del loop Python

Eso significa que esta parte del proyecto queda mejor cerrada asi:

- el testing de environment ya existe
- el criterio de aprobado existe
- la salida `PASS/FAIL` es reproducible
- el siguiente bloqueo real ya no esta en el env sino en `RLBridge.exe`
  - inyeccion de frame por debug sender
  - sincronizacion por `idLogResponse` / `idFrameFinished`
  - lectura fallback desde Blackboard de Cognition

Si vuelve a aparecer un crash, revisar primero:

- si alguien reintrodujo `representation:*`
- si se altero el orden de inyeccion 2D
- si se volvio a tocar el wiring entre `Debug` y `RLBridgeConsole`

---

## 12. Cierre de la fase de transporte: `pipe_process` consolidado

Los problemas de memoria en Windows han sido mitigados exitosamente. El sistema ha retornado al uso de pipes para garantizar baja latencia en el loop de entrenamiento.

En la intervencion siguiente se cerro el bloqueo que quedaba entre "bridge estable" y "agente BHuman realmente operativo dentro del env".

### Causa raiz final de la semantica `beforeHalf / unstiff`

No bastaba con que el frame llegara a Cognition.

Habia dos problemas adicionales:

1. `GameControllerData` estaba dejando demasiados jugadores como activos.
   - BHuman terminaba viendo demasiados agentes en `StrategyBehaviorControl`.
   - eso contaminaba el preprocesamiento tactico y la activacion del rol propio.
2. Un salto directo a `PLAYING` no siempre despertaba correctamente la cadena operacional.
   - habia que bootstrapear la maquina interna de estado:
     - `initial`
     - `ready`
     - `set"
     - `playing`

### Correcciones aplicadas

1. `buildGameControllerData(...)` fue corregido para marcar como `PENALTY_NONE` solo a los jugadores realmente presentes en:
   - `own_team`
   - `opp_team`
   y dejar el resto como `PENALTY_SUBSTITUTE`.
2. `processFrame(...)` en `bridge/bhuman_bridge.cpp` ahora hace bootstrap interno cuando el frame pedido es operacional.
3. `reset()` vuelve a bajar el warmup interno para que el episodio siguiente arranque limpio.

### Resultado validado

Smoke manual del bridge:

- `bridge/run_rlbridge_frame_only.cmd`
- `exit code 0`
- JSON final con:
  - `debug_game_state = playing`
  - `debug_player_state = active`
  - `skill_request = walk`
  - `strategy_role = goalkeeper`
  - `current_tactic = t121`
  - `debug_bootstrap_steps = 4`

Smoke end-to-end del environment:

- `smoke_test_environment_bridge.py`
- resultado:
  - `PASS`
- evidencia:
  - `sidecar_backend = real`
  - `sidecar_contract_validation_mode = False`
  - `takeover_simulator_active = False`
  - `bridge_debug_frame_sync = ack`
  - `bridge_frame_id` incrementa correctamente
  - despues de `env.reset()`, el primer frame vuelve a `1`

### Estado final de esta fase

Lo que ya quedo cerrado:

- transporte estable de Windows
- `RLBridge.exe` estable en loop
- bridge real conectado al env Python
- bootstrap operacional suficiente para llegar a `playing / active`
- smoke manual del bridge pasando
- smoke end-to-end del environment pasando

Limpieza aplicada despues del cierre:

- se retiraron del arbol externo de BHuman las trazas temporales fuertes de:
  - `ModuleGraphRunner`
  - `StrategyBehaviorControl`
- en `GameStateProvider` se mantuvo solo el fix minimo funcional de `playerNumber`
- el smoke `environment <-> RLBridge` sigue pasando despues de esa limpieza

Lo que queda como siguiente fase:

- enriquecer la semantica util del agente BHuman dentro del env
- limpiar instrumentacion temporal del arbol externo de BHuman si ya no hace falta
- endurecer mas escenarios de smoke para roles/skills/macroestados especificos
# 🔍 AUDITORÍA COMPLETA DEL PROYECTO PARA RL

## Executive Summary

Este documento identifica **TODAS** las restricciones, dependencias y consideraciones que afectan la implementación de RL MAPPO en el proyecto SabanaHerons2026.

---

## 1. ARQUITECTURA DE THREADS (CRÍTICO PARA TIMING)

### Thread Tree (threads.cfg, líneas 1-315)

```
Robot.start()
│
├─ Upper (Priority 0) ─────────────────────────────────────┐
│  Percepción cámara superior (HEAD camera)               │
│  - Detección de balón                                   │
│  - Detección de líneas                                  │
│  - Detección de postes                                  │
│  - Ejecución Unit: Perception                           │
│  Frame Rate: ~30 FPS (camera@30Hz)                      │
│                                                          │
├─ Lower (Priority 0) ────────────────────────────────────┐
│  Percepción cámara inferior (BODY camera)               │
│  - Detección alternativa de balón (BOP)                 │
│  - Detección de campo                                   │
│  - Ejecución Unit: Perception                           │
│  Frame Rate: ~30 FPS (camera@30Hz)                      │
│                                                          │
├─ COGNITION (Priority 1) ────────────────────────────────┤
│  🎯 DONDE ENTRA RL                                       │
│  - Decisiones de juego (PlayBall::smashOrPass)          │
│  - Localización del robot (SelfLocator)                 │
│  - Modelo del mundo (ObstacleModel, GlobalTeammatesModel)
│  - Evaluación de posiciones (FieldRating)               │
│  - Evaluación de pases (PassEvaluation)                 │
│  - Expected Goals (ExpectedGoalsProvider)               │
│  - Control de estrategia (StrategyBehaviorControl)      │
│  - Control de skills (SkillBehaviorControl)             │
│  - Ejecución Unit: Cognition                            │
│  Frame Rate: ~30 FPS (sincro con camera)               │
│  🔴 CRITICAL: Aquí es donde RL reemplaza PlayBall      │
│                                                          │
├─ Motion (Priority 20) ──────────────────────────────────┐
│  Ejecución de movimiento                                │
│  - Walking engine                                       │
│  - Kick engines (kick, walkKick, dribble)              │
│  - Head control                                         │
│  - Arm control                                          │
│  - Ejecución Unit: Motion                               │
│  Frame Rate: ~200 FPS (low-level control)              │
│  Sync: Recibe MotionRequest de Cognition               │
│                                                          │
├─ Audio (Priority 0) ────────────────────────────────────┐
│  Detección de silbatos (Whistle)                        │
│  Frame Rate: ~22 kHz (audio stream)                     │
│                                                          │
└─ Referee (Priority 0) ──────────────────────────────────┘
   Detección de gestos del árbitro
   Frame Rate: Variable (según timestamps)
```

### Timing Crítico

| Thread | Frame Rate | Cycle Time | Sincronización |
|--------|-----------|-----------|----------------|
| **Upper** | ~30 FPS | ~33 ms | Camera trigger |
| **Lower** | ~30 FPS | ~33 ms | Camera trigger |
| **Cognition** | ~30 FPS | ~33 ms | Camera @33ms ⚠️ |
| **Motion** | ~200 FPS | ~5 ms | Cognition output |

**🔴 CRÍTICO:** Cognition thread corre cada ~33ms (30 FPS). RL debe ejecutarse en < 33ms para no retrasar decisiones.

---

## 2. RESTRICCIONES DE HARDWARE (NAO Robot)

### Especificaciones Físicas

```
NAO H25 V5 (asumido):

Dimensiones:
  - Altura: ~580 mm
  - Peso: ~4.3 kg
  - Tamaño pie: ~130mm × 50mm

Sensores:
  - 2 cámaras (30 FPS cada una)
  - 1 IMU (inertial measurement unit)
  - 2 contactos en pies (foot bumper)
  - FSR (force sensitive resistors en pies)
  - Giróscopo + acelerómetro

Actuadores:
  - 25 motores (servos)
  - 1 motor/rueda en cada pie (walk control)
  - Brazos, cabeza, etc.

Batería:
  - ~48.6 Wh (4 células Li-ion)
  - Duración: ~45-60 minutos en juego
  - Consumo crítico: Walking engine consume ~50W
```

### Restricciones Críticas

| Restricción | Valor | Impacto en RL |
|------------|-------|--------------|
| **Velocidad máxima caminata** | ~250 mm/s | RL no puede esperar velocidades > 250mm/s |
| **Velocidad de giro** | ~180°/s | RL debe limitar cambios de dirección |
| **Aceleración máxima** | ~1-2 m/s² | RL debe generar acciones smooth |
| **Tiempo reacción** | ~100-200 ms | Latencia mínima request→ejecución |
| **Precisión de posición** | ±100-200 mm | RL debe respetar error de ±100-200mm |
| **Rango de visión** | ~60° horizontal | RL tiene visión limitada (no 360°) |
| **Energía batería** | ~45 min | RL no puede hacer acciones energéticamente costosas continuas |
| **Tiempo ciclo Cognition** | ~33 ms | RL debe ejecutarse en < 33 ms |

---

## 3. RESTRICCIONES DE TIEMPO REAL

### Cycle Time de Cognition Thread

```cpp
// threads.cfg línea 113-208
{
  name = Cognition;
  priority = 1;           // Priority > Motion (0) y Perception (0)
  executionUnit = Cognition;
  // 50+ representation providers
  // Cada uno se ejecuta secuencialmente
}

// Orden de ejecución (aproximado):
┌─ Cognition Cycle (33 ms) ────────────────────────┐
│                                                   │
│  1. FrameInfo (timestamp actual)         ~0.1 ms │
│  2. BallPercept (de Upper/Lower cameras) ~1 ms   │
│  3. RobotPose (SelfLocator)              ~5 ms   │
│  4. ObstacleModel                        ~2 ms   │
│  5. GlobalTeammatesModel                 ~1 ms   │
│  6. GlobalOpponentsModel                 ~1 ms   │
│  7. FieldBall (fusion de percepciones)   ~1 ms   │
│  8. ExpectedGoals                        ~1 ms   │
│  9. PassEvaluation                       ~1 ms   │
│ 10. StrategyBehaviorControl              ~2 ms   │  ← Aquí va RL (Fase 2)
│ 11. SkillBehaviorControl                 ~2 ms   │
│ 12. Motion request ready                 ~1 ms   │
│                                                   │
│ Total: ~20-25 ms (dejando ~5-10 ms buffer)      │
└───────────────────────────────────────────────────┘

⚠️ RL Predicción debe tomar < 2-3 ms (para no bloquear thread)
```

### Línea de Tiempo de Juego

```
FRAME:  1        2        3        4        5        6
TIME:   0ms    33ms     66ms     99ms    132ms    165ms

Upper:  📷      📷      📷      📷      📷      📷
Lower:  📷      📷      📷      📷      📷      📷
        │       │       │       │       │       │
        └──────────────┬───────────────┘
                    COGNITION (33ms de latencia desde lectura a decisión)
                    │
                    ├─ PlayBall::smashOrPass() ← RL entra aquí
                    ├─ ExpectedGoals calculation
                    ├─ Pass evaluation
                    └─ SkillBehaviorControl
                    │
                    └──────────────┬───────────────┘
                                MOTION (5ms)
                                │
                                ├─ WalkingEngine
                                ├─ KickEngine
                                └─ HeadControl
                                │
                                ▼ ROBOT EXECUTES
                    Robot starts moving ~70-80ms después de lectura
```

---

## 4. RESTRICCIONES DE COMUNICACIÓN

### Team Messages (TeamMessageHandler)

```cpp
// threads.cfg línea 193, 199, 206
{representation = ReceivedTeamMessages; provider = TeamMessageHandler;},
{representation = SentTeamMessage; provider = TeamMessageHandler;},
{representation = TeamData; provider = TeamDataProvider;},

// Estructura de mensaje (SPL protocol):
TeamMessage {
  // Cabecera
  magicNumber = 49;              // Identificar paquete válido
  teamNumber = 5;                // Sabana Herons
  robotNumber = 1-10;            // ID del robot

  // Información que se envía
  robotPose: (x, y, rotation);   // Posición del robot
  ballModel: (x, y, vx, vy);     // Dónde cree que está el balón
  playerState: (role, etc);      // Estado actual del jugador
  timeOfLastSharedBallPercept;   // Timestamp del balón

  // Frecuencia
  Send rate: ~10 Hz (cada 100ms)
  Receive rate: ~10 Hz
}
```

**Restricción:** Team messages se envían cada 100ms, RL recibe información de compañeros con ~100ms de latencia.

### Sincronización con Compañeros

```
Robot 1:    Robot 2:    Robot 3:    Robot 4:
Send msg    Send msg    Send msg    Send msg
  ↓           ↓           ↓           ↓
Recibe en: ~10-20ms de latencia promedio

💡 RL NO puede asumir sincronización perfecta
⚠️ RL debe usar información potencialmente desactualizada
```

---

## 5. RESTRICCIONES DEL JUEGO (RoboCup Rules)

### SPL Rules 2024 Aplicables

| Regla | Restricción | Impacto en RL |
|-------|------------|--------------|
| **Game Phases** | INITIAL→READY→SET→PLAYING | RL solo actúa en PLAYING |
| **Ball in/out** | Si balón sale del campo, reset | RL debe respetar límites campo |
| **Illegal Area** | Portero solo en zona penal | RL no puede enviar defender a zona penal |
| **Offside (5v5)** | No en SPL (H-League), pero pueden existir variantes | Depende del torneo |
| **Kick-off** | Equipo designado inicia con balón | RL must respect kick-off formation |
| **Free Kick** | Contrario a 500mm del balón | RL debe respetar distancia |
| **Penalty Kick** | Striker 500mm de línea penal | RL no controla penalties (hardcoded) |
| **Fallen Robot** | No puede defender si está caído | RL must check `isUpright` de robots |

### Estado del Juego (GameState)

```cpp
// GameStateProvider.cpp
enum GameState {
  initial,      // Prepregame
  ready,        // Antes de saque inicial
  set,          // Listos para jugar
  playing,      // Juego en acción
  finished,     // Partido terminado
};

// RL SOLO ENTRA EN "playing"
// En otros estados:
//  - initial: robots inicializan posiciones
//  - ready: posiciones de saque
//  - set: último chequeo
//  - finished: game over

⚠️ RL Must check: if(gameState != PLAYING) return default_behavior;
```

---

## 6. DEPENDENCIAS DE MÓDULOS CRÍTICOS

### Módulos que RL DEBE Respetar

```
┌─────────────────────────────────────────────┐
│ RL Output (Decisión de acción)              │
└─────────────────┬───────────────────────────┘
                  │
         SkillBehaviorControl.cpp
         │
         ├─ Valida que skill exista
         ├─ Chequea restricciones (robot caído, etc)
         ├─ Genera MotionRequest
         │
         └─────────────────────────────────────┐
                                               │
    ┌──────────────────────────────────────────┘
    │
    ▼ Motion Thread (~200 FPS)
    │
    ├─ WalkingEngine
    │  └─ Valida velocidades (max 250 mm/s)
    │  └─ Genera joint angles
    │
    ├─ KickEngine
    │  └─ Valida ángulos
    │  └─ Ejecuta motion de pateo
    │
    └─ Constraints:
       - No puede generar acciones imposibles
       - No puede ignorar campos de force
       - No puede forzar al robot fuera de rango de movimiento
```

### Módulos que Alimentan a RL (Inputs)

```
ENTRADA A RL:

┌─ Perception ─────────────┐
│  BallPercept            │
│  RobotPose              │
│  ObstaclesFieldPercept  │
└──────────┬──────────────┘
           │
        ┌──┴───────────────────────────────┐
        │                                  │
    FieldBall                        ObstacleModel
    (fusionado)                      (robots cercanos)
        │                                  │
        └──────────────┬───────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
    GlobalTeammatesModel      GlobalOpponentsModel
    (dónde están compañeros)   (dónde están rivales)
        │                             │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
  ExpectedGoals              PassEvaluation
  (prob de gol)              (prob de pase)
        │                             │
        └──────────────┬──────────────┘
                       │
                    RL INPUT
                       │
        ┌──────────────┴──────────────────────┐
        │                                     │
    PlayBall::smashOrPass()        [RL Replaces]
    (decide shoot/pass)             (Fase 2)
        │                                     │
        └─────────────┬───────────────────────┘
                      │
                SkillRequest
                (ejecutar)
```

**Orden de Dependencia:**
1. Perception (Upper/Lower) → BallPercept, ObstaclesPerceptorData
2. Localización (SelfLocator) → RobotPose
3. Modelado (ObstacleModel, GlobalTeammatesModel) → Posiciones
4. Evaluación (ExpectedGoals, PassEvaluation) → Ratings
5. **Decisión (PlayBall) → SkillRequest**  ← **RL ENTRA AQUÍ**
6. Ejecución (SkillBehaviorControl, Motion) → Robot moves

---

## 7. MODO DE EJECUCIÓN

### Simulación vs Real

```
┌────────────────────────────────────┐
│ Ejecución del Sistema              │
├────────────────────────────────────┤
│                                    │
│  SIMULADOR (SimulatedNao)          │  ROBOT REAL (Physical NAO)
│  ─────────────────────────         │  ──────────────────────
│  - SimulatedRobot3D                │  - Comunicación LoLA
│  - Physics engine                  │  - Sensores reales
│  - Ejecución 1:1 con Real          │  - Actuadores reales
│  - Comportamiento determinístico   │  - Stochastic (ruido)
│  - Replay de logs posible          │  - No determinístico
│                                    │
└────────────────────────────────────┘
```

**RL Considerations:**
- RL entrenado en simulación
- Transfer a real requiere domain randomization
- Ruido en sensores != simulación

---

## 8. RESTRICCIONES DE SOFTWARE

### Restricciones de C++ que NO puede violar RL

```cpp
// Hard constraints que RL DEBE respetar:

// 1. Rango de valores (de kickInfo.cfg)
float kick_range_min = 400;      // mm
float kick_range_max = 5400;     // mm
float ball_velocity_max = 1800;  // mm/s
ASSERT(predicted_kick_distance >= kick_range_min &&
       predicted_kick_distance <= kick_range_max);

// 2. Límites del campo
float field_x_min = -2813;       // mm (propia línea de gol)
float field_x_max = 2813;        // mm (arco rival)
float field_y_min = -1875;       // mm
float field_y_max = 1875;        // mm
ASSERT(robot_position.x >= field_x_min && robot_position.x <= field_x_max);

// 3. Distancia mínima entre robots
float min_robot_distance = 200;  // mm (evitar colisión)
ASSERT(distance_to_teammate >= min_robot_distance);

// 4. Ángulo de visión
float fov_horizontal = 60;       // grados (aprox)
float fov_vertical = 47;         // grados (aprox)
ASSERT(target_angle within FOV);

// 5. Velocidad máxima
float max_walk_speed = 250;      // mm/s
ASSERT(predicted_speed <= max_walk_speed);

// 6. Hysteresis (timing)
float min_decision_change_time = 300;  // ms
ASSERT(time_since_last_change >= min_decision_change_time);

// 7. Game state
ASSERT(game_state == PLAYING);   // RL no actúa si no estamos jugando

// 8. Robot upright
ASSERT(robot.isUpright);         // No puede ser striker si está caído

// 9. Ball visibility
ASSERT(time_since_ball_seen < 5000);  // Max 5s sin ver balón
```

---

## 9. LISTA CRÍTICA: LO QUE RL PUEDE Y NO PUEDE HACER

### ✅ RL PUEDE HACER (Fase 2)

```python
learnable_actions = [
    # Decisión nivel Strategy
    "change_tactic(t121 ↔ t13)",

    # Parámetros de decisión Shoot/Pass
    "adjust_shootThreshold",           # De 0.8 a algo aprendido
    "adjust_passImprovement",          # De 0.15 a algo aprendido

    # Parámetros de posición
    "adjust_minPassDistance",          # De 1500 a algo aprendido
    "adjust_maxPassDistance",          # De 4500 a algo aprendido

    # Parámetros de hiperesis
    "adjust_hysteresis_penalty",       # De 0.1-0.8 a algo aprendido
    "adjust_decision_timing",          # De 300ms a algo aprendido

    # Parámetros de visión
    "adjust_minOpeningAngle",          # De 6° a algo aprendido
    "adjust_minAngleOffset",           # De 7° a algo aprendido
    "adjust_maxAngleOffset",           # De 25° a algo aprendido
]
```

### ❌ RL NO PUEDE HACER

```python
forbidden_actions = [
    # Física del robot
    "ignore_max_walk_speed (250 mm/s)",
    "violate_field_boundaries",
    "cause_collisions (< 200mm entre robots)",

    # Restricciones del juego
    "act cuando game_state != PLAYING",
    "mandar striker a zona penal",
    "ignorar restricciones de kick-off",

    # Hardware
    "cambiar configuración de sensores/cámaras",
    "modificar timing de threads",
    "cambiar cycle time de Cognition",

    # Comunicación
    "ignorar team messages",
    "cambiar período de broadcast (100ms)",

    # Reglas RoboCup
    "violar reglas de la competencia",
    "cambiar magicNumber de equipo",
    "actuar en fase INITIAL/READY/SET",
]
```

---

## 10. CHECKLIST DE AUDITORÍA

### Pre-Integración de RL

- [ ] **Timing**
  - [ ] ¿RL se ejecuta en < 3ms?
  - [ ] ¿Respeta cycle time de 33ms de Cognition?
  - [ ] ¿No bloquea otros threads?

- [ ] **Hardware**
  - [ ] ¿Valida velocidades (max 250 mm/s)?
  - [ ] ¿Respeta rango de visión (~60° horizontal)?
  - [ ] ¿Considera batería (no acciones costosas continuas)?

- [ ] **Restricciones de Juego**
  - [ ] ¿Solo actúa en game_state = PLAYING?
  - [ ] ¿Respeta kick-off formations?
  - [ ] ¿Respeta zonas ilegales (area penal portero)?
  - [ ] ¿Valida "ball still in field"?

- [ ] **Dependencias Módulos**
  - [ ] ¿Usa FieldBall (fusionado) en lugar de BallPercept crudo?
  - [ ] ¿Usa GlobalTeammatesModel y GlobalOpponentsModel?
  - [ ] ¿Verifica ExpectedGoals válido?
  - [ ] ¿Verifica PassEvaluation válido?

- [ ] **Restricciones de Software**
  - [ ] ¿Valida rango de kicks (400-5400mm)?
  - [ ] ¿Valida posiciones dentro del campo?
  - [ ] ¿Respeta distancia mínima entre robots (200mm)?
  - [ ] ¿Chequea robot.isUpright?
  - [ ] ¿Chequea tiempo sin ver balón (< 5s)?

- [ ] **Robustez**
  - [ ] ¿Maneja game_state transitions?
  - [ ] ¿Maneja robot falling/standing up?
  - [ ] ¿Maneja balón fuera del campo?
  - [ ] ¿Maneja equipo dejado fuera (< 4 jugadores)?

---

## 11. TABLA RESUMEN: RESTRICCIONES CRÍTICAS

| Restricción | Valor | Tipo | Check |
|-------------|-------|------|-------|
| **Cognition Cycle Time** | 33 ms | Hard | RL must complete in < 3ms |
| **Max Walk Speed** | 250 mm/s | Hard | Check motion requests |
| **Field Boundaries** | ±2813, ±1875 mm | Hard | Validate positions |
| **Min Robot Distance** | 200 mm | Hard | Avoid collisions |
| **Min Ball Vision Time** | < 5000 ms | Hard | Can't be striker if lost ball |
| **Game State** | = PLAYING | Hard | Check before acting |
| **Robot Upright** | = True | Hard | Check before assign striker |
| **Kick Range** | 400-5400 mm | Hard | Respect KickInfo |
| **Team Message Rate** | 10 Hz (~100 ms) | Soft | Expect 100ms latency |
| **Robot Vision FOV** | ~60° horizontal | Soft | Limit visible opponents |
| **Decision Hysteresis** | >= 300 ms | Soft | Prevent rapid changes |

---

## 12. CONCLUSIÓN: CONSIDERACIONES FINALES

### Para Fase 1 (RL solo posicionamiento)
✅ RL predice (x, y) de cada robot
✅ C++ ejecuta todas las decisiones (shoot/pass)
✅ Nivel de integración: BAJO
⚠️ Restricciones: Mantener validaciones C++ intactas

### Para Fase 2 (RL influencia en decisiones)
✅ RL ajusta parámetros (shootThreshold, distancias, etc)
✅ C++ usa esos parámetros en PlayBall::smashOrPass
⚠️ Restricciones: MUCHAS (timing, juego, hardware)
🔴 Crítico: Validar hard constraints en cada ciclo

### Para Fase 3 (RL reemplaza SmashOrPass)
✅ RL predice directamente: {shoot, pass_to_X, dribble}
✅ C++ valida y ejecuta
❌ Muchas restricciones rompen
🔴 Muy complejo - probablemente no viable en Fase 2

**Recomendación:** Mantener Fase 2 conservador. Implementar como:
- RL ajusta 3-4 parámetros clave (shootThreshold, distancias)
- C++ valida restricciones hard
- Incrementar complejidad con paciencia

