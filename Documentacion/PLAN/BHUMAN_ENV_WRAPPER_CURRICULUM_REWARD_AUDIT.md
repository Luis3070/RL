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

Ademas de lo ya listado en la seccion 9, el proyecto debe:

- agregar un loader real del contrato parseado desde BHuman
- introducir una representacion explicita de `handoff_ready` y `takeover_active`
- modelar `shoot`, `pass`, `dribble` y `duel` como macro-acciones de BHuman, no como fisica local
- incorporar gating por estado de juego
- validar presupuesto temporal de inferencia
- alinear observaciones cooperativas con `TeamData` y `team ball`
- reemplazar por completo las metricas legacy de avance del curriculum

### 12.4 Criterio Final de Exito del Proyecto Python

Este proyecto solo puede considerarse preparado para transplante cuando:

- entrena sobre campo, tacticas y roles reales de 4v4 Scaled
- el RL aprende a maximizar al ejecutor BHuman en lugar de reemplazarlo
- el takeover existe como contrato explicito del env
- la reward mide calidad de handoff y exito de ejecucion real
- los stages cooperativos se apoyan en semantica realista de companeros y balon compartido
- la inferencia respeta las restricciones temporales de Cognition

Mientras eso no se cumpla, el stack sigue siendo una base util de experimentacion, pero no una plataforma lista para un transplante exitoso.

---

## 13. Recomendaciones

Antes de entrar a corregir codigo, se recomienda fijar estas decisiones de arquitectura:

### 13.1 Separar tres capas

Debe quedar separada la siguiente cadena:

- `BHUMANPARSER`: extrae y normaliza la fuente de verdad del repo BHuman
- contrato Python: transforma esa verdad en estructuras utilizables por el env
- entorno RL: consume el contrato, entrena y mide

Si estas tres capas se mezclan, el proyecto se vuelve dificil de auditar y facil de romper.

### 13.2 Que significa "arreglar BHUMANPARSER"

Arreglar `BHUMANPARSER` NO significa rehacer el proyecto RL.
Significa:

- corregir cualquier dato falso o ambiguo
- marcar claramente que es `verificado` y que es `propuesto`
- cubrir todos los CFGs, `.cpp` y `.h` realmente importantes para el contrato
- eliminar flags muertos o parametros no vigentes del diseno activo
- incluir `SkillBehaviorControl` como parte del contrato real de ejecucion
- dejar claro que `DemoGoToBallAndKick` no es flujo real de 4v4 match play

### 13.3 Definir primero el contrato de control

Antes de reescribir `environment.py`, hay que congelar estas respuestas:

- que controla RL
- que controla BHuman
- cuando se activa `handoff_ready`
- cuando empieza `takeover_active`
- cuando termina el takeover
- como vuelve el control al RL

### 13.3b Definir `handoff/takeover` con histeresis

La activacion y salida del takeover NO deben depender de un solo umbral duro.

Se recomienda usar histeresis:

- umbral de entrada a `handoff_ready`
- umbral de entrada a `takeover_active`
- umbral de salida de takeover ligeramente distinto
- timeout maximo de macro-accion

Variables recomendadas para la condicion:

- distancia al balon
- error angular del robot hacia el balon o corredor util
- velocidad del robot
- `GameState`
- espacio libre o presion local

Esto evita:

- parpadeo entre control `RL/BHuman`
- activaciones espurias
- salidas prematuras del takeover
- politicas inestables por cambios bruscos de modo

### 13.3c Regla de entrenamiento durante takeover

Si el takeover se activa por logica del sistema y NO por una accion elegida por la policy, la recomendacion es:

- enmascarar la `actor loss` del agente activo mientras BHuman controla
- mantener el critic entrenando normalmente
- mantener a los companeros no activos entrenando normalmente
- permitir una accion dummy en el buffer solo por consistencia de tensores, pero no usarla para `policy loss`

No se recomienda modelar `WAIT_FOR_BHUMAN` como accion aprendible cuando el handoff es automatico.

La razon es:

- evita atribuir credito a acciones sin efecto causal real
- evita que la policy aprenda a "esperar" como habilidad artificial
- reduce ruido en PPO/MAPPO
- mantiene el credito en el estado previo al takeover y en la reorganizacion posterior

`WAIT_FOR_BHUMAN` solo tiene sentido si ceder control fuese una decision explicita del agente, lo cual no es el contrato recomendado en este proyecto.

### 13.3d Granularidad temporal del bridge: esquema hibrido

La recomendacion final para el proyecto es un esquema **hibrido**:

- `env.step()` representa **1 frame de Cognition** de BHuman
- el takeover y las skills de BHuman se observan frame a frame
- por encima de eso, el entorno agrega **macroestados** mientras siga activo `playBall`

Esto permite:

- maxima fidelidad temporal con el robot real
- masking preciso de la `actor loss`
- observabilidad frame a frame de `SkillRequest` y `MotionRequest`
- metricas agregadas por skill completa

El bridge debe devolver el **siguiente estado completo** del mundo en cada step, no un delta parcial.

Ademas, el entorno debe registrar por macroestado:

- `macro_active`
- `macro_type`
- `macro_start_frame`
- `macro_duration_frames`
- `macro_end_reason`
- `macro_success`

Este esquema cierra mejor sim2real que un fast-forward por macro completa, pero conserva la trazabilidad agregada necesaria para reward y auditoria.

### 13.3e Decisiones tecnicas congeladas antes de implementar

Tras la auditoria y la definicion del bridge, estas decisiones deben considerarse cerradas:

- **Autoridad del mundo por step:** Python entrega el estado actual al bridge y BHuman devuelve el siguiente estado completo por frame de Cognition
- **Mecanismo del bridge:** debe elegirse explicitamente entre `pybind11`, proceso embebido o IPC/shared memory; no dejarlo implicito
- **Schema congelado:** nombres, unidades, enums, campos opcionales y campos derivados quedan fijados antes de tocar `environment.py`
- **Ruta de despliegue:** la inferencia final en BHuman debe respetar presupuesto temporal de Cognition y formato de export consistente con deploy

Estas cuatro decisiones no son detalle de implementacion.
Son precondiciones para evitar duplicacion de logica entre Python y C++.

### 13.4 No permitir que el env siga siendo `dribble-only`

Si el env no simula `pass`, `shoot`, `dribble` y `duel` como macro-acciones de BHuman, el agente aprendera una interfaz incompleta.

La consecuencia seria:

- curriculum aparentemente estable
- policy aparentemente competente
- transplante fallido en juego real

### 13.5 Mantener el objetivo correcto

El objetivo del RL no es reemplazar a BHuman.
El objetivo del RL es dejar al ejecutor BHuman en estados donde:

- sus decisiones sean coherentes
- sus macro-acciones tengan alta probabilidad de exito
- la cooperacion emerja alrededor de ese ejecutor fijo

### 13.6 Orden recomendado de correccion

1. corregir `BHUMANPARSER_GUIDE.md`
2. cerrar el contrato Python derivado del parser
3. introducir `handoff/takeover` en el env
4. reescribir reward
5. reescribir wrapper y metrics
6. adaptar `train.py`
7. adaptar `visualizer.py`
8. agregar tests y smoke tests

---

## 14. Smoke Test

Se recomienda agregar un `contract_smoke_test.py` pequeno y automatico.

Su objetivo no es medir performance.
Su objetivo es verificar que el transplante no se rompio en lo esencial.

### 14.1 Que debe verificar

- el parser carga sin errores
- el contrato Python se construye sin hardcodes legacy
- el campo cargado es 4v4 Scaled
- las tacticas cargadas son las reales del escenario
- existe `handoff_ready`
- existe `takeover_active`
- el bridge devuelve el siguiente estado completo por frame de Cognition
- el entorno registra macroestados agregados por takeover
- el env puede crear un episodio sin usar fisica local de `kick/dribble`
- el rollout de una macro-accion BHuman devuelve un resultado valido
- el visualizer puede mostrar la bandera `RL/BHuman`
- el tiempo de inferencia queda dentro del presupuesto objetivo

### 14.2 Criterio de aprobado

El smoke test debe pasar solo si:

- no quedan dependencias criticas en hardcodes viejos
- el takeover existe como estado del sistema
- `reward.py` no depende de `touch/kick/dribble` del env anterior
- `wrapper.py` reporta metricas del contrato
- `train.py` puede arrancar sobre el contrato nuevo sin romperse

### 14.3 Momento recomendado para ejecutarlo

Debe correrse:

- despues de corregir el parser
- despues de introducir takeover
- antes de arrancar entrenamiento largo
- despues de cualquier cambio grande en `environment.py`, `reward.py` o `wrapper.py`

---

## 15. Tabla Semantica-Contractual Auditada

> Objetivo: dejar una matriz unica que conecte:
> `documento -> linea documental -> estado semantico -> ubicacion actual en el proyecto Python`.
>
> **Lectura de la tabla:**
> - `CERRADO`: ya hay definicion suficiente en la documentacion
> - `PARCIAL`: la intencion esta definida pero falta congelar detalle operativo
> - `ABIERTO`: falta una definicion formal o un fallback/criterio exacto

| Punto | Estado | Documento principal | Linea doc | Soporte documental | Linea doc | Proyecto actual | Linea codigo | Diagnostico |
|------|--------|---------------------|-----------|--------------------|-----------|-----------------|-------------|-------------|
| Parser -> contract loader -> env -> bridge separados | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 10 | `RL_TRANSPLANT_GUIDE_COMPLETE.md` | 163 | `environment.py` | 588 | La separacion esta documentada; el proyecto actual aun no implementa esa cadena y solo la menciona en comentario |
| `next_full_state` por frame de Cognition | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1093 | `RL_TRANSPLANT_GUIDE_COMPLETE.md` | 163 | `environment.py` | 537 | La doc congela step por frame y estado completo; el env actual construye obs locales sin bridge BHuman real |
| `handoff_ready` oracle-driven | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1024 | `BHUMAN_RL_CURRICULUM_CONTRACT.md` | 77 | `environment.py` | 1231 | La doc ya lo fija; el proyecto actual sigue usando proximidad local al balon como disparador implicito de interaccion |
| `takeover_active` evidence-driven con histeresis y timeout 90 | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1038 | `BHUMANPARSER_GUIDE.md` | 1065 | `environment.py` | 1261 | La doc ahora lo congela; el proyecto actual no tiene takeover explicito y conserva kick/dribble locales |
| Schema exacto del oracle por frame | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1128 | `RL_TRANSPLANT_GUIDE_COMPLETE.md` | 165 | `environment.py` | 537 | El schema esta congelado; el env actual no consume ese contrato |
| Unidades `mm`, `mm/s`, `rad`, `rad/s`, `ms`, `frames` | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1170 | `MAPPO_Auditoria.md` | 340 | `config.py` | 69 | La documentacion es consistente; el proyecto actual aun mezcla thresholds legacy de interaccion local sobre esas unidades |
| Semantica `active_agent`, `role`, `position`, `playBall` | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1254 | `MAPPO_Auditoria.md` | 123 | `environment.py` | 628 | La doc ya separa `position` de `playBall`; el proyecto actual marca teammate activo con una heuristica de rol `forward` |
| Fuente real de teammates activos | `CERRADO` | `MAPPO_Auditoria.md` | 147 | `MAPPO_Plan_3Fases.md` | 463 | `environment.py` | 628 | En deployment debe venir de `TeamData`; el proyecto actual no respeta esa fuente |
| Gating por `GameState`, `HandleGameState`, set plays | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 995 | `CFG_UNIFIED_ANALYSIS.md` | 149 | `environment.py` | 772 | La precedencia documental ya existe; el proyecto actual fija `t4` y no implementa gating real de partido |
| Tacticas validas 4v4 scaled | `CERRADO` | `CFG_UNIFIED_ANALYSIS.md` | 817 | `BHUMANPARSER_GUIDE.md` | 3144 | `config.py` | 251 | La doc cierra que solo valen `t121_4v4_scaled` y `t13_4v4_scaled`; el proyecto actual usa `TACTIC_NAMES` legacy |
| Criterios de exito / fallo / timeout / aborto del takeover | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1065 | `BHUMANPARSER_GUIDE.md` | 1335 | `reward.py` | 344 | Ya quedaron formalizados en doc; el proyecto actual sigue premiando interacciones del env viejo |
| Trazabilidad de campos parseados -> uso RL | `PARCIAL` | `BHUMANPARSER_GUIDE.md` | 958 | `BHUMANPARSER_GUIDE.md` | 1389 | `config.py` | 152 | El contrato exige trazabilidad, pero el proyecto actual aun solo conserva metadatos simplificados `ball_age` y `using_team_ball` |
| Observabilidad minima para reward y metrics | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1338 | `BHUMAN_RL_CURRICULUM_CONTRACT.md` | 99 | `wrapper.py` | 72 | La doc ya define eventos y macroestado; el wrapper actual sigue midiendo `touch_rate/goal_rate/win_rate` |
| Presupuesto temporal inferencia + bridge | `CERRADO` | `PROJECT_AUDIT_FOR_RL.md` | 544 | `RL_TRANSPLANT_GUIDE_COMPLETE.md` | 167 | `train.py` | 1263 | El presupuesto temporal esta claro; el proyecto actual no valida bridge ni inferencia dentro de presupuesto |
| Fallback parser / bridge / takeover | `CERRADO` | `BHUMANPARSER_GUIDE.md` | 1404 | `BHUMANPARSER_GUIDE.md` | 2811 | `environment.py` | 195 | La doc ya distingue los 3 casos; el proyecto actual solo tiene fallbacks locales de simulacion, no contractuales |
| Ausencia de fisica local inventada para `kick/dribble` | `CERRADO` | `BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md` | 98 | `BHUMAN_RL_CURRICULUM_CONTRACT.md` | 21 | `environment.py` | 1359 | La auditoria lo declara bloqueante; el proyecto actual aun usa `ball_kick_range` y `ball_dribble_range` como fuente de verdad |
| Metricas BHuman-aware en wrapper/train | `CERRADO` | `BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md` | 173 | `BHUMAN_RL_CURRICULUM_CONTRACT.md` | 90 | `wrapper.py` | 72 | La documentacion ya define takeover-aware metrics; el proyecto actual sigue en metricas legacy |
| Smoke test contractual | `PARCIAL` | `BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md` | 745 | `RL_TRANSPLANT_GUIDE_COMPLETE.md` | 1693 | `condaTester.py` | 1 | La idea y checklist ya existen; aun falta un smoke contractual unico y pequeno en el repo |

### 15.1 Lectura ejecutiva de la matriz

- Lo semantico importante del bridge y takeover ya no esta abierto: quedo **cerrado en documentacion**.
- Lo que sigue faltando esta del lado del **proyecto Python actual**, no del contrato documental.
- Los dos huecos que aun merecen seguimiento son:
  - trazabilidad ejecutable `campo parseado -> uso real`
  - smoke contractual unico del repo

### 15.2 Criterio de cierre documental

La documentacion puede considerarse suficientemente cerrada para implementacion si se mantiene este criterio:

- `BHUMANPARSER_GUIDE.md` queda como fuente canonica del bridge, schema, takeover, histeresis y fallback
- `BHUMAN_ENV_WRAPPER_CURRICULUM_REWARD_AUDIT.md` queda como fuente canonica de auditoria del proyecto Python y de la matriz semantico-contractual
- cualquier cambio futuro al contrato debe reflejarse primero en estas dos tablas antes de tocar `environment.py`, `reward.py`, `wrapper.py` o `train.py`
