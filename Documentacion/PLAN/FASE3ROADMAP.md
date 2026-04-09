# FASE3ROADMAP — Hoja de Ruta Operativa para una Cognición RL Completa

> **Objetivo de este documento**
>
> Definir una ruta realista para llevar el proyecto desde el estado actual
> "RL posiciona y BHuman decide la cognición principal" hacia un estado futuro
> "RL decide la mayor parte de la cognición futbolística y BHuman ejecuta,
> valida y protege".
>
> Este roadmap NO plantea un salto único. Plantea un **curriculum de cognición**
> con trasplantes coordinados entre:
>
> - el stack Python de entrenamiento/simulación/telemetría
> - el stack BHuman de decisión táctica y ejecución real

---

## 1. Tesis central

La Fase 3 no debe entenderse como:

- "reemplazar BHuman de una vez"

La Fase 3 debe entenderse como:

- "migrar progresivamente funciones cognitivas de BHuman hacia RL"
- "conservar en BHuman la ejecución física, la validación y los guard rails"
- "abrir nuevas cabezas de decisión RL solo cuando el contrato operativo y la métrica estén cerrados"

En otras palabras:

- **BHuman actual** es el cerebro táctico principal.
- **Fase 3** busca que ese cerebro pase a ser **RL**, pero de manera incremental.

---

## 2. Qué significa "toda la cognición"

En el contexto de este proyecto, "toda la cognición" significa que RL termina
decidiendo explícitamente:

- `x, y, theta` objetivo
- rol funcional
- intención táctica local o global
- selección de macro-acción
- parámetros de la macro-acción
- receptor de pase o target de tiro
- prioridad de presión / cobertura / apoyo
- eventualmente tipo de kick o variante de skill

Mientras tanto, BHuman conserva:

- cinemática y control motor
- skill execution real
- constraints físicos
- constraints reglamentarios y de safety
- aborts y fallbacks
- interfaces con percepción, team comm y timing de ciclo

La meta final no es "RL hace todo". La meta final es:

- **RL decide**
- **BHuman ejecuta y protege**

---

## 3. Principio arquitectónico

La Fase 3 debe implementarse como **policy jerárquica y curriculum jerárquico**.

No se recomienda una sola policy plana que aprenda simultáneamente:

- locomoción
- control de balón
- roles
- táctica
- selección de skill
- parámetros de pase/tiro
- coordinación multiagente

Eso haría muy difícil:

- el credit assignment
- la depuración
- el análisis de cobertura
- el aislamiento de regresiones
- el trasplante progresivo a BHuman

La política recomendada es multi-head:

1. `pose head`
- `target_x`
- `target_y`
- `target_theta`

2. `macro head`
- `walk`
- `dribble`
- `shoot`
- `pass`
- `hold`
- `block`

3. `argument head`
- `pass_target`
- `shoot_target`
- `kick_type` en etapas tardías

4. `role / intention head`
- primero intención local
- luego rol estable

5. `tactic head`
- solo cuando la coordinación local ya funcione

---

## 4. Estado actual del proyecto

El proyecto actual ya avanzó bastante en la base necesaria para esta transición:

- `environment.py` ya modela mejor el takeover y la evidencia operacional
- `bhuman_sidecar.py` ya aproxima mejor la decisión BHuman y la telemetría asociada
- `wrapper.py` y `train.py` ya separan mejor métricas de takeover por tipo
- `visualizer.py` ya puede mostrar cobertura y éxito por familias de macro
- `takeover_trace.py` ya permite auditar episodios de takeover y éxito

Sin embargo, todavía NO conviene abrir toda la cognición de una vez porque:

- `shoot` y `pass` todavía no tienen la misma madurez operativa que `dribble/duel`
- la coordinación multiagente real todavía debe consolidarse mejor
- roles y táctica aún están más cerca de BHuman que de una policy propia

Por eso, la Fase 3 debe dividirse en subfases.

---

## 5. Visión general de subfases

La Fase 3 propuesta se divide en:

1. `Fase 3A`
- locomoción cognitiva y dribble simple

2. `Fase 3B`
- interacción real con balón
- unlock progresivo de `shoot`

3. `Fase 3C`
- `pass` real en contexto multiagente
- coordinación local

4. `Fase 3D`
- roles emergentes o semi-supervisados

5. `Fase 3E`
- táctica de equipo
- sustitución de la lógica táctica principal de BHuman

---

## 6. Fase 3A — Locomoción cognitiva y dribble simple

### 6.1 Objetivo

Que RL aprenda primero la relación entre:

- su cuerpo
- el balón
- la orientación
- el progreso espacial útil

Antes de pedirle decisiones complejas de pase, tiro o táctica.

### 6.2 Qué decide RL

- `target_x`
- `target_y`
- `target_theta`
- macro restringida a:
  - `walk`
  - `dribble`

### 6.3 Qué conserva BHuman

- selección fina de skill real
- control físico del dribble
- validación de estado
- fallbacks si la ejecución es inviable

### 6.4 Curriculum recomendado

1. `Stage 3A-0`
- 1 agente
- balón estático
- solo `walk`
- objetivo: llegar al balón con pose útil

2. `Stage 3A-1`
- 1 agente
- `walk + dribble`
- balón estático
- sin oponente

3. `Stage 3A-2`
- 1 agente
- `walk + dribble`
- balón con pequeña velocidad inicial
- objetivo: reenganchar y sostener control

4. `Stage 3A-3`
- 1 agente
- dribble hacia targets simples
- oposición muy ligera o pasiva

### 6.5 Reward principal

- acercamiento útil al balón
- orientación correcta al target
- control sostenido
- progreso del balón
- progreso del balón alineado al target
- penalización por zigzag, oscilación y pérdida inmediata de control

### 6.6 Métricas de gating

- `takeover_success_rate_dribble_duel`
- `takeover_coverage_rate_dribble_duel`
- progreso medio del balón durante dribble
- ratio de finalizaciones `completed` frente a `timeout`
- ratio de `skill_request_cleared` recuperado correctamente

### 6.7 Criterio de paso

Pasar a 3B solo si:

- la cobertura `D+D` es alta y estable
- el éxito `D+D` es estable
- el dribble deja de depender de heurísticas laxas
- el visualizador y los traces muestran progreso espacial coherente

---

## 7. Fase 3B — Interacción real con balón

### 7.1 Objetivo

Expandir la policy desde "moverse bien con el balón" hacia:

- decidir entre avanzar y rematar
- reconocer contexto de tiro
- separar locomoción de interacción ofensiva

### 7.2 Qué decide RL

- `target_x`
- `target_y`
- `target_theta`
- macro:
  - `walk`
  - `dribble`
  - `shoot`

### 7.3 Qué conserva BHuman

- ejecución real del tiro
- selección física de contacto
- timing fino del kick
- constraints de locomoción y estabilidad

### 7.4 Curriculum recomendado

1. `Stage 3B-0`
- 1 agente
- arco libre
- tiro desde posiciones favorables

2. `Stage 3B-1`
- 1 agente
- tiro desde varios ángulos
- ligero desalineamiento inicial

3. `Stage 3B-2`
- 1 agente
- defensor pasivo
- RL debe elegir entre seguir dribble o tirar

4. `Stage 3B-3`
- 1 agente
- oposición ligera activa
- decisión `dribble vs shoot`

### 7.5 Reward principal

- éxito real del macro `shoot`
- valor territorial
- reducción de distancia al gol
- calidad del tiro
- penalización por tiros de baja calidad
- penalización por perder balón antes del intento

### 7.6 Métricas de gating

- `takeover_success_rate_shoot`
- `takeover_coverage_rate_shoot`
- tasa de `kick_event` real
- separación entre `shoot coverage` y `shoot success`
- calidad de finalización

### 7.7 Criterio de paso

Pasar a 3C solo si:

- existe evidencia operacional fuerte de `shoot`
- la cobertura de `shoot` no es anecdótica
- la policy no colapsa eligiendo `shoot` en todo

---

## 8. Fase 3C — Pass real y coordinación local

### 8.1 Objetivo

Introducir por primera vez cognición multiagente explícita:

- receptores válidos
- spacing local
- apoyo
- recepción
- decisión `pass vs dribble vs shoot`

### 8.2 Regla fundamental

`pass` solo se habilita en contextos multiagente reales.

No debe entrenarse `pass` como habilidad monolítica en escenarios donde no existe:

- receptor real
- línea de pase válida
- contexto cooperativo

### 8.3 Qué decide RL

- `target_x`
- `target_y`
- `target_theta`
- macro:
  - `walk`
  - `dribble`
  - `shoot`
  - `pass`
- `pass_target`

### 8.4 Qué conserva BHuman

- ejecución real del pase
- tipo de contacto/kick
- verificación de target factible
- safety y aborts

### 8.5 Curriculum recomendado

1. `Stage 3C-0`
- 2 agentes propios
- escenarios de pase simples
- receptor abierto

2. `Stage 3C-1`
- 2 agentes propios
- varios receptores posibles
- masking estricto de receptor válido

3. `Stage 3C-2`
- 3 agentes propios
- apoyo y reposicionamiento local
- oponente pasivo

4. `Stage 3C-3`
- 3 o 4 agentes propios
- oposición ligera
- decisión `dribble/shoot/pass`

### 8.6 Reward principal

- éxito real de `pass`
- recepción utilizable
- continuidad de posesión
- mejora del estado ofensivo después del pase
- penalización por pase al receptor incorrecto
- penalización por pase forzado o bloqueado

### 8.7 Métricas de gating

- `takeover_coverage_rate_pass`
- `takeover_success_rate_pass`
- tasa de receptor válido
- continuidad de posesión después del pase
- spacing del equipo

### 8.8 Criterio de paso

Pasar a 3D solo si:

- `pass` deja de ser anecdótico
- existe telemetría BHuman fuerte de escenarios de pase
- la elección de receptor es estable y con masking correcto

---

## 9. Fase 3D — Roles emergentes o semi-supervisados

### 9.1 Objetivo

Hacer que RL deje de actuar como varios agentes "idénticos que reaccionan" y
empiece a organizar funcionalmente al equipo.

### 9.2 Qué decide RL

- pose
- macro
- intención local
- rol

### 9.3 Recomendación

No abrir de entrada roles finos tipo:

- `defenderL`
- `defenderR`
- `forwardL`
- `forwardR`

Es mejor empezar por roles funcionales:

- `holder`
- `supporter`
- `cover`
- `receiver`

Luego mapearlos a roles estables del sistema.

### 9.4 Curriculum recomendado

1. `Stage 3D-0`
- 3 agentes
- intención local
- no colisionar ni duplicar función

2. `Stage 3D-1`
- 4 agentes
- cobertura y apoyo
- mantenimiento de triángulos útiles

3. `Stage 3D-2`
- 4 agentes
- RL ya emite rol funcional
- BHuman todavía conserva compatibilidad con roles clásicos

4. `Stage 3D-3`
- 4v4
- rol funcional estable por contexto

### 9.5 Reward principal

- spacing útil
- cobertura defensiva
- líneas de pase
- evitar duplicidad de función
- ocupación de carriles

### 9.6 Métricas de gating

- dispersión útil del equipo
- frecuencia de colisión o superposición
- consistencia de rol por ventana temporal
- mejora de posesión y continuidad ofensiva

### 9.7 Criterio de paso

Pasar a 3E solo si:

- los roles aportan orden y no caos
- la coordinación local ya funciona sin depender de heurística fuerte
- el equipo ya no necesita que BHuman reasigne roles como red de seguridad primaria

---

## 10. Fase 3E — Táctica de equipo

### 10.1 Objetivo

Mover la decisión táctica principal desde BHuman hacia RL.

### 10.2 Qué decide RL

- pose
- macro
- receptor / target
- rol
- táctica
- agente activo implícito o explícito

### 10.3 Qué conserva BHuman

- ejecución
- validación
- safeguards
- integración con game state y reglamento

### 10.4 Intenciones tácticas sugeridas

- `build_up`
- `direct_attack`
- `switch_play`
- `press`
- `recover`
- `stabilize_possession`

### 10.5 Curriculum recomendado

1. `Stage 3E-0`
- táctica como variable observada o asistida
- RL todavía no la gobierna totalmente

2. `Stage 3E-1`
- RL propone táctica
- BHuman puede rechazar o degradar

3. `Stage 3E-2`
- RL gobierna táctica normal play
- BHuman conserva set plays y safety

4. `Stage 3E-3`
- self-play completo
- evaluación ELO
- validación robusta sim2real

### 10.6 Reward principal

- valor de posesión
- progresión estructurada
- recuperación tras pérdida
- balance ataque/defensa
- calidad colectiva de decisión

### 10.7 Criterio de cierre

La Fase 3 solo puede declararse cerrada si:

- RL ya decide la cognición principal
- BHuman ya no es la fuente principal de rol/táctica
- el sistema mantiene seguridad y robustez operacional

---

## 11. Orquestación de trasplantes — idea general

La Fase 3 requiere dos trasplantes paralelos:

1. `Trasplante Python`
- cambiar el contrato de entrenamiento y el espacio de acción

2. `Trasplante BHuman`
- cambiar quién decide y cómo se inyectan esas decisiones en runtime

La regla de oro es:

- **primero** se abre el contrato en Python
- **luego** se observa y valida
- **después** se trasplanta esa decisión a BHuman
- **finalmente** se mantiene fallback por fase

Nunca al revés.

---

## 12. Trasplante Python — qué cambia por capas

### 12.1 Archivos núcleo a intervenir

- `RL/environment.py`
- `RL/wrapper.py`
- `RL/train.py`
- `RL/visualizer.py`
- `RL/bhuman_sidecar.py`
- `RL/takeover_trace.py`
- `RL/config.py`

### 12.2 Responsabilidad de cada uno

#### `environment.py`

Debe evolucionar de:

- "RL emite target continuo y takeover usa macro derivada"

A:

- "RL emite pose + macro + argumentos"
- "el env separa claramente coverage, readiness, success y resultado por tipo"
- "cada subfase habilita o bloquea heads"

Debe absorber:

- masking por macro
- masking por `pass_target`
- contractos explícitos de observación por subfase
- reward shaping específico por capacidad
- episodic accounting por macro y por rol

#### `wrapper.py`

Debe evolucionar de:

- métricas agregadas de takeover

A:

- métricas de curriculum cognitivo por capacidad
- gating por subhead
- criterios de avance por subfase

Debe mantener:

- métricas separadas de `D+D`, `shoot`, `pass`
- y luego agregar:
  - cobertura de rol
  - consistencia táctica
  - coordinación local

#### `train.py`

Debe evolucionar de:

- actor continuo simple

A:

- actor multi-head
- losses por head
- logging por familia de outputs
- curriculum por capacidades

Debe incorporar:

- sampling condicionado por fase
- disabled heads por stage
- métricas de entropía por head
- plots por subcapacidad

#### `bhuman_sidecar.py`

Debe dejar de ser solo:

- observador y espejo de decisión BHuman

Y pasar a ser también:

- validador semántico del contrato de trasplante
- oracle temporal mientras RL absorbe funciones

Debe servir para:

- comparar RL vs BHuman
- etiquetar dataset para behavioral cloning
- validar cuándo una capacidad ya está lista para trasplante real

#### `takeover_trace.py`

Debe crecer de:

- trazas de takeover y macro

A:

- trazas por decisión cognitiva
- logs por head
- argumentos de macro
- razón de fallback
- mismatch RL vs BHuman

#### `visualizer.py`

Debe mostrar:

- macro elegida por RL
- macro ejecutada por BHuman
- cobertura por macro
- rol local
- táctica actual
- fallback activo

### 12.3 Orden de trasplante Python

#### Paso P1

Separar el `action space` en heads.

#### Paso P2

Permitir habilitar heads por stage.

#### Paso P3

Separar reward por familia de decisión.

#### Paso P4

Separar métricas por:

- locomoción
- balón
- pass
- rol
- táctica

#### Paso P5

Introducir datasets de imitation / behavioral cloning para:

- `shoot`
- `pass`
- quizá `role / intention`

#### Paso P6

Introducir entrenamiento híbrido:

- BC para bootstrap
- RL para superar el clon

---

## 13. Trasplante BHuman — qué cambia por capas

### 13.1 Archivos BHuman principales a intervenir

- `SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.h`
- `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/StrategyBehaviorControl.h`
- `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.h`
- `SabanaHerons2026/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/PlayBall.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/PassToTeammate.cpp`
- `SabanaHerons2026/Src/Modules/BehaviorControl/FieldRatingProvider/FieldRatingProvider.cpp`

### 13.2 Qué hace cada módulo hoy

#### `Behavior.cpp / Behavior.h`

Hoy concentran buena parte de la cognición táctica clásica:

- `assignPositions`
- `assignRoles`
- `determineActiveAgent`

#### `PlayBall.cpp`

Hoy concentra la decisión de macro ofensiva:

- `smashOrPass`

#### `SkillBehaviorControl/Skills/PlayBall.cpp`

Hace el despacho hacia skills reales:

- `PassToTeammate`
- ejecución de playBall

#### `PassToTeammate.cpp`

Resuelve target y ejecución real del pase.

#### `FieldRatingProvider.cpp`

Evalúa y estructura información táctica relevante para:

- targets
- compañeros
- pase
- valoración espacial

#### `bhuman_bridge.cpp`

Es el punto natural para:

- exportar telemetría
- importar decisiones RL
- comparar BHuman vs RL
- habilitar o deshabilitar trasplantes por subfase

### 13.3 Principio de trasplante BHuman

En BHuman no se debe hacer:

- "borrar Behavior.cpp y ya"

Se debe hacer:

- introducir flags por capacidad
- permitir propuesta RL con fallback BHuman
- instrumentar mismatches
- migrar una decisión a la vez

### 13.4 Orden de trasplante BHuman

#### Paso B1 — Shadow mode

RL decide internamente pero BHuman no obedece todavía.

Se loggea:

- decisión RL
- decisión BHuman
- divergencia
- contexto

Objetivo:

- medir si RL ya coincide o mejora sobre BHuman

#### Paso B2 — Advisory mode

RL propone:

- pose
- macro
- target

BHuman todavía valida y puede degradar.

Objetivo:

- usar RL como recomendador

#### Paso B3 — Controlled ownership

RL se vuelve dueño de una sola capacidad:

- primero pose
- luego macro simple
- luego pass target
- luego rol

BHuman conserva fallback específico por capacidad.

#### Paso B4 — Tactical ownership

RL se vuelve dueño de táctica normal play.

BHuman conserva:

- set plays
- estados reglamentarios
- safety
- emergencia

---

## 14. Mapa explícito de trasplantes por subfase

## 14.1 Fase 3A

### En Python

- `environment.py`
  - habilitar macro head solo con `walk/dribble`
  - reward centrado en control y progreso
- `train.py`
  - actor multi-head mínimo
  - logging por head
- `wrapper.py`
  - gating solo con `D+D`
- `visualizer.py`
  - mostrar macro RL y progreso de dribble

### En BHuman

- `bhuman_bridge.cpp`
  - exportar decisión de locomoción y control
  - shadow mode para pose
- `Behavior.cpp`
  - NO reemplazar roles ni táctica aún
- `PlayBall.cpp`
  - NO reemplazar `smashOrPass` aún

### Trasplante efectivo

- solo pose y dribble simple

## 14.2 Fase 3B

### En Python

- abrir `shoot`
- medir `shoot coverage` y `shoot success`
- añadir BC opcional para macro `shoot`

### En BHuman

- `bhuman_bridge.cpp`
  - exportar evidencia real de kick y target
- `PlayBall.cpp`
  - mantener decisión BHuman como oracle comparativo

### Trasplante efectivo

- RL puede proponer `shoot`
- BHuman ejecuta y valida

## 14.3 Fase 3C

### En Python

- abrir `pass`
- habilitar `pass_target`
- masking multiagente estricto
- datasets de telemetría real de pase

### En BHuman

- `bhuman_bridge.cpp`
  - exportar `passTarget`, target real, kick elegido y resultado
- `PassToTeammate.cpp`
  - conservar ejecución física real
- `FieldRatingProvider.cpp`
  - usarlo como fuente de validación/telemetría

### Trasplante efectivo

- RL decide si pasar y a quién
- BHuman sigue ejecutando el skill real

## 14.4 Fase 3D

### En Python

- abrir `role head`
- reward por spacing y cobertura
- métricas de consistencia de rol

### En BHuman

- `Behavior.cpp`
  - introducir bypass o inyección parcial sobre `assignRoles`
- `StrategyBehaviorControl.cpp`
  - mantener compatibilidad con roles legacy mientras dura la transición

### Trasplante efectivo

- RL empieza a ser dueño del rol

## 14.5 Fase 3E

### En Python

- abrir `tactic head`
- self-play completo
- evaluación por ELO y estabilidad

### En BHuman

- `Behavior.cpp`
  - bypass progresivo de `assignPositions`, `assignRoles`, `determineActiveAgent`
- `PlayBall.cpp`
  - bypass progresivo de `smashOrPass`
- `StrategyBehaviorControl.cpp`
  - conectar inferencia RL como fuente principal de decisión

### Trasplante efectivo

- RL pasa a ser el origen de la cognición táctica principal

---

## 15. Behavioral cloning dentro de la Fase 3

BC no debe verse como un reemplazo de RL.

Debe verse como:

- bootstrap de decisiones difíciles
- prior sobre decisiones humanas/BHuman ya razonables
- mecanismo para reducir exploración caótica

### Dónde usarlo primero

- `shoot`
- `pass`
- quizá selección inicial de intención local

### Dónde NO usarlo como solución final

- táctica completa del equipo
- coordinación emergente final

### Secuencia recomendada

1. capturar telemetría real BHuman
2. construir dataset etiquetado por contexto
3. preentrenar heads difíciles
4. fine-tuning con RL
5. comparar RL vs BHuman en shadow mode

---

## 16. Qué NO trasplantar demasiado pronto

No conviene trasplantar al inicio:

- `assignRoles` completo
- `determineActiveAgent` completo
- `smashOrPass` completo
- tipo de kick fino
- táctica global completa

Razón:

- el sistema pierde interpretabilidad
- el debugging se vuelve mucho más difícil
- se mezcla error táctico con error de ejecución

---

## 17. Riesgos principales

### 17.1 Riesgo de credit assignment

Si RL falla una jugada, puede haber fallado:

- la pose
- la selección de macro
- el rol
- la táctica
- el receptor
- o la ejecución física posterior

Por eso la jerarquía es obligatoria.

### 17.2 Riesgo de cobertura falsa

Si el sistema sigue mirando solo métricas agregadas:

- parecerá que Fase 3 avanza
- pero en realidad solo avanzó `dribble/duel`

### 17.3 Riesgo de transplante prematuro

Si BHuman entrega ownership completo antes de tiempo:

- el robot puede volverse inestable tácticamente
- el debugging se rompe
- las métricas dejan de ser comparables

### 17.4 Riesgo de BC fuera de distribución

Si el dataset de BHuman no cubre estados raros:

- RL puede imitar bien lo frecuente
- pero degradarse fuerte en estados nuevos

---

## 18. Criterios de go/no-go por subfase

### Go para 3A

- `D+D` estable
- traces coherentes
- pose útil

### Go para 3B

- `shoot coverage` no anecdótico
- evidencia real de kick

### Go para 3C

- `pass` solo en contextos multiagente
- receptor válido y continuidad de posesión

### Go para 3D

- roles agregan orden y no ruido

### Go para 3E

- RL ya domina coordinación local
- fallback BHuman se activa poco y con sentido

---

## 19. Recomendación operativa para el equipo

Si hay dos tracks de trabajo:

### Ingeniero A — Track de entrega y robustez

Responsable de:

- entorno Python
- contrato operacional
- takeover
- métricas
- visualización
- smoke tests

Su misión es dejar el stack entrenable y auditable.

### Ingeniero B — Track de Fase 3

Responsable de:

- actor multi-head
- datasets BC
- curriculum cognitivo
- trasplantes progresivos a BHuman
- shadow mode y advisory mode

Su misión es abrir la cognición propia sin romper la base.

---

## 20. Recomendación final

La Fase 3 sí es viable como visión del proyecto.

Pero debe ejecutarse con esta secuencia:

1. `walk`
2. `dribble`
3. `shoot`
4. `pass`
5. coordinación local
6. roles
7. táctica

Y con esta regla:

- cada capacidad se entrena primero en Python
- luego se audita
- luego entra en shadow mode en BHuman
- luego advisory mode
- luego ownership parcial
- y solo al final ownership principal

Ese es el camino más serio para que RL termine siendo la cognición principal
del equipo sin convertir el proyecto en una caja negra imposible de depurar.

---

## 21. Tabla ejecutiva por subfase

## 21.1 Fase 3A

| Campo | Definición operativa |
| --- | --- |
| Objetivo | Que RL domine pose útil y dribble simple con evidencia operacional robusta |
| Inputs nuevos | Pose propia, balón, target simple, métricas de control de balón |
| Outputs RL | `target_x`, `target_y`, `target_theta`, `macro={walk,dribble}` |
| Reward dominante | Control, facing, progreso de balón, progreso alineado |
| Gating | `takeover_success_rate_dribble_duel`, cobertura `D+D`, progreso medio |
| Python | `environment.py`, `train.py`, `wrapper.py`, `visualizer.py` |
| BHuman | `bhuman_bridge.cpp` en shadow mode, sin reemplazar táctica |
| Riesgo principal | confundir contacto con dribble útil |
| Go/No-Go | solo se abre 3B si el dribble es estable y auditado |

## 21.2 Fase 3B

| Campo | Definición operativa |
| --- | --- |
| Objetivo | Introducir decisión ofensiva local `dribble vs shoot` |
| Inputs nuevos | estado ofensivo, contexto de tiro, target de arco |
| Outputs RL | pose + `macro={walk,dribble,shoot}` |
| Reward dominante | éxito real de shoot, calidad de tiro, castigo por tiro malo |
| Gating | `takeover_success_rate_shoot`, `takeover_coverage_rate_shoot`, kick evidence |
| Python | abrir head de `shoot`, logging y coverage dedicados |
| BHuman | sigue ejecutando el kick y validando timing real |
| Riesgo principal | cobertura baja de `shoot` o evidencia falsa de kick |
| Go/No-Go | no se abre `pass` hasta tener `shoot` con evidencia real robusta |

## 21.3 Fase 3C

| Campo | Definición operativa |
| --- | --- |
| Objetivo | Introducir `pass` real solo en contexto multiagente válido |
| Inputs nuevos | teammates activos, líneas de pase, receptor válido |
| Outputs RL | pose + `macro={walk,dribble,shoot,pass}` + `pass_target` |
| Reward dominante | continuidad de posesión, calidad del pase, mejora del estado ofensivo |
| Gating | `takeover_success_rate_pass`, `takeover_coverage_rate_pass`, receptor válido |
| Python | masking multiagente, telemetría real de pase, BC opcional |
| BHuman | `PassToTeammate.cpp` ejecuta, `bhuman_bridge.cpp` exporta target y resultado |
| Riesgo principal | entrenar `pass` fuera de distribución o sin contexto cooperativo |
| Go/No-Go | no se abre rol hasta que `pass` sea estable y no anecdótico |

## 21.4 Fase 3D

| Campo | Definición operativa |
| --- | --- |
| Objetivo | Organizar al equipo por roles funcionales emergentes |
| Inputs nuevos | contexto multiagente estable, spacing, posesión, apoyo/cobertura |
| Outputs RL | pose + macro + intención local + rol |
| Reward dominante | spacing, cobertura, no duplicidad, líneas de pase |
| Gating | consistencia de rol, colisiones bajas, ocupación útil |
| Python | role head, métricas de consistencia, visualización de rol |
| BHuman | bypass parcial de `assignRoles`, fallback legacy |
| Riesgo principal | ruido de rol sin mejora real colectiva |
| Go/No-Go | no se abre táctica hasta que el rol agregue orden observable |

## 21.5 Fase 3E

| Campo | Definición operativa |
| --- | --- |
| Objetivo | Reemplazar progresivamente la táctica principal de BHuman |
| Inputs nuevos | contexto de juego completo, roles, macro y estado colectivo |
| Outputs RL | pose + macro + target + rol + táctica |
| Reward dominante | valor de posesión, progresión estructurada, balance colectivo |
| Gating | estabilidad en self-play, fallback bajo, mejora sostenida |
| Python | tactic head, self-play, evaluación fuerte |
| BHuman | bypass parcial/total de `assignPositions`, `assignRoles`, `determineActiveAgent`, `smashOrPass` |
| Riesgo principal | pérdida de interpretabilidad y regressions tácticas |
| Go/No-Go | solo se declara cierre cuando RL sea la fuente principal de cognición sin comprometer seguridad |

---

## 22. Validación de factibilidad real del trasplante

Este roadmap no está basado solo en intuición. Tiene anclaje real en el código actual.

### 22.1 Factores que hacen el roadmap lograble

1. El puente BHuman ya exporta `SkillRequest`, `MotionRequest`, `ActivationGraph` y `MotionInfo`.
- Esto existe en `SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp`.

2. El codebase ya soporta semántica explícita de `passTarget`.
- `SkillRequest` ya incluye `passTarget` en:
  - `SabanaHerons2026/Src/Representations/BehaviorControl/SkillRequest.h`
  - `SabanaHerons2026/Src/Representations/BehaviorControl/SkillRequest.cpp`

3. El bridge ya lee/exporta `pass_target_player_number`.
- Esto aparece en `bhuman_bridge.cpp`.

4. El stack BHuman ya tiene skills reales diferenciados para `shoot`, `pass`, `dribble`, `block`.
- `SkillBehaviorControl.cpp`
- `Skills/PlayBall.cpp`
- `Skills/PassToTeammate.cpp`

5. El codebase sí contiene soporte de `CompiledNN` / ONNX.
- Hay evidencia en:
  - `JointAnglePredictor.h`
  - `JointAnglePredictor.cpp`

6. El proyecto Python ya tiene una base operacional para takeover por macro.
- `RL/environment.py`
- `RL/wrapper.py`
- `RL/train.py`
- `RL/visualizer.py`

### 22.2 Factores que obligan a cautela

1. Que una representación exista no significa que ya exista ownership RL sobre ella.

2. `passTarget` existe, pero eso no resuelve por sí mismo:
- masking correcto
- telemetría suficiente
- receptor válido en contexto real

3. `CompiledNN` existe, pero no hay todavía un módulo Fase 3 listo para inferencia táctica multi-head.

4. El bridge hoy está mucho más preparado para observación y auditoría que para ownership táctico pleno.

Conclusión operativa:

- **la integración es realista**
- **el trasplante total es posible**
- **pero solo si se respeta la secuencia incremental**

---

## 23. Criterios de robustez mínimos

La Fase 3 no debe avanzar solo porque el reward subió.

Cada subfase debe pasar estos mínimos:

### 23.1 Robustez operacional

- el env compila y corre sin rutas especiales ocultas
- hay smoke tests positivos y negativos
- el `visualizer` refleja la nueva capacidad
- el `takeover_trace` deja ver por qué hubo éxito o fallo

### 23.2 Robustez semántica

- el nombre de la macro coincide con la ejecución observada
- coverage y success están separados
- la métrica agregada no oculta falta de cobertura en skills avanzados

### 23.3 Robustez de trasplante

- existe shadow mode antes de ownership
- existe advisory mode antes de bypass
- existe fallback explícito por capacidad
- existe mismatch logging entre RL y BHuman

### 23.4 Robustez de entrenamiento

- losses por head disponibles
- entropía por head disponible
- masking correcto en heads discretos
- curriculum que realmente desactive outputs no habilitados

### 23.5 Robustez de deployment

- inferencia cabe en presupuesto temporal
- decisiones inválidas no rompen el comportamiento
- BHuman puede degradar o rechazar la orden RL

---

## 24. Checklist de creación real del proyecto Fase 3

Este checklist define cuándo puede decirse que la Fase 3 dejó de ser idea y se volvió proyecto real.

### 24.1 Para abrir 3A

- [ ] `train.py` soporta actor multi-head mínimo
- [ ] `environment.py` acepta macro head restringido
- [ ] `wrapper.py` gatea por `D+D`
- [ ] `visualizer.py` muestra `D+D`
- [ ] smoke tests `dribble/duel` estables

### 24.2 Para abrir 3B

- [ ] existe `shoot coverage`
- [ ] existe evidencia de kick real
- [ ] la policy no colapsa a `shoot`
- [ ] traces separan `coverage` y `success`

### 24.3 Para abrir 3C

- [ ] existe telemetría real de pase
- [ ] existe fixture/snapshot derivado de telemetría real
- [ ] el masking de `pass_target` es correcto
- [ ] `pass` solo aparece en contextos multiagente válidos

### 24.4 Para abrir 3D

- [ ] existe role head
- [ ] existe métrica de consistencia de rol
- [ ] hay visualización de rol
- [ ] `assignRoles` puede entrar en bypass parcial

### 24.5 Para abrir 3E

- [ ] existe tactic head
- [ ] existe self-play serio
- [ ] `Behavior.cpp` admite ownership parcial por capacidad
- [ ] `smashOrPass` puede compararse contra RL en shadow mode

---

## 25. Recomendación final de implementación realista

La Fase 3 será real, robusta y lograble solo si se acepta este principio:

- **no se trasplanta una abstracción**
- **se trasplanta una capacidad validada**

Eso implica:

1. cada capacidad debe tener:
- contrato de entrada
- output claro
- reward específico
- coverage
- success
- smoke test
- visualización
- fallback

2. cada capacidad debe pasar por:
- Python only
- shadow mode en BHuman
- advisory mode
- ownership parcial
- ownership principal

3. el proyecto no debe vender como "Fase 3 cerrada" algo que siga dependiendo de:
- heurística dominante en BHuman
- coverage marginal de `shoot/pass`
- ausencia de fallback
- ausencia de validación operacional

La versión robusta y alcanzable de la Fase 3 no es:

- "RL reemplaza todo de golpe"

La versión robusta y alcanzable de la Fase 3 es:

- "RL absorbe una función cognitiva a la vez, con contrato, métrica, visualización, smoke test y fallback"

Ese enfoque sí es compatible con la arquitectura real del repo actual y con el código real disponible en BHuman.

---

## 26. Por qué `shadow mode` y `advisory mode`

Estas dos etapas no son burocracia. Son el mecanismo que hace que el trasplante
sea técnicamente controlable.

### 26.1 Por qué `shadow mode`

En `shadow mode`:

- RL decide
- BHuman decide
- pero solo BHuman ejecuta

Esto permite responder preguntas críticas antes de tocar ownership real:

- ¿RL está viendo la misma semántica que BHuman?
- ¿RL coincide con BHuman en estados normales?
- ¿RL mejora o empeora en estados difíciles?
- ¿la policy falla por decisión o por ejecución?
- ¿hay heads que parecen bien en Python pero colapsan en runtime real?

Sin `shadow mode`, el primer error de RL ya cae directamente sobre el robot.
Con `shadow mode`, el sistema todavía es seguro y además produce dataset de:

- divergencia RL vs BHuman
- estados donde RL supera a BHuman
- estados donde RL se desvía de forma peligrosa

En resumen:

- `shadow mode` existe para **medir antes de mandar**

### 26.2 Por qué `advisory mode`

En `advisory mode`:

- RL propone
- BHuman valida
- BHuman puede aceptar, degradar o rechazar

Esto sirve porque entre:

- "RL predice bien offline"

y

- "RL puede gobernar el robot"

hay una zona intermedia donde:

- la inferencia puede llegar tarde
- la acción puede ser semánticamente correcta pero físicamente mala
- la decisión puede ser válida en promedio pero peligrosa en edge cases

`advisory mode` permite:

- introducir ownership sin perder safeguards
- medir tasa de aceptación de propuestas RL
- medir por qué BHuman rechaza o degrada
- descubrir qué parte del contrato todavía no está lista

En resumen:

- `advisory mode` existe para **mandar con red de seguridad**

### 26.3 Diferencia práctica entre ambos

`Shadow mode`:

- RL no manda
- solo observa y se compara

`Advisory mode`:

- RL ya influye
- pero no tiene autoridad total

`Ownership parcial`:

- RL manda sobre una capacidad concreta
- BHuman conserva fallback específico

### 26.4 Por qué son obligatorios en este proyecto

Porque este proyecto no está reemplazando un componente aislado.
Está intentando reemplazar:

- pose
- macro
- pass target
- rol
- táctica

Eso hace que un error de RL pueda parecer:

- un error de locomoción
- un error de pase
- un error de coordinación
- o un error táctico

Si se salta `shadow mode` y `advisory mode`, se pierde la única zona de
diagnóstico donde todavía se puede aislar qué falló.

---

## 27. Backlog técnico por subfase

Este backlog está organizado para que cada subfase produzca:

- código entrenable
- código trasplantable
- métricas auditables
- fallback explícito

## 27.1 Backlog Fase 3A

### Ticket 3A-PY-01 — Separar action space por heads mínimos

- Archivos:
  - `RL/train.py`
  - `RL/environment.py`
  - `RL/config.py`
- Entrega:
  - head continuo `pose`
  - head discreto `macro` con `walk/dribble`
- Criterio de listo:
  - entrenamiento corre con heads deshabilitados/habilitados por stage

### Ticket 3A-PY-02 — Reward y métricas de dribble útiles

- Archivos:
  - `RL/environment.py`
  - `RL/wrapper.py`
- Entrega:
  - reward por control
  - reward por progreso del balón
  - coverage y success `D+D`
- Criterio de listo:
  - `D+D` queda trazable sin depender de métrica agregada

### Ticket 3A-PY-03 — Visualización y traces de dribble

- Archivos:
  - `RL/visualizer.py`
  - `RL/takeover_trace.py`
- Entrega:
  - HUD de macro RL
  - progreso de balón
  - razón de finalización
- Criterio de listo:
  - se puede depurar un fallo de dribble sin inspección manual profunda

### Ticket 3A-BH-01 — Shadow mode de pose y dribble

- Archivos:
  - `SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
- Entrega:
  - exportar decisión RL simulada
  - loggear divergencia contra BHuman
- Criterio de listo:
  - mismatch RL vs BHuman aparece en telemetría

## 27.2 Backlog Fase 3B

### Ticket 3B-PY-01 — Abrir `shoot` en macro head

- Archivos:
  - `RL/train.py`
  - `RL/environment.py`
- Entrega:
  - `macro={walk,dribble,shoot}`
  - masking por stage
- Criterio de listo:
  - `shoot` no existe antes de 3B

### Ticket 3B-PY-02 — Métricas y smoke tests de shoot

- Archivos:
  - `RL/wrapper.py`
  - `RL/train.py`
  - smoke tests relacionados
- Entrega:
  - `shoot coverage`
  - `shoot success`
  - smoke positivo y negativo
- Criterio de listo:
  - coverage y success se separan claramente

### Ticket 3B-BH-01 — Exportar evidencia fuerte de kick

- Archivos:
  - `SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
- Entrega:
  - timestamps de kick
  - target real
  - tipo de kick cuando aplique
- Criterio de listo:
  - `shoot success` puede apoyarse en evidencia fuerte del bridge

## 27.3 Backlog Fase 3C

### Ticket 3C-PY-01 — Abrir `pass_target`

- Archivos:
  - `RL/train.py`
  - `RL/environment.py`
- Entrega:
  - head de `pass_target`
  - masking multiagente
- Criterio de listo:
  - no existen targets inválidos durante entrenamiento

### Ticket 3C-PY-02 — Dataset real de pase

- Archivos:
  - nuevos scripts de captura/export
  - `RL/takeover_trace.py`
  - artefactos de telemetría
- Entrega:
  - snapshots y CSV nacidos de telemetría BHuman real
- Criterio de listo:
  - el fixture de `pass` no es hardcodeado inventado

### Ticket 3C-PY-03 — Bootstrap BC para pass

- Archivos:
  - scripts de entrenamiento o pretraining
  - `RL/train.py`
- Entrega:
  - preentrenamiento opcional para `pass` y `pass_target`
- Criterio de listo:
  - BC puede activarse/desactivarse sin romper PPO/MAPPO

### Ticket 3C-BH-01 — Advisory mode de pass

- Archivos:
  - `SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
  - `Skills/PassToTeammate.cpp`
- Entrega:
  - RL propone `pass_target`
  - BHuman acepta/rechaza/degrada
- Criterio de listo:
  - existe tasa de aceptación de propuestas RL

## 27.4 Backlog Fase 3D

### Ticket 3D-PY-01 — Role head funcional

- Archivos:
  - `RL/train.py`
  - `RL/environment.py`
  - `RL/config.py`
- Entrega:
  - role head inicial
  - roles funcionales en vez de roles finos
- Criterio de listo:
  - el role head puede activarse sin abrir táctica completa

### Ticket 3D-PY-02 — Métricas de consistencia de rol

- Archivos:
  - `RL/wrapper.py`
  - `RL/train.py`
  - `RL/visualizer.py`
- Entrega:
  - estabilidad temporal de rol
  - dispersión/ocupación útil
- Criterio de listo:
  - el equipo puede verse y medirse como sistema organizado

### Ticket 3D-BH-01 — Bypass parcial de `assignRoles`

- Archivos:
  - `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp`
  - `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.h`
- Entrega:
  - modo híbrido donde RL propone rol y BHuman conserva fallback
- Criterio de listo:
  - se puede aislar cuándo el rol RL fue usado o rechazado

## 27.5 Backlog Fase 3E

### Ticket 3E-PY-01 — Tactic head

- Archivos:
  - `RL/train.py`
  - `RL/environment.py`
  - `RL/config.py`
- Entrega:
  - head táctico activable por stage
- Criterio de listo:
  - el tactic head no contamina fases previas

### Ticket 3E-PY-02 — Self-play y evaluación fuerte

- Archivos:
  - `RL/train.py`
  - scripts auxiliares de evaluación
- Entrega:
  - self-play reproducible
  - evaluación comparativa
- Criterio de listo:
  - la política táctica no se evalúa solo contra scripted opponents simples

### Ticket 3E-BH-01 — Advisory mode táctico

- Archivos:
  - `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp`
  - `SabanaHerons2026/Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.cpp`
  - `SabanaHerons2026/Src/Apps/RLBridge/bhuman_bridge.cpp`
- Entrega:
  - RL propone táctica y macro ofensiva
  - BHuman valida y fallback
- Criterio de listo:
  - ya existe ownership parcial sobre táctica normal play

### Ticket 3E-BH-02 — Ownership progresivo de táctica

- Archivos:
  - `Behavior.cpp`
  - `PlayBall.cpp`
  - `StrategyBehaviorControl.cpp`
- Entrega:
  - bypass explícito de:
    - `assignPositions`
    - `assignRoles`
    - `determineActiveAgent`
    - `smashOrPass`
- Criterio de listo:
  - cada bypass se puede activar por flag y revertir sin romper el sistema

---

## 28. Orden recomendado de ejecución del backlog

Para mantener el proyecto realista, robusto y lograble, el orden recomendado es:

1. `3A-PY-01`
2. `3A-PY-02`
3. `3A-PY-03`
4. `3A-BH-01`
5. `3B-PY-01`
6. `3B-PY-02`
7. `3B-BH-01`
8. `3C-PY-01`
9. `3C-PY-02`
10. `3C-PY-03`
11. `3C-BH-01`
12. `3D-PY-01`
13. `3D-PY-02`
14. `3D-BH-01`
15. `3E-PY-01`
16. `3E-PY-02`
17. `3E-BH-01`
18. `3E-BH-02`

La idea del orden es simple:

- primero abrir capacidad
- luego medirla
- luego visualizarla
- luego compararla contra BHuman
- y solo después darle ownership parcial
