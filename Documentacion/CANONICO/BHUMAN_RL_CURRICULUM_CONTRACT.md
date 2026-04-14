# Curriculum RL Alineado al Contrato de BHuman

> Fecha: 22 de marzo de 2026
> Objetivo: entrenar agentes MARL para que maximicen el rendimiento del ejecutor real de BHuman, en lugar de competir contra el.

---

## 1. Contrato No Negociable

### 1.1 Que controla BHuman

Durante este curriculum, BHuman conserva la ejecucion fisica del juego con balon:

- `SkillBehaviorControl` gatea y enruta la ejecucion real
- `PlayBall` skill layer selecciona la skill ejecutora segun `SkillRequest`
- `KickAtGoal` resuelve disparo y puede caer a `DribbleToGoal`
- `PassToTeammate` resuelve pase
- `GoToBallAndDribble` y `Dribble` resuelven el dribble fisico
- `Zweikampf` resuelve duelo bajo presion

### 1.2 Que NO debe controlar el RL en este curriculum

Hasta terminar Stage 12, el RL NO debe emitir:

- tipo de kick
- parametros de `KickAtGoal`
- parametros de `PassToTeammate`
- parametros de `Zweikampf`
- `MotionRequest::dribble`
- comandos motores o de joints

El RL aprende a producir estados y posicionamientos que sean faciles de explotar por BHuman.

### 1.3 Que si aprende el RL

El RL aprende:

- approach al balon
- orientacion previa a takeover
- posicionamiento ofensivo y defensivo
- spacing con companeros
- evitar bloquear lineas de pase o de dribble
- elegir poses que aumenten la probabilidad de exito del ejecutor BHuman

---

## 2. Handoff BHuman

### 2.1 Idea central

El environment debe tener dos modos:

1. `RL-navigation mode`
2. `BHuman-takeover mode`

Mientras el agente aun no deja una pose buena para el ejecutor, actua RL.
Cuando la pose ya es "executor-friendly", el control local del balon pasa a BHuman.

### 2.2 Envelope de takeover recomendado

Esto es una recomendacion de disenio inferida del flujo real de `PlayBall`, `GoToBallAndDribble`, `DribbleToGoal` y `Zweikampf`, no una ley exacta del sistema:

- balon visto recientemente
- balon delante del robot
- error lateral pequeno respecto al balon
- orientacion razonable hacia la direccion objetivo
- sin colision inmediata con borde u obstaculo

Envelope conservador recomendado para curriculum:

- `ball_seen_age <= 300 ms` en stages con presion
- `ball_rel_x in [0, 500] mm`
- `abs(ball_rel_y) <= 300 mm`
- `abs(angle_to_ball) <= 80 deg`
- margen minimo a borde de campo

### 2.3 Semantica del environment

```text
si handoff_ready == false:
    aplicar accion RL de navegacion/posicionamiento

si handoff_ready == true:
    congelar decision local del agente por una ventana corta
    dejar que BHuman seleccione y ejecute:
        shoot / pass / dribble / duel
    recompensar al agente por la calidad del takeover y por el resultado aguas abajo
```

### 2.4 Implicacion clave

La reward NO debe premiar "yo toque el balon".
Debe premiar "deje al ejecutor BHuman en una situacion de alta calidad".

---

## 3. Espacios Base

### 3.1 Observaciones base por agente

Para todos los stages:

- `self_pose`
- `self_velocity`
- `ball_rel`
- `ball_global`
- `ball_seen_age`
- `heading_to_ball`
- `heading_to_opponent_goal`
- `distance_to_field_border`
- `last_bhuman_takeover_success`
- `nearest_obstacle_features`

Desde stages cooperativos:

- `teammate_i_rel_pose`
- `teammate_i_rel_velocity`
- `opponent_j_rel_pose`
- `opponent_j_rel_velocity`
- `active_agent_flag`
- `team_spread`
- `open_lane_features`
- `support_slot_features`

### 3.2 Acciones base

#### Stages 0-4

Accion continua de approach:

- `target_x_rel_ball`
- `target_y_rel_ball`
- `target_heading`
- `speed_scale`

El agente no decide kick, pass ni dribble. Solo decide como llegar y con que orientacion entregar el estado al ejecutor.

#### Stages 5-12

Accion continua de posicionamiento:

- `target_x_field`
- `target_y_field`
- `target_heading`
- `support_aggression`

Interpretacion:

- agente activo: pose de approach/handoff al balon
- agentes soporte: pose de apoyo para abrir linea, cubrir rebote o dar opcion de pase

### 3.3 Reward base

```text
R =
  + w_progress        * progreso hacia pose de takeover
  + w_ready           * tiempo dentro de handoff envelope
  + w_takeover        * takeover BHuman exitoso
  + w_exec_outcome    * progreso del balon / pase viable / disparo viable tras takeover
  + w_support         * spacing y lineas de apoyo
  - w_collision       * choques
  - w_out_of_bounds   * salir de zona valida
  - w_oscillation     * dithering / zig-zag inutil
  - w_blocking        * bloquear linea de tiro/pase/dribble de BHuman
```

Pesos recomendados:

- early stages: dominar `w_progress`, `w_ready`, `w_takeover`
- middle stages: subir `w_exec_outcome`
- team stages: subir `w_support` y `w_blocking`

---

## 4. Curriculum por Stages

## Stage 0

- Escenario: balon a radio <= 800 mm, sin obstaculos
- Obs: observaciones base individuales
- Action: `target_x_rel_ball`, `target_y_rel_ball`, `target_heading`, `speed_scale`
- Reward: progreso al balon, entrar en handoff envelope, takeover BHuman exitoso, penalizacion por oscilacion
- Gate de promocion: >= 95% de episodios con takeover limpio y sin colision

## Stage 1

- Escenario: balon a radio <= 1600 mm, sin obstaculos
- Obs: igual a Stage 0
- Action: igual a Stage 0
- Reward: igual a Stage 0, pero con mayor peso en eficiencia temporal
- Gate de promocion: >= 90% de episodios con takeover antes de timeout

## Stage 2

- Escenario: balon a radio <= 3200 mm, obstaculos estaticos
- Obs: observaciones base + features de obstaculos y borde
- Action: igual a Stage 1
- Reward: progreso, clearance a obstaculos, takeover limpio, penalizacion fuerte por colision o cierre de camino a BHuman
- Gate de promocion: >= 85% de takeovers limpios con obstaculos

## Stage 3

- Escenario: balon en cualquier parte del campo, obstaculos dinamicos
- Obs: base + obstaculos dinamicos + velocidad relativa de obstaculos
- Action: igual a Stage 2
- Reward: takeover limpio, robustez ante trayectorias curvas/rectas, evitar cierres de corredor, estabilidad temporal
- Gate de promocion: win-rate positivo contra obstaculos dinamicos scripted y >= 80% de takeovers limpios

## Stage 4

- Escenario: 1v1 self-play con ejecutor BHuman congelado
- Obs: base + un oponente + features de control territorial
- Action: igual a Stage 3
- Reward: takeover limpio, llegar antes al balon, progresar el balon tras takeover, no exponerse a duelos malos
- Gate de promocion: win-rate > 55% contra snapshots anteriores + takeover estable

## Stage 5

- Escenario: 2v0 cooperacion
- Obs: base + un companero + active-agent flag
- Action: agente activo usa approach; agente soporte usa target pose de apoyo
- Reward: takeover del agente activo, spacing, carriles de pase abiertos, no bloquear al ejecutor
- Gate de promocion: >= 85% de episodios con soporte util y takeover limpio

## Stage 6

- Escenario: 2v2 self-play
- Obs: base + companero + 2 oponentes + open lanes
- Action: igual a Stage 5
- Reward: takeover limpio, spacing, presion, soporte, recuperacion tras perdida
- Gate de promocion: win-rate > 52% contra pool de oponentes + mejora sostenida en takeover success

## Stage 7

- Escenario: 3v0 cooperacion
- Obs: base + 2 companeros + team spread
- Action: approach para activo, support pose para no activos
- Reward: triangulacion, amplitud, linea de pase, takeover limpio del agente activo
- Gate de promocion: >= 85% de episodios con estructura ofensiva util

## Stage 8

- Escenario: 3v2
- Obs: base + 2 companeros + 2 oponentes
- Action: igual a Stage 7
- Reward: takeover bajo presion moderada, apoyo al poseedor, cobertura de rebote y contraataque
- Gate de promocion: ventaja territorial positiva + takeover success >= 75%

## Stage 9

- Escenario: 3v3 self-play
- Obs: base + 2 companeros + 3 oponentes + lanes
- Action: igual a Stage 8
- Reward: coordinacion completa 3v3, presion, cobertura, no bloquear decisiones de BHuman
- Gate de promocion: win-rate > 52% contra mezcla de snapshots y scripted teams

## Stage 10

- Escenario: 4v0 cooperacion
- Obs: base + 3 companeros + geometricas de estructura 4v4
- Action: igual a Stage 9
- Reward: amplitud, profundidad, soporte escalonado, takeover limpio del agente activo
- Gate de promocion: >= 85% de episodios con ocupacion correcta de carriles

## Stage 11

- Escenario: 4v3
- Obs: base + 3 companeros + 3 oponentes
- Action: igual a Stage 10
- Reward: takeover bajo presion, soporte realista de 4v4, balance ataque/seguridad
- Gate de promocion: ventaja territorial y de ocasiones positiva, takeover success >= 70%

## Stage 12

- Escenario: self-play 4v4 completo
- Obs: observacion completa multiagente
- Action: posicionamiento completo por agente, manteniendo BHuman como ejecutor de balon
- Reward: goal differential, takeover quality, territorial control, support quality, recovery, anti-blocking
- Gate de cierre: estabilidad contra snapshots, scripted baselines y seeds distintos

---

## 5. Que Cambiaria a Tu Curriculum

Tu curriculum base es bueno. La progresion por distancia, obstaculos y numero de agentes tiene sentido. Yo cambiaria esto:

1. Pondria criterios de promocion obligatorios por stage.
2. Agregaria sub-stages intermedios `2v1`, `3v1` y `4v2`.
3. Mantendria congelada toda decision `shoot/pass/dribble` del RL hasta terminar Stage 12.
4. Haria replay de stages viejos cada vez que se avance para evitar olvido catastrofico.
5. Usaria pool de snapshots en self-play, no solo el ultimo rival.

Version recomendada:

- `2v0 -> 2v1 -> 2v2`
- `3v0 -> 3v1 -> 3v2 -> 3v3`
- `4v0 -> 4v2 -> 4v3 -> 4v4`

---

## 6. Por Que Este Curriculum Maximiza BHuman

Porque entrena al agente a:

- llegar al balon con una pose que BHuman puede explotar
- no cerrar lineas de kick, pass o dribble
- dejar al ejecutor local en estados de baja incertidumbre
- aprender soporte y spacing en torno a la logica real de partido

No entrena al agente a reemplazar el ejecutor.
Lo entrena a ser el mejor "proveedor de estados" posible para el ejecutor real.

---

## 7. Lo Que NO Se Puede Garantizar

Ningun curriculum garantiza convergencia matematica o buen juego por si solo.

Lo que si se puede garantizar con este diseno es:

- preservacion del contrato con BHuman
- estabilidad mayor que si el RL intenta controlar el balon directamente
- mejor probabilidad de transferencia sim2real
- menor riesgo de que el policy aprenda hacks incompatibles con el NAO real

Para acercarse a convergencia estable hacen falta ademas:

- promotion gates por stage
- normalizacion de rewards
- replays de regresion
- evaluacion con seeds distintos
- self-play con population/snapshots

---

## 8. Recomendacion Final

Si el objetivo principal es transplante seguro a BHuman:

- usar este curriculum con BHuman executor congelado
- no aprender `shoot/pass/dribble` todavia
- tratar el takeover BHuman como el evento central del reward

Cuando el 4v4 sea estable, ahi si abrir una rama separada para Fase 2/Fase 3.
