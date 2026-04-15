# Artifact: telemetría del entorno y traza takeover (NaoSoccerEnv)

**Versión:** coherente con el código en `RL/environment.py`, `RL/takeover_trace.py`, `RL/reward.py`, `RL/train.py`, `RL/visualizer.py`, `RL/analyze_takeover_run.py`.  
**Objetivo operativo:** dejar el **entorno auditable en ~2 días** para producción (regresiones, postmortems y tuning sin depender del visualizer).

---

## 1. Para qué sirve esta telemetría

| Necesidad | Cómo la cubre |
|-----------|----------------|
| Reproducir “qué pasó” en un step concreto | `takeover_trace.csv` + columnas `diag_*` / `rw_*` |
| Separar bug de **RL**, **handoff**, **BHuman** o **física 2D** | `control_owner`, `diag_control_*`, `diag_ball_collision_tags` |
| Ver si el dolor viene del **reward** o del **bridge** | `rw_*` vs `sidecar_*` / `diag_activation_*` |
| Entrenar o firmar releases con **criterios cuantitativos** | Agregados en `analyze_takeover_run.py` + `metrics.csv` por episodio |

---

## 2. Capas de datos (de abajo arriba)

```
┌─────────────────────────────────────────────────────────────────┐
│  Capa 0 — Ground truth simulación 2D (Python)                    │
│  BallState, AgentState, OpponentState, GameState, obstáculos   │
│  Lectura: arrays pos/vel/theta en NaoSoccerEnv                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Capa 1 — Física / integración por step                         │
│  Movimiento RL, colisiones balón-agente-opp-obstáculo, kicks     │
│  Escritura: _diag_ball_collision_tags (tags de resolución)     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Capa 2 — Contrato BHuman (sidecar)                             │
│  BHumanDecisionSidecar.decide → SidecarOutput                   │
│  Lectura: skill/motion/macro, targets, activation_graph, bridge  │
│  Enriquecimiento: RealSidecarPhysicsExecutor (targets campo)    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Capa 3 — Handoff / takeover (HandoffTakeoverManager)           │
│  handoff_ready, takeover_active, macro_type, duraciones, etc.   │
│  Derivado: control_owner (RL | HANDOFF | BHUMAN)                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Capa 4 — Reward (reward.py)                                    │
│  compute_reward → RewardComponents → rw_* en info                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Capa 5 — Diagnósticos por step (_compute_step_diagnostics)     │
│  Teleports, overlaps, θ, kicks semánticos, histeresis control  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Capa 6 — Info dict por agente (_build_infos)                   │
│  Misma estructura para todos los agent_id; train toma una muestra│
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
  TakeoverTraceLogger.log_step                   visualizer.control_debug
  (primer agente del dict infos)                  (HUD + diag_compact)
```

**Nota importante:** en `train.py`, la traza usa `sample = next(iter(infos.values()))`. En entornos multi-agente, es **un** agente representativo (orden de inserción del dict). Para análisis crítico del activo, el CSV ya incluye `active_agent_id` y posiciones del activo en columnas `agent_*`.

---

## 3. Orden temporal dentro de `NaoSoccerEnv.step` (lecturas relevantes)

Orden simplificado (lo que afecta a la telemetría):

1. Guardar estado al **inicio** del step: `_prev_ball_pos`, `_prev_ball_speed`, `_diag_step_start_active_*` (id, pos, θ del activo al inicio).
2. Control RL → `SidecarOutput` (`decide`).
3. Takeover / handoff / aplicación de kicks (según modo).
4. **`_resolve_ball_collisions()`** → actualiza **`_diag_ball_collision_tags`** (qué contactos se resolvieron).
5. Oponentes, goles, flags `ball_out`, etc.
6. **`_compute_rewards`** → **`_last_reward_components_dict`** (`rw_*` = `asdict(RewardComponents)` con prefijo).
7. Incremento `step_count`, etc.
8. **`_compute_step_diagnostics`** → casi todos los **`diag_*`** por step y contadores episódicos `episode_diag_*`.
9. **`_build_infos`** → mezcla macro, sidecar, diagnósticos, reward, activation resumida, inactivos JSON.

Así, en una fila CSV, **balón y agente** son estado **al final del step** salvo donde se documenta explícitamente (p. ej. salto de balón vs `_prev_ball_pos`).

---

## 4. Variables de entorno y archivos generados

| Variable | Archivo | Efecto |
|----------|---------|--------|
| `TRACE_TAKEOVER_STEPS` | `RL/.env` o entorno OS | `1` (default): habilita logger. `0`: no se crea `takeover_trace.csv`. |
| `TRACE_TAKEOVER_ALL_STEPS` | igual | `1`: **una fila por step** si `sidecar_backend == real` (CSV grande, auditoría completa). |
| `TRACE_TAKEOVER_SAMPLE_EVERY` | igual | Entero `N>0`: además se loguea **1 de cada N** steps con backend real (stages sin handoff donde `should_log` quedaría vacío). |
| (ninguna) | `runs/.../metrics.csv` | Agregados **por episodio** desde el loop de entrenamiento (tasas de handoff, takeover, etc.). |

**Salida principal de telemetría por step:**  
`<log_dir>/takeover_trace.csv` — columnas fijas en `TakeoverTraceLogger.COLUMNS`.

---

## 5. Criterio de escritura en CSV (`should_log`)

Se escribe fila si:

- `sidecar_backend == "real"` **y**
- (`TRACE_TAKEOVER_ALL_STEPS` **o** al menos una de las condiciones “interesantes”: handoff, takeover, macro distinto de `none`, playball, kicks, ball_out, diagnósticos anómalos, colisión etiquetada, zweikampf en activation, etc.)

Con el default **sin** `TRACE_TAKEOVER_ALL_STEPS`, muchos steps “solo RL lejos del contrato” **no** generan fila (ahorra tamaño).

---

## 6. Umbrales numéricos (`environment.py`, prefijo `_DIAG_`)

| Constante | Valor (aprox.) | Uso |
|-----------|----------------|-----|
| `_DIAG_BALL_TELEPORT_JUMP_MM` | 520 | Salto de posición del balón sin kick-like / goal / out → `diag_ball_teleport_flag` |
| `_DIAG_AGENT_TELEPORT_JUMP_MM` | 400 | Salto del activo inicio→fin de step → `diag_agent_teleport_flag` |
| `_DIAG_SHOOT_BACKWARD_DOT_THRESH` | 0.18 | Cosine vel vs target semántico en kick-like → `diag_shoot_backward_flag` |
| `_DIAG_SHOOT_MIN_VEL_MM_S` | 80 | Mínimo |vel| balón para evaluar semántica de shoot |
| `_DIAG_SHOOT_MIN_TARGET_DIST_MM` | 80 | Mínima distancia balón→target para vector unitario válido |
| `_DIAG_THETA_SNAP_PREP_DEG` | 18 | PREP shoot: Δθ **cuerpo** (inicio vs fin step) |
| `_DIAG_BALL_SOFT_JUMP_MM` | 120 | Salto “sospechoso” sin ser teleport duro → `diag_ball_large_jump_flag` |
| `_DIAG_OVERLAP_SEP_EPS_MM` | 6 | Separación centro-centro − radios < esto → overlap |
| `_DIAG_PENETRATION_EPS_MM` | 1 | Separación < −ε → penetración |
| `_DIAG_MOTION_VS_BODY_DEG` | 22 | Desalineación motion_walk_theta vs cuerpo (no PREP snap) |
| `_DIAG_THETA_SNAP_MOTION_PREP_DEG` | 18 | Snap motion en PREP shoot |
| `_DIAG_RAPID_KICK_PROXY_FRAMES` | 12 | Otro `kick_proxy` antes de N frames → `diag_rapid_kick_proxy_flag` |
| `_DIAG_POST_KICK_ALIGN_DOT_THRESH` | 0.14 | Step **siguiente** al kick: alineación vel vs `u_tgt` guardado → inversión |
| `_DIAG_RETROGRADE_VEL_X_MM_S` | −95 | En kick shoot-like, `vx` muy negativa (heurística ataque +x) |
| `_DIAG_OPP_PRESSURE_MM` | 520 | Mínima distancia activo–oponente ≤ esto → presión |

Ajustar aquí si el CSV es demasiado ruidoso o demasiado silencioso.

---

## 7. Diccionario de columnas `takeover_trace.csv`

### 7.1 Identidad del step

| Columna | Origen | Función |
|---------|--------|---------|
| `episode` | `train.py` (`episode + 1`) | Episodio 1-based alineado con métricas. |
| `stage_id`, `stage_name` | Currículo | Etapa de entrenamiento. |
| `env_step` | `info["step"]` | Paso dentro del episodio. |
| `bridge_frame_id` | `SidecarOutput` | Correlación con logs del bridge. |

### 7.2 Propiedad del control y macro

| Columna | Cómo se obtiene | Función |
|---------|------------------|---------|
| `control_owner` | `BHUMAN` si `macro.takeover_active`; si no `HANDOFF` si `handoff_ready`; si no `RL` | Quién “manda” en la UI lógica. |
| `handoff_ready`, `takeover_active` | `HandoffTakeoverManager.macro` | Contrato listo vs ejecución BHuman. |
| `macro_type`, `takeover_macro_type`, `macro_duration_frames`, `macro_end_reason`, `last_macro_end_reason` | macro + histórico | Fin de macro, timeouts, skill cleared, etc. |
| `playball_entry`, `ready_but_walk`, `ready_touch_but_walk` | flags del step | Plateaus típicos de política vs contrato. |
| `takeover_kick`, `takeover_kick_proxy` | env step | Evidencia de kick real vs proxy de velocidad. |
| `pending_kick_*`, `post_kick_release_frames` | env | Cola alrededor del contacto. |
| `takeover_control_frames`, `takeover_progress_frames`, `takeover_ball_progress_*` | macro | Progreso durante takeover. |

### 7.3 Estado balón y agente activo (fin de step)

| Columna | Función |
|---------|---------|
| `ball_*`, `ball_heading_deg` | Posición/velocidad global del balón. |
| `agent_*`, `agent_ball_*` | Estado del **agente activo** (posiblemente distinto del `agent_id` de la fila si multi-agent; priorizar `active_agent_id`). |

### 7.4 Salida BHuman / bridge (contrato)

| Prefijo | Función |
|---------|---------|
| `sidecar_backend`, `sidecar_current_tactic` | Proveniencia (`real` es requisito para traza). |
| `sidecar_macro_type`, `sidecar_macro_type_effective` | Macro declarado vs macro operacional inferido. |
| `sidecar_skill_request`, `sidecar_motion_request`, `sidecar_motion_executed_phase`, `sidecar_behavior_status` | Capas skill / motion / behavior. |
| `behavior_walk_*`, `motion_walk_*`, `motion_target_direction`, `skill_target_direction` | Targets relativos y en campo. |
| `kick_length`, `kick_length_raw` | Intención de kick. |
| `bridge_*`, `field_ball_*`, `shoot_target_*`, `ball_target_*`, `effective_*` | Cadena de targets hasta el ejecutor 2D. |

### 7.5 Diagnósticos “clásicos” (primer bloque `diag_*`)

| Campo | Función |
|-------|---------|
| `diag_ball_step_jump_mm` | ‖pos_fin − pos_inicio‖ del balón en el step. |
| `diag_ball_teleport_flag` | Salto grande **sin** kick-like/goal/out (umbral 520 mm). |
| `diag_active_agent_step_jump_mm`, `diag_agent_teleport_flag` | Salto del activo (inicio vs fin). |
| `diag_shoot_semantic_dot`, `diag_shoot_semantic_error`, `diag_shoot_backward_flag` | Alineación velocidad balón vs target semántico (`_resolve_playball_ball_target`) en frames kick-like. |
| `episode_diag_*_frames` / `*_rate` | Contadores acumulados en el episodio hasta este step. |
| `diag_active_agent_theta_step_delta_deg` | Δθ cuerpo del mismo agente que al inicio del step. |
| `diag_theta_snap_prep_flag` | PREP shoot + Δθ cuerpo ≥ 18°. |
| `diag_ball_agent_center_dist_mm` | Distancia centro balón–**activo al fin** de step. |

### 7.6 Diagnósticos extendidos (control, geometría, kicks)

| Campo | Función |
|-------|---------|
| `diag_prev_control_owner`, `diag_current_control_owner`, `diag_control_transition`, `diag_control_owner_changed`, `episode_diag_control_owner_changes`, `episode_diag_control_owner_change_rate` | Histeresis RL↔HANDOFF↔BHUMAN. |
| `diag_active_agent_changed_step` | Cambio de `active_agent_id` intra-step. |
| `diag_ball_speed_step_delta` | Cambio de rapidez del balón. |
| `diag_ball_large_jump_flag` | Salto ≥ 120 mm sin kick-like (sospecha de snap suave). |
| `diag_ball_jump_when_kick_proxy_mm` | Magnitud de salto cuando hay `kick_proxy`. |
| `diag_ball_agent_sep_mm` | `dist − (R_agent + R_ball)`; negativo = penetración. |
| `diag_ball_agent_overlap_flag`, `diag_ball_agent_penetration_flag` | Contacto fuerte / traspaso geométrico. |
| `diag_min_opp_dist_mm` | −1 si no hay oponentes. |
| `diag_opp_pressure_flag` | Oponente cerca del activo. |
| `diag_motion_walk_theta_deg`, `diag_motion_walk_theta_vs_body_deg` | “Recto al arco” vs cuerpo en número. |
| `diag_effective_target_theta_vs_body_deg` | θ de `_resolve_motion_target` vs cuerpo. |
| `diag_skill_target_dir_vs_body_deg`, `diag_motion_target_dir_vs_body_deg` | Direcciones declaradas vs cuerpo. |
| `diag_theta_snap_motion_prep_flag` | Snap motion en PREP. |
| `diag_frames_since_kick_proxy` | Frames desde el último kick-like **al inicio** del step. |
| `diag_rapid_kick_proxy_flag` | Burst de kicks. |
| `diag_sidecar_macro_neq_takeover_macro` | Desalineación macro sidecar vs macro takeover bajo takeover. |
| `diag_handoff_ready_rl_control_flag` | Listo para handoff pero control aún RL. |
| `diag_macro_active_without_takeover_flag` | Macro activo sin takeover (tensión de estado). |
| `diag_ball_vel_x_mm_s`, `diag_shoot_vel_retrograde_x_flag` | Heurística “patada hacia atrás” en eje X. |
| `diag_post_kick_vel_align_dot`, `diag_post_kick_inversion_flag` | Step **posterior** al kick: vel vs `u_tgt` guardado. |

### 7.7 Física explícita, multi-agente, activation graph, reward

| Campo | Función |
|-------|---------|
| `diag_ball_collision_tags` | Lista `|` de resoluciones: `rl_agent_sep:…`, `rl_agent_coincident:…`, `opp_sep:…`, `static_obs_sep:…`. Explica **qué rama** de empuje se aplicó. |
| `diag_inactive_ball_dists_json` | JSON `{agent_id: dist_mm}` para RL **no** activos (relevos / 4v4 futuro). |
| `diag_activation_graph_len`, `diag_activation_states`, `diag_activation_options`, `diag_activation_zweikampf_flag` | Resumen del `activation_graph` del bridge (duel/shoot vía estados como `zweikampf`). |
| `diag_macro_from_activation_graph` | Macro inferido con `infer_real_macro_type(skill, motion, activation_graph)` (evidencia BHuman agregada, sin C++). |
| `diag_macro_bridge_vs_activation_flag` | `1` si `sidecar.macro_type` ≠ macro inferido y ambos distintos de `none` (desalineación skill/graph vs etiqueta bridge). |
| `rw_sparse` … `rw_fov_bonus`, `rw_total` | Desglose **sin shaping_weight aplicado por campo** en el CSV: los valores son los del dataclass `RewardComponents`; el `total` ya incluye `sparse + shaping_weight * shaped` como en `reward.compute_reward`. |

*(Si necesitas el detalle matemático de cada término, ver `reward.py` y `ENV_REWARD_CURRICULUM_GUIDE.md`.)*

---

## 8. `metrics.csv` (otra capa, por episodio)

Generado por el logger de entrenamiento: tasas como `takeover_ready_rate`, `ready_but_walk_rate`, éxitos por tipo de macro, etc.  
**No sustituye** `takeover_trace.csv`: resume **episodio**; la traza resume **step** donde hubo contrato o anomalía.

**Uso conjunto:** `metrics.csv` para tendencias; traza para causa raíz de un episodio outlier.

---

## 9. Visualizer (`visualizer.py`)

- **`control_debug()`**: construye strings para HUD (macro, skill, motion, `BallAct`, kicks, tasas de episodio).
- **`diag_compact`**: abreviaturas (`TELball`, `Overlap`, `MotSnap`, `CtlChg`, `RapidKick`, `PostInv`, `RetroX`, …).
- **`diag_control_transition`**: borde de control en texto.

No es fuente de verdad para auditoría legal del run: la fuente es el **CSV** + `infos` si se vuelca a JSON.

---

## 10. Análisis offline (`analyze_takeover_run.py`)

```bash
cd RL
python analyze_takeover_run.py <ruta_run_o_metrics.csv> [--window N] [--bucket-mm M] [--top-episodes K]
```

Salidas:

- Resumen de `metrics.csv` (ventanas, por stage).
- Frecuencias de macros / walk / shoot targets en traza.
- Si existen columnas extendidas: ranking de episodios “dolorosos”, percentiles de separación balón–agente, etc.
- Si existen `rw_*` y `diag_ball_collision_tags`: bloque **trace_reward_physics** (tags de colisión, zweikampf, magnitudes de `rw_*` relevantes).

**Runs antiguos** sin columnas nuevas: el script indica que hay que **re-ejecutar** con el `environment` actual.

---

## 11. Checklist orientado a “env listo en 2 días”

1. **Una corrida corta** con `TRACE_TAKEOVER_ALL_STEPS=1` sobre stage objetivo → revisar `diag_ball_collision_tags` y `rw_ready_but_walk_penalty` en episodios rankeados.
2. **Corrida larga** con default de traza (filtrada) → validar que `metrics.csv` no degrada `takeover_ready_rate` sin explicación en `diag_control_*`.
3. **Antes de producción:** congelar versión de este artifact + commit SHA en el run log.
4. **No bloquear** el cierre por telemetría del binario BHuman interno: lo que falta se obtiene con dumps nativos en otra iteración.

---

## 12. Limitaciones explícitas (honestidad técnica)

- No se loguea el **árbol completo de skills C++** ni timers internos del ejecutable; solo lo que expone el bridge (`activation_graph`, metadata, motion/skill).
- `diag_shoot_semantic_*` y post-kick usan el **mismo** `_resolve_playball_ball_target` que el ejecutor 2D, pero no sustituyen replay determinista del bridge.
- `takeover_kick` puede ir en cero en traza según evidencia; muchos contactos van por **`takeover_kick_proxy`** — diseñado así.
- Muestra de `infos` en entrenamiento: **un** agente por step en CSV; diseño aceptable si el foco es el activo y el balón global.

---

## 13. Referencias de código

| Pieza | Archivo |
|-------|---------|
| Constantes `_DIAG_*`, `_compute_step_diagnostics`, `_resolve_ball_collisions`, `_build_infos` | `environment.py` |
| `RewardComponents`, `compute_reward` | `reward.py` |
| Columnas CSV, `should_log`, env vars | `takeover_trace.py` |
| Llamada `log_step` | `train.py` |
| HUD diagnóstico | `visualizer.py` |
| Agregados CLI | `analyze_takeover_run.py` |

---

*Fin del artifact.*
