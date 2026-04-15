# REPORTE DE DIAGNÓSTICO ENTRANTE

**ID de Corrida:** `practical_stage0_fovTrue_1776181732`
**Analista:** Antigravity (Senior MLOps Analyst)
**Ubicación Oficial:** `C:\ALEJO\RL\Documentacion\ARTIFACTS\reporte_diagnostico_entrante.md`

---

## 1. CORRELACIÓN: Datos Técnicos vs. Observaciones Visuales (Pains)

| Pain Point (Manifest) | Evidencia Técnica (Logs/Código) | Archivo Sospechoso |
| :--- | :--- | :--- |
| **P1: Múltiples shots / teletransportación** | **Ep 1, Step 120:** El agente salta **1936mm** en un solo frame (de `1414` a `-522`). Coincide con activación de `HANDOFF`. | [environment.py](file:///c:/ALEJO/RL/environment.py#L1132) |
| **P2: Colisión con obstáculo** | No existe penalización implementada. `reward.py` solo tiene comentarios sobre colisiones. | [reward.py](file:///c:/ALEJO/RL/reward.py#L17) |
| **P4: Rotaciones absurdas (dribble)** | `ready_touch_but_walk = 1` detectado. Conflicto de targets entre RL y BHuman en banda. | [environment.py](file:///c:/ALEJO/RL/environment.py#L1521) |
| **P6: Handoff speed (stage 1)** | Latencia crítica. El bridge permanece en `walk` con el robot tocando el balón. | [bhuman_sidecar.py](file:///c:/ALEJO/RL/bhuman_sidecar.py) |

---

## 2. FILTRADO DE HIPÓTESIS

1.  **[VALIDADA]** **Inconsistencia de Pose:** El executor 2D inyecta la pose de BHuman (`robot_pose_field`) directamente al sim sin reconciliación de frames, causando el "Teleport" del paso 120.
2.  **[VALIDADA]** **Recompensas Persistentes:** La recompensa terminal de takeover se aplica cada frame mientras `macro_end_reason != "none"`, en lugar de ser un impulso único.
3.  **[VALIDADA]** **Falta de Castigo por Colisión:** El agente ignora obstáculos porque el entorno solo los "empuja" físicamente pero no reporta penalización al aprendizaje.

---

## 3. MAPA DE INVESTIGACIÓN

### A. Capa de Reward (`reward.py`)
*   Implementar `_collision_penalty` real (actualmente es solo un comentario).
*   Convertir `takeover_success/failure` en un evento tipo *impulse* (1 frame) para evitar el "multi-castigo".

### B. Capa de Ejecución (`environment.py`)
*   Reducir `max_translation_step` de `8.0x` a un valor físicamente plausible durante el handover.
*   Revisar la integración de `robot_pose_field` en `_stabilize_pre_kick_pose`.

### C. Capa de Sidecar (`bhuman_sidecar.py`)
*   Auditar por qué el `SkillRequest` tarda tanto en cambiar de `walk` a `shoot` cuando el robot ya está en posición de handoff.

---

## 4. EVIDENCIA CRUDA (Dump de Trazas)

### Salto de Posición (Teleport Ep 1):
```csv
episode,env_step,agent_x,agent_y,control_owner
1,119,1414.42,-120.88,RL
1,120,-522.55,466.07,HANDOFF  <-- Salto detectado de ~2 metros
```

### Persistencia de Fallo (Ep 5):
```csv
episode,env_step,takeover_active,macro_end_reason
5,336,1,timeout
5,337,1,timeout  <-- Aplicación duplicada de penalización
```

---
**Próximo Paso:** Proceder con la corrección estructural en `reward.py` y `environment.py`.
