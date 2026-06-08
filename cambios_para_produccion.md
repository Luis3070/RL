# Cambios para lograr 4/5 criterios de evaluación — PPO Producción RoboCup HSL

> Documento histórico. El baseline vigente de transplante ya no es este.
> Referencia actual: `runs/production_stability_repair_v1/checkpoint_stage4_step5000.pt`
> con validación canónica parcial en
> `runs/behavioral_evaluator/behavior_20260608_181747/summary.csv`.

**Checkpoint base:** `runs/production_readiness_v2/checkpoint_stage4_step10000.pt`  
**Resultado:** 4/5 escenarios aprobados en `behavioral_evaluator`  
**Evaluación:** `runs/behavioral_evaluator/behavior_20260605_022521/summary.csv`

---

## Resultado final por escenario

| Escenario | Aprobó | Criterio bloqueante |
|-----------|--------|---------------------|
| `finish_band_2p5_3m_clean` | ✅ | — |
| `finish_band_2p5_3m_static_lane` | ✅ | — |
| `dynamic_full_field_pressure` | ✅ | — |
| `blocked_corridor_repair` | ✅ | — |
| `preshoot_bad_angle_boundary` | ❌ | `collision_rate > 0.12` |

---

## Cambios principales realizados y por qué

### 1. Rebalanceo de escenarios en `train_production.py`

**Qué se hizo:** Se ajustaron los pesos de los escenarios de entrenamiento para que la distribución de episodios fuera más representativa de lo que evalúa `behavioral_evaluator`.

**Pesos anteriores (dominados por topups):**
```
finish_near_arm_clean_topup:5, finish_bridge_clean_topup:6, finish_bridge_static_topup:4
finish_band_2p5_3m_clean:4, finish_band_2p5_3m_static_lane:3
```

**Pesos nuevos (cargados a escenarios canónicos):**
```
finish_band_2p5_3m_clean:8
finish_band_2p5_3m_static_lane:6
blocked_corridor_repair:5
preshoot_bad_angle_boundary:3
```

**Por qué:** El evaluador canónico usaba exactamente los escenarios `finish_band_2p5_3m_clean`, `finish_band_2p5_3m_static_lane`, `preshoot_bad_angle_boundary`, `dynamic_full_field_pressure` y `blocked_corridor_repair`. Cuando los topups dominaban el entrenamiento, el modelo mejoraba en el eval interno del training loop pero fallaba en el `behavioral_evaluator` porque nunca veía suficientes episodios de los escenarios reales. El mismatch entre distribución de entrenamiento y distribución de evaluación era el cuello principal.

---

### 2. Corrección de la máscara de gate en inferencia — `train.py` (función `gate_skill_mask_from_obs`)

**Qué se hizo:** Se modificó la función `gate_skill_mask_from_obs` para que cuando `dribble_armed` o `shoot_armed` estén activos, se bloqueen también `walk` y `stand` en la máscara de la política.

**Antes:**
```python
mask[:, _SHOOT_SKILL_IDX] = shoot_armed.to(mask.dtype)
mask[:, _DRIBBLE_SKILL_IDX] = dribble_armed.to(mask.dtype)
# walk y stand siempre permitidos
```

**Después:**
```python
finish_armed = shoot_armed | dribble_armed
mask[:, _SHOOT_SKILL_IDX] = shoot_armed.to(mask.dtype)
mask[:, _DRIBBLE_SKILL_IDX] = dribble_armed.to(mask.dtype)
mask[:, SKILL_ORDER.index("stand")] = (~finish_armed).to(mask.dtype)
mask[:, SKILL_ORDER.index("walk")] = (~finish_armed).to(mask.dtype)
```

**Por qué:** El `behavioral_evaluator` reportaba `clean_takeover_rate=0.0` e `incompatible_takeover_rate` alto a pesar de que el agente llegaba al balón y a veces disparaba. El problema era que una vez dentro del takeover envelope (con `dribble_armed=True`), la política seguía eligiendo `walk` o `stand` porque esas acciones no estaban prohibidas. El evaluador considera eso una toma "sucia" — el agente tiene el balón pero sigue caminando en lugar de driblarlo. Bloqueando `walk` y `stand` cuando hay un gate activo se fuerza al agente a comprometerse con la skill correcta, lo que eleva `clean_takeover_rate` directamente.

Este cambio opera en inferencia sin necesidad de reentrenar — cambia la máscara que aplica la política al muestrear acciones, no los pesos de la red.

---


## Métrica raíz que desbloqueó los 4 escenarios: `clean_takeover_rate`

Todos los escenarios que fallaban antes del cambio de máscara tenían `clean_takeover_rate` bajo o nulo, a pesar de que `takeover_success` era alto. La diferencia:

- **`takeover_success`**: el agente *llegó* al balón en algún momento del episodio
- **`clean_takeover_rate`**: el agente llegó al balón **y** ejecutó inmediatamente dribble/shoot, sin pasar por walk/stand dentro del takeover envelope

El gate bloqueando `walk`/`stand` cuando `finish_armed` transformó `takeover_success` alto en `clean_takeover_rate` alto, lo que desbloqueó los criterios de todos los escenarios excepto `preshoot_bad_angle_boundary`.

---

## Por qué `preshoot_bad_angle_boundary` sigue fallando

Este escenario coloca al robot en ángulo difícil frente a la portería con obstáculos dinámicos. El criterio que falla es `collision_rate <= 0.12` — el agente choca con los NAOs que se mueven. Esto no se resuelve con la máscara de gate: es un problema de la política frente a obstáculos dinámicos con ángulos de aproximación adversos, que requiere más entrenamiento en ese escenario específico o ajustar la penalización de colisión en Stage 4.

---

## Resumen de archivos modificados

| Archivo | Cambio | Efecto |
|---------|--------|--------|
| `train_production.py` | Rebalanceo de pesos de escenarios | Alineó distribución de entrenamiento con distribución del evaluator canónico |
| `train.py` (`gate_skill_mask_from_obs`) | Bloqueo de `walk`/`stand` cuando `finish_armed` | Subió `clean_takeover_rate` de ~0 a >0.75 en 4 escenarios |
| `gate.py` (`_allowed_skills`) | Cambio intermedio en lógica de entorno | No fue la palanca correcta para la política; se mantuvo como consistencia semántica |
| `reward.py` | missed_dribble_handoff y dribble_skill_intent existían, pero solo se activaban para stages {0, 1} | En Stage 4, sus pesos estaban definidos pero no hacían nada. |
| `curriculum.py` | castigar más el óptimo local de caminar/parquearse encima de la pelota en el Stage 4 | subir clean_takeover_rate|

> Tambien se aumentaron los steps a 1000
