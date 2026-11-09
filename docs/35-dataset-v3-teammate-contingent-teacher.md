# 35 — Dataset v3: teammate-contingent teacher + contrastive worlds

**Para:** Codex (con Luis + Claude) · **Fecha:** 2026-06-13 · **Rama base:** `rl-v3a-multiagent-defender`

## Por qué (la conclusión empírica de v3a/v3b)

Dos pasadas de cambios reward/BC (v3a, v3b) NO movieron el hold de supports (open
~0.1 plano, triangle 0.0) y no subieron conversión + floor a la vez — sólo barajaron
tradeoffs. **El techo de la capa reward/BC está confirmado.** La causa raíz no es el
reward: es que **el dataset no contiene ejemplos de cooperación**. El teacher actual
etiqueta:

- al **striker** como finalizador en solitario (`smash_or_pass` mira xG propio vs un
  proxy de posición del compañero), y
- a los **supports** como *seguidores de ancla* (camina/párate hacia un slot geométrico),

con el hold premiado **por geometría, no por utilidad**. No hay filas donde la acción
correcta del striker *dependa* del estado del compañero de forma que produzca acciones
distintas para el mismo ego-estado. Por eso la política "lee" a los compañeros un poco
(ablación v2b: 7.9% de skill-flips) pero **no combina**: nunca vio combinaciones.

**Idea central de v3:** el teacher debe tomar decisiones **contingentes al compañero**,
y el dataset debe contener **pares contrastivos** — mismo ego-estado, compañero
abierto vs cubierto → acción distinta. Eso es lo único que enseña la contingencia.

Pipeline igual que siempre: heurística → dataset → BC → mask/gate → PPO. Esto SÍ es un
regen del corpus de ataque (stages 6/7), warm-start v2b@7500.

⚠️ Memoria [[graph-coordination-pipeline-hostility]]: los fallos de este tipo de cambio
son **silenciosos** (BC loss baja, audit verde, promoción ok, y la política ignora la
señal). Por eso cada etapa lleva una verificación dura ANTES de pasar a la siguiente.

---

## 1. Conductas nuevas del teacher (`multiagent_teacher.py`)

### 1.1 Striker: pase contingente real (no proxy de posición)
`smash_or_pass` hoy pasa si un compañero adelantado supera el xG propio por margen. v3:
hacer que la decisión dependa de la **apertura real del carril del compañero** (lane
libre de oponentes hacia el receptor) y de **si el receptor está en posición de recibir**
(no de espaldas, no fuera de rango). Mismo gate (`pass_armed`), pero la SEÑAL que arma el
pase ahora distingue compañero-abierto de compañero-cubierto. Resultado: en el mundo A
(abierto) etiqueta `pass`; en el mundo B (cubierto, idéntico salvo un oponente en el
carril) etiqueta `dribble`/`shoot`.

### 1.2 Give-and-go (pase y desmarque)
Tras un pase, el **pasador** NO se queda: se mueve a un slot de recepción para el
posible retorno (`teacher_open_support` con target = punto de apoyo adelantado tras el
receptor). El **receptor** (ahora portador vía el role-flip por control de balón, D1 de
doc-34) decide retorno vs conducción con `smash_or_pass`. Esto produce cadenas de 2-3
pasos de cooperación en el dataset.

### 1.3 Anticipación del receptor (liderar el pase)
El support que es el **target probable de pase** se mueve a una posición *recibible*
ANTES del pase (no después). Hoy los supports se anclan a slots fijos; v3: el slot del
support-receptor se sesga hacia el carril de pase del striker cuando el striker controla
y mira hacia él.

### 1.4 Hold CONTINGENTE (el fix real del hold-rate)
El hold deja de ser geométrico. Un support **mantiene posición (stand) sólo cuando es una
opción de pase viva**: striker con balón + carril abierto hacia el support + support en
rango de pase. Si el carril se cierra o el striker no puede pasar, el support **se
reposiciona** (camina a otro slot). Así el "quedarse quieto" tiene una RAZÓN, y la
política aprende a holdear cuando holdear sirve — no a caminar por defecto ni a holdear
en el vacío. Esto reemplaza la heurística de slot fijo + hold por tolerancia.

---

## 2. Mundos contrastivos (`production_curriculum.py`)

Generar escenarios en **PARES** idénticos salvo un grado de libertad (la apertura de un
compañero/carril), etiquetados con un `contrast_id` y una bandera de variante (A=abierto,
B=cubierto). La infraestructura parcial ya existe: el env expone
`production_scenario_contrast` (ver `wrapper.py`), úsalo para marcar la variante.

Pares mínimos:

| Par | Variante A (abierto) | Variante B (cubierto) | Acción correcta que se contrasta |
|---|---|---|---|
| `giveandgo_lane` | support en carril limpio | oponente tapa el carril | striker pass (A) vs dribble (B) |
| `support_open_vs_marked` | support libre adelantado | oponente marcando al support | hold del support (A) vs reposicionar (B) |
| `return_pass` | pasador con línea de retorno limpia | retorno tapado | receptor devuelve (A) vs conduce (B) |

Cada par se genera con la MISMA semilla base + el MISMO ego-spawn, cambiando sólo la
posición del oponente del carril. El collector debe emitir ambas variantes (no dejar que
el sampler omita una).

---

## 3. Reward (capa TEAM, `team_parallel_environment.py`)

Hacer el **hold bonus contingente** (no geométrico): pagar el hold sólo cuando el support
es una opción de pase viva (mismas condiciones que §1.4 — striker con balón, carril
abierto al support). Así el reward refuerza exactamente la conducta que el teacher
etiqueta, y deja de premiar holdear en el vacío (que es parte de por qué v3a/v3b no
movieron el hold). Mantener `RL_MA_TEAM_GOAL_BONUS=2.5`, el finishing bonus (P6) y el
drift penalty reworked (v3b) tal cual.

---

## 4. Secuencia con verificaciones duras (no saltarse ninguna)

1. **Teacher dry-run contrastivo (ANTES de regen):** correr el teacher en un par
   contrastivo y verificar que **emite acciones DISTINTAS en A vs B** (pass vs dribble,
   hold vs reposicionar). Si A y B dan la misma acción, la contingencia no existe — el
   resto del pipeline será silenciosamente inútil. **Gate: ≥80% de pares con acción
   distinta.**
2. **Regen chunked** (gate 0.30/0.20, mismas reglas que el corpus de ataque actual),
   stages 6 y 7, con los pares contrastivos en el pool.
3. **Audit de balance contrastivo:** además del audit normal (liveness, role-mask 0 bad
   rows), verificar que **ambas variantes A y B están presentes en proporción ~1:1** por
   par, y que la distribución de skills difiere entre A y B (si son iguales, el contraste
   se perdió en el filtrado). **Gate: |frac_A − frac_B| < 0.15 por par.**
4. **BC + PPO** warm-start v2b@7500, KL anchor v2b@7500, evaluar TODOS los checkpoints.
5. **Validación de cooperación (el gauge que importa):**
   - **Ablación de teammates** (zero obs[26:39]): skill-flips deben subir MUY por encima
     del 7.9% de v2b — eso prueba que la política *condiciona* en los compañeros, no que
     imita. **Gate: skill-flips ≥ ~20%.** Si sigue en ~8%, la contingencia no se aprendió
     (fallo silencioso) → revisar §1, no alargar PPO.
   - **Hold-rate** (eval_summary, ya agregado por P0): debe salir por fin de ~0.1 →
     objetivo ≥0.4 open, triangle off-zero.
   - **Conversión** shoot_finish ≥0.25 y **floor single-agent ≥4/5** (sin regresión).

---

## 5. Alcance, riesgo y relación con producción

- Esto es la pista **pesada** (memoria: 3-4 semanas realista si se secuencia bien;
  fallos silenciosos). NO es para el demo inmediato — el candidato de ataque sigue siendo
  **v2b@7500** (mejor floor 4/5; v3a/v3b son sidegrades).
- El wire audit ya es role-aware (doc de este commit, `RL_WIRE_AUDIT_TEAM_MODE`), así que
  las eval de equipo de v3 no van a false-fail.
- Encaja con la ideología per-role: este corpus v3 mejora el **striker/supports** ONNX
  (`ppo_team_hsl2026.onnx`); el defender (doc-34) es pista paralela; mismo contrato 47-dim
  para toda la familia.
- Regla dura intacta: `OBS_SIZE=47`, `SKILL_ORDER`, Design-A (una sola máscara en
  `train.gate_skill_mask_from_obs`), gate 0.30/0.20 en todo, checkpoints inmutables.
