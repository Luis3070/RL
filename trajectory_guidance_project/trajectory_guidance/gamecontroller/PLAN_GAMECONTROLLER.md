# Plan: GameController — Control Humano del Agente

## Objetivo
Crear un controlador humano tipo "click-to-move" (estilo LoL) conectado al environment real (`NaoSoccerEnv` vía `NaoSoccerEnvWrapper`) para diagnosticar si los problemas de trayectorias erráticas vienen de A* o del environment.

## Principio Fundamental
**Todo pasa por el action space real del environment.** No se hace bypass de la física, no se teletransporta al agente, no se modifica el balón directamente. El controlador humano genera acciones `[dx, dy, dtheta]` en `[-1, 1]` que se procesan por `_decode_action` → `step()` → `_handle_ball_interactions()` exactamente igual que si vinieran de A* o de la policy RL.

Esto es **mandatorio** para que el diagnóstico sea válido: si el humano puede hacer goles y dribbleos correctos usando el mismo pipeline de acciones, el problema es A*. Si el humano también sufre comportamientos erráticos, el problema es el environment.

---

## Arquitectura (3 archivos)

### 1. `env_bridge.py` — Bridge al Environment
**Responsabilidad**: Conectar el gamecontroller al environment real sin duplicar lógica del bridge existente.

- Reutiliza `NaoSoccerEnvWrapper` del proyecto (misma instancia que usa A*)
- Reutiliza `EnvStateReader` de `visualizer.py` para leer estado (posiciones, balón, game)
- Reutiliza `CoordTransform` de `visualizer.py` (añadiendo método inverso `to_mm(px_x, px_y)` para convertir clicks de pantalla a coordenadas de campo)
- Expone:
  - `reset(stage_id, seed)` → resetea environment al stage dado
  - `step(action_normalized)` → ejecuta un paso con acción `[-1,1]^3`, retorna `(obs, rewards, terms, truncs, infos)`
  - `get_agent_pos()` → posición actual del agente controlado
  - `get_ball_pos()` → posición actual del balón
  - `get_state_reader()` → `EnvStateReader` para el renderer

### 2. `controller.py` — Lógica de Control Humano
**Responsabilidad**: Convertir inputs del usuario (clicks, teclas) en acciones normalizadas `[-1,1]^3`.

#### Esquema de control:
| Input | Acción |
|-------|--------|
| **Click izquierdo** | Mover agente a posición clickeada (sin teletransporte — camina) |
| **Space** | Tiro a portería: acción con max throttle hacia arco rival a través del balón |
| **P** | Pase: acción con max throttle en dirección del cursor a través del balón |
| **R** | Reiniciar episodio (mismo stage, nueva seed) |
| **N** | Siguiente stage |
| **0-3** | Saltar a stage específico |
| **Q / ESC** | Salir |

#### Conversión click → acción:
```
1. click_mm = CoordTransform.to_mm(click_px)          # px → mm del campo
2. displacement = click_mm - agent_pos                  # vector de desplazamiento
3. max_disp = stage.local_horizon o ACTION.max_displacement_mm
4. action[0] = clip(displacement[0] / max_disp, -1, 1) # normalizar a [-1,1]
5. action[1] = clip(displacement[1] / max_disp, -1, 1)
6. action[2] = 0.0  (theta es face-ball automático en stages 0-2)
```

#### Tiro (Space):
```
1. Calcular dirección: agente → balón → arco rival (x=4500, y=0)
2. target = posición lejana en esa dirección (max throttle)
3. Convertir a acción normalizada con action[0:2] saturado a ±1
   (esto genera throttle > 280mm → kick automático del environment)
```

#### Pase (P):
```
1. Calcular dirección: agente → balón → posición del cursor
2. target = posición lejana en esa dirección (max throttle)
3. Convertir a acción normalizada con action[0:2] saturado a ±1
```

#### Estado del controlador:
- `target_mm`: posición meta actual del agente (se actualiza con cada click)
- `kick_pending`: bool — si se activó kick, se genera acción de kick en los próximos N pasos hasta que el agente contacte el balón o se agote
- `pass_pending`: similar para pase
- `pass_direction`: dirección del pase (se captura del cursor al presionar P)

### 3. `game.py` — Visualizador + Loop Principal
**Responsabilidad**: Renderizado pygame + game loop.

- Reutiliza `Renderer` de `visualizer.py` (campo, agentes, balón, FOV, HUD)
- Añade:
  - Indicador visual del target actual del agente (cursor/waypoint)
  - Indicador de modo actual (movimiento / kick pendiente / pase pendiente)
  - Trace del recorrido del agente (línea amarilla como en replay)
  - HUD con: stage actual, step count, score, reward acumulado, modo de control
  - Indicador visual de rango de kick (círculo alrededor del agente cuando está cerca del balón)

#### Game Loop:
```python
while running:
    # 1. Procesar eventos pygame (clicks, teclas)
    # 2. Controlador genera acción normalizada [-1,1]^3
    # 3. Bridge ejecuta step(acción) en el environment REAL
    # 4. Renderer dibuja el estado actual del environment
    # 5. Clock tick (FPS configurable, default 10 = tiempo real con dt=0.1s)
```

**FPS = 10** por defecto para que cada frame corresponda a un step del environment (dt=0.1s). El usuario puede subir/bajar la velocidad del juego.

---

## Conexión Exacta con el Environment

### Action Space (sin modificar):
```
action[0] ∈ [-1, 1] → dx relativo al agente (mm, escalado por max_displacement_mm)
action[1] ∈ [-1, 1] → dy relativo al agente (mm, escalado por max_displacement_mm)
action[2] ∈ [-1, 1] → dtheta (rad, si theta_free=True; si no, face-ball automático)
```

### Pipeline que se respeta íntegro:
```
Acción humana [-1,1]^3
  → environment._decode_action()     # relativo → absoluto
    → EMA suavizado                   # action smoothing
    → local_horizon clamp             # subgoal local
    → _movement.step()                # física del NAO (velocidad, rotación)
    → _handle_ball_interactions()     # dribble/kick automático
    → _ball_physics.step()            # fricción, rebotes
    → check_goal()                    # detección de gol
    → _compute_rewards()              # rewards reales del environment
```

### Lo que el humano controla vs lo que el environment decide:
| Aspecto | Controlado por |
|---------|---------------|
| Dirección de movimiento | Humano (click) |
| Velocidad de movimiento | Environment (max_forward_speed=250mm/s) |
| Theta (orientación) | Environment (face-ball automático en stages 0-2) |
| Dribble | Environment (automático si dist < dribble_range + hemisferio frontal) |
| Kick | Environment (automático si throttle ≥ 280mm + angle_diff ≤ kick_half_rad) |
| Caídas | Environment (fall_probability si speed > 280mm/s) |
| Física del balón | Environment (fricción, respawn si OOB) |

---

## Estructura de Archivos

```
trajectory_guidance_project/
└── trajectory_guidance/
    └── gamecontroller/
        ├── __init__.py          # vacío
        ├── PLAN_GAMECONTROLLER.md  # este documento
        ├── env_bridge.py        # bridge al environment
        ├── controller.py        # lógica de control humano
        └── game.py              # visualizador + game loop (entry point)
```

### Ejecución:
```bash
# Desde la raíz del proyecto (RL/)
python -m trajectory_guidance.gamecontroller.game --stage 0
# O directamente:
python trajectory_guidance/trajectory_guidance/gamecontroller/game.py
```

---

## Dependencias
- `pygame` (ya instalado — usado por `replay_dataset_pygame.py` y `visualizer.py`)
- `numpy` (ya instalado)
- Módulos del proyecto: `config`, `wrapper`, `environment`, `visualizer`, `reward`

## Restricciones que se respetan
- **No teletransportes**: el agente camina paso a paso con la física del NAO
- **No bypass de acciones**: todo pasa por el action space [-1,1]^3
- **No modificación del environment**: se usa la misma instancia que A*
- **Mismos rewards**: se muestran los rewards reales del environment
- **Misma física**: kick/dribble determinado por el environment, no por el controlador
