# Documentacion del estado actual RL x B-Human

Esta carpeta describe el estado actual del proyecto RL integrado con B-Human y SimRobot.

Lectura recomendada:

1. [Estado actual](./00-estado-actual.md)
2. [Flujo natural de B-Human](./01-flujo-natural-bhuman.md)
3. [Fisica SimRobot 2D vs 3D](./02-fisica-simrobot-2d-vs-3d.md)
4. [Contrato RL, backend y entorno](./03-contrato-rl-env.md)
5. [Verificacion operativa](./04-verificacion-operativa.md)
6. [Premisas irrompibles](./05-premisas-irrompibles.md)
7. [Matriz de skills y comparativo 2D vs 3D](./06-simrobot-skill-matrix.md)
8. [Observation space realista y fallback de canary](./07-observation-space-realista-y-fallback-canary.md)
9. [Handoff tecnico PPO 23 vars + MAPPO 35 vars](./08-codex-handoff-ppo-mappo.md)
10. [Implementacion PPO observation space 23](./09-implementacion-ppo-observation-space-23.md)
11. [Escenarios RL sin oracle world model](./10-escenarios-rl-sin-oracle-world-model.md)
12. [RL1v0_3D: percepcion natural y diagnostico](./11-rl1v0-3d-percepcion-natural.md)
13. [Wire RL realista final](./12-wire-rl-realista-final.md)
14. [Pruebas GUI en SimRobot](./13-pruebas-gui-simrobot.md)

Resumen corto:

- RL reemplaza la capa de estrategia, no el motor de movimiento.
- Python manda `SkillRequest` por `RLSharedState`.
- `SkillBehaviorControl` convierte el skill en `MotionRequest`.
- `MotionEngine` y SimRobot ejecutan la caminata natural.
- El backend rapido `cpp` usa `RLSim2D`, pero ahora tambien avanza desde el `MotionRequest` final.
- Los perfiles SimRobot reales (`2d`, `3d`) tienen `sim2D.enabled = false`.
