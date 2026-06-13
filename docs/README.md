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
9. [Implementacion PPO observation space 23](./09-implementacion-ppo-observation-space-23.md)
10. [Escenarios RL sin oracle world model](./10-escenarios-rl-sin-oracle-world-model.md)
11. [RL1v0_3D: percepcion natural y diagnostico](./11-rl1v0-3d-percepcion-natural.md)
12. [Wire RL realista final](./12-wire-rl-realista-final.md)
13. [Pruebas GUI en SimRobot](./13-pruebas-gui-simrobot.md)
14. [Guia practica para armar escenas y escenarios RL](./15-guia-escenas-y-escenarios-rl.md)
15. [Curriculum dinamico: balon y rivales](./16-curriculum-dinamico-balon-y-rivales.md)
16. [Handoff visible SimRobot GUI](./17-handoff-visible-simrobot-gui.md)
17. [SimRobot estable y partido hibrido RL vs B-Human](./21-simrobot-estable-y-rl-vs-bhuman.md)
18. [Stage 8 defender: SimRobot visible single-process](./31-stage8-defender-visible-debug.md)
19. [Stage 8 defender: checklist operativo](./32-stage8-defender-checklist.md)
20. [Stage 8 defender: evaluacion del train inicial](./33-stage8-defender-initial-train-eval.md)

Resumen corto:

- RL reemplaza la capa de estrategia, no el motor de movimiento.
- Python manda `SkillRequest` por `RLSharedState`.
- `SkillBehaviorControl` convierte el skill en `MotionRequest`.
- `MotionEngine` y SimRobot ejecutan la caminata natural.
- Los backends soportados son SimRobot real: `simrobot` y `simrobot_visible`.
- Los perfiles operativos son `2d` y `3d`.
