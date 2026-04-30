# Documentacion del estado actual RL x B-Human

Esta carpeta describe el estado actual del proyecto RL integrado con B-Human y SimRobot.

Lectura recomendada:

1. [Estado actual](./00-estado-actual.md)
2. [Flujo natural de B-Human](./01-flujo-natural-bhuman.md)
3. [Fisica SimRobot 2D vs 3D](./02-fisica-simrobot-2d-vs-3d.md)
4. [Contrato RL, backend y entorno](./03-contrato-rl-env.md)
5. [Verificacion operativa](./04-verificacion-operativa.md)
6. [Premisas irrompibles](./05-premisas-irrompibles.md)

Resumen corto:

- RL reemplaza la capa de estrategia, no el motor de movimiento.
- Python manda `SkillRequest` por `RLSharedState`.
- `SkillBehaviorControl` convierte el skill en `MotionRequest`.
- `MotionEngine` y SimRobot ejecutan la caminata natural.
- El backend rapido `cpp` usa `RLSim2D`, pero ahora tambien avanza desde el `MotionRequest` final.
- Los perfiles SimRobot reales (`2d`, `3d`) tienen `sim2D.enabled = false`.

