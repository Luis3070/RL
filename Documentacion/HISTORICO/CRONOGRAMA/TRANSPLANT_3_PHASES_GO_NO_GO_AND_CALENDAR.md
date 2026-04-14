# Go/No-Go y Calendario de 2 Meses para el Transplante RL

Fecha de consolidacion: 2026-03-22
Horizonte analizado: 2026-03-23 a 2026-05-22

## Veredicto ejecutivo

### Estado real hoy

- GO VERDE para arrancar produccion de Fase 1.
- NO GO VERDE para declarar en produccion el transplante completo de las 3 fases.

### Significado practico

- La documentacion de Fase 1 ya esta suficientemente cerrada para implementar parser, entorno e integracion inicial sin falsas fuentes criticas.
- El transplante completo de 3 fases aun NO esta completamente alineado a nivel documental.

## Confirmacion de si el transplante va por 3 fases

### Respuesta corta

Si, ambos documentos hablan de 3 etapas, pero NO son las mismas 3 fases.

### Canon actual del proyecto

Segun `MAPPO_Plan_3Fases.md`, el proyecto canonico es:

1. Fase 1: RL solo define posiciones `(x, y, theta)`.
2. Fase 2: RL define posiciones + roles.
3. Fase 3: RL suplanta tactica y estrategia completas, incluyendo skills.

### Estructura actual del guide de transplante

Segun `RL_TRANSPLANT_GUIDE_COMPLETE.md`, el transplante hoy esta descrito como:

1. Fase 1: RL predice posiciones `(x, y, theta)`.
2. Fase 2: RL ajusta parametros de decision de `PlayBall` y skills asociados.
3. Fase 3: RL reemplaza `SmashOrPass`.

### Conclusión

El transplante SI esta escrito en 3 etapas, pero NO sigue exactamente las mismas 3 fases del plan maestro.

Eso significa:

- Fase 1 SI esta alineada.
- Fase 2 NO esta alineada:
  - plan maestro = posiciones + roles
  - guide de transplante = tuning/inyeccion de parametros de decision
- Fase 3 NO esta alineada:
  - plan maestro = positions + roles + skills / estrategia completa
  - guide de transplante = reemplazo de `SmashOrPass`

## Decision recomendada de gobernanza

Para evitar desvio de scope, congelar esta jerarquia:

1. Documento canonico del proyecto:
   - `MAPPO_Plan_3Fases.md`
2. Documento canonico de arranque:
   - `FASE1_DOCUMENTATION_SEMAFORO.md`
3. Documento tecnico auxiliar de transplante:
   - `RL_TRANSPLANT_GUIDE_COMPLETE.md`

Regla operativa:

- `RL_TRANSPLANT_GUIDE_COMPLETE.md` debe leerse como una guia tecnica de integracion vertical sobre `PlayBall`, no como la definicion canonica de las fases del proyecto.
- Si se quiere que el transplante "vaya por las 3 fases" canonicas, luego habra que reetiquetar o reestructurar ese guide para que:
  - su Fase 2 corresponda a posiciones + roles
  - su Fase 3 corresponda a positions + roles + skills

## Luz verde a produccion

### Luz verde real

Hay luz verde para:

- producir Fase 1
- montar parser core
- montar environment
- entrenar policy Fase 1
- empezar integracion C++/Nao para Fase 1

### Aun no hay luz verde total

No hay luz verde total para declarar desde hoy:

- "transplante 3 fases listo"
- "roadmap documental completamente alineado para Fase 2 y Fase 3"

## Calendario de 2 meses

## Supuesto base

Este calendario usa dos carriles:

- Carril ideal:
  - en 8 semanas, Fase 3 operativa en un Nao como demo funcional de laboratorio
  - NO implica robustez de torneo ni estabilidad final
- Carril realista:
  - en 8 semanas, Fase 2 operativa como meta alcanzable
  - Fase 3 quedaria como spike o prototipo parcial

## Semana 1: 2026-03-23 a 2026-03-29

Objetivo:
- congelar documentalmente Fase 1 como baseline
- arrancar implementacion del parser core

Entregables:
- `cfg_tokenizer.py`
- `cfg_parser.py`
- `strategy_parser.py`
- `tactic_parser.py`
- `walk_parser.py`
- `load_phase1()`

Gate:
- parse correcto de `s4v4_scaled.cfg`, `t121_4v4_scaled.cfg`, `t13_4v4_scaled.cfg`
- resolucion correcta de expresiones simbolicas

## Semana 2: 2026-03-30 a 2026-04-05

Objetivo:
- cerrar environment Fase 1 en Python

Entregables:
- observation space Fase 1
- action space `(x, y, theta)`
- limites de campo
- modelo de movimiento
- heuristica de portero intacta

Gate:
- episodios simulados completos sin crash
- validacion de bounds y orientacion

## Semana 3: 2026-04-06 a 2026-04-12

Objetivo:
- primer entrenamiento Fase 1

Entregables:
- pipeline MAPPO funcional
- checkpoints
- evaluacion contra baseline heuristico

Gate:
- training estable
- mejora visible frente a home poses estaticas o baseline minimo

## Semana 4: 2026-04-13 a 2026-04-19

Objetivo:
- puente C++/Python para Fase 1

Entregables:
- canal IPC/shared memory o mecanismo equivalente
- validacion C++ de posiciones RL
- reemplazo controlado de poses de tactica

Gate:
- replay offline o dry-run sincronizado entre Python y C++
- sin violar FSM ni skills existentes

## Semana 5: 2026-04-20 a 2026-04-26

Objetivo:
- primeras pruebas en Nao para Fase 1

Entregables:
- test de caminata a targets RL
- estabilidad de orientacion
- logging de latencia y divergencia sim2real

Gate:
- robot ejecuta posiciones RL sin romper navegacion
- latencia de inferencia e integracion dentro del presupuesto del hilo

## Semana 6: 2026-04-27 a 2026-05-03

Objetivo:
- bifurcacion de carriles

Carril realista:
- definir Fase 2 canonica como posiciones + roles
- agregar cabeza discreta de roles en Python
- mantener skills heuristicas

Carril ideal:
- ademas abrir el carril agresivo de decisiones `PlayBall`
- preparar `shootThreshold`, `minPassDistance`, `maxPassDistance`, `ratingThreshold`

Gate:
- decision formal de arquitectura:
  - o se sigue Fase 2 canonica del plan maestro
  - o se acepta que el guide de transplante es una subruta tecnica separada

## Semana 7: 2026-05-04 a 2026-05-10

Objetivo:
- consolidar Fase 2 realista

Carril realista:
- RL emite posiciones + roles
- constraint de striker unico
- evaluacion de estabilidad de roles

Carril ideal:
- RL ya toca decisiones vivas de `PlayBall`
- wiring de `KickAtGoal` y `PassToTeammate`

Gate:
- Fase 2 supera Fase 1 en al menos una metrica clara:
  - cobertura
  - estabilidad de asignacion
  - ventaja ofensiva

## Semana 8: 2026-05-11 a 2026-05-17

Objetivo:
- cierre del objetivo de 2 meses

Carril realista:
- Fase 2 operativa
- pruebas controladas en Nao o en pipeline muy cercano a Nao

Carril ideal:
- prototipo Fase 3 operativo en Nao para laboratorio
- fallback duro a heuristico cuando RL falle

Gate:
- realista:
  - Fase 2 usable y medible
- ideal:
  - Fase 3 ejecuta decisiones sin colapsar el sistema
  - fallback funciona

## Buffer final: 2026-05-18 a 2026-05-22

Objetivo:
- estabilizacion
- documentacion final
- decision de continuidad

Entregables:
- reporte comparativo Fase 1 vs Fase 2
- estado de Fase 3:
  - demo funcional
  - o spike inconcluso

## Resultado esperado al dia 2026-05-22

### Caso realista

Debe existir:

- Fase 1 desplegada y validada
- Fase 2 operativa
- Fase 3 apenas en prototipo o laboratorio

Este es el escenario que considero serio y defendible.

### Caso ideal

Podria existir:

- Fase 1 estable
- Fase 2 funcional
- Fase 3 operativa en un Nao como demo controlada

Pero solo si se cumplen TODAS estas condiciones:

- cero bloqueos fuertes en parser o bridge
- entrenamiento converge rapido
- latencia en Nao no rompe el hilo
- fallback a heuristico funciona desde el primer intento
- el equipo acepta una Fase 3 de laboratorio, no de competencia

## Riesgo principal del roadmap

El mayor riesgo NO es Fase 1.

El mayor riesgo es de definicion:

- hoy el plan maestro y el guide de transplante divergen en Fase 2 y Fase 3

Si eso no se alinea, pueden pasar dos cosas malas:

- entrenar una Fase 2 que no corresponde al canon del proyecto
- creer que Fase 3 ya esta lograda cuando solo se reemplazo `SmashOrPass`

## Recomendacion final

Decision recomendada:

1. Declarar GO VERDE para producir Fase 1 desde ya.
2. Declarar que la meta realista de 2 meses es Fase 2 operativa.
3. Tratar Fase 3 en 2 meses solo como objetivo ideal de laboratorio.
4. Antes de abrir Semana 6, congelar por escrito una de estas dos opciones:
   - opcion A: Fase 2 canonica = posiciones + roles
   - opcion B: camino alterno experimental = tuning `PlayBall`

Sin esa decision, el transplante no esta realmente alineado a las 3 fases canonicas.
