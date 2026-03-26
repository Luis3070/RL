# Guia de Implementacion: BhumanParser — SabanaHerons2026

> **Fecha:** 21 de Marzo de 2026
> **Objetivo:** Guiar la creacion de un BhumanParser funcional que cierre la brecha entre los archivos `.cfg` de BHuman y el entorno RL Python.
> **Restriccion:** El parser es el UNICO puente sim2real. Sin el, el RL entrena en un mundo que no corresponde al robot real.

> **Estado del documento para implementacion:**
> - Las secciones de parser, CFGs, dataclasses, `SkillBehaviorControl`, `kickInfo`, trazabilidad a archivos reales y validacion estructural describen informacion **verificada** contra el repo BHuman.
> - Las secciones con `rl_predict_*`, `action_space`, `Bridge`, rangos RL y funciones Python de decision describen una **especificacion propuesta** del bridge RL, no una implementacion existente.
> - Recomendacion de arquitectura para quien implemente: separar `parser core -> contract loader -> env/bridge RL`. El parser NO debe contener reward, curriculum, ni logica de takeover.
> - Orden canonico de implementacion: cerrar primero `Fase 1` (`cfg_parser` + `strategy_parser` + `tactic_parser` + `walk_parser` + `load_phase1()`), y extender a Fase 2/3 con cambios minimos sobre ese mismo core.

---

## Tabla de Contenidos

1. [Por que un Parser y no Hardcoding](#1-por-que-parser)
2. [Arquitectura del Parser](#2-arquitectura)
3. [Diagnostico del Parser Actual](#3-diagnostico)
4. [Especificacion del Formato InExprMapFile](#4-formato)
5. [Tabla de Simbolos 4v4 Scaled](#5-simbolos)
6. [Modulo 1: Parser Base de CFGs](#6-parser-base)
7. [Modulo 2: Strategy Parser (FSM)](#7-strategy)
8. [Modulo 3: Tactics Parser (Posiciones)](#8-tactics)
9. [Modulo 4: Actions Parser (Shoot/Pass)](#9-actions)
10. [Modulo 5: KickInfo Parser (Pateos)](#10-kickinfo)
11. [Modulo 6: Walk/Behavior Parser (Movimiento)](#11-walk)
12. [Modulo 7: Validacion y Roundtrip](#12-validacion)
13. [Integracion con environment.py y config.py](#13-integracion)
14. [Dataclasses de Salida](#14-dataclasses)
15. [Fallbacks y Manejo de Errores](#15-fallbacks)
16. [Testing del Parser](#16-testing)
17. [Checklist de Implementacion](#17-checklist)

---

<a name="1-por-que-parser"></a>
## 1. Por que un Parser y no Hardcoding

### El Problema Actual

El proyecto tiene **dos fuentes de verdad desconectadas** (no existe parser Python funcional):

```
config.py               environment.py
(valores 5v5)            (hardcoded arbitrario)
      │                         │
      └── No usa parser         └── Valores inventados
```

### Por que Parsear

1. **Los .cfg son la fuente de verdad.** Si alguien modifica `t121.cfg` para mover un defensor, el parser lo refleja automaticamente. Un hardcode se desincroniza.

2. **Expresiones simbolicas.** Las poses usan expresiones como `xPosOwnPenaltyArea + 500` que dependen de `fieldDimensions.cfg`. Si se cambia el tamano del campo, todas las poses cambian. Un hardcode tendria que recalcular a mano.

3. **Son pocos archivos** (~10 CFGs criticos). El parser es ~300-400 lineas de Python.

4. **Reutilizacion.** El parser sirve para cargar home poses, FSM de transiciones, set plays, kick info, y cualquier .cfg futuro.

5. **Trazabilidad.** Cada valor en el entorno RL tiene trazabilidad directa al .cfg original.

### El Objetivo

```
Config/Locations/4v4_Scaled/fieldDimensions.cfg ──┐
Config/Behavior/Strategies/s4v4_scaled.cfg ───────┤
Config/Behavior/Tactics/t121_4v4_scaled.cfg ──────┤
Config/Behavior/Tactics/t13_4v4_scaled.cfg ───────┤    BhumanParser
Config/Scenarios/4v4_Scaled/.../kickAtGoalImpl.cfg┼──→ (Python)
Config/Scenarios/4v4_Scaled/.../passToTeammate.cfg┤       │
Config/Scenarios/4v4_Scaled/.../zweikampfImpl.cfg ┤       ▼
Config/Scenarios/4v4_Scaled/kickInfo.cfg ─────────┤  AllConfigs
Config/Scenarios/4v4_Scaled/libWalkProvider.cfg ──┤  (dataclass)
Config/Robots/Default/walkingEngine.cfg ──────────┘
                                                         │
                                              ┌──────────┼──────────┐
                                              ▼          ▼          ▼
                                         config.py  environment.py  reward.py
                                         (FIELD)    (home poses,    (hard
                                                     FSM, roles)    constraints)
```

---

<a name="2-arquitectura"></a>
## 2. Arquitectura del Parser

### 2.1 Modulos

```
BhumanParser/
├── __init__.py                 # Exporta API publica
├── cfg_tokenizer.py            # Tokenizador del formato InExprMapFile (#, //, /**/)
├── cfg_parser.py               # Parser base: tokens → dict Python (con FIELD_SYMBOLS)
├── field_symbols.py            # Tabla de simbolos de fieldDimensions.cfg
├── strategy_parser.py          # Parsea s4v4_scaled.cfg → StrategyConfig
├── tactic_parser.py            # Parsea t*_4v4_scaled.cfg → TacticConfig
├── action_parser.py            # Parsea 3 action CFGs activos en partidos:
│                               #   kickAtGoalImpl → KickAtGoalConfig
│                               #   passToTeammateImpl → PassToTeammateConfig
│                               #   zweikampfImpl → ZweikampfConfig (6 subsistemas)
│                               # NOTA: demoGoToBallAndKickImpl.cfg NO se parsea —
│                               #   solo activo en LibDemo/AnyPlaceDemo, nunca en partidos 4v4.
├── kickinfo_parser.py          # Parsea kickInfo.cfg → Dict[str, KickInfo] (named map completo, 23 entradas)
├── walk_parser.py              # Parsea libWalkProvider + walkingEngine → WalkConfig
├── behavior_parser.py          # Parsea behaviorParameters.cfg → BehaviorParams
├── strategy_bc_parser.py       # Parsea strategyBehaviorControl.cfg → StrategyBehaviorControlConfig
├── skill_bc_parser.py          # Parsea skillBehaviorControl.cfg → SkillBehaviorControlConfig (gates/options de ejecucion)
├── dataclasses.py              # Todas las dataclasses (17 tipos)
├── validator.py                # Validacion de rangos y roundtrip
├── fallbacks.py                # Valores por defecto si .cfg no disponible
└── loader.py                   # load_all() → AllConfigs (12 grupos de CFG / 13 archivos contando ambas tacticas + 1 C++ header)
```

### 2.2 Flujo de Datos

```
.cfg file (texto)
    │
    ▼
cfg_tokenizer.py ── tokens (key=value, arrays, structs)
    │
    ▼
cfg_parser.py ── dict Python (con expresiones resueltas via field_symbols)
    │
    ▼
{strategy,tactic,action,kickinfo,walk}_parser.py ── dataclass tipada
    │
    ▼
validator.py ── validacion de rangos, tipos, consistencia
    │
    ▼
loader.py ── AllConfigs (struct con todo lo parseado)
    │
    ▼
environment.py / config.py (consumidores)
```

### 2.3 Principios de Diseno

1. **Single Source of Truth:** Los .cfg son la unica fuente. Todo lo demas se deriva.
2. **Fail-Loud:** Si un .cfg no parsea correctamente, ERROR explicito (no fallback silencioso).
3. **Fallback Explicito:** Si los .cfg no estan disponibles (desarrollo sin BHuman), usar fallbacks DOCUMENTADOS con warning.
4. **Float32:** Usar `numpy.float32` para consistencia con C++.
5. **Idempotente:** Parsear el mismo .cfg siempre da el mismo resultado.

### 2.4 Limite del Parser

El parser termina en `AllConfigs`.

NO le corresponde:

- decidir rewards
- decidir curriculum
- ejecutar takeover
- simular `shoot/pass/dribble/duel`
- reemplazar `SkillBehaviorControl`

Esas responsabilidades pertenecen a capas posteriores:

- `contract loader`: adapta `AllConfigs` a estructuras de runtime Python
- `environment/bridge`: usa ese contrato para entrenar y evaluar RL

---

<a name="3-diagnostico"></a>
## 3. Requisitos del Parser a Implementar

**Nota:** No existe ningun parser Python funcional en el repositorio actualmente. El parser debe construirse desde cero siguiendo esta guia.

### 3.1 Capacidades Requeridas

El parser debe soportar:

- Formato InExprMapFile de BHuman (tokenizer, expresiones aritmeticas, arrays, structs anidados)
- Comentarios C-style (`//` y `/* */`)
- Evaluacion de expresiones aritmeticas con simbolos de `fieldDimensions.cfg`
- Sufijo `deg` para angulos
- Funciones de alto nivel: `load_all_tactics()`, `load_strategy_fsm()`, `load_kick_info()`, etc.
- Fallback hardcodeado (con WARNING explicito) si los .cfg no estan disponibles
- Dataclasses tipadas como salida (no dicts crudos)
- Validacion de rangos para cada parametro parseado
- Hash de CFGs para trazabilidad training↔CFG version

### 3.2 Requisitos Funcionales

| Requisito | Prioridad | Detalle |
|-----------|-----------|---------|
| FIELD_SYMBOLS de 4v4 Scaled | CRITICO | Usar valores de `Config/Locations/4v4_Scaled/fieldDimensions.cfg` (2813/1875), NO Default 5v5 |
| Integracion con environment.py | CRITICO | El parser debe ser importado y usado por el entorno RL |
| Estrategia default "s4v4_scaled" | ALTO | La FSM carga `s4v4_scaled.cfg`, no `s5v5` |
| TACTIC_NAMES incluye variantes `_4v4_scaled` | ALTO | `t121_4v4_scaled` y `t13_4v4_scaled` deben estar en la lista |
| Fallback con tacticas scaled | ALTO | Si no hay .cfg, los fallbacks deben ser de tacticas 4v4 Scaled |
| Fuente unica de TACTIC_NAMES | MEDIO | Definido SOLO en el parser, importado por config.py (no duplicar) |
| Dataclasses tipadas | MEDIO | Retornar structs tipados, no dicts crudos |
| Validacion de rangos | MEDIO | Verificar que valores parseados esten en rangos esperados |
| Hash de CFGs | BAJO | Guardar hash determinista de los CFGs usados junto al modelo entrenado |

### 3.3 Plan de Implementacion

Construir el parser desde cero siguiendo la arquitectura modular descrita en la Seccion 2:
- Implementar parser base (tokenizer + evaluador de expresiones)
- Agregar parsers especializados por capa (strategy, tactics, actions, kickinfo, walk)
- Agregar dataclasses tipadas para cada tipo de configuracion
- Agregar validacion de rangos y consistency checks
- Integrar con environment.py y config.py
- Implementar fallbacks con valores 4v4 Scaled verificados

---

<a name="4-formato"></a>
## 4. Especificacion del Formato InExprMapFile

### 4.1 Sintaxis

Los archivos `.cfg` de BHuman usan un formato propietario llamado **InExprMapFile**:

```cfg
// Comentario de linea
/* Comentario de bloque */

// Key-value simple
paramName = value;

// Valor numerico con sufijo
angle = 6deg;              // 6 grados (convertir a float)

// Expresion aritmetica con simbolos
translation = {x = xPosOwnGoalArea - 94; y = 0;};

// Array
positions = [
    {type = goalkeeper; pose = {...};},
    {type = defenderL; pose = {...};},
];

// Struct anidado
pose = {rotation = 0deg; translation = {x = 500; y = 200;};};

// Enum sin comillas
kickType = forwardFastRight;

// Bool
lookAhead = false;

// Numeros cientificos
min = 1.17549e-38;
```

### 4.2 Reglas del Tokenizer

1. **Comentarios:** `//` hasta fin de linea, `/* */` bloque, **`#` hasta fin de linea** (usado en strategy .cfg)
2. **Identificadores:** `[a-zA-Z_][a-zA-Z0-9_]*`
3. **Numeros:** enteros, floats, notacion cientifica
4. **Sufijo `deg`:** `6deg` → `6.0` (almacenar en grados, convertir a radianes donde sea necesario)
5. **Strings:** enums sin comillas (ej: `forwardFastRight`)
6. **Separador:** `;` despues de cada valor
7. **Operadores:** `+`, `-`, `*`, `/` (en expresiones)
8. **Simbolos:** nombres que aparecen en FIELD_SYMBOLS se reemplazan por su valor numerico

### 4.3 Resolucion de Expresiones

```python
# Ejemplo de expresion en .cfg:
# x = xPosOwnGoalArea - 94;
# Con FIELD_SYMBOLS["xPosOwnGoalArea"] = -2438
# Resultado: x = -2438 - 94 = -2532

# Ejemplo con multiplicacion:
# x = centerCircleRadius * 1.5;
# Con FIELD_SYMBOLS["centerCircleRadius"] = 469
# Resultado: x = 469 * 1.5 = 703.5

# Ejemplo con multiplicacion y signo negativo:
# x = -centerCircleRadius * 0.5;
# Resultado: x = -(469) * 0.5 = -234.5
# CUIDADO: el '-' es unario sobre el simbolo, no sobre el resultado

# Ejemplo de negacion unaria de simbolo (fieldDimensions.cfg):
# xPosOwnPenaltyArea = -xPosOpponentPenaltyArea;
# Con FIELD_SYMBOLS["xPosOpponentPenaltyArea"] = 1781
# Resultado: xPosOwnPenaltyArea = -1781

# Ejemplo de referencia a otro simbolo (fieldDimensions.cfg):
# crossBarRadius = goalPostRadius;
# Con FIELD_SYMBOLS["goalPostRadius"] = 50
# Resultado: crossBarRadius = 50

# Ejemplo de IEEE float limits (kickAtGoalImpl.cfg):
# min = 1.17549e-38; max = 3.40282e+38;
# Resultado: min = float("1.17549e-38"), max = float("3.40282e+38")
```

---

<a name="5-simbolos"></a>
## 5. Tabla de Simbolos 4v4 Scaled

**Fuente:** `Config/Locations/4v4_Scaled/fieldDimensions.cfg`

```python
FIELD_SYMBOLS_4V4_SCALED: dict[str, float] = {
    # === Eje X (largo del campo: 5626mm total) ===
    "xPosOpponentFieldBorder":  3313.0,
    "xPosOpponentGoal":         3368.0,
    "xPosOpponentGoalPost":     2838.0,
    "xPosOpponentGroundLine":   2813.0,
    "xPosOpponentGoalArea":     2438.0,
    "xPosOpponentPenaltyMark":  2000.0,
    "xPosOpponentPenaltyArea":  1781.0,
    "xPosPenaltyStrikerStartPosition": 1781.0,
    "xPosHalfWayLine":          0.0,
    "xPosOwnPenaltyArea":      -1781.0,
    "xPosOwnPenaltyMark":      -2000.0,
    "xPosOwnGoalArea":         -2438.0,
    "xPosOwnGroundLine":       -2813.0,
    "xPosOwnGoalPost":         -2838.0,
    "xPosOwnGoal":             -3368.0,
    "xPosOwnFieldBorder":      -3313.0,

    # === Eje Y (ancho del campo: 3750mm total) ===
    "yPosLeftFieldBorder":      1960.0,
    "yPosLeftSideline":         1875.0,
    "yPosLeftPenaltyArea":      1250.0,
    "yPosLeftGoalArea":          688.0,
    "yPosLeftGoal":              800.0,
    "yPosRightGoal":            -800.0,
    "yPosRightGoalArea":        -688.0,
    "yPosRightPenaltyArea":    -1250.0,
    "yPosRightSideline":       -1875.0,
    "yPosRightFieldBorder":    -1960.0,

    # === Otros ===
    "centerCircleRadius":        469.0,
    "fieldLinesWidth":            50.0,
    "goalPostRadius":             50.0,
    "penaltyMarkSize":           100.0,
}
```

**CRITICO:** Estos valores REEMPLAZAN los del Default 5v5 que tiene el parser actual. El parser debe usar ESTOS y solo estos para el proyecto 4v4 Scaled.

**Nota sobre self-referencing:** El archivo `fieldDimensions.cfg` define simbolos que referencian a otros definidos anteriormente:
```cfg
xPosOpponentPenaltyArea = 1781;
xPosOwnPenaltyArea = -xPosOpponentPenaltyArea;   // = -1781
crossBarRadius = goalPostRadius;                   // = 50
```
Si se desea parsear `fieldDimensions.cfg` dinamicamente (en vez de hardcodear FIELD_SYMBOLS), el parser debe:
1. Parsear linea por linea, en orden secuencial
2. Agregar cada simbolo resuelto al dict de simbolos
3. Al encontrar `-identifier`, resolver como negacion unaria del simbolo
4. Al encontrar `identifier` sin operadores, resolver como copia del simbolo

Los simbolos hardcodeados en FIELD_SYMBOLS_4V4_SCALED ya tienen los valores finales resueltos, asi que el parseo dinamico de fieldDimensions es OPCIONAL. Pero si se implementa, debe respetar el orden secuencial.

---

<a name="6-parser-base"></a>
## 6. Modulo 1: Parser Base de CFGs

### 6.1 Responsabilidad

Tomar un archivo `.cfg` y convertirlo en un `dict[str, Any]` Python donde:
- Strings se quedan como strings
- Numeros se convierten a float/int
- Arrays se convierten a listas
- Structs se convierten a dicts anidados
- Expresiones se evaluan con FIELD_SYMBOLS
- Sufijo `deg` se almacena como float (grados)

### 6.2 Interfaz

```python
class BHumanCfgParser:
    """Parser del formato InExprMapFile de BHuman."""

    def __init__(self, symbols: dict[str, float] = None):
        """
        Args:
            symbols: Tabla de simbolos para resolver expresiones.
                     Default: FIELD_SYMBOLS_4V4_SCALED
        """

    def parse_file(self, filepath: str | Path) -> dict[str, Any]:
        """Parsear un .cfg completo y retornar dict."""

    def parse_string(self, text: str) -> dict[str, Any]:
        """Parsear texto .cfg directamente (util para tests)."""
```

### 6.3 Metodos Internos Necesarios

```python
def _skip_whitespace_and_comments(self) -> None
def _parse_map(self) -> dict          # key = value; key = value; ...
def _parse_identifier(self) -> str     # [a-zA-Z_][a-zA-Z0-9_]*
def _parse_value(self) -> Any          # dispatch: array, struct, expression
def _parse_array(self) -> list         # [ value, value, ... ]
def _parse_struct(self) -> dict        # { key = value; key = value; }
def _parse_expression(self) -> float | str | bool
    # Evaluar: numeros, simbolos, operaciones aritmeticas, bools, enums
def _evaluate_arithmetic(self, tokens: list) -> float
    # Resolver: symbol + 500, radius * 1.5, -symbol, etc.
```

### 6.3b Distincion: Array vs Struct vs Named Map

El parser debe distinguir tres patrones de valor compuesto:

```python
# 1. ARRAY: empieza con '[', contiene lista de valores
positions = [ {type = goalkeeper; ...}, {type = defenderL; ...} ];
# Resultado: list[dict]

# 2. STRUCT: empieza con '{', contiene key=value pairs
pose = {rotation = 0deg; translation = {x = 500; y = 200;};};
# Resultado: dict[str, Any]

# 3. NAMED MAP: una key cuyo valor es un struct donde cada sub-key
#    tiene a su vez un struct como valor. Usado en kickInfo.cfg:
kicks = {
  forwardFastRight = { kickType = forwardFastRight; range = {min = 1600; max = 4500;}; ... };
  forwardFastLeft  = { kickType = forwardFastLeft;  range = {min = 1600; max = 4500;}; ... };
};
# Resultado: dict[str, dict[str, Any]]  (mapa de nombre → propiedades)
```

La diferencia entre STRUCT y NAMED MAP es semantica, no sintactica — ambos son `{ key = value; }`. El parser base los trata igual como `dict`. Es el parser especializado (KickInfoParser) quien interpreta la estructura.

### 6.4 Casos Edge a Manejar

1. **`true`/`false`** → Python `True`/`False`
2. **`6deg`** → `6.0` (float, almacenar en grados)
3. **`1.17549e-38`** → `float("1.17549e-38")` (notacion cientifica)
4. **`none`** → string `"none"` (es un enum, no Python None)
5. **Expresion `xPosOwnGoalArea - 94`** → `-2438.0 - 94.0 = -2532.0`
6. **Expresion `centerCircleRadius * 0.5`** → `469.0 * 0.5 = 234.5`
7. **Identificadores sin valor** en arrays: `[forwardFastRight, walkForwardsLeft]` → `["forwardFastRight", "walkForwardsLeft"]`
8. **Array de structs:** `[{type = x; ...}, {type = y; ...}]`
9. **Trailing commas** en arrays/structs (permitidas por BHuman)
10. **Named struct maps:** `kicks = { forwardFastRight = {...}; forwardFastLeft = {...}; }` — NO es array, es dict de structs nombrados. Usado en `kickInfo.cfg`. La key del map es un identificador, no un indice numerico.
11. **Unary minus en simbolos:** `-xPosOpponentPenaltyArea` → `-1781.0`. Usado en `fieldDimensions.cfg` para definir valores simetricos como `xPosOwnPenaltyArea = -xPosOpponentPenaltyArea`. El tokenizer debe manejar `-` antes de un simbolo como operador unario.
12. **Asignacion simbolo-a-simbolo:** `crossBarRadius = goalPostRadius;` — el valor es otro simbolo ya definido (no una expresion aritmetica). El parser debe resolver buscando en el dict de simbolos ya parseados.
13. **Referencia circular en fieldDimensions:** Las definiciones en `fieldDimensions.cfg` se referencian entre si (ej: `xPosOwnPenaltyArea = -xPosOpponentPenaltyArea`). El parser de fieldDimensions debe parsear en ORDEN SECUENCIAL, agregando cada simbolo resuelto al dict antes de evaluar el siguiente.
14. **Comentarios `#`:** `s4v4_scaled.cfg` usa `#` como marcador de comentario de linea (ademas de `//`). El tokenizer debe aceptar ambos.

---

<a name="7-strategy"></a>
## 7. Modulo 2: Strategy Parser (FSM)

### 7.1 Input

`Config/Behavior/Strategies/s4v4_scaled.cfg`

**Estructura real del archivo (extracto critico):**
```cfg
# Comentario con hash (NO //)
tactics = [
  {
    tactic = t121_4v4_scaled;
    transitions = [
      {
        to = t13_4v4_scaled;
        conditions = [
          {
            numOfFieldPlayersGE = 3;
            ballXThreshold = 500;
            timeSinceBallAheadOfThresholdGE = 280;
          }
        ];
      }
    ];
  },
  {
    tactic = t13_4v4_scaled;
    transitions = [
      { to = t121_4v4_scaled; conditions = [{ numOfFieldPlayersLE = 2; }]; },
      { to = t121_4v4_scaled; conditions = [{
          numOfFieldPlayersGE = 3;
          ballXThreshold = -500;
          timeSinceBallBehindThresholdGE = 280;
      }]; }
    ];
  }
];
ownKickOffs = [ { type = attackKickOff_scaled; weight = 1; } ];
opponentKickOffs = [ { type = kiteKickOff_scaled; weight = 1; } ];
ownPenaltyKicks = [ { type = theOneTrueOwnPenaltyKick_scaled; weight = 1; } ];
opponentPenaltyKicks = [ { type = theOneTrueOpponentPenaltyKick_scaled; weight = 1; } ];
ownFreeKicks = [
  { type = ownCornerKick_scaled; weight = 1; },
  { type = ownGoalKick_scaled; weight = 1; }
];
opponentFreeKicks = [ { type = opponentCornerKick_scaled; weight = 1; } ];
```

**NOTA:** Este archivo usa comentarios `#` (no `//`). El tokenizer debe soportar ambos.

### 7.2 Output: StrategyConfig

```python
@dataclass
class TransitionCondition:
    numOfFieldPlayersGE: int | None = None
    numOfFieldPlayersLE: int | None = None
    ballXThreshold: float | None = None          # mm
    timeSinceBallAheadOfThresholdGE: int | None = None   # ms
    timeSinceBallBehindThresholdGE: int | None = None    # ms
    ownScoreGE: int | None = None
    opponentScoreGE: int | None = None
    scoreDifferenceGE: int | None = None
    scoreDifferenceLE: int | None = None

@dataclass
class Transition:
    to: str                                    # "t13_4v4_scaled"
    conditions: list[TransitionCondition]

@dataclass
class TacticState:
    tactic: str                                # "t121_4v4_scaled"
    transitions: list[Transition]

@dataclass
class StrategyConfig:
    name: str                                  # "s4v4_scaled"
    tactics: list[TacticState]
    own_kickoffs: list[dict]
    opponent_kickoffs: list[dict]
    own_penalty_kicks: list[dict]
    opponent_penalty_kicks: list[dict]
    own_free_kicks: list[dict]
    opponent_free_kicks: list[dict]
```

### 7.3 Logica de Parseo

```python
def parse_strategy(self, filepath: Path) -> StrategyConfig:
    raw = self.base_parser.parse_file(filepath)

    tactic_states = []
    for tactic_entry in raw.get("tactics", []):
        transitions = []
        for trans in tactic_entry.get("transitions", []):
            conditions = []
            for cond in trans.get("conditions", []):
                conditions.append(TransitionCondition(
                    numOfFieldPlayersGE=cond.get("numOfFieldPlayersGE"),
                    numOfFieldPlayersLE=cond.get("numOfFieldPlayersLE"),
                    ballXThreshold=cond.get("ballXThreshold"),
                    timeSinceBallAheadOfThresholdGE=cond.get("timeSinceBallAheadOfThresholdGE"),
                    timeSinceBallBehindThresholdGE=cond.get("timeSinceBallBehindThresholdGE"),
                    # ... demas campos
                ))
            transitions.append(Transition(to=trans["to"], conditions=conditions))
        tactic_states.append(TacticState(
            tactic=tactic_entry["tactic"],
            transitions=transitions,
        ))

    return StrategyConfig(
        name=filepath.stem,
        tactics=tactic_states,
        own_kickoffs=raw.get("ownKickOffs", []),
        opponent_kickoffs=raw.get("opponentKickOffs", []),
        own_penalty_kicks=raw.get("ownPenaltyKicks", []),
        opponent_penalty_kicks=raw.get("opponentPenaltyKicks", []),
        own_free_kicks=raw.get("ownFreeKicks", []),
        opponent_free_kicks=raw.get("opponentFreeKicks", []),
    )
```

### 7.4 Uso en environment.py

```python
# En lugar de: self._current_tactic = "t4"  (hardcoded)

strategy = parser.parse_strategy("s4v4_scaled.cfg")
self._current_tactic = strategy.tactics[0].tactic  # "t121_4v4_scaled" (estado inicial)
self._strategy_fsm = strategy  # Para evaluar transiciones

def _evaluate_tactic_transition(self, ball_x, num_field_players, time_ms):
    """Evaluar si debe cambiar de tactica segun FSM del .cfg"""
    current = self._find_tactic_state(self._current_tactic)
    for transition in current.transitions:
        if self._check_conditions(transition.conditions, ball_x, num_field_players, time_ms):
            self._current_tactic = transition.to
            return True
    return False
```

---

<a name="8-tactics"></a>
## 8. Modulo 3: Tactics Parser (Posiciones)

### 8.1 Input

- `Config/Behavior/Tactics/t121_4v4_scaled.cfg`
- `Config/Behavior/Tactics/t13_4v4_scaled.cfg`

**Estructura real de t121_4v4_scaled.cfg (completo):**
```cfg
positions = [
  {
    type = goalkeeper;
    pose = { rotation = 0deg; translation = {x = xPosOwnGoalArea - 94; y = 0;}; };
  },
  {
    type = defenderL;
    pose = { rotation = 0deg; translation = {x = xPosOwnPenaltyArea + 313; y = 0;}; };
  },
  {
    type = defenderR;
    pose = { rotation = 0deg; translation = {x = -centerCircleRadius * 0.5; y = -625;}; };
  },
  {
    type = forwardL;
    pose = { rotation = 0deg; translation = {x = -centerCircleRadius * 0.5; y = 625;}; };
  },
  {
    type = forwardR;
    pose = { rotation = 0deg; translation = {x = centerCircleRadius * 1.5; y = 0;}; };
  }
];
priorityGroups = [
  { positions = [defenderR]; priorities = [1]; },
  { positions = [forwardR];  priorities = [2]; },
  { positions = [forwardL];  priorities = [3]; },
  { positions = [defenderL]; priorities = [4]; }
];
```

**Valores resueltos (con FIELD_SYMBOLS_4V4_SCALED):**
- goalkeeper: (-2532, 0) — `xPosOwnGoalArea(-2438) - 94`
- defenderL: (-1468, 0) — `xPosOwnPenaltyArea(-1781) + 313`
- defenderR: (-234.5, -625) — `-centerCircleRadius(469) * 0.5`
- forwardL: (-234.5, 625) — `-centerCircleRadius(469) * 0.5`
- forwardR: (703.5, 0) — `centerCircleRadius(469) * 1.5`

### 8.2 Output: TacticConfig

```python
@dataclass
class PoseInfo:
    x: float        # mm (expresion resuelta)
    y: float        # mm
    rotation: float  # grados

@dataclass
class RolePosition:
    role_type: str   # "goalkeeper", "defenderL", "defenderR", "forwardL", "forwardR"
    pose: PoseInfo

@dataclass
class PriorityGroup:
    positions: list[str]   # ["defenderR"]
    priorities: list[int]  # [1]

@dataclass
class TacticConfig:
    name: str
    positions: list[RolePosition]
    priority_groups: list[PriorityGroup]

    def get_home_poses(self) -> dict[str, tuple[float, float]]:
        """Retorna {role_type: (x, y)} para uso directo en environment.py"""
        return {rp.role_type: (rp.pose.x, rp.pose.y) for rp in self.positions}

    def get_field_player_positions(self) -> dict[str, tuple[float, float]]:
        """Retorna solo field players (excluir goalkeeper)"""
        return {rp.role_type: (rp.pose.x, rp.pose.y)
                for rp in self.positions if rp.role_type != "goalkeeper"}
```

### 8.3 Logica de Parseo

```python
def parse_tactic(self, filepath: Path) -> TacticConfig:
    raw = self.base_parser.parse_file(filepath)

    positions = []
    for pos_entry in raw.get("positions", []):
        pose = pos_entry["pose"]
        translation = pose["translation"]
        positions.append(RolePosition(
            role_type=pos_entry["type"],
            pose=PoseInfo(
                x=float(translation["x"]),   # Ya resuelta por parser base
                y=float(translation["y"]),
                rotation=float(pose.get("rotation", 0)),
            )
        ))

    priority_groups = []
    for pg in raw.get("priorityGroups", []):
        priority_groups.append(PriorityGroup(
            positions=pg["positions"] if isinstance(pg["positions"], list) else [pg["positions"]],
            priorities=pg["priorities"] if isinstance(pg["priorities"], list) else [pg["priorities"]],
        ))

    return TacticConfig(
        name=filepath.stem,
        positions=positions,
        priority_groups=priority_groups,
    )
```

### 8.4 Uso en environment.py

```python
# En lugar de:
#   home_poses = {"agent_0": np.array([-2350, 0]), ...}  (hardcoded)

tactics = {
    "t121_4v4_scaled": parser.parse_tactic("t121_4v4_scaled.cfg"),
    "t13_4v4_scaled": parser.parse_tactic("t13_4v4_scaled.cfg"),
}

def _get_home_poses(self):
    tactic = self.tactics[self._current_tactic]
    return tactic.get_field_player_positions()
    # Retorna: {"defenderL": (-1468, 0), "defenderR": (-234.5, -625), ...}
```

---

<a name="9-actions"></a>
## 9. Modulo 4: Actions Parser (Shoot/Pass)

### 9.1 Input

- `Config/Scenarios/4v4_Scaled/BehaviorControl/kickAtGoalImpl.cfg`
- `Config/Scenarios/4v4_Scaled/BehaviorControl/passToTeammateImpl.cfg`
- `Config/Scenarios/4v4_Scaled/BehaviorControl/zweikampfImpl.cfg`
- ~~`demoGoToBallAndKickImpl.cfg`~~ — NO se parsea (solo LibDemo/AnyPlaceDemo, inactivo en 4v4)

### 9.2 Output: ActionConfig

```python
@dataclass
class KickAtGoalConfig:
    hysteresis_number: int           # ms (300)
    hysteresis_kick_range_ext: float # mm (500)
    hysteresis_angle: float          # deg (10)
    min_opening_angle: float         # deg (6)
    kick_inaccuracy: float           # deg (10)
    hysteresis_decision_penalty: int # ms (500)
    allowed_kicks: list[str]
    allowed_kick_ranges: list[dict]  # [{min, max}]

@dataclass
class PassToTeammateConfig:
    """`PassToTeammateImpl` carga 28 parametros top-level en C++.
    Esta dataclass los aplana a mas campos Python cuando el tipo real es
    compuesto (`Vector2f` -> x/y, `Rangef` -> min/max)."""
    # --- Kicks aprendibles (Fase 2) ---
    min_kick_length_for_long: float  # mm (2813)
    rating_threshold: float          # 0-1 (0.7) — umbral para intentar pase
    pass_ahead_distance: float       # mm (0)
    min_angle_offset: float          # deg (7)
    max_angle_offset: float          # deg (25)
    max_angle_offset_behind: float   # deg (10)
    max_precision_offset: float      # deg (5)
    min_time_waiting: int            # ms (3000)
    max_time_waiting: int            # ms (8000)
    max_time_waiting_factor: float   # factor multiplicativo MAXIMO de espera (0.1)
    time_left_to_adjust: int         # ms (7000)
    ignore_obstacles_threshold: float     # mm (250)
    ignore_dynamic_obstacles_threshold: float  # mm (500)
    do_special_kick_checks: bool          # true
    kick_range_offset: float              # mm (313)
    average_kick_execution_time: int      # ms (800)
    average_kick_ball_offset_x: float     # mm (-185) — offset del balon al ejecutar kick
    average_kick_ball_offset_y: float     # mm (0)
    waiting_position_offset_x: float      # mm (-94) — offset de posicion de espera
    waiting_position_offset_y: float      # mm (0)
    target_offset_to_field_border: float  # mm (500)
    interception_distance_threshold_min: float  # mm (100)
    interception_distance_threshold_max: float  # mm (400)
    target_distance_scale_min: float      # mm (0)
    target_distance_scale_max: float      # mm (2500)
    min_rating: float                     # (0.01)
    look_ahead: bool                      # false en 4v4 actual; SI implementado en PassToTeammate.cpp
    max_look_ahead_time: int              # ms (3000) — usado solo si look_ahead=true
    # --- Head tracking (sim2real context, NO aprendible) ---
    max_head_angle: float                 # deg (60) — robot gira cabeza para ver receptor de pase
    head_speed: float                     # deg/s (60)
    head_tilt: float                      # deg (15)
    # --- Kicks validos ---
    allowed_kicks: list[str]

@dataclass
class SmashOrPassParams:
    """Parametros de PlayBall.h para la decision shoot vs pass.
    FUENTE: C++ default (PlayBall.h:L16-34), NO hay .cfg override en 4v4_Scaled."""
    shoot_threshold: float = 0.8          # L20
    min_pass_distance: float = 1500.0     # L22 mm
    max_pass_distance: float = 4500.0     # L23 mm
    max_free_kick_distance: float = 6000.0 # L24 mm (HARD)
    max_goal_kick_distance: float = 9000.0 # L25 mm (HARD)
    min_penalty: float = 0.1              # L26
    max_penalty: float = 0.8              # L27
    ball_dist_min_penalty: float = 300.0  # L28 (HARD)
    ball_dist_max_penalty: float = 200.0  # L29 (HARD)
    min_rating: float = 0.01             # L30 (HARD)
    sample_distance: float = 500.0        # L33 mm
    sample_steps_min: int = -1            # L34
    sample_steps_max: int = 2             # L34

@dataclass
class ZweikampfSearchParams:
    """Sub-sistema 1: busqueda de angulo optimo de kick."""
    num_angles_near_best: int = 3
    range_of_best_duel_pose: float = 10.0            # deg
    num_overall_search: int = 5
    overall_search_range: float = 180.0               # deg
    bonus_steal_small: float = 0.01                  # deg
    bonus_steal_big: float = 3.0                     # deg
    more_search_after_doing_nothing: int = 50
    goal_sector_width: float = 1.0                   # deg
    min_max_angle_range_min: float = 4.0             # deg
    min_max_angle_range_max: float = 10.0            # deg

@dataclass
class ZweikampfDuelRatings:
    """Sub-sistema 2: puntuacion de candidatos de kick (negativo = mejor)."""
    rating_min_max_difference: float = 0.3
    rating_ball_lands_in_own_half: float = 0.2
    rating_pose_blocked_small_kick_angle: float = 0.3
    rating_pose_blocked: float = 1.0
    rating_opponent_faster: float = 0.1
    rating_same_kick: float = -0.1
    rating_same_kick_angle: float = -0.1
    rating_steal_ball: float = -0.4
    goal_area_long_kick_rating: float = -0.5
    rating_steal_ball_kick: float = -0.9
    rating_steal_ball_kick_better_side: float = -0.05
    rating_goal_shot: float = -1.5

@dataclass
class ZweikampfTimings:
    """Sub-sistema 3: hysteresis y cooldowns de timing."""
    max_time_doing_nothing: int = 1000           # ms — tras este tiempo sin kick valido, se desactiva
    deactive_after_doing_nothing: int = 5000     # ms — espera antes de reactivarse
    sideward_restriction_time: int = 4000        # ms
    no_kick_min_time: int = 200                  # ms — cooldown entre kicks normales
    no_kick_steal_min_time: int = 250            # ms — cooldown entre steal kicks
    ignore_skill_request_time: int = 2000        # ms

@dataclass
class ZweikampfObstacleHandling:
    """Sub-sistema 4: geometria de pie y manejo de obstaculos."""
    left_foot_edge: tuple[float, float] = (110.0, 100.0)
    right_foot_edge: tuple[float, float] = (110.0, -100.0)
    left_foot_kick_edge: tuple[float, float] = (170.0, 100.0)
    right_foot_kick_edge: tuple[float, float] = (170.0, -100.0)
    max_obstacle_dist_for_walk_steal: float = 100.0    # mm
    max_obstacle_dist_for_side_steal: float = 450.0    # mm
    obstacle_dist_for_blocking_pose: float = 100.0     # mm
    obstacle_blocking_radius: float = 200.0            # mm
    obstacle_blocking_radius_sidewards_increase: float = 30.0  # mm
    no_kick_blocking_pose: float = 180.0               # mm
    obstacle_min_dist_for_back_pass: float = 600.0     # mm
    obstacle_shift_range_min: float = -100.0
    obstacle_shift_range_max: float = 30.0
    max_obstacle_dist_to_ball_for_risky_kicks: float = 600.0   # mm
    max_obstacle_dist_to_force_forward_steal: float = 200.0    # mm

@dataclass
class ZweikampfStealBallParams:
    """Sub-sistema 5: parametros del forwardSteal (robo de balon)."""
    steal_ball_type_too_much_rotation_min: float = 25.0    # deg
    bonus_for_steal_ball_min_max_angle: float = 120.0      # deg
    steal_ball_range_min: float = 88.0                     # deg
    steal_ball_range_max: float = 95.0                     # deg
    steal_ball_min_robot_rotation: float = 70.0            # deg
    steal_ball_min_position_rotation: float = 45.0         # deg
    forward_steal_opponent_normal_min: float = 50.0        # deg
    forward_steal_opponent_normal_max: float = 70.0        # deg
    forward_steal_opponent_malus: float = 40.0             # deg

@dataclass
class ZweikampfLastKickHysterese:
    """Sub-sistema 6: hysteresis del ultimo kick seleccionado."""
    kick_length_hysterese: float = 500.0    # mm — si nuevo rango difiere < 500mm, mantener anterior
    kick_direction_hysterese: float = 20.0  # deg — si nueva direccion difiere < 20deg, mantener anterior

@dataclass
class ZweikampfConfig:
    """Parametros COMPLETOS de zweikampfImpl.cfg — sistema de duelo, dribble y pateo en presion.
    FUENTE: Config/Scenarios/4v4_Scaled/BehaviorControl/zweikampfImpl.cfg
    NOTA: Este es el skill ACTIVO en partidos 4v4. NO confundir con demoGoToBallAndKickImpl.cfg
    que solo se activa en modo exhibicion (LibDemo/AnyPlaceDemo), nunca durante un partido.
    Zweikampf define TODA la logica de kick bajo presion: busqueda, puntuacion, timing,
    geometria de pie, robo de balon e hysteresis."""
    # Sub-sistemas (ver dataclasses individuales)
    search: ZweikampfSearchParams = None
    ratings: ZweikampfDuelRatings = None
    timings: ZweikampfTimings = None
    obstacle_handling: ZweikampfObstacleHandling = None
    steal_ball: ZweikampfStealBallParams = None
    last_kick_hysterese: ZweikampfLastKickHysterese = None
    # Parametros sueltos
    smaller_kick_angle_preference: float = 15.0    # deg
    forward_steal_preference_range: float = 10.0   # deg
    replace_normal_to_long_goal_shot: float = 10.0 # deg
    sector_wheel_obstacle_ball_radius_factor: float = 3.0
    allow_pass_angle_min: float = -45.0            # deg
    allow_pass_angle_max: float = 45.0             # deg
    field_border_safe_distance: float = 625.0      # mm — constraint real de approach (NOT demo cfg)
    min_forward_turn_range: float = 625.0          # mm
    sidewards_rotation_correction_bonus: float = 5.0  # deg
    ball_position_interpolation_range_min: float = 250.0
    ball_position_interpolation_range_max: float = 500.0
    look_active_min_ball_distance: float = 500.0   # mm
    replace_forward_with_long_goal_shot: bool = True
    # Allowed kicks en duelo (8 tipos activos en 4v4)
    allowed_kicks: list[str] = None
    # ["walkForwardsLeft", "walkForwardsRight", "walkForwardsLeftLong",
    #  "walkForwardsRightLong", "walkForwardsRightAlternative", "walkForwardsLeftAlternative",
    #  "walkForwardStealBallLeft", "walkForwardStealBallRight"]

    def __post_init__(self):
        if self.search is None:
            self.search = ZweikampfSearchParams()
        if self.ratings is None:
            self.ratings = ZweikampfDuelRatings()
        if self.timings is None:
            self.timings = ZweikampfTimings()
        if self.obstacle_handling is None:
            self.obstacle_handling = ZweikampfObstacleHandling()
        if self.steal_ball is None:
            self.steal_ball = ZweikampfStealBallParams()
        if self.last_kick_hysterese is None:
            self.last_kick_hysterese = ZweikampfLastKickHysterese()
        if self.allowed_kicks is None:
            self.allowed_kicks = [
                "walkForwardsLeft", "walkForwardsRight",
                "walkForwardsLeftLong", "walkForwardsRightLong",
                "walkForwardsRightAlternative", "walkForwardsLeftAlternative",
                "walkForwardStealBallLeft", "walkForwardStealBallRight",
            ]

# ELIMINADO: DemoGoToBallAndKickConfig — demoGoToBallAndKickImpl.cfg no se parsea.
# Este cfg solo se activa en LibDemo::soccer (AnyPlaceDemo), que NO figura en
# skillBehaviorControl.cfg del escenario 4v4_Scaled. El cfg demo no tiene rol en partidos.
# Los constraints de approach timing reales provienen de ZweikampfTimings
# (noKickMinTime=200ms, deactiveAfterDoingNothing=5000ms).
```

### 9.3 OBLIGATORIO: Puntos de Decision C++ que el Parser/Bridge DEBE Cubrir

> **CONTRATO DE IMPLEMENTACION:** Esta seccion NO es informativa — es una OBLIGACION. Cada punto listado aqui DEBE tener un campo correspondiente en `AllConfigs` (Seccion 13.1) y una entrada en el Bridge Contract (Seccion 9.4). Si falta alguno, el bridge esta ROTO y el transplante sim2real FALLA.

El parser no solo consume CFGs — tambien debe conocer los **puntos exactos del C++ donde se toman decisiones**, porque esos son los puntos donde RL inyecta valores en Fase 2/3. Estos puntos definen que parametros el bridge debe exponer al RL.

**Archivo: `PlayBall.cpp` (Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/)**

| Linea | Punto de Decision | Parametro(s) | Fuente del Default | Clasificacion |
|-------|-------------------|-------------|-------------------|---------------|
| L44 | `theExpectedGoals.getRating(ballPos)` | xG rating | ExpectedGoalsProvider (modelo neural) | Observable (no aprendible directamente) |
| L56 | `if(maxValue > p.shootThreshold)` | `shootThreshold` | PlayBall.h:20 → C++ default 0.8f | **Aprendible (Fase 2)** |
| L70-72 | `passDistance < p.minPassDistance \|\| > maxPassDistance` | `minPassDistance`, `maxPassDistance` | PlayBall.h:22-23 → C++ defaults 1500/4500 | **Aprendible (Fase 2)** |
| L77 | `agentGoalLine(..., p.sampleDistance)` | `sampleDistance` | PlayBall.h:33 → C++ default 500 | **Aprendible (Fase 2)** |
| L78 | `for(t = p.sampleSteps.min; ...)` | `sampleSteps` | PlayBall.h:34 → C++ default (-1, 2) | **Aprendible (Fase 2)** |
| L82-84 | `sampleValue = PassEval × xG × oppRating` | rating compuesto | PassEvaluationProvider + ExpectedGoalsProvider | **Reemplazable por RL (Fase 2/3)** |
| L95-98 | `if(applyDecisionPenalty) agentValue -= penalty` | `minPenalty`, `maxPenalty` | PlayBall.h:26-27 → C++ defaults 0.1/0.8 | **Aprendible (Fase 2)** |

**Archivo: `PlayBall.h` (parametros completos L16-34)**

| Linea | Parametro | Default C++ | Clasificacion |
|-------|-----------|------------|---------------|
| L18 | `alwaysShoot` | false | Hard constraint (debug only) |
| L19 | `ignoreObstacles` | false | Hard constraint (legacy switch) |
| L20 | `shootThreshold` | 0.8f | **Aprendible** |
| L22 | `minPassDistance` | 1500.f | **Aprendible** |
| L23 | `maxPassDistance` | 4500.f | **Aprendible** |
| L24 | `maxFreeKickDistance` | 6000.f | Hard constraint (regla) |
| L25 | `maxGoalKickDistance` | 9000.f | Hard constraint (regla) |
| L26 | `minPenalty` | 0.1f | **Aprendible** |
| L27 | `maxPenalty` | 0.8f | **Aprendible** |
| L28 | `ballDistMinPenalty` | 300.f | Hard constraint |
| L29 | `ballDistMaxPenalty` | 200.f | Hard constraint |
| L30 | `minRating` | 0.01f | Hard constraint |
| L31 | `lookAhead` | true | **FLAG MUERTO** — declarado pero sin `if(p.lookAhead)` en PlayBall.cpp. TODO nunca completado. No es constraint ni aprendible. |
| L32 | `lookAheadTime` | 2000 | **FLAG MUERTO** — idem L31. |
| L33 | `sampleDistance` | 500.f | **Aprendible** |
| L34 | `sampleSteps` | (-1, 2) | **Aprendible** |

### 9.3b OBLIGATORIO: Flujo runtime de partido que el bridge debe conocer

Estos archivos no todos se parsean, pero SI deben aparecer en la documentacion del bridge porque son quienes consumen el `SkillRequest` y determinan que skill se ejecuta realmente en partido:

1. `StrategyBehaviorControl.cpp` llama `theBehavior.update(...)` y produce `SkillRequest`
2. `Behavior.cpp` fija tactica, roles y `determineActiveAgent()`
3. `SkillBehaviorControl.cpp` ejecuta `PlaySoccer()`
4. `PlaySoccer.h` hace `select_option(options)`
5. `HandleGameState.h` decide si una `playingOption` intercepta el flujo; si no, llama `executeRequest()`
6. `SkillBehaviorControl.cpp::executeRequest()` enruta `shoot/pass/dribble` hacia `thePlayBallSkill()`
7. `SkillBehaviorControl/Skills/PlayBall.cpp` llama segun el caso a `KickAtGoal`, `PassToTeammate`, `GoToBallAndDribble`, `HandleBallAtOwnGoalPost` o `Zweikampf`

**Implicacion para el parser/RL:**
- `skillBehaviorControl.cfg` no es un output del RL
- pero SI forma parte del contrato de ejecucion y debe entrar a `AllConfigs` como referencia/gating
- `demoGoToBallAndKickImpl.cfg` queda fuera del parser de match-play porque no entra en este flujo

**Si el environment quiere modelar `dribble` con rigor, tambien debe auditar estos archivos runtime:**
- `SkillBehaviorControl/Skills/Ball/GoToBallAndDribble.cpp`
- `SkillBehaviorControl/Skills/DribbleToGoal.cpp`
- `SkillBehaviorControl/Skills/Output/MotionRequest/Dribble.cpp`
- `Representations/MotionControl/MotionRequest.h`

### 9.3c OBLIGATORIO: Senales BHuman para `handoff_ready` y `takeover_active` del bridge

> **DECISION DE ARQUITECTURA ADOPTADA:** Para maximizar sim2real, el bridge debe usar **oracle temprano**.
> El takeover comienza **apenas BHuman asigna `playBall`** al agente activo en `PLAYING`.
> NO se espera a una heuristica Python de proximidad fina al balon.

#### Senal de entrada al macroestado (`handoff_ready`)

La senal primaria debe venir del propio flujo de BHuman:

- `GameState.isPlaying()` activo
- el agente fue asignado al rol activo `playBall` por `Behavior::determineActiveAgent()`
- el flujo runtime entra en la cadena `SkillRequest -> SkillBehaviorControl -> PlayBall`

En otras palabras:

- **senal BHuman de handoff:** `playBall`

Esto debe tratarse en el bridge como la frontera donde RL deja de controlar al agente activo y BHuman pasa a ser el oracle ejecutor.

#### Senal de takeover en ejecucion (`takeover_active`)

El bridge no debe inventar un flag.
Debe derivarlo de evidencia aguas abajo ya producida por BHuman, por ejemplo:

- `SkillRequest.skill in {shoot, pass, dribble}`
- `SkillBehaviorControl::executeRequest()` enruta a `thePlayBallSkill()`
- `BehaviorStatus.passTarget` o `BehaviorStatus.shootingTo` estan activos
- `MotionRequest.motion in {walkToBallAndKick, dribble}`

En otras palabras:

- **senal BHuman de takeover:** evidencia de ejecucion real aguas abajo

#### Terminacion del macroestado

El macroestado termina cuando BHuman deja de sostener esa cadena de control. El bridge debe considerar terminado el takeover si ocurre cualquiera de estas condiciones:

- el agente deja de ser `playBall`
- `GameState` deja de estar en `PLAYING`
- el flujo deja de emitir evidencia aguas abajo de ejecucion (`SkillRequest`, `BehaviorStatus`, `MotionRequest`)
- cambia el agente activo

En otras palabras:

- **fin del takeover:** BHuman deja de emitir `playBall` para ese agente o desaparece la evidencia de ejecucion aguas abajo

#### Histeresis obligatoria para `handoff_ready` y `takeover_active`

> **DECISION OPERATIVA CERRADA:** La arquitectura sigue siendo de **oracle temprano**.
> La histeresis NO redefine el takeover.
> Solo evita flicker en flags derivados por frame.

Para no contradecir el contrato ya fijado:

- `handoff_ready` sigue siendo una senal **oracle-driven** basada en `PLAYING + playBall`
- `takeover_active` sigue siendo una senal **evidence-driven** basada en `SkillRequest` / `BehaviorStatus` / `MotionRequest`
- la envelope geometrica del curriculum NO gatea `handoff_ready`; debe registrarse aparte como calidad del handoff

##### Definiciones raw

- `handoff_ready_raw`:
  - `game.phase == playing`
  - `bhuman_oracle.active_agent_player_number == game.player_number`
  - `bhuman_oracle.active_role == playBall`
  - el flujo no esta interceptado por set play / `HandleGameState`

- `takeover_evidence_raw`:
  - `bhuman_oracle.skill_request.skill in {shoot, pass, dribble, duel}`
  - o `bhuman_oracle.motion_request.motion in {walkToBallAndKick, dribble}`
  - o `bhuman_oracle.behavior_status.pass_target_player_number != null`
  - o `bhuman_oracle.behavior_status.shooting_to != null`

##### Tabla exacta de histeresis

| Senal | Entrada | Permanencia | Salida soft | Salida hard | Timeout |
|------|---------|-------------|-------------|-------------|---------|
| `handoff_ready` | entra en el **primer frame** donde `handoff_ready_raw == true` | se mantiene mientras `handoff_ready_raw == true` | no aplica | sale **inmediato** si `phase != playing`, cambia `active_agent`, `active_role != playBall`, o un set play intercepta el flujo | no aplica |
| `takeover_active` | entra en el **primer frame** donde `handoff_ready == true` y `takeover_evidence_raw == true` | se mantiene si sigue habiendo evidencia, o si solo falta evidencia por **1 frame de gracia** | termina si hay **2 frames consecutivos** sin evidencia aguas abajo | termina **inmediato** si `phase != playing`, cambia `active_agent`, `active_role != playBall`, `integrity.frame_valid == false`, o el bridge entra en fallback | `macro_timeout_frames = 90` frames de Cognition (`~3.0 s` si `dt ~= 33 ms`) |

##### Variables derivadas recomendadas del bridge

- `handoff_quality_ok`: bool derivado de envelope geometrica del curriculum
- `takeover_evidence_missing_frames`: contador entero
- `bridge_fallback_active`: bool
- `bridge_fallback_reason`: enum string

##### Envelope geometrica: uso correcto

La envelope sugerida por el curriculum:

- `ball_seen_age <= 300 ms`
- `ball_rel_x in [0, 500] mm`
- `abs(ball_rel_y) <= 300 mm`
- `abs(angle_to_ball) <= 80 deg`
- margen minimo a borde de campo

NO decide por si sola la toma de control.
Debe usarse para:

- shaping de reward
- metricas de calidad de handoff
- auditoria del approach del agente

##### Razon del valor `90` frames

Se fija un timeout inicial conservador de **90 frames** porque:

- respeta granularidad base de `1 frame de Cognition`
- cubre macro-acciones reales cortas y medianas sin fast-forward artificial
- da una frontera de seguridad clara para auditoria y fallback

Si una skill o escenario exige un timeout mas fino, se puede especializar por `macro_type`, pero el valor global por defecto queda congelado en `90`.

#### Tabla exacta de terminacion, exito, fallo, timeout y aborto

| Caso | Condicion exacta | `macro_end_reason` | `macro_success` | Efecto operativo |
|------|-------------------|--------------------|-----------------|------------------|
| Exito por disparo | `events.goal_scored == true` y `macro_owner_player_number == active_agent_player_number` | `completed` | `true` | cerrar macro, registrar exito de tiro |
| Exito por pase | `events.pass_completed == true` | `completed` | `true` | cerrar macro, registrar exito de pase |
| Exito por duelo | `events.duel_won == true` y el equipo mantiene posesion | `completed` | `true` | cerrar macro, registrar exito de duelo |
| Exito por dribble | la macro termina sin perdida y con progreso territorial o posesion retenida | `completed` | `true` | cerrar macro, registrar exito de avance |
| Fin neutro por limpieza de request | desaparece la evidencia aguas abajo por 2 frames y no hubo evento terminal fuerte | `skill_request_cleared` o `motion_request_cleared` | `false` | devolver control al RL |
| Fallo por cambio de rol | `active_role != playBall` antes de completar la macro | `role_changed` | `false` | terminar takeover |
| Fallo por cambio de agente | cambia `active_agent_player_number` | `active_agent_changed` | `false` | terminar takeover |
| Fallo por estado de juego | `game.phase != playing` | `game_state_changed` | `false` | takeover cancelado por flujo del partido |
| Fallo por perdida del balon | `ball_model.ball_position_consistent_with_game_state == false` o posesion rival tras la macro | `ball_lost` | `false` | takeover fallido |
| Fallo por balon fuera | `events.ball_out == true` | `out` | `false` | takeover termina por salida |
| Timeout | `macro_duration_frames >= 90` | `timeout` | `false` | cerrar macro y activar fallback seguro del bridge |
| Aborto por integridad | `integrity.frame_valid == false` o frame incompleto | `aborted_integrity` | `false` | no confiar en el frame, fallback inmediato |
| Aborto por bridge | `bridge_fallback_active == true` | `aborted_bridge_fallback` | `false` | ceder control completo a BHuman |

> **Nota de contrato:** `aborted_integrity` y `aborted_bridge_fallback` deben tratarse como abortos explicitos, no como completions neutrales.

#### Implicacion para el entrenamiento

Con esta decision de `oracle temprano`:

- el agente activo entra a takeover apenas BHuman toma el control de `playBall`
- durante takeover se enmascara la `actor loss` del agente activo
- el critic sigue normal
- los companeros no activos siguen entrenando normal

#### Implicacion para el bridge C++/Python

El bridge debe exportar al menos estas senales del contrato BHuman:

- `oracle_active_agent`
- `oracle_role`
- `oracle_skill_request`
- `oracle_motion_request`
- `oracle_pass_target`
- `oracle_shoot_target`
- `oracle_takeover_active`

Estas variables NO son nuevas decisiones del RL.
Son el reflejo Python del contrato ya existente en BHuman.

#### Granularidad temporal adoptada: esquema hibrido

> **DECISION FINAL:** El bridge usa un esquema **hibrido**.
>
> - **step base del entorno = 1 frame de Cognition**
> - **macroestado = agregacion de esos frames mientras el agente siga bajo `playBall` y exista evidencia aguas abajo de ejecucion**

Esto significa:

- Python `env.step()` llama al oracle una vez por frame de Cognition
- el bridge devuelve el **siguiente estado completo** del mundo, no solo deltas
- en paralelo, el bridge mantiene metadatos de macroestado para metricas y reward agregada

Campos minimos del macroestado agregado:

- `macro_active`
- `macro_type`
- `macro_start_frame`
- `macro_duration_frames`
- `macro_end_reason`
- `macro_success`

#### Salida recomendada del oracle por frame

Para evitar ambiguedad, la salida recomendada del bridge es el **estado siguiente completo**:

- poses de agentes
- velocidades o estado de locomocion relevante
- pose y velocidad del balon
- posesion
- agente activo
- rol activo
- `SkillRequest`
- `MotionRequest`
- `takeover_active`
- estado agregado del macroestado

No se recomienda devolver solo `delta_x`, `delta_y`, etc., porque eso complica la reconstruccion del estado y aumenta el riesgo de desalineacion sim2real.

#### Schema JSON final recomendado del bridge

> **IMPORTANTE:** Este JSON NO describe una representacion nativa unica de BHuman.
> Describe la **salida estructurada final del bridge** hacia Python.
>
> Mezcla tres tipos de campos:
> - campos nativos o directamente derivables del repo BHuman
> - campos estructurados por el bridge para Python
> - campos agregados de macroestado para reward, auditoria y visualizacion

```json
{
  "schema_name": "BHumanOracleStepOutput",
    "schema_version": "1.1.1",
  "temporal_contract": {
    "step_base": "1_cognition_frame",
    "macro_tracking": "enabled",
    "state_return_mode": "next_full_state"
  },
  "frame": {
    "frame_id": "int",
    "timestamp_ms": "int",
    "dt_ms": "int"
  },
  "game": {
    "phase": "initial|standby|ready|set|playing|finished",
    "player_number": "int",
    "team_is_kicking": "bool|null",
    "set_play": "none|kickoff|free_kick|goal_kick|corner_kick|penalty_kick|other",
    "is_penalty_shootout": "bool"
  },
  "strategy_status": {
    "proposed_tactic": "string|null",
    "accepted_tactic": "string|null",
    "proposed_mirror": "bool|null",
    "accepted_mirror": "bool|null",
    "proposed_set_play": "string|null",
    "accepted_set_play": "string|null",
    "set_play_step": "int|null"
  },
  "env_next_state": {
    "ball": {
      "position_mm": {
        "x": "float",
        "y": "float"
      },
      "velocity_mm_s": {
        "x": "float",
        "y": "float"
      },
      "owner_player_number": "int|null",
      "in_play": "bool"
    },
    "own_team_agents": [
      {
        "player_number": "int",
        "pose": {
          "x_mm": "float",
          "y_mm": "float",
          "theta_rad": "float"
        },
        "velocity": {
          "vx_mm_s": "float|null",
          "vy_mm_s": "float|null",
          "omega_rad_s": "float|null"
        },
        "is_upright": "bool|null",
        "controller_mode": "RL|BHUMAN",
        "has_ball_contact": "bool|null",
        "strategy": {
          "position": "string|null",
          "role": "string|null"
        }
      }
    ],
    "teammates": [
      {
        "player_number": "int",
        "robot_pose": {
          "x_mm": "float",
          "y_mm": "float",
          "theta_rad": "float"
        },
        "behavior_status": {
          "walking_to": {
            "x_mm": "float|null",
            "y_mm": "float|null"
          },
          "speed_mm_s": "float|null",
          "pass_target_player_number": "int|null",
          "pass_origin_player_number": "int|null",
          "shooting_to": {
            "x_mm": "float|null",
            "y_mm": "float|null"
          }
        },
        "strategy_status": {
          "position": "string|null",
          "role": "string|null"
        }
      }
    ],
    "obstacles": [
      {
        "type": "unknown|someRobot|opponent|teammate|fallenSomeRobot|fallenOpponent|fallenTeammate",
        "center_relative_mm": {
          "x": "float",
          "y": "float"
        },
        "left_relative_mm": {
          "x": "float",
          "y": "float"
        },
        "right_relative_mm": {
          "x": "float",
          "y": "float"
        },
        "velocity_mm_per_ms": {
          "x": "float|null",
          "y": "float|null"
        },
        "last_seen_ms": "int|null"
      }
    ]
  },
  "bhuman_oracle": {
    "active_agent_player_number": "int|null",
    "active_role": "playBall|freeKickWall|closestToTeammatesBall|goalkeeper|position_role|none",
    "handoff_ready": "bool",
    "takeover_active": "bool",
    "skill_request": {
      "skill": "none|stand|walk|shoot|pass|dribble|duel|block|mark|observe",
      "pass_target_player_number": "int|null",
      "target_pose": {
        "x_mm": "float|null",
        "y_mm": "float|null",
        "theta_rad": "float|null"
      },
      "target_direction_rad": "float|null"
    },
    "behavior_status": {
      "pass_target_player_number": "int|null",
      "pass_origin_player_number": "int|null",
      "walking_to": {
        "x_mm": "float|null",
        "y_mm": "float|null"
      },
      "speed_mm_s": "float|null",
      "shooting_to": {
        "x_mm": "float|null",
        "y_mm": "float|null"
      }
    },
    "motion_request": {
      "motion": "playDead|stand|walkAtAbsoluteSpeed|walkAtRelativeSpeed|walkToPose|walkToBallAndKick|dribble|dive|special|replayWalk|calibration",
      "walk_speed": {
        "x": "float|null",
        "y": "float|null",
        "theta": "float|null"
      },
      "walk_target": {
        "x_mm": "float|null",
        "y_mm": "float|null",
        "theta_rad": "float|null"
      },
      "target_direction_rad": "float|null",
      "direction_precision": {
        "min_rad": "float|null",
        "max_rad": "float|null"
      },
      "kick_type": "string|null",
      "kick_length_mm": "float|null",
      "align_precisely": "notPrecise|precise|null",
      "pre_step_allowed": "bool|null",
      "turn_kick_allowed": "bool|null"
    }
  },
  "ball_model": {
    "position_relative_mm": {
      "x": "float",
      "y": "float"
    },
    "velocity_relative_mm_s": {
      "x": "float|null",
      "y": "float|null"
    },
    "end_position_relative_mm": {
      "x": "float",
      "y": "float"
    },
    "intercepted_end_position_relative_mm": {
      "x": "float",
      "y": "float"
    },
    "time_since_ball_was_seen_ms": "int",
    "time_since_ball_disappeared_ms": "int",
    "use_teammates_ball": "bool",
    "ball_position_consistent_with_game_state": "bool",
    "teammates_ball_is_valid": "bool",
    "intercept_ball": "bool"
  },
  "macrostate": {
    "macro_active": "bool",
    "macro_owner_player_number": "int|null",
    "macro_type": "playBall_shoot|playBall_pass|playBall_dribble|playBall_duel|playBall_goalpost|none",
    "macro_start_frame_id": "int|null",
    "macro_duration_frames": "int",
    "macro_end_reason": "none|role_changed|game_state_changed|skill_request_cleared|motion_request_cleared|active_agent_changed|ball_lost|goal|out|timeout|completed|aborted_integrity|aborted_bridge_fallback",
    "macro_success": "bool|null"
  },
  "events": {
    "handoff_started": "bool",
    "handoff_ended": "bool",
    "takeover_started": "bool",
    "takeover_ended": "bool",
    "active_agent_changed": "bool",
    "possession_changed": "bool",
    "pass_started": "bool",
    "pass_completed": "bool",
    "pass_intercepted": "bool",
    "shot_started": "bool",
    "goal_scored": "bool",
    "goal_conceded": "bool",
    "dribble_started": "bool",
    "duel_started": "bool",
    "duel_won": "bool",
    "duel_lost": "bool",
    "ball_out": "bool"
  },
  "integrity": {
    "oracle_source": "bhuman",
    "state_source": "bridge_next_full_state",
    "frame_valid": "bool",
    "contract_version": "string"
  }
}
```

#### Clasificacion de campos del schema

**Nativos o directamente derivables de BHuman:**

- `game`
- `strategy_status`
- `env_next_state.ball`
- `env_next_state.teammates`
- `env_next_state.obstacles`
- `bhuman_oracle.skill_request`
- `bhuman_oracle.behavior_status`
- `bhuman_oracle.motion_request`
- `ball_model`

**Derivados/estructurados por el bridge:**

- `env_next_state.own_team_agents`
- `bhuman_oracle.handoff_ready`
- `bhuman_oracle.takeover_active`
- `macrostate`
- `events`
- `integrity`

**Decision de implementacion:**

- el bridge C++ debe producir este JSON logico por frame
- Python debe consumirlo como fuente unica de verdad del oracle
- `reward.py`, `wrapper.py` y `visualizer.py` deben leer de esta estructura, no reconstruir logica paralela

#### Decisiones tecnicas congeladas del bridge

Para que esta guia sea implementable sin ambiguedad, se fijan estas decisiones:

- **Autoridad del mundo por step:** Python entrega el estado actual, BHuman procesa un frame de Cognition y devuelve el siguiente estado completo
- **Mecanismo de transporte:** la implementacion final debe elegir explicitamente `pybind11`, proceso embebido o IPC/shared memory
- **Schema como contrato:** el JSON anterior queda congelado antes de integrar con `environment.py`
- **Despliegue real:** el bridge y la inferencia final deben respetar las restricciones temporales del hilo de Cognition

Si alguna de estas decisiones cambia a mitad de implementacion, el parser deja de ser fuente estable del contrato y el transplante vuelve a abrir una brecha sim2real.

#### Fallback operativo del bridge: 3 casos obligatorios

##### Caso 1: falla parser / contract loader

Condicion:

- no se pudo parsear un `.cfg` obligatorio
- o `load_all()` / contract loader no puede construir `AllConfigs`

Fallback:

- usar `load_all_fallback()` **solo** si la causa es indisponibilidad del repo/configs
- si la causa es parse error o inconsistencia estructural, **fail-loud** y no arrancar entrenamiento/despliegue

Salida requerida:

- `integrity.frame_valid = false`
- `bridge_fallback_active = true`
- `bridge_fallback_reason = "parser_unavailable"` o `"contract_build_failed"`

##### Caso 2: falla bridge / frame invalido

Condicion:

- frame incompleto
- serializacion rota
- `next_full_state` no cumple schema
- error de IPC / `pybind11` / shared memory

Fallback:

- abortar la macro en curso
- ceder control completo a BHuman para ese frame y frames siguientes hasta recuperar integridad
- no fabricar estado parcial en Python

Salida requerida:

- `integrity.frame_valid = false`
- `macro_end_reason = aborted_bridge_fallback`
- `bridge_fallback_active = true`
- `bridge_fallback_reason = "bridge_frame_invalid"` o `"bridge_transport_error"`

##### Caso 3: takeover invalido o excede timeout

Condicion:

- `macro_duration_frames >= 90`
- o takeover pierde integridad semantica (`playBall` sin evidencia util sostenida, owner ambiguo, state drift)

Fallback:

- cerrar takeover actual como fallido
- devolver control al flujo base BHuman
- bloquear reward positiva de completion para esa macro

Salida requerida:

- `takeover_active = false`
- `macro_end_reason = timeout` o `aborted_integrity`
- `macro_success = false`
- `bridge_fallback_active = true`
- `bridge_fallback_reason = "takeover_timeout"` o `"takeover_integrity_lost"`

**Archivo: `KickAtGoal.cpp` (Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/)**

| Linea | Punto de Decision | Parametro(s) | Fuente del Default | Clasificacion |
|-------|-------------------|-------------|-------------------|---------------|
| L45 | `minOpeningAngle` — angulo minimo al arco | `minOpeningAngle` | kickAtGoalImpl.cfg → 6deg | **Aprendible (Fase 2)** |
| L47 | `hysteresisDecisionPenalty` — inercia | `hysteresisDecisionPenalty` | kickAtGoalImpl.cfg → 500ms | Hard constraint |
| L151/282 | `calcTTRP() + executionTime + penalty` | TTRP compuesto | KickSelection + kickInfo.cfg | Observable |

**Archivo: `PassToTeammate.cpp` (Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/)**

| Linea | Punto de Decision | Parametro(s) | Fuente del Default | Clasificacion |
|-------|-------------------|-------------|-------------------|---------------|
| L61 | `ratingThreshold` — rating minimo de pase | `ratingThreshold` | passToTeammateImpl.cfg → 0.7 | **Aprendible (Fase 2)** |
| L62 | `passAheadDistance` — anticipar companero | `passAheadDistance` | passToTeammateImpl.cfg → 0 | **Aprendible (Fase 2)** |
| L63 | `minAngleOffset` — desviacion minima | `minAngleOffset` | passToTeammateImpl.cfg → 7deg | **Aprendible (Fase 2)** |

**Archivo: `PassEvaluationProvider.h` (Src/Modules/BehaviorControl/PassEvaluationProvider/)**

| Linea | Parametro | Default C++ | Clasificacion |
|-------|-----------|------------|---------------|
| L34 | `minValue` | 0.01f | Hard constraint |
| L35 | `opponentDistToTargetThreshold` | 1400.f | Observable (no aprendible — modelo de oponentes) |
| L36 | `opponentDistToLineThreshold` | 1000.f | Observable |
| L37 | `distancePenaltyThresholdPlaying` | 4000.f | Observable |

### 9.4 CONTRATO OPERATIVO FINAL: Decision Point C++ → Parser → AllConfigs → Funcion Python RL

> **REQUISITO BLOQUEANTE:** Esta tabla es el contrato vinculante FINAL entre el C++, el parser, y el RL. Cada fila cierra la cadena completa: linea C++ exacta → parametro → campo AllConfigs → funcion Python RL concreta → rango de salida. Si CUALQUIER columna esta vacia, el bridge esta ROTO y el proyecto FALLA en sim2real.
>
> **Fuente de verdad C++:** Los decision points listados aqui coinciden exactamente con `DECISION_POINTS_CPP.md` Seccion 7 y `CFG_UNIFIED_ANALYSIS.md` Seccion 10.2. Si existe discrepancia, es un BUG.
>
> **Como verificar:** Ejecutar `test_bridge_contract_complete()` (Seccion 16).
>
> **Nota de rigor:** Desde esta seccion en adelante, los `action_space`, `rl_predict_*`, rangos y contratos Python son una **especificacion propuesta de bridge** derivada del C++ real. No describen una implementacion Python ya existente en el repo actual.
>
> **Nota de implementacion para dev senior / Claude:** Los bloques `Bridge`, `rl_predict_*` y ejemplos Python de esta seccion deben leerse como pseudocodigo de interfaz. La implementacion recomendada es en DOS capas:
> 1. `parser core` que solo retorna `AllConfigs`
> 2. `contract loader / env bridge` que consume `AllConfigs` y refleja el contrato BHuman: `playBall` como senal temprana de handoff, `takeover_active` derivado de evidencia aguas abajo, masking del actor activo y salida estructurada del oracle

---

#### DECISION 1: ¿Disparar o Pasar? (`PlayBall.cpp:smashOrPass()`)

```
┌─────────┬──────────────────────┬────────────┬───────────────────────────────────┬──────────────────────────────────────┬────────────┐
│ LINEA   │ PARAMETRO C++        │ DEFAULT    │ AllConfigs.campo                  │ FUNCION PYTHON RL                    │ RANGO RL   │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L56     │ shootThreshold       │ 0.8        │ .smash_or_pass.shoot_threshold   │ rl_predict_threshold(state) → float  │ [0.3, 1.0] │
│         │ if(maxValue > thres) │            │                                   │ Fase 2: action_space[0]              │            │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L70     │ minPassDistance       │ 1500mm     │ .smash_or_pass.min_pass_distance │ rl_predict_distances(state)[0]       │ [500,3000] │
│ L71     │ maxPassDistance       │ 4500mm     │ .smash_or_pass.max_pass_distance │ rl_predict_distances(state)[1]       │ [2000,8000]│
│         │ passDistance < min    │            │                                   │ Fase 2: action_space[1:3]            │            │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L77     │ sampleDistance        │ 500mm      │ .smash_or_pass.sample_distance   │ rl_predict_sampling(state)[0]        │ [200,1000] │
│ L78     │ sampleSteps.min/max  │ (-1, 2)    │ .smash_or_pass.sample_steps_min  │ rl_predict_sampling(state)[1:3]      │ [-3, 5]    │
│         │ for(t=min; t<=max)   │            │ .smash_or_pass.sample_steps_max  │ Fase 2: action_space[3:6]            │            │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L82-84  │ sampleValue =        │ (compuesto)│ N/A — RL REEMPLAZA calculo       │ rl_predict_pass_quality(state)       │ [0.0, 1.0] │
│         │ PassEval×xG×oppRat   │            │ Inputs: teammate_pos, ball_pos,  │ Fase 2/3: reemplaza rating completo  │            │
│         │                      │            │ opponent_positions → obs_space    │ bridge inyecta float en vez del      │            │
│         │                      │            │                                   │ producto PassEval×xG×oppRating       │            │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L95-98  │ minPenalty            │ 0.1        │ .smash_or_pass.min_penalty       │ rl_predict_hysteresis(state)[0]      │ [0.0, 0.3] │
│ (via    │ maxPenalty            │ 0.8        │ .smash_or_pass.max_penalty       │ rl_predict_hysteresis(state)[1]      │ [0.3, 1.0] │
│ L26-27) │ decisionPenalty =    │            │                                   │ Fase 2: action_space[6:8]            │            │
│         │ mapToRange(...)      │            │                                   │                                      │            │
└─────────┴──────────────────────┴────────────┴───────────────────────────────────┴──────────────────────────────────────┴────────────┘
```

**Implementacion Python concreta para Decision 1:**

```python
# bridge.py — Fase 2: RL ajusta parametros de smashOrPass

class SmashOrPassBridge:
    """Bridge entre RL y PlayBall.cpp:smashOrPass()."""

    def __init__(self, cfgs: AllConfigs):
        self.defaults = cfgs.smash_or_pass  # baseline del C++

    def get_rl_action_space(self) -> dict:
        """Define action space para Fase 2 — 8 outputs continuos."""
        return {
            "threshold_output":   Box(0.3, 1.0, shape=(1,)),    # L56
            "distance_output":    Box([500, 2000], [3000, 8000], shape=(2,)),  # L70-71
            "sampling_output":    Box([200, -3, 0], [1000, 5, 5], shape=(3,)),# L77-78
            "hysteresis_output":  Box([0.0, 0.3], [0.3, 1.0], shape=(2,)),    # L95-98
        }

    def apply_rl_action(self, action: np.ndarray) -> SmashOrPassParams:
        """Convierte RL output → SmashOrPassParams para inyectar en C++."""
        return SmashOrPassParams(
            shoot_threshold=float(action[0]),         # → PlayBall.cpp:L56
            min_pass_distance=float(action[1]),       # → PlayBall.cpp:L70
            max_pass_distance=float(action[2]),       # → PlayBall.cpp:L71
            sample_distance=float(action[3]),         # → PlayBall.cpp:L77
            sample_steps_min=int(action[4]),           # → PlayBall.cpp:L78
            sample_steps_max=int(action[5]),           # → PlayBall.cpp:L78
            min_penalty=float(action[6]),              # → PlayBall.h:L26
            max_penalty=float(action[7]),              # → PlayBall.h:L27
            # HARD constraints — NUNCA modificados por RL:
            min_rating=self.defaults.min_rating,
            ball_dist_min_penalty=self.defaults.ball_dist_min_penalty,
            ball_dist_max_penalty=self.defaults.ball_dist_max_penalty,
            max_free_kick_distance=self.defaults.max_free_kick_distance,
            max_goal_kick_distance=self.defaults.max_goal_kick_distance,
        )

    def predict_pass_quality(self, state: np.ndarray) -> float:
        """Fase 2/3: RL reemplaza PlayBall.cpp:L82-84 (PassEval×xG×oppRating)."""
        # state contiene: teammate_positions, ball_position, opponent_positions
        return float(self.policy.predict(state))  # → [0.0, 1.0]
```

---

#### DECISION 2: ¿Donde Pasar? (`PassToTeammate.cpp`)

```
┌─────────┬──────────────────────┬────────────┬───────────────────────────────────┬──────────────────────────────────────┬────────────┐
│ LINEA   │ PARAMETRO C++        │ DEFAULT    │ AllConfigs.campo                  │ FUNCION PYTHON RL                    │ RANGO RL   │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L61     │ ratingThreshold      │ 0.7        │ .pass_to_teammate                │ rl_predict_pass_params(state)[0]     │ [0.3, 0.9] │
│         │ if(rating < thres)   │            │   .rating_threshold              │ Fase 2: action_space[0]              │            │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L62     │ passAheadDistance     │ 0mm        │ .pass_to_teammate                │ rl_predict_pass_params(state)[1]     │ [-500,500] │
│         │ anticipar companero  │            │   .pass_ahead_distance           │ Fase 2: action_space[1]              │            │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L63     │ minAngleOffset       │ 7deg       │ .pass_to_teammate                │ rl_predict_pass_params(state)[2]     │ [2, 15]deg │
│         │ desviacion minima    │            │   .min_angle_offset              │ Fase 2: action_space[2]              │            │
└─────────┴──────────────────────┴────────────┴───────────────────────────────────┴──────────────────────────────────────┴────────────┘
```

**Implementacion Python concreta para Decision 2:**

```python
class PassToTeammateBridge:
    """Bridge entre RL y PassToTeammate.cpp."""

    def __init__(self, cfgs: AllConfigs):
        self.defaults = cfgs.pass_to_teammate

    def apply_rl_action(self, action: np.ndarray) -> dict:
        """Convierte RL output → parametros de pase."""
        return {
            "rating_threshold": float(action[0]),    # → PassToTeammate.cpp:L61
            "pass_ahead_distance": float(action[1]), # → PassToTeammate.cpp:L62
            "min_angle_offset": float(action[2]),    # → PassToTeammate.cpp:L63
        }
```

---

#### DECISION 3: ¿Cuando Disparar? (`KickAtGoal.cpp`)

```
┌─────────┬──────────────────────┬────────────┬───────────────────────────────────┬──────────────────────────────────────┬────────────┐
│ LINEA   │ PARAMETRO C++        │ DEFAULT    │ AllConfigs.campo                  │ FUNCION PYTHON RL                    │ RANGO RL   │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L45     │ minOpeningAngle      │ 6deg       │ .kick_at_goal.min_opening_angle  │ rl_predict_kick_angle(state) → float │ [2, 15]deg │
│         │ angulo libre al arco │            │                                   │ Fase 2: action_space[0]              │            │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L47     │ hysteresisDecision   │ 500ms      │ .kick_at_goal                    │ N/A — HARD CONSTRAINT                │ FIJO       │
│         │ Penalty              │            │   .hysteresis_decision_penalty    │ No modificable por RL                │            │
└─────────┴──────────────────────┴────────────┴───────────────────────────────────┴──────────────────────────────────────┴────────────┘
```

**Implementacion Python concreta para Decision 3:**

```python
class KickAtGoalBridge:
    """Bridge entre RL y KickAtGoal.cpp."""

    def __init__(self, cfgs: AllConfigs):
        self.defaults = cfgs.kick_at_goal

    def apply_rl_action(self, action: np.ndarray) -> dict:
        return {
            "min_opening_angle": float(action[0]),  # → KickAtGoal.cpp:L45
            # hysteresisDecisionPenalty: FIJO en 500ms (L47, HARD constraint)
        }
```

---

#### DECISION 4: Fase 3 — RL Reemplaza smashOrPass() Entero

```
┌─────────┬──────────────────────┬────────────┬───────────────────────────────────┬──────────────────────────────────────┬────────────┐
│ LINEA   │ ALCANCE              │ DEFAULT    │ AllConfigs USO                    │ FUNCION PYTHON RL                    │ OUTPUT     │
├─────────┼──────────────────────┼────────────┼───────────────────────────────────┼──────────────────────────────────────┼────────────┤
│ L33-118 │ smashOrPass() entero │ C++ logica │ constraint validator + fallback   │ rl_predict_action(state) → int       │ {0=shoot,  │
│         │ PlayBall.cpp         │ completa   │ Si RL falla → usar defaults      │ Fase 3: action_space discreto        │  1=pass_0, │
│         │                      │            │ Validar contra HARD constraints   │                                      │  2=pass_1, │
│         │                      │            │ (kickInfo ranges, field limits)   │                                      │  3=pass_2, │
│         │                      │            │                                   │                                      │  4=dribble}│
└─────────┴──────────────────────┴────────────┴───────────────────────────────────┴──────────────────────────────────────┴────────────┘
```

**Implementacion Python concreta para Decision 4:**

```python
class FullDecisionBridge:
    """Fase 3: RL reemplaza PlayBall.cpp:smashOrPass() completo."""

    def __init__(self, cfgs: AllConfigs):
        self.cfgs = cfgs

    def predict_action(self, state: np.ndarray) -> int:
        """RL toma la decision shoot/pass/dribble directamente."""
        action_id = int(self.policy.predict(state))
        return self._validate(action_id, state)

    def _validate(self, action_id: int, state: np.ndarray) -> int:
        """Validar contra HARD constraints antes de ejecutar."""
        if action_id == 0:  # shoot
            # Validar que hay angulo libre >= minOpeningAngle (KickAtGoal.cpp:L45)
            if not self._has_opening(state, self.cfgs.kick_at_goal.min_opening_angle):
                return 4  # fallback: dribble
        elif action_id in (1, 2, 3):  # pass
            # Validar distancia dentro de [minPassDistance, maxPassDistance]
            teammate_dist = self._get_pass_distance(state, action_id)
            if not (self.cfgs.smash_or_pass.min_pass_distance
                    <= teammate_dist
                    <= self.cfgs.smash_or_pass.max_pass_distance):
                return 4  # fallback: dribble
            # Validar kick range fisicamente posible (kickInfo.cfg)
            if not self._kick_in_range(teammate_dist):
                return 4
        return action_id
```

---

**Resumen del contrato — 12 parametros aprendibles/reemplazables en 4 decisiones:**

```
Decision 1 (smashOrPass):     8 params → 8 action outputs continuos
Decision 2 (PassToTeammate):  3 params → 3 action outputs continuos
Decision 3 (KickAtGoal):      1 param  → 1 action output continuo
Decision 4 (Fase 3 full):     1 output → 1 action output discreto (5 clases)
                              ─────────
TOTAL Fase 2: 12 outputs continuos
TOTAL Fase 3: 12 continuos + 1 discreto (o solo discreto si reemplaza entero)
```

**Verificacion de completitud:** Esta seccion cubre los 7 puntos de inyeccion de `DECISION_POINTS_CPP.md` Seccion 7-8 y los 7 de `CFG_UNIFIED_ANALYSIS.md` Seccion 10.2. Si un decision point aparece en esos documentos pero NO aqui, es un BUG de la guia.

### 9.5 CONTRATO DE TRAZABILIDAD COMPLETO: Ruta/Linea → Parser → AllConfigs → Python RL → Punto de Inyeccion

> **TABLA MAESTRA DE TRAZABILIDAD.** Cada fila es una cadena verificable end-to-end. Si una columna esta vacia, el contrato tiene un hueco. Esta tabla cubre 12 grupos de CFG (13 archivos contando ambas tacticas) + 1 C++ header — NO solo Fase 2.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ #  │ FUENTE (archivo:linea)                   │ PARSER                          │ AllConfigs.campo                              │
│    │                                          │                                 │                                               │
│    │ VARIABLE PYTHON RL                       │ USO EN ENTORNO RL               │ FASE │ CLASIFICACION                         │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 1  │ fieldDimensions.cfg:L1-45               │ field_symbols.py                │ .field_symbols["xPos...", "yPos...", etc.]    │
│    │ FIELD_SYMBOLS_4V4_SCALED                │ config.py: limites de campo     │ 1    │ HARD CONSTRAINT (no aprendible)       │
│    │                                          │ environment.py: normalizacion   │      │                                       │
│    │                                          │ reward.py: out-of-bounds        │      │                                       │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 2  │ s4v4_scaled.cfg:L3-30                   │ StrategyParser.parse()          │ .strategy: StrategyConfig                    │
│    │ strategy.tactics[].transitions           │ environment.py: _step_fsm()     │ 1    │ MANDATORIO (FSM control)              │
│    │                                          │ tactic one_hot en obs_space     │      │                                       │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 3  │ t121_4v4_scaled.cfg:L1-20               │ TacticParser.parse()            │ .tactics["t121_4v4_scaled"]: TacticConfig    │
│    │ positions[].pose.(x,y)                   │ environment.py: _get_home_poses │ 1    │ APRENDIBLE (Fase 1 positioning)       │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 4  │ t13_4v4_scaled.cfg:L1-15                │ TacticParser.parse()            │ .tactics["t13_4v4_scaled"]: TacticConfig     │
│    │ positions[].pose.(x,y)                   │ environment.py: _get_home_poses │ 1    │ APRENDIBLE (Fase 1 positioning)       │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 5  │ kickAtGoalImpl.cfg:L1-30                │ ActionParser                    │ .kick_at_goal: KickAtGoalConfig              │
│    │ .kick_at_goal.min_opening_angle          │ .parse_kick_at_goal()           │      │                                       │
│    │   → action_space: angle_output[0]        │ KickAtGoal.cpp:L45 inyeccion   │ 2    │ APRENDIBLE (angulo minimo arco)       │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 6  │ passToTeammateImpl.cfg:L1-45            │ ActionParser                    │ .pass_to_teammate: PassToTeammateConfig      │
│    │ .pass_to_teammate.rating_threshold       │ .parse_pass_to_teammate()       │      │                                       │
│    │   → action_space: pass_output[0]         │ PassToTeammate.cpp:L61          │ 2    │ APRENDIBLE                            │
│    │ .pass_to_teammate.pass_ahead_distance    │                                 │      │                                       │
│    │   → action_space: pass_output[1]         │ PassToTeammate.cpp:L62          │ 2    │ APRENDIBLE                            │
│    │ .pass_to_teammate.min_angle_offset       │                                 │      │                                       │
│    │   → action_space: pass_output[2]         │ PassToTeammate.cpp:L63          │ 2    │ APRENDIBLE                            │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 7  │ zweikampfImpl.cfg:L1-84                 │ ActionParser                    │ .zweikampf: ZweikampfConfig                  │
│    │ .zweikampf.allowed_kicks                 │ .parse_zweikampf()              │      │                                       │
│    │   → constraint: kicks validos en duelo   │ _validate_duel_kick()           │ 3    │ CONSTRAINT (sin reward en parser)     │
│    │ .zweikampf.rating_steal_ball/goal_shot   │ reward.py (fuera del parser)    │ 3    │ METADATA para reward futuro           │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 8  │ zweikampfImpl.cfg (timings)             │ ActionParser                    │ .zweikampf.timings: ZweikampfTimings         │
│    │ .timings.no_kick_min_time (200ms)        │ .parse_zweikampf()              │      │                                       │
│    │   → constraint: cooldown entre kicks     │ _validate_kick_timing()         │ 2    │ CONSTRAINT (timing real en partidos)  │
│    │ .timings.deactive_after_doing_nothing    │ _check_zweikampf_active()       │ 2    │ CONSTRAINT (desactivacion por timeout)│
│    │ NOTA: demoGoToBallAndKickImpl.cfg ELIMINADO — no activo en partidos 4v4.               │      │                                       │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 9  │ PlayBall.h:L20-34 (C++ header)          │ SmashOrPassParams()             │ .smash_or_pass: SmashOrPassParams            │
│    │ .smash_or_pass.shoot_threshold           │ (instanciado directo)           │      │                                       │
│    │   → action_space: threshold_output[0]    │ PlayBall.cpp:L56 inyeccion     │ 2    │ APRENDIBLE                            │
│    │ .smash_or_pass.min_pass_distance         │                                 │      │                                       │
│    │   → action_space: distance_output[0]     │ PlayBall.cpp:L70 inyeccion     │ 2    │ APRENDIBLE                            │
│    │ .smash_or_pass.max_pass_distance         │                                 │      │                                       │
│    │   → action_space: distance_output[1]     │ PlayBall.cpp:L71 inyeccion     │ 2    │ APRENDIBLE                            │
│    │ .smash_or_pass.sample_distance           │                                 │      │                                       │
│    │   → action_space: sampling_output[0]     │ PlayBall.cpp:L77 inyeccion     │ 2    │ APRENDIBLE                            │
│    │ .smash_or_pass.sample_steps_min/max      │                                 │      │                                       │
│    │   → action_space: sampling_output[1]     │ PlayBall.cpp:L78 inyeccion     │ 2    │ APRENDIBLE                            │
│    │ .smash_or_pass.min/max_penalty           │                                 │      │                                       │
│    │   → action_space: hysteresis_output      │ PlayBall.h:L26-27 inyeccion    │ 2    │ APRENDIBLE                            │
│    │ PlayBall.cpp:L82-84 (rating compuesto)   │ N/A — RL REEMPLAZA calculo     │      │                                       │
│    │   → RL predice pass_quality [0.0, 1.0]   │ bridge: rl_predict_rating()     │ 2/3  │ REEMPLAZABLE POR RL                  │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 10 │ kickInfo.cfg:L1-347                     │ KickInfoParser.parse()          │ .kicks: dict[str, KickInfo] (23 pateos)      │
│    │ kicks[tipo].range_min/max                │ sim: simulate_kick() validation │ 1    │ CONSTRAINT (rangos fisicos)           │
│    │ kicks[tipo].ball_velocity_min/max        │ reward: kick feasibility check  │ 1    │ CONSTRAINT (velocidades)              │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 11 │ walkingEngine.cfg + libWalkProvider.cfg  │ WalkParser.parse()              │ .walk: WalkConfig                            │
│    │ .walk.max_forward/sideways/backwards     │ config.py: speed limits         │ 1    │ CONSTRAINT (velocidades NAO)          │
│    │ .walk.field_border_expansion             │ _clamp_position()               │ 1    │ CONSTRAINT (borde campo)              │
│    │ .walk.obstacle_avoidance_distance        │ _avoid_obstacle()               │ 1    │ CONSTRAINT (evasion)                  │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 12 │ behaviorParameters.cfg:L1-11            │ BehaviorParser.parse()          │ .behavior: BehaviorParams                    │
│    │ .behavior.jump_radius/stand_radius       │ _get_keeper_action_zone()       │ 1    │ CONSTRAINT (zonas portero)            │
│    │ .behavior.keeper_jumping_on              │ reward: keeper behavior          │ 1    │ CONSTRAINT (flag portero)             │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 13 │ strategyBehaviorControl.cfg:L1          │ StrategyBehaviorControl         │ .strategy_behavior_control:                  │
│    │ .strategy_behavior_control.strategy      │ Parser.parse()                  │   StrategyBehaviorControlConfig              │
│    │   → assertion: == "s4v4_scaled"          │ load_all(): AssertionError si   │ 1    │ ASSERTION (integridad scenario)       │
│    │                                          │ strategy != s4v4_scaled         │      │                                       │
├────┼──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┤
│ 14 │ skillBehaviorControl.cfg:L1-19          │ SkillBehaviorControl            │ .skill_behavior_control:                     │
│    │ .skill_behavior_control.playing_options  │ Parser.parse()                  │   SkillBehaviorControlConfig                 │
│    │   → constraint: opciones validas         │ _get_available_playing_options() │ 1   │ CONSTRAINT (opciones disponibles)     │
└────┴──────────────────────────────────────────┴─────────────────────────────────┴───────────────────────────────────────────────┘

RESUMEN DE COBERTURA:
  12 grupos de CFG parseados (13 archivos contando ambas tacticas) + 1 C++ header → 14 filas de trazabilidad
  7 puntos de inyeccion C++ Fase 2 (filas 5, 6, 8, 9)
  12 parametros aprendibles/reemplazables (Fase 2+)
  Cada fila tiene: fuente → parser → AllConfigs.campo → uso Python RL → fase → clasificacion
  Si una fila no tiene parser o AllConfigs.campo, el contrato esta ROTO.
```

### 9.6 Uso en environment.py (Fase 2+)

```python
# Fase 2: RL ajusta estos parametros
action_params = parser.parse_kick_at_goal("kickAtGoalImpl.cfg")
pass_params = parser.parse_pass_to_teammate("passToTeammateImpl.cfg")

# Defaults del CFG (baseline)
defaults = {
    "shootThreshold": smash_or_pass.shoot_threshold,    # 0.8
    "minPassDistance": smash_or_pass.min_pass_distance,  # 1500
    "maxPassDistance": smash_or_pass.max_pass_distance,  # 4500
    "minOpeningAngle": action_params.min_opening_angle,  # 6
    "ratingThreshold": pass_params.rating_threshold,     # 0.7
}
```

---

<a name="10-kickinfo"></a>
## 10. Modulo 5: KickInfo Parser (Pateos)

### 10.1 Input

`Config/Scenarios/4v4_Scaled/kickInfo.cfg`

**Estructura real (extracto):** Es un NAMED MAP, no un array.
```cfg
kicks = {
  forwardFastRight = {
    kickType = forwardFastRight;
    rotationOffset = 3deg;
    ballOffset = {x = -185; y = 50;};
    range = {min = 1600.01; max = 4500;};
    ballVelocity = {min = 979.799; max = 1643.17;};
    exclusionRange = {min = 0; max = 0;};
    executionTime = 1500;
    postRotationOffset = 0deg;
    motion = kick;
    kickMotionType = kickForwardFast;
    walkKickType = none;
    kickLeg = right;
    mirror = false;
  };
  walkForwardsRight = {
    kickType = walkForwardsRight;
    rotationOffset = 0deg;
    ballOffset = {x = -180; y = 50;};
    range = {min = 400; max = 1000;};
    ballVelocity = {min = 489.898; max = 774.597;};
    exclusionRange = {min = 0; max = 0;};
    executionTime = 500;
    postRotationOffset = 0deg;
    motion = walk;
    kickMotionType = none;
    walkKickType = forward;
    kickLeg = right;
    mirror = false;
  };
  // ... 21 entradas mas (23 total)
};
```

**Parseo:** El parser base retorna `{"kicks": {"forwardFastRight": {...}, "walkForwardsRight": {...}, ...}}`. El KickInfoParser itera sobre `raw["kicks"].items()` para construir el dict completo de KickInfo (23 entradas reales del archivo).

### 10.2 Output: dict[str, KickInfo]

```python
@dataclass
class KickInfo:
    kick_type: str
    rotation_offset: float       # deg
    ball_offset: tuple[float, float]  # (x, y) mm
    range_min: float             # mm
    range_max: float             # mm
    ball_velocity_min: float     # mm/s
    ball_velocity_max: float     # mm/s
    execution_time: int          # ms
    motion_type: str             # "kick" | "walk"
    kick_leg: str                # "left" | "right"
    is_mirror: bool
    exclusion_range: tuple[float, float]  # (min, max) mm
    post_rotation_offset: float  # deg
    kick_motion_type: str        # "kickForwardFast" | "kickForwardFastLong" | "none" | etc.
    walk_kick_type: str          # "forward" | "forwardLong" | "forwardAlternative" | "none" | etc.
```

### 10.3 Uso en Simulacion

```python
kicks = parser.parse_kick_info("kickInfo.cfg")

# En sim, cuando striker dispara:
def simulate_kick(kick_type: str, distance: float) -> tuple[float, float]:
    ki = kicks[kick_type]
    assert ki.range_min <= distance <= ki.range_max, \
        f"Distance {distance} out of range [{ki.range_min}, {ki.range_max}]"
    # Velocidad proporcional a distancia dentro del rango
    t = (distance - ki.range_min) / max(ki.range_max - ki.range_min, 1)
    velocity = ki.ball_velocity_min + t * (ki.ball_velocity_max - ki.ball_velocity_min)
    return velocity, ki.execution_time
```

---

<a name="11-walk"></a>
## 11. Modulo 6: Walk/Behavior Parser (Movimiento y Configuracion Global)

### 11.1 Inputs

- `Config/Robots/Default/walkingEngine.cfg` → velocidades maximas
- `Config/Scenarios/4v4_Scaled/libWalkProvider.cfg` → evasion de obstaculos, bordes
- `Config/Scenarios/4v4_Scaled/behaviorParameters.cfg` → config global portero/striker
- `Config/Scenarios/4v4_Scaled/strategyBehaviorControl.cfg` → strategy activa
- `Config/Scenarios/4v4_Scaled/skillBehaviorControl.cfg` → opciones de comportamiento

### 11.2 Estructura real de libWalkProvider.cfg (completo)

```cfg
ballTimeout = 2000;
obstacleAvoidanceDistance = 480;
obstacleAvoidanceMinRadius = 220;
obstacleAvoidanceMaxRadius = 600;
goalPostAvoidanceDistance = 400;
goalPostAvoidanceMinRadius = 300;
goalPostAvoidanceMaxRadius = 1000;
ballAvoidanceDistance = 465;
ballAvoidanceMinRadius = 465;
ballAvoidanceMaxRadius = 765;
penaltyAreaExpansion = 0;
penaltyAreaAvoidanceDistance = 400;
penaltyAreaAvoidanceMinRadius = 300;
penaltyAreaAvoidanceMaxRadius = 1000;
fieldBorderExpansion = 300;
fieldBorderAvoidanceDistance = 400;
fieldBorderAvoidanceMinRadius = 300;
fieldBorderAvoidanceMaxRadius = 1000;
```

### 11.3 Estructura real de behaviorParameters.cfg (completo)

```cfg
keeperJumpingOn = true;
penaltyStrikerWalkSpeed = 0.5;
penaltyStrikerAngleToLeftPostOffset = 10deg;
penaltyStrikerAngleToRightPostOffset = 10deg;
penaltyStrikerUseObstacles = true;
ballCatchMaxWalkDistance = 1200;
standRadius = 60;
walkRadius = 125;
genuflectRadius = 200;
genuflectStandRadius = 200;
jumpRadius = 600;
```

### 11.4 Output: Dataclasses

```python
@dataclass
class WalkConfig:
    """Combinacion de walkingEngine efectivo + libWalkProvider.cfg.
    FUENTE: walkingEngine resuelto por search path de B-Human.
            En este repo, para robots nombrados de Config/Robots/robots.cfg,
            el efectivo cae en Config/Robots/Default/walkingEngine.cfg porque
            no existe override Config/Robots/<robot>/<robot>/walkingEngine.cfg.
            Config/Robots/Nao/Nao/walkingEngine.cfg solo aplica si
            headName == bodyName == "Nao".
            Config/Scenarios/4v4_Scaled/libWalkProvider.cfg (evasion)"""
    # Velocidades maximas (de walkingEngine.cfg)
    max_forward: float = 250.0      # mm/s
    max_sideways: float = 200.0     # mm/s
    max_backwards: float = 150.0    # mm/s
    max_rotation: float = 120.0     # deg/s
    # Evasion (de libWalkProvider.cfg — TODOS parseados, no defaults)
    ball_timeout: int = 2000                       # ms
    obstacle_avoidance_distance: float = 480.0     # mm
    obstacle_avoidance_min_radius: float = 220.0   # mm
    obstacle_avoidance_max_radius: float = 600.0   # mm
    goal_post_avoidance_distance: float = 400.0    # mm
    ball_avoidance_distance: float = 465.0         # mm
    ball_avoidance_min_radius: float = 465.0       # mm
    ball_avoidance_max_radius: float = 765.0       # mm
    penalty_area_expansion: float = 0.0            # mm
    field_border_expansion: float = 300.0          # mm
    field_border_avoidance_distance: float = 400.0 # mm

@dataclass
class BehaviorParams:
    """FUENTE: Config/Scenarios/4v4_Scaled/behaviorParameters.cfg — TODOS parseados."""
    keeper_jumping_on: bool = True
    penalty_striker_walk_speed: float = 0.5
    penalty_striker_angle_to_left_post_offset: float = 10.0   # deg
    penalty_striker_angle_to_right_post_offset: float = 10.0  # deg
    penalty_striker_use_obstacles: bool = True
    ball_catch_max_walk_distance: float = 1200.0  # mm
    stand_radius: float = 60.0     # mm
    walk_radius: float = 125.0     # mm
    genuflect_radius: float = 200.0 # mm
    genuflect_stand_radius: float = 200.0 # mm
    jump_radius: float = 600.0     # mm
```

### 11.5 Logica de Parseo

```python
class WalkParser:
    def __init__(self, base: BHumanCfgParser):
        self.base = base

    def parse(self, walk_engine_path: Path, lib_walk_path: Path) -> WalkConfig:
        """Parsear AMBOS archivos y combinar en WalkConfig."""
        # walkingEngine.cfg — estructura ANIDADA:
        #   configuratedParameters = { maxSpeed = { rotation = 120deg; translation = {x=250; y=200;}; }; maxSpeedBackwards = 150; };
        engine = self.base.parse_file(walk_engine_path)
        cfg_params = engine.get("configuratedParameters", {})
        max_speed = cfg_params.get("maxSpeed", {})
        translation = max_speed.get("translation", {})

        # libWalkProvider.cfg — estructura plana: key = value;
        lib = self.base.parse_file(lib_walk_path)

        return WalkConfig(
            max_forward=float(translation.get("x", 250)),
            max_sideways=float(translation.get("y", 200)),
            max_backwards=float(cfg_params.get("maxSpeedBackwards", 150)),
            max_rotation=float(max_speed.get("rotation", 120)),
            ball_timeout=int(lib.get("ballTimeout", 2000)),
            obstacle_avoidance_distance=float(lib.get("obstacleAvoidanceDistance", 480)),
            obstacle_avoidance_min_radius=float(lib.get("obstacleAvoidanceMinRadius", 220)),
            obstacle_avoidance_max_radius=float(lib.get("obstacleAvoidanceMaxRadius", 600)),
            goal_post_avoidance_distance=float(lib.get("goalPostAvoidanceDistance", 400)),
            ball_avoidance_distance=float(lib.get("ballAvoidanceDistance", 465)),
            ball_avoidance_min_radius=float(lib.get("ballAvoidanceMinRadius", 465)),
            ball_avoidance_max_radius=float(lib.get("ballAvoidanceMaxRadius", 765)),
            penalty_area_expansion=float(lib.get("penaltyAreaExpansion", 0)),
            field_border_expansion=float(lib.get("fieldBorderExpansion", 300)),
            field_border_avoidance_distance=float(lib.get("fieldBorderAvoidanceDistance", 400)),
        )

class BehaviorParser:
    def __init__(self, base: BHumanCfgParser):
        self.base = base

    def parse(self, filepath: Path) -> BehaviorParams:
        """Parsear behaviorParameters.cfg completo."""
        raw = self.base.parse_file(filepath)
        return BehaviorParams(
            keeper_jumping_on=bool(raw.get("keeperJumpingOn", True)),
            penalty_striker_walk_speed=float(raw.get("penaltyStrikerWalkSpeed", 0.5)),
            penalty_striker_angle_to_left_post_offset=float(raw.get("penaltyStrikerAngleToLeftPostOffset", 10)),
            penalty_striker_angle_to_right_post_offset=float(raw.get("penaltyStrikerAngleToRightPostOffset", 10)),
            penalty_striker_use_obstacles=bool(raw.get("penaltyStrikerUseObstacles", True)),
            ball_catch_max_walk_distance=float(raw.get("ballCatchMaxWalkDistance", 1200)),
            stand_radius=float(raw.get("standRadius", 60)),
            walk_radius=float(raw.get("walkRadius", 125)),
            genuflect_radius=float(raw.get("genuflectRadius", 200)),
            genuflect_stand_radius=float(raw.get("genuflectStandRadius", 200)),
            jump_radius=float(raw.get("jumpRadius", 600)),
        )
```

### 11.6 Logica de Parseo: Zweikampf (3 CFGs activos en partidos)

> **NOTA:** ActionParser parsea SOLO los 3 CFGs activos en partidos 4v4.
> `demoGoToBallAndKickImpl.cfg` NO se parsea — no forma parte del pipeline RL.

```python
class ActionParser:
    """Parsea los 3 action CFGs activos en partidos 4v4: kickAtGoal, passToTeammate, zweikampf."""

    def __init__(self, base: BHumanCfgParser):
        self.base = base

    def parse_kick_at_goal(self, filepath: Path) -> KickAtGoalConfig:
        raw = self.base.parse_file(filepath)
        return KickAtGoalConfig(
            hysteresis_number=int(raw.get("hysteresisNumber", 300)),
            hysteresis_kick_range_ext=float(raw.get("hysteresisKickRangeExtension", 500)),
            hysteresis_angle=float(raw.get("hysteresisAngle", 10)),
            min_opening_angle=float(raw.get("minOpeningAngle", 6)),
            kick_inaccuracy=float(raw.get("kickInaccuracy", 10)),
            hysteresis_decision_penalty=int(raw.get("hysteresisDecisionPenalty", 500)),
            allowed_kicks=raw.get("allowedKicks", []),  # identifiers planos, no structs
            allowed_kick_ranges=raw.get("allowedKickRanges", []),
        )

    def parse_pass_to_teammate(self, filepath: Path) -> PassToTeammateConfig:
        """Parsear passToTeammateImpl.cfg — incluye TODOS los campos del archivo real."""
        raw = self.base.parse_file(filepath)
        return PassToTeammateConfig(
            min_kick_length_for_long=float(raw.get("minKickLengthForLong", 2813)),
            rating_threshold=float(raw.get("ratingThreshold", 0.7)),
            pass_ahead_distance=float(raw.get("passAheadDistance", 0)),
            min_angle_offset=float(raw.get("minAngleOffset", 7)),
            max_angle_offset=float(raw.get("maxAngleOffset", 25)),
            max_angle_offset_behind=float(raw.get("maxAngleOffsetBehind", 10)),
            max_precision_offset=float(raw.get("maxPrecisionOffset", 5)),
            waiting_position_offset_x=float(raw.get("waitingPositionOffset", {}).get("x", -94)),
            waiting_position_offset_y=float(raw.get("waitingPositionOffset", {}).get("y", 0)),
            target_offset_to_field_border=float(raw.get("targetOffsetToFieldBorder", 500)),
            time_left_to_adjust=int(raw.get("timeLeftToAdjust", 7000)),
            min_time_waiting=int(raw.get("minTimeWaiting", 3000)),
            max_time_waiting=int(raw.get("maxTimeWaiting", 8000)),
            max_time_waiting_factor=float(raw.get("maxTimeWaitingFactor", 0.1)),
            kick_range_offset=float(raw.get("kickRangeOffset", 313)),
            average_kick_execution_time=int(raw.get("averageKickExecutionTime", 800)),
            average_kick_ball_offset_x=float(raw.get("averageKickBallOffset", {}).get("x", -185)),
            average_kick_ball_offset_y=float(raw.get("averageKickBallOffset", {}).get("y", 0)),
            ignore_obstacles_threshold=float(raw.get("ignoreObstaclesThreshold", 250)),
            ignore_dynamic_obstacles_threshold=float(raw.get("ignoreDynamicObstaclesThreshold", 500)),
            do_special_kick_checks=bool(raw.get("doSpecialKickChecks", True)),
            look_ahead=bool(raw.get("lookAhead", False)),  # false en 4v4 — NO es dead flag aqui
            max_look_ahead_time=int(raw.get("maxLookAheadTime", 3000)),
            interception_distance_threshold_min=float(raw.get("interceptionDistanceThreshold", {}).get("min", 100)),
            interception_distance_threshold_max=float(raw.get("interceptionDistanceThreshold", {}).get("max", 400)),
            target_distance_scale_min=float(raw.get("targetDistanceScale", {}).get("min", 0)),
            target_distance_scale_max=float(raw.get("targetDistanceScale", {}).get("max", 2500)),
            min_rating=float(raw.get("minRating", 0.01)),
            max_head_angle=float(raw.get("maxHeadAngle", 60)),   # deg — contexto sim2real cabeza
            head_speed=float(raw.get("headSpeed", 60)),          # deg — NO aprendible, contexto
            head_tilt=float(raw.get("headTilt", 15)),            # deg — NO aprendible, contexto
            allowed_kicks=[k if isinstance(k, str) else k.get("kick", "")
                          for k in raw.get("allowedKicks", [])],
        )

    def parse_zweikampf(self, filepath: Path) -> ZweikampfConfig:
        """Parsear zweikampfImpl.cfg — 6 subsistemas + parametros sueltos.
        TODOS los campos del archivo real estan cubiertos."""
        raw = self.base.parse_file(filepath)
        s = raw.get("searchParameters", {})
        r = raw.get("duelRatings", {})
        t = raw.get("duelTimings", {})
        o = raw.get("obstacleHandling", {})
        sb = raw.get("stealBallParameters", {})
        lk = raw.get("lastKickHysterese", {})
        return ZweikampfConfig(
            search=ZweikampfSearchParams(
                num_angles_near_best=int(s.get("numOfAnglesNearBestDuelPose", 3)),
                range_of_best_duel_pose=float(s.get("rangeOfBestDuelPose", 10)),
                num_overall_search=int(s.get("numOfOverallSearch", 5)),
                overall_search_range=float(s.get("overallSearchRange", 180)),
                bonus_steal_small=float(s.get("bonusForStealBallDirectionAdjustmentSmall", 0.01)),
                bonus_steal_big=float(s.get("bonusForStealBallDirectionAdjustmentBig", 3)),
                more_search_after_doing_nothing=int(s.get("moreSearchAfterDoingNothing", 50)),
                goal_sector_width=float(s.get("goalSectorWidth", 1)),
                min_max_angle_range_min=float(s.get("minMaxAngleAngleRange", {}).get("min", 4)),
                min_max_angle_range_max=float(s.get("minMaxAngleAngleRange", {}).get("max", 10)),
            ),
            ratings=ZweikampfDuelRatings(
                rating_min_max_difference=float(r.get("ratingMinMaxDifference", 0.3)),
                rating_ball_lands_in_own_half=float(r.get("ratingBallLandsInOwnHalf", 0.2)),
                rating_pose_blocked_small_kick_angle=float(r.get("ratingPoseBlockedSmallKickAngle", 0.3)),
                rating_pose_blocked=float(r.get("ratingPoseBlocked", 1.0)),
                rating_opponent_faster=float(r.get("ratingOpponentFaster", 0.1)),
                rating_same_kick=float(r.get("ratingSameKick", -0.1)),
                rating_same_kick_angle=float(r.get("ratingSameKickAngle", -0.1)),
                rating_steal_ball=float(r.get("ratingStealBall", -0.4)),
                goal_area_long_kick_rating=float(r.get("goalAreaLongKickRating", -0.5)),
                rating_steal_ball_kick=float(r.get("ratingStealBallKick", -0.9)),
                rating_steal_ball_kick_better_side=float(r.get("ratingStealBallKickBetterSide", -0.05)),
                rating_goal_shot=float(r.get("ratingGoalShot", -1.5)),
            ),
            timings=ZweikampfTimings(
                max_time_doing_nothing=int(t.get("maxTimeDoingNothing", 1000)),
                deactive_after_doing_nothing=int(t.get("deactiveAfterDoingNothing", 5000)),
                sideward_restriction_time=int(t.get("sidewardRestrictionTime", 4000)),
                no_kick_min_time=int(t.get("noKickMinTime", 200)),
                no_kick_steal_min_time=int(t.get("noKickStealMinTime", 250)),
                ignore_skill_request_time=int(t.get("ignoreSkillRequestTime", 2000)),
            ),
            obstacle_handling=ZweikampfObstacleHandling(
                left_foot_edge=(float(o.get("leftFootEdge", {}).get("x", 110)),
                                float(o.get("leftFootEdge", {}).get("y", 100))),
                right_foot_edge=(float(o.get("rightFootEdge", {}).get("x", 110)),
                                 float(o.get("rightFootEdge", {}).get("y", -100))),
                left_foot_kick_edge=(float(o.get("leftFootKickEdge", {}).get("x", 170)),
                                     float(o.get("leftFootKickEdge", {}).get("y", 100))),
                right_foot_kick_edge=(float(o.get("rightFootKickEdge", {}).get("x", 170)),
                                      float(o.get("rightFootKickEdge", {}).get("y", -100))),
                max_obstacle_dist_for_walk_steal=float(o.get("maxObstacleDistanceForWalkStealBallKick", 100)),
                max_obstacle_dist_for_side_steal=float(o.get("maxObstacleDistanceForSideStealBallKick", 450)),
                obstacle_dist_for_blocking_pose=float(o.get("obstacleDistanceForBlockingPose", 100)),
                obstacle_blocking_radius=float(o.get("obstacleBlockingRadius", 200)),
                obstacle_blocking_radius_sidewards_increase=float(o.get("obstacleBlockingRadiusSidewardsIncrease", 30)),
                no_kick_blocking_pose=float(o.get("noKickBlockingPose", 180)),
                obstacle_min_dist_for_back_pass=float(o.get("obstacleMinDistanceForBackPass", 600)),
                obstacle_shift_range_min=float(o.get("obstacleShiftRange", {}).get("min", -100)),
                obstacle_shift_range_max=float(o.get("obstacleShiftRange", {}).get("max", 30)),
                max_obstacle_dist_to_ball_for_risky_kicks=float(o.get("maxObstacleDistanceToBallForRiskyKicks", 600)),
                max_obstacle_dist_to_force_forward_steal=float(o.get("maxObstacleDistanceToForceForwardSteal", 200)),
            ),
            steal_ball=ZweikampfStealBallParams(
                steal_ball_type_too_much_rotation_min=float(sb.get("stealBallTypeTooMuchRotationMinValue", 25)),
                bonus_for_steal_ball_min_max_angle=float(sb.get("bonusForStealBallMinMaxAngle", 120)),
                steal_ball_range_min=float(sb.get("stealBallRange", {}).get("min", 88)),
                steal_ball_range_max=float(sb.get("stealBallRange", {}).get("max", 95)),
                steal_ball_min_robot_rotation=float(sb.get("stealBallMinRobotRotation", 70)),
                steal_ball_min_position_rotation=float(sb.get("stealBallMinPositionRotation", 45)),
                forward_steal_opponent_normal_min=float(sb.get("forwardStealBallOpponentPositionNormal", {}).get("min", 50)),
                forward_steal_opponent_normal_max=float(sb.get("forwardStealBallOpponentPositionNormal", {}).get("max", 70)),
                forward_steal_opponent_malus=float(sb.get("forwardStealBallOpponentPositionMalus", 40)),
            ),
            last_kick_hysterese=ZweikampfLastKickHysterese(
                kick_length_hysterese=float(lk.get("kickLengthHysterese", 500)),
                kick_direction_hysterese=float(lk.get("kickDirectionHysterese", 20)),
            ),
            smaller_kick_angle_preference=float(raw.get("smallerKickAnglePreference", 15)),
            forward_steal_preference_range=float(raw.get("forwardStealPreferenceRange", 10)),
            replace_normal_to_long_goal_shot=float(raw.get("replaceNormalToLongKickGoalShot", 10)),
            sector_wheel_obstacle_ball_radius_factor=float(raw.get("sectorWheelObstacleBallRadiusFactor", 3)),
            allow_pass_angle_min=float(raw.get("allowPassAngle", {}).get("min", -45)),
            allow_pass_angle_max=float(raw.get("allowPassAngle", {}).get("max", 45)),
            field_border_safe_distance=float(raw.get("fieldBorderSafeDistance", 625)),
            min_forward_turn_range=float(raw.get("minForwardTurnRange", 625)),
            sidewards_rotation_correction_bonus=float(raw.get("sidewardsRotationCorrectionBonus", 5)),
            ball_position_interpolation_range_min=float(raw.get("ballPositionInterpolationRange", {}).get("min", 250)),
            ball_position_interpolation_range_max=float(raw.get("ballPositionInterpolationRange", {}).get("max", 500)),
            look_active_min_ball_distance=float(raw.get("lookActiveMinBallDistance", 500)),
            replace_forward_with_long_goal_shot=bool(raw.get("replaceForwardWithLongGoalShot", True)),
            allowed_kicks=raw.get("allowedKicks", []),
        )

    # parse_demo_go_to_ball ELIMINADO — demoGoToBallAndKickImpl.cfg no se parsea.
    # No activo en partidos 4v4. Ver nota en ZweikampfConfig.
```

### 11.7 Output: Dataclasses para strategyBehaviorControl y skillBehaviorControl

```python
@dataclass
class StrategyBehaviorControlConfig:
    """FUENTE: Config/Scenarios/4v4_Scaled/strategyBehaviorControl.cfg
    Confirma que la strategy activa es s4v4_scaled. Parseo OBLIGATORIO."""
    strategy: str = "s4v4_scaled"

@dataclass
class SkillBehaviorControlConfig:
    """FUENTE: Config/Scenarios/4v4_Scaled/skillBehaviorControl.cfg (19 lineas)
    Define opciones/gates de comportamiento disponibles y activas en match-play."""
    options: list[str] = None          # [HandlePhysicalRobot, HandlePlayerState, ...]
    playing_options: list[str] = None  # [HandleGoalkeeperCatchBall, HandleStrikerLostBall, ...]
    use_new_handle_catch_ball: bool = True
```

### 11.7b Logica de Parseo: strategyBehaviorControl y skillBehaviorControl

```python
class StrategyBehaviorControlParser:
    def __init__(self, base: BHumanCfgParser):
        self.base = base

    def parse(self, filepath: Path) -> StrategyBehaviorControlConfig:
        raw = self.base.parse_file(filepath)
        strategy_name = raw.get("strategy", "")
        assert strategy_name == "s4v4_scaled", \
            f"strategyBehaviorControl.cfg apunta a '{strategy_name}', esperado 's4v4_scaled'"
        return StrategyBehaviorControlConfig(strategy=strategy_name)

class SkillBehaviorControlParser:
    def __init__(self, base: BHumanCfgParser):
        self.base = base

    def parse(self, filepath: Path) -> SkillBehaviorControlConfig:
        raw = self.base.parse_file(filepath)
        return SkillBehaviorControlConfig(
            options=raw.get("options", []),
            playing_options=raw.get("playingOptions", []),
            use_new_handle_catch_ball=bool(raw.get("useNewHandleCatchBallBehavior", True)),
        )
```

### 11.8 Uso en environment.py: Los 6 CFGs que completan el cierre

Cada uno de los 6 CFGs que antes quedaban "flojos" tiene un uso concreto en el sistema RL:

```python
# === zweikampfImpl.cfg -> Fase 3: metadata y constraints de duelo ===
# En environment.py / wrapper.py: NO calcular reward aqui.
def _validate_duel_kick(self, kick_type: str) -> bool:
    """Solo permite kicks que zweikampfImpl.cfg declara validos."""
    return kick_type in self._cfgs.zweikampf.allowed_kicks

def _get_duel_metadata(self, stole_ball: bool, scored_goal: bool) -> dict:
    """Expone senales y defaults de duelo para que reward.py decida si las usa."""
    return {
        "stole_ball": stole_ball,
        "scored_goal": scored_goal,
        "rating_steal_ball": self._cfgs.zweikampf.rating_steal_ball,
        "rating_goal_shot": self._cfgs.zweikampf.rating_goal_shot,
    }

# === zweikampfImpl.cfg → constraint: approach timing en duelo ===
def _can_kick_again(self, time_since_last_kick_ms: int) -> bool:
    """Cooldown minimo entre kicks en duelo, respeta zweikampfImpl.cfg noKickMinTime."""
    return time_since_last_kick_ms >= self._cfgs.zweikampf.timings.no_kick_min_time

def _is_zweikampf_active(self, time_doing_nothing_ms: int) -> bool:
    """Zweikampf se desactiva si el robot no hace nada mas de deactiveAfterDoingNothing."""
    return time_doing_nothing_ms < self._cfgs.zweikampf.timings.deactive_after_doing_nothing

# === libWalkProvider.cfg → movement constraints ===
def _clamp_position(self, x: float, y: float) -> tuple[float, float]:
    """Aplica bordes del campo con expansion de libWalkProvider.cfg."""
    border = self._cfgs.walk.field_border_expansion  # 300mm
    x_limit = self._cfgs.field_symbols["xPosOpponentGroundLine"] + border
    y_limit = self._cfgs.field_symbols["yPosLeftSideline"] + border
    return (
        np.clip(x, -x_limit, x_limit),
        np.clip(y, -y_limit, y_limit),
    )

def _avoid_obstacle(self, pos: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """Evasion de obstaculos usando radios de libWalkProvider.cfg."""
    dist = np.linalg.norm(pos - obstacle)
    if dist < self._cfgs.walk.obstacle_avoidance_min_radius:
        direction = (pos - obstacle) / max(dist, 1e-6)
        return obstacle + direction * self._cfgs.walk.obstacle_avoidance_distance
    return pos

# === behaviorParameters.cfg → keeper zones y constraints ===
def _get_keeper_action_zone(self) -> dict[str, float]:
    """Radios de accion del portero desde behaviorParameters.cfg."""
    return {
        "stand": self._cfgs.behavior.stand_radius,       # 60mm
        "walk": self._cfgs.behavior.walk_radius,          # 125mm
        "genuflect": self._cfgs.behavior.genuflect_radius,# 200mm
        "jump": self._cfgs.behavior.jump_radius,          # 600mm
    }

# === strategyBehaviorControl.cfg → assertion de integridad ===
# Verificado en load_all(): strategy == s4v4_scaled
# Si falla, load_all() lanza AssertionError ANTES de que el RL inicie.
# Uso adicional: confirmar que el scenario cargado es el correcto.
assert self._cfgs.strategy_behavior_control.strategy == "s4v4_scaled"

# === skillBehaviorControl.cfg → opciones y gates de ejecucion ===
def _get_available_playing_options(self) -> list[str]:
    """Opciones de juego activas desde skillBehaviorControl.cfg."""
    return self._cfgs.skill_behavior_control.playing_options
    # → ["HandleGoalkeeperCatchBall", "HandleStrikerLostBall", "HandleCatchBall"]
```

Nota de arquitectura:
- `reward.py` puede consumir estas senales mas adelante, pero el parser, el loader y el bridge NO calculan reward.
- En Fase 1 este bloque queda fuera del camino critico; se conserva solo como extension futura para Fase 3.

### 11.9 Uso en config.py

```python
# En lugar de:
#   x_opponent_ground: float = 4500.0  (5v5 INCORRECTO)

walk = parser.parse_walk_config()
field = parser.field_symbols

config = FieldConfig(
    x_opponent_ground=field["xPosOpponentGroundLine"],  # 2813 (4v4)
    x_own_ground=field["xPosOwnGroundLine"],            # -2813
    y_left_sideline=field["yPosLeftSideline"],           # 1875
    y_right_sideline=field["yPosRightSideline"],         # -1875
    max_forward_speed=walk.max_forward,                  # 250
)
```

---

<a name="12-validacion"></a>
## 12. Modulo 7: Validacion y Roundtrip

### 12.1 Validacion de Rangos

```python
VALIDATION_SCHEMA = {
    # Strategy
    "ballXThreshold": {"type": float, "min": -2813, "max": 2813},
    "timeSinceBallAheadOfThresholdGE": {"type": int, "min": 0, "max": 10000},
    "numOfFieldPlayersGE": {"type": int, "min": 0, "max": 4},

    # Tactics
    "position_x": {"type": float, "min": -3500, "max": 3500},
    "position_y": {"type": float, "min": -2000, "max": 2000},

    # Actions
    "shootThreshold": {"type": float, "min": 0.0, "max": 1.0},
    "minOpeningAngle": {"type": float, "min": 0, "max": 90},
    "ratingThreshold": {"type": float, "min": 0.0, "max": 1.0},
    "minPassDistance": {"type": float, "min": 0, "max": 10000},
    "maxPassDistance": {"type": float, "min": 0, "max": 10000},

    # Kicks
    "range_min": {"type": float, "min": 0, "max": 10000},
    "range_max": {"type": float, "min": 0, "max": 10000},
    "ball_velocity_min": {"type": float, "min": 0, "max": 5000},
}
```

### 12.2 Hash de CFGs para Trazabilidad

```python
import hashlib

def compute_cfg_hash(cfg_paths: list[Path]) -> str:
    """Hash determinista de todos los CFGs usados en training."""
    combined = hashlib.sha256()
    for path in sorted(cfg_paths):
        with open(path, "rb") as f:
            combined.update(f.read())
    return combined.hexdigest()

# Guardar hash con el modelo entrenado:
model_metadata = {
    "cfg_hash": compute_cfg_hash(cfg_files),
    "field_symbols": "4v4_scaled",
    "training_date": "2026-03-21",
}
```

### 12.3 Consistency Checks

```python
def validate_consistency(strategy: StrategyConfig, tactics: dict[str, TacticConfig]):
    """Verificar que strategy referencia tacticas que existen."""
    errors = []
    for state in strategy.tactics:
        if state.tactic not in tactics:
            errors.append(f"Strategy references tactic '{state.tactic}' but it wasn't parsed")
        for trans in state.transitions:
            if trans.to not in tactics:
                errors.append(f"Transition to '{trans.to}' but tactic not found")
    if errors:
        raise ValueError(f"Consistency check failed:\n" + "\n".join(errors))
```

---

<a name="13-integracion"></a>
## 13. Integracion con environment.py y config.py

### 13.1 API de Alto Nivel: loader.py

```python
@dataclass
class AllConfigs:
    # --- Capa 1: Strategy ---
    field_symbols: dict[str, float]                # fieldDimensions.cfg (30+ simbolos)
    strategy: StrategyConfig                       # s4v4_scaled.cfg (FSM)
    # --- Capa 2: Tactics ---
    tactics: dict[str, TacticConfig]               # t121/t13_4v4_scaled.cfg (posiciones)
    # --- Capa 3: Actions ---
    kick_at_goal: KickAtGoalConfig                 # kickAtGoalImpl.cfg (disparo)
    pass_to_teammate: PassToTeammateConfig         # passToTeammateImpl.cfg (pase)
    zweikampf: ZweikampfConfig                     # zweikampfImpl.cfg (duelo — 6 subsistemas)
    # demo_go_to_ball ELIMINADO — demoGoToBallAndKickImpl.cfg no activo en partidos 4v4
    smash_or_pass: SmashOrPassParams               # PlayBall.h defaults (decision C++)
    # --- Capa 4: Physics ---
    kicks: dict[str, KickInfo]                     # kickInfo.cfg (23 pateos)
    # --- Capa 5: Movement ---
    walk: WalkConfig                               # walkingEngine.cfg + libWalkProvider.cfg
    # --- Capa 6: Global ---
    behavior: BehaviorParams                       # behaviorParameters.cfg
    strategy_behavior_control: StrategyBehaviorControlConfig  # strategyBehaviorControl.cfg
    skill_behavior_control: SkillBehaviorControlConfig       # skillBehaviorControl.cfg

def load_all(config_root: str | Path, scenario: str = "4v4_Scaled") -> AllConfigs:
    """
    Parsear TODOS los CFGs in-scope del RL/bridge.

    Implementacion recomendada:
    - Primero cerrar `load_phase1()` con `field_symbols`, `strategy`, `tactics` y `walk`.
    - Luego extender a `load_all()` agregando `action_parser`, `kickinfo` y el bridge contract.
    - El salto a Fase 2/3 NO debe reescribir `cfg_tokenizer.py`, `cfg_parser.py`,
      `strategy_parser.py` ni `tactic_parser.py`.

    Archivos parseados (12 grupos de CFG / 13 archivos contando ambas tacticas + 1 C++ header):
    - Config/Locations/4v4_Scaled/fieldDimensions.cfg                    → field_symbols
    - Config/Behavior/Strategies/s4v4_scaled.cfg                         → strategy
    - Config/Behavior/Tactics/t*_4v4_scaled.cfg                          → tactics
    - Config/Scenarios/{scenario}/BehaviorControl/kickAtGoalImpl.cfg     → kick_at_goal
    - Config/Scenarios/{scenario}/BehaviorControl/passToTeammateImpl.cfg → pass_to_teammate
    - Config/Scenarios/{scenario}/BehaviorControl/zweikampfImpl.cfg      → zweikampf
    - Config/Scenarios/{scenario}/kickInfo.cfg                           → kicks
    # NOTA: demoGoToBallAndKickImpl.cfg NO se parsea — solo activo en LibDemo/AnyPlaceDemo.
    - Config/Scenarios/{scenario}/libWalkProvider.cfg                    → walk (evasion)
    - Config/Scenarios/{scenario}/behaviorParameters.cfg                 → behavior
    - Config/Scenarios/{scenario}/strategyBehaviorControl.cfg            → strategy_behavior_control
    - Config/Scenarios/{scenario}/skillBehaviorControl.cfg               → skill_behavior_control
    - walkingEngine efectivo segun search path de B-Human                → walk (velocidades)
      En este repo, para robots nombrados de Config/Robots/robots.cfg,
      el efectivo es Config/Robots/Default/walkingEngine.cfg.
      Config/Robots/Nao/Nao/walkingEngine.cfg solo aplica si
      headName == bodyName == "Nao".

    Archivos NO parseados (declaracion explicita):
    - PlayBall.h: Parametros de SmashOrPass leidos del C++ fuente, no de .cfg.
      Se instancian como SmashOrPassParams con los defaults del header.
      Razon: Estos parametros no tienen .cfg override en 4v4_Scaled.
    - ExpectedGoalsProvider.h / PassEvaluationProvider.h: Modulos de evaluacion
      con parametros propios. No se parsean porque RL no los ajusta — los usa
      como cajas negras via theExpectedGoals/thePassEvaluation.

    Args:
        config_root: Raiz del directorio Config/ de BHuman
        scenario: "4v4_Scaled" (default) o "Default"

    Returns:
        AllConfigs con toda la informacion parseada y validada.
    """
    root = Path(config_root)
    base = BHumanCfgParser(symbols=FIELD_SYMBOLS_4V4_SCALED)

    # Parsear cada capa
    strategy = StrategyParser(base).parse(root / "Behavior/Strategies/s4v4_scaled.cfg")
    tactics = {
        f.stem: TacticParser(base).parse(f)
        for f in (root / "Behavior/Tactics").glob("t*_4v4_scaled.cfg")
    }
    kick_at_goal = ActionParser(base).parse_kick_at_goal(
        root / f"Scenarios/{scenario}/BehaviorControl/kickAtGoalImpl.cfg"
    )
    pass_to_teammate = ActionParser(base).parse_pass_to_teammate(
        root / f"Scenarios/{scenario}/BehaviorControl/passToTeammateImpl.cfg"
    )
    zweikampf = ActionParser(base).parse_zweikampf(
        root / f"Scenarios/{scenario}/BehaviorControl/zweikampfImpl.cfg"
    )
    # demoGoToBallAndKickImpl.cfg NO se parsea — no activo en partidos 4v4.
    kicks = KickInfoParser(base).parse(root / f"Scenarios/{scenario}/kickInfo.cfg")

    # Walk: combinar libWalkProvider (evasion) + walkingEngine efectivo.
    # En este repo, para robots nombrados, el efectivo cae en Robots/Default.
    walk = WalkParser(base).parse(
        walk_engine_path=root / "Robots/Default/walkingEngine.cfg",
        lib_walk_path=root / f"Scenarios/{scenario}/libWalkProvider.cfg",
    )

    # Behavior params: behaviorParameters.cfg del scenario
    behavior = BehaviorParser(base).parse(
        root / f"Scenarios/{scenario}/behaviorParameters.cfg"
    )

    # Strategy behavior control: confirma que strategy = s4v4_scaled (TIPADO)
    strategy_bc = StrategyBehaviorControlParser(base).parse(
        root / f"Scenarios/{scenario}/strategyBehaviorControl.cfg"
    )

    # Skill behavior control: opciones de comportamiento (TIPADO)
    skill_bc = SkillBehaviorControlParser(base).parse(
        root / f"Scenarios/{scenario}/skillBehaviorControl.cfg"
    )

    # Validar consistencia
    validate_consistency(strategy, tactics)

    return AllConfigs(
        field_symbols=FIELD_SYMBOLS_4V4_SCALED,
        strategy=strategy,
        tactics=tactics,
        kick_at_goal=kick_at_goal,
        pass_to_teammate=pass_to_teammate,
        zweikampf=zweikampf,
        smash_or_pass=SmashOrPassParams(),  # defaults de PlayBall.h (no hay .cfg override)
        kicks=kicks,
        walk=walk,
        behavior=behavior,
        strategy_behavior_control=strategy_bc,
        skill_behavior_control=skill_bc,
    )
```

### 13.2 Integracion en config.py

```python
# config.py — ANTES (incorrecto):
# x_opponent_ground: float = 4500.0   ← 5v5

# config.py — DESPUES:
from bhumanparser.loader import load_all, FIELD_SYMBOLS_4V4_SCALED

# Si CFGs disponibles:
try:
    _cfgs = load_all(CFG_ROOT)
    FIELD = _cfgs.field_symbols
except FileNotFoundError:
    logger.warning("CFGs no disponibles, usando fallback 4v4 Scaled")
    FIELD = FIELD_SYMBOLS_4V4_SCALED  # Fallback con valores correctos

@dataclass
class FieldConfig:
    x_opponent_ground: float = FIELD["xPosOpponentGroundLine"]  # 2813
    x_own_ground: float = FIELD["xPosOwnGroundLine"]            # -2813
    y_left_sideline: float = FIELD["yPosLeftSideline"]           # 1875
    y_right_sideline: float = FIELD["yPosRightSideline"]         # -1875
```

### 13.3 Integracion en environment.py

```python
# environment.py — ANTES (incorrecto):
# self._current_tactic = "t4"
# home_poses = {"agent_0": np.array([-2350.0, 0.0]), ...}

# environment.py — DESPUES:
from bhumanparser.loader import load_all

class RoboCupEnv:
    def __init__(self, cfg_root=None):
        if cfg_root:
            self._cfgs = load_all(cfg_root)
        else:
            self._cfgs = load_all_fallback()  # Valores hardcoded correctos

        # FSM activa
        self._strategy = self._cfgs.strategy
        self._current_tactic = self._strategy.tactics[0].tactic  # "t121_4v4_scaled"
        self._time_in_tactic = 0

        # Tacticas con posiciones reales
        self._tactics = self._cfgs.tactics

        # Tactic names para one_hot (solo las 2 activas)
        self._tactic_names = [s.tactic for s in self._strategy.tactics]
        # → ["t121_4v4_scaled", "t13_4v4_scaled"]

    def _get_home_poses(self) -> dict[str, tuple[float, float]]:
        """Home poses de la tactica actual (desde .cfg)"""
        return self._tactics[self._current_tactic].get_field_player_positions()

    def _write_tactic_obs(self, obs: np.ndarray, offset: int) -> int:
        """Escribir one_hot de tactica actual (2 bits, no 9)"""
        idx = self._tactic_names.index(self._current_tactic)
        obs[offset + idx] = 1.0
        return offset + len(self._tactic_names)

    def _step_fsm(self, ball_x, num_field_players, dt_ms):
        """Evaluar transicion de tactica cada step"""
        self._time_in_tactic += dt_ms
        current = next(s for s in self._strategy.tactics
                       if s.tactic == self._current_tactic)
        for trans in current.transitions:
            if self._check_transition(trans, ball_x, num_field_players):
                self._current_tactic = trans.to
                self._time_in_tactic = 0
                break
```

---

<a name="14-dataclasses"></a>
## 14. Resumen de Dataclasses de Salida

```
AllConfigs                                    # 12 grupos de CFG parseados + 1 C++ header
│
├── CAPA 1: STRATEGY
│   ├── field_symbols: dict[str, float]       # fieldDimensions.cfg (30+ simbolos)
│   └── strategy: StrategyConfig              # s4v4_scaled.cfg
│       ├── tactics: list[TacticState]        #   2 estados (t121, t13)
│       │   └── transitions: list[Transition]
│       │       └── conditions: list[TransitionCondition]
│       └── set_plays: kickoffs, penalties, corners
│
├── CAPA 2: TACTICS
│   └── tactics: dict[str, TacticConfig]      # t121/t13_4v4_scaled.cfg
│       ├── positions: list[RolePosition]     #   4-5 posiciones por tactica
│       │   └── pose: PoseInfo (x, y, rotation)
│       └── priority_groups: list[PriorityGroup]
│
├── CAPA 3: ACTIONS
│   ├── kick_at_goal: KickAtGoalConfig        # kickAtGoalImpl.cfg → disparo
│   ├── pass_to_teammate: PassToTeammateConfig # passToTeammateImpl.cfg → pase
│   ├── zweikampf: ZweikampfConfig            # zweikampfImpl.cfg → duelo (Fase 3, 6 subsistemas)
│   │   ├── search_params: ZweikampfSearchParams       # busqueda de angulo de duelo
│   │   ├── ratings: ZweikampfDuelRatings              # puntuacion de acciones (-1.5 a +1.0)
│   │   ├── timings: ZweikampfTimings                  # timeouts y cooldowns
│   │   ├── obstacle_handling: ZweikampfObstacleHandling # geometria de pies y obstaculos
│   │   ├── steal_ball: ZweikampfStealBallParams       # angulos para robo de balon
│   │   └── last_kick: ZweikampfLastKickHysterese      # histeresis de ultimo kick
│   │   # demo_go_to_ball ELIMINADO — demoGoToBallAndKickImpl.cfg no activo en partidos 4v4
│   └── smash_or_pass: SmashOrPassParams      # PlayBall.h defaults (C++, no .cfg)
│       ├── shoot_threshold, min/max_pass_distance
│       ├── min/max_penalty, sample_distance, sample_steps_min/max
│       └── min_rating, ball_dist_min/max_penalty (HARD constraints)
│
├── CAPA 4: PHYSICS
│   └── kicks: dict[str, KickInfo]            # kickInfo.cfg → 23 tipos de pateo
│       └── range, velocity, time, leg, motion
│
├── CAPA 5: MOVEMENT
│   └── walk: WalkConfig                      # walkingEngine.cfg + libWalkProvider.cfg
│       └── max_forward/sideways/backwards, field_border_expansion, avoidance
│
└── CAPA 6: GLOBAL
    ├── behavior: BehaviorParams                          # behaviorParameters.cfg
    ├── strategy_behavior_control: StrategyBehaviorControlConfig  # strategyBehaviorControl.cfg
    └── skill_behavior_control: SkillBehaviorControlConfig       # skillBehaviorControl.cfg
```

---

<a name="14b-traceability"></a>
## 14b. Mapa de Trazabilidad: CFG/C++ → Dataclass → Tensor RL

Esta seccion cierra la cadena completa: **ruta → linea → parametro → parser → dataclass → tensor RL → punto de uso**.

### 14b.1 Cadena Completa por Parametro

```
FUENTE (CFG o C++)            → PARSER              → DATACLASS FIELD          → USO EN RL
─────────────────────────────────────────────────────────────────────────────────────────────
fieldDimensions.cfg:L4        → field_symbols.py     → field_symbols["xPos..."] → config.py: limites campo
                                                                                  environment.py: normalizacion
                                                                                  reward.py: out-of-bounds penalty

s4v4_scaled.cfg:L3-30        → strategy_parser.py   → StrategyConfig.tactics    → environment.py: _step_fsm()
                                                       .transitions.conditions     tactic one_hot en obs

t121_4v4_scaled.cfg:L1-20    → tactic_parser.py     → TacticConfig.positions    → environment.py: _get_home_poses()
                                                       [].pose.(x,y)              baseline para Fase 1 positioning

PlayBall.h:L20                → SmashOrPassParams()  → smash_or_pass            → environment.py: default si RL falla
(C++ default, no .cfg)          (instanciado directo)  .shoot_threshold           Fase 2: rl_predict_threshold()

PlayBall.h:L22-23             → SmashOrPassParams()  → smash_or_pass            → Fase 2: rl_predict_distances()
                                                       .min/max_pass_distance

PlayBall.h:L26-27             → SmashOrPassParams()  → smash_or_pass            → Fase 2: rl_predict_hysteresis()
                                                       .min/max_penalty

PlayBall.h:L33-34             → SmashOrPassParams()  → smash_or_pass            → Fase 2: rl_predict_sampling()
                                                       .sample_distance/steps

kickAtGoalImpl.cfg:L4         → action_parser.py     → KickAtGoalConfig         → Fase 2: rl_predict_angle()
                                                       .min_opening_angle

passToTeammateImpl.cfg:L15-17  → action_parser.py     → PassToTeammateConfig     → Fase 2: rl_predict_pass_params()
                                                       .rating_threshold
                                                       .pass_ahead_distance
                                                       .min_angle_offset

zweikampfImpl.cfg:L1-84       → action_parser.py     → ZweikampfConfig           → Fase 3: metadata de duelo
                                                       .rating_steal_ball          reward.py decide si lo usa
                                                       .allowed_kicks              constraint: kicks validos en duelo

zweikampfImpl.cfg:timings     → action_parser.py     → ZweikampfTimings          → constraint: cooldown entre kicks
                                                       .no_kick_min_time           _can_kick_again() en duelo
                                                       .deactive_after_doing_nothing _is_zweikampf_active()

kickInfo.cfg:L1-347           → kickinfo_parser.py   → kicks["forwardFastRight"] → sim: simulate_kick() validation
                                                       .range_min/max             reward: kick feasibility check
                                                       .ball_velocity_min/max

walkingEngine.cfg:L...        → walk_parser.py       → WalkConfig.max_forward   → config.py: speed limits
libWalkProvider.cfg:L...                               .field_border_expansion    environment.py: movement clamp

behaviorParameters.cfg:L1-11  → behavior_parser.py   → BehaviorParams            → constraint: portero zones
                                                       .jump_radius, .stand_radius  reward: keeper behavior

strategyBehaviorControl.cfg   → StrategyBehaviorControlParser → strategy_behavior_control  → assertion: strategy == s4v4_scaled
                                .parse()                         .strategy                     load_all() falla si != s4v4_scaled

skillBehaviorControl.cfg      → SkillBehaviorControlParser    → skill_behavior_control        → constraint: playing_options validas
                                .parse()                         .options, .playing_options       _get_available_playing_options()
```

### 14b.2 Clasificacion Formal de Parametros

Cada parametro del sistema cae en exactamente UNA de estas categorias:

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ CATEGORIA 1: FUENTE CFG REAL                                                         │
│ Parametros que vienen de archivos .cfg parseados por el BhumanParser.                │
│ Son la fuente de verdad operativa. Si alguien modifica el .cfg, el RL lo refleja.    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ fieldDimensions.cfg      → Todos los simbolos del campo (30+)                        │
│ s4v4_scaled.cfg          → ballXThreshold, timeSinceBall*, numOfFieldPlayers*        │
│ t121/t13_4v4_scaled.cfg  → positions[].pose, priorityGroups                          │
│ kickAtGoalImpl.cfg       → hysteresisNumber, minOpeningAngle, kickInaccuracy, etc.   │
│ passToTeammateImpl.cfg   → minKickLengthForLong, ratingThreshold, minAngleOffset,    │
│                            maxAngleOffset, minTimeWaiting, maxTimeWaiting, etc.      │
│ zweikampfImpl.cfg        → searchParameters, duelRatings, duelTimings,               │
│                            obstacleHandling, stealBallParameters, lastKickHysterese  │
│ kickInfo.cfg             → 23 tipos de pateo: range, velocity, executionTime         │
│ libWalkProvider.cfg      → fieldBorderExpansion, avoidance distances                 │
│ walkingEngine.cfg        → maxSpeed forward/sideways/backwards/rotation              │
│ behaviorParameters.cfg   → keeperJumpingOn, standRadius, walkRadius, jumpRadius      │
│ strategyBehaviorControl  → strategy = s4v4_scaled (confirmacion)                     │
│ skillBehaviorControl     → options[], playingOptions[] (gating y referencia runtime)   │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│ CATEGORIA 2: DEFAULT DESDE C++ (sin .cfg override)                                   │
│ Parametros definidos en headers C++ que NO tienen override en .cfg de 4v4_Scaled.     │
│ El parser los instancia directamente desde los valores del header.                   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ PlayBall.h:L20    shootThreshold = 0.8f                                              │
│ PlayBall.h:L22    minPassDistance = 1500.f                                            │
│ PlayBall.h:L23    maxPassDistance = 4500.f                                            │
│ PlayBall.h:L24    maxFreeKickDistance = 6000.f                                        │
│ PlayBall.h:L25    maxGoalKickDistance = 9000.f                                        │
│ PlayBall.h:L26    minPenalty = 0.1f                                                   │
│ PlayBall.h:L27    maxPenalty = 0.8f                                                   │
│ PlayBall.h:L28    ballDistMinPenalty = 300.f                                          │
│ PlayBall.h:L29    ballDistMaxPenalty = 200.f                                          │
│ PlayBall.h:L30    minRating = 0.01f                                                   │
│ PlayBall.h:L33    sampleDistance = 500.f                                              │
│ PlayBall.h:L34    sampleSteps = (-1, 2)                                               │
│ PassEvaluationProvider.h:L34-43  (minValue, opponentDist*, distancePenalty*, etc.)    │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│ CATEGORIA 3: HARD CONSTRAINTS (RL NUNCA los modifica)                                │
│ Restricciones fisicas, reglamentarias, o de seguridad que RL debe RESPETAR.          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Campo:         field_x ∈ [-2813, 2813], field_y ∈ [-1875, 1875]                     │
│ Fisico:        max_walk_speed=250mm/s, max_rotation=120deg/s                         │
│ Pateo:         kick_range ∈ [400, 5400]mm, ball_velocity ≤ 1800mm/s                 │
│ Evasion:       fieldBorderExpansion=300mm, ballAvoidance=465mm                       │
│ Juego:         game_state==PLAYING, robot_upright==true, ball_seen < 5000ms          │
│ Reglamento:    maxFreeKickDistance=6000, maxGoalKickDistance=9000                     │
│ Thread:        cognition_cycle=33ms, rl_max_inference=3ms                            │
│ Hysteresis:    ballDistMin/MaxPenalty (controladas por proximidad, no aprendibles)    │
│ Decision:      alwaysShoot, ignoreObstacles (switches de debug, no aprendibles)      │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│ CATEGORIA 4: APRENDIBLE POR RL (parametros que RL puede ajustar o reemplazar)        │
│ El RL recibe el default como baseline y predice valores alternativos.                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Fase 1:                                                                              │
│   positions[].pose.(x,y)    — RL predice posiciones directamente                    │
│   (ballXThreshold, numFieldPlayers se mantienen del CFG)                             │
│                                                                                      │
│ Fase 2 (ademas de Fase 1):                                                           │
│   shootThreshold            — PlayBall.cpp:56                    rango [0.3, 1.0]    │
│   minPassDistance            — PlayBall.cpp:70                    rango [500, 3000]   │
│   maxPassDistance            — PlayBall.cpp:71                    rango [2000, 8000]  │
│   minPenalty                 — PlayBall.h:26                      rango [0.0, 0.3]    │
│   maxPenalty                 — PlayBall.h:27                      rango [0.3, 1.0]    │
│   sampleDistance             — PlayBall.h:33                      rango [200, 1000]   │
│   sampleSteps                — PlayBall.h:34                      rango variable      │
│   sampleValue (L82-84)      — RL puede reemplazar rating         rango [0.0, 1.0]    │
│   minOpeningAngle            — KickAtGoal.cpp:45                  rango [2, 15] deg   │
│   ratingThreshold            — PassToTeammate.cpp:61              rango [0.3, 0.9]    │
│   passAheadDistance          — PassToTeammate.cpp:62              rango [-500, 500]   │
│   minAngleOffset             — PassToTeammate.cpp:63              rango [2, 15] deg   │
│                                                                                      │
│ Fase 3 (ademas de Fase 2):                                                           │
│   ballXThreshold            — s4v4_scaled.cfg                    rango [0, 1000]     │
│   timeSinceBallAhead        — s4v4_scaled.cfg                    rango [100, 500]    │
│   Decision completa (shoot/pass/dribble) — reemplaza smashOrPass entero              │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

<a name="15-fallbacks"></a>
## 15. Fallbacks y Manejo de Errores

### 15.1 Cuando usar Fallback

El fallback se usa SOLO cuando los archivos .cfg de BHuman no estan disponibles (desarrollo local sin acceso al repo BHuman). El fallback SIEMPRE usa valores 4v4 Scaled verificados.

```python
def load_all_fallback() -> AllConfigs:
    """Valores hardcoded correctos (4v4 Scaled) para cuando no hay .cfg."""
    logger.warning(
        "Usando fallback hardcoded. Valores verificados contra 4v4 Scaled CFGs. "
        "Para maxima trazabilidad, usar load_all() con ruta a los .cfg reales."
    )
    return AllConfigs(
        field_symbols=FIELD_SYMBOLS_4V4_SCALED,
        strategy=_fallback_strategy(),
        tactics=_fallback_tactics(),
        kick_at_goal=_fallback_kick_at_goal(),
        pass_to_teammate=_fallback_pass_to_teammate(),
        zweikampf=ZweikampfConfig(),             # defaults de zweikampfImpl.cfg (6 subsistemas)
        # demo_go_to_ball ELIMINADO
        smash_or_pass=SmashOrPassParams(),        # defaults de PlayBall.h
        kicks=_fallback_kicks(),
        walk=WalkConfig(),                        # defaults
        behavior=BehaviorParams(),                # defaults
        strategy_behavior_control=StrategyBehaviorControlConfig(),  # default: s4v4_scaled
        skill_behavior_control=SkillBehaviorControlConfig(
            options=["HandlePhysicalRobot", "HandlePlayerState", "HandleReturnFromSideline",
                     "HandlePenaltyShootout", "HandlePenaltyKick", "HandleRefereeSignal",
                     "HandleGameState", "HandleIllegalAreas"],
            playing_options=["HandleGoalkeeperCatchBall", "HandleStrikerLostBall", "HandleCatchBall"],
        ),
    )

def _fallback_tactics() -> dict[str, TacticConfig]:
    return {
        "t121_4v4_scaled": TacticConfig(
            name="t121_4v4_scaled",
            positions=[
                RolePosition("goalkeeper", PoseInfo(-2532, 0, 0)),
                RolePosition("defenderL", PoseInfo(-1468, 0, 0)),
                RolePosition("defenderR", PoseInfo(-234.5, -625, 0)),
                RolePosition("forwardL", PoseInfo(-234.5, 625, 0)),
                RolePosition("forwardR", PoseInfo(703.5, 0, 0)),
            ],
            priority_groups=[
                PriorityGroup(["defenderR"], [1]),
                PriorityGroup(["forwardR"], [2]),
                PriorityGroup(["forwardL"], [3]),
                PriorityGroup(["defenderL"], [4]),
            ],
        ),
        "t13_4v4_scaled": TacticConfig(
            name="t13_4v4_scaled",
            positions=[
                RolePosition("goalkeeper", PoseInfo(-2532, 0, 0)),
                RolePosition("defenderL", PoseInfo(-1938, 0, 0)),
                RolePosition("forwardL", PoseInfo(234.5, 500, 0)),
                RolePosition("forwardR", PoseInfo(234.5, -500, 0)),
            ],
            priority_groups=[
                PriorityGroup(["defenderL"], [1]),
                PriorityGroup(["forwardL"], [2]),
                PriorityGroup(["forwardR"], [3]),
            ],
        ),
    }
```

### 15.2 Manejo de Errores

```python
# PRINCIPIO: Fail-loud, never silently degrade

# MAL (parser actual):
tactic_idx = TACTIC_NAMES.index(tactic) if tactic in TACTIC_NAMES else 0  # ← Silencioso!

# BIEN:
if tactic not in self._tactic_names:
    raise ValueError(
        f"Tactic '{tactic}' not found in strategy. "
        f"Available: {self._tactic_names}. "
        f"Did you mean one of the _4v4_scaled variants?"
    )
```

---

<a name="16-testing"></a>
## 16. Testing del Parser

### 16.1 Tests Minimos Requeridos

```python
# test_parser.py

def test_bridge_contract_complete():
    """Verificar que el Bridge Contract (Seccion 9.4) esta cubierto end-to-end.
    Cada decision point C++ tiene: AllConfigs.campo presente + rango valido."""
    cfgs = load_all(CFG_ROOT)

    # Decision 1: PlayBall.cpp smashOrPass — 8 params aprendibles
    assert 0.0 < cfgs.smash_or_pass.shoot_threshold <= 1.0       # L56
    assert cfgs.smash_or_pass.min_pass_distance > 0               # L70
    assert cfgs.smash_or_pass.max_pass_distance > cfgs.smash_or_pass.min_pass_distance  # L71
    assert cfgs.smash_or_pass.sample_distance > 0                 # L77
    assert cfgs.smash_or_pass.sample_steps_min < cfgs.smash_or_pass.sample_steps_max    # L78
    assert 0.0 <= cfgs.smash_or_pass.min_penalty < cfgs.smash_or_pass.max_penalty <= 1.0  # L26-27

    # Decision 2: PassToTeammate.cpp — 3 params aprendibles
    assert 0.0 < cfgs.pass_to_teammate.rating_threshold <= 1.0   # L61
    assert hasattr(cfgs.pass_to_teammate, 'pass_ahead_distance')  # L62
    assert cfgs.pass_to_teammate.min_angle_offset > 0             # L63

    # Decision 3: KickAtGoal.cpp — 1 param aprendible + 1 hard
    assert cfgs.kick_at_goal.min_opening_angle > 0                # L45
    assert cfgs.kick_at_goal.hysteresis_decision_penalty > 0      # L47 (HARD)

    # HARD constraints que RL NUNCA modifica:
    assert cfgs.smash_or_pass.min_rating > 0                      # L30
    assert cfgs.smash_or_pass.max_free_kick_distance > 0          # L24
    assert cfgs.smash_or_pass.max_goal_kick_distance > 0          # L25

def test_field_symbols_are_4v4_scaled():
    """FIELD_SYMBOLS deben ser 4v4 Scaled, NO 5v5 Default."""
    assert FIELD_SYMBOLS_4V4_SCALED["xPosOpponentGroundLine"] == 2813
    assert FIELD_SYMBOLS_4V4_SCALED["yPosLeftSideline"] == 1875
    assert FIELD_SYMBOLS_4V4_SCALED["centerCircleRadius"] == 469

def test_parse_expression_with_symbols():
    """Expresiones con simbolos se resuelven correctamente."""
    parser = BHumanCfgParser(symbols=FIELD_SYMBOLS_4V4_SCALED)
    result = parser.parse_string("x = xPosOwnGoalArea - 94;")
    assert result["x"] == -2438 - 94  # = -2532

def test_parse_tactic_t121():
    """Tactica t121_4v4_scaled se parsea con 5 posiciones."""
    tactic = parser.parse_tactic("t121_4v4_scaled.cfg")
    assert len(tactic.positions) == 5
    goalkeeper = next(p for p in tactic.positions if p.role_type == "goalkeeper")
    assert goalkeeper.pose.x == pytest.approx(-2532, abs=50)

def test_parse_tactic_t13():
    """Tactica t13_4v4_scaled se parsea con 4 posiciones (sin defenderR)."""
    tactic = parser.parse_tactic("t13_4v4_scaled.cfg")
    assert len(tactic.positions) == 4
    roles = [p.role_type for p in tactic.positions]
    assert "defenderR" not in roles

def test_strategy_references_valid_tactics():
    """Strategy solo referencia tacticas que existen."""
    cfgs = load_all(CFG_ROOT)
    for state in cfgs.strategy.tactics:
        assert state.tactic in cfgs.tactics

def test_home_poses_within_field():
    """Todas las home poses estan dentro del campo."""
    cfgs = load_all(CFG_ROOT)
    for name, tactic in cfgs.tactics.items():
        for pos in tactic.positions:
            assert -3500 <= pos.pose.x <= 3500, f"{name}/{pos.role_type} x={pos.pose.x}"
            assert -2000 <= pos.pose.y <= 2000, f"{name}/{pos.role_type} y={pos.pose.y}"

def test_tactic_names_match_strategy():
    """Las 2 tacticas activas del strategy existen en tactics parseados."""
    cfgs = load_all(CFG_ROOT)
    for state in cfgs.strategy.tactics:
        assert state.tactic in cfgs.tactics, f"'{state.tactic}' not in tactics"

def test_load_all_has_all_fields():
    """load_all() debe retornar TODOS los campos del Bridge Contract (Seccion 9.4)."""
    cfgs = load_all(CFG_ROOT)
    # Capa 3: Actions — todos los CFGs parseados
    assert cfgs.kick_at_goal is not None
    assert cfgs.pass_to_teammate is not None
    assert cfgs.zweikampf is not None
    assert cfgs.smash_or_pass is not None
    # Bridge contract fields
    assert hasattr(cfgs.smash_or_pass, 'shoot_threshold')
    assert hasattr(cfgs.smash_or_pass, 'sample_distance')
    assert hasattr(cfgs.smash_or_pass, 'sample_steps_min')
    assert hasattr(cfgs.kick_at_goal, 'min_opening_angle')
    assert hasattr(cfgs.pass_to_teammate, 'rating_threshold')
    assert hasattr(cfgs.pass_to_teammate, 'pass_ahead_distance')
    assert hasattr(cfgs.pass_to_teammate, 'min_angle_offset')
    # Capa 6: Global — no deben ser None
    assert cfgs.strategy_behavior_control.strategy == "s4v4_scaled"
    assert cfgs.behavior is not None

def test_zweikampf_params():
    """zweikampfImpl.cfg se parsea correctamente con todos los 6 subsistemas."""
    cfgs = load_all(CFG_ROOT)
    # duelRatings
    assert cfgs.zweikampf.ratings.rating_steal_ball == pytest.approx(-0.4)
    assert cfgs.zweikampf.ratings.rating_goal_shot == pytest.approx(-1.5)
    assert cfgs.zweikampf.ratings.rating_pose_blocked == pytest.approx(1.0)
    # duelTimings
    assert cfgs.zweikampf.timings.max_time_doing_nothing == 1000
    assert cfgs.zweikampf.timings.deactive_after_doing_nothing == 5000
    assert cfgs.zweikampf.timings.no_kick_min_time == 200
    # searchParameters
    assert cfgs.zweikampf.search_params.num_of_angles_near_best_duel_pose == 3
    # stealBallParameters
    assert cfgs.zweikampf.steal_ball.steal_ball_min_robot_rotation == pytest.approx(70.0)
    # lastKickHysterese
    assert cfgs.zweikampf.last_kick.kick_length_hysterese == pytest.approx(500)
    # allowedKicks (8 tipos)
    assert len(cfgs.zweikampf.allowed_kicks) == 8
    assert "walkForwardStealBallLeft" in cfgs.zweikampf.allowed_kicks

def test_strategy_behavior_control_typed():
    """strategyBehaviorControl.cfg se parsea como dataclass tipada."""
    cfgs = load_all(CFG_ROOT)
    assert isinstance(cfgs.strategy_behavior_control, StrategyBehaviorControlConfig)
    assert cfgs.strategy_behavior_control.strategy == "s4v4_scaled"

def test_skill_behavior_control_typed():
    """skillBehaviorControl.cfg se parsea como dataclass tipada con opciones."""
    cfgs = load_all(CFG_ROOT)
    assert isinstance(cfgs.skill_behavior_control, SkillBehaviorControlConfig)
    assert "HandleGoalkeeperCatchBall" in cfgs.skill_behavior_control.playing_options
    assert "HandlePlayerState" in cfgs.skill_behavior_control.options
    assert cfgs.skill_behavior_control.use_new_handle_catch_ball is True

def test_walk_has_avoidance_params():
    """WalkConfig tiene TODOS los parametros de evasion de libWalkProvider.cfg."""
    cfgs = load_all(CFG_ROOT)
    assert cfgs.walk.obstacle_avoidance_distance == pytest.approx(480)
    assert cfgs.walk.field_border_expansion == pytest.approx(300)
    assert cfgs.walk.ball_avoidance_distance == pytest.approx(465)

def test_behavior_has_keeper_zones():
    """BehaviorParams tiene radios de portero de behaviorParameters.cfg."""
    cfgs = load_all(CFG_ROOT)
    assert cfgs.behavior.stand_radius == pytest.approx(60)
    assert cfgs.behavior.jump_radius == pytest.approx(600)
    assert cfgs.behavior.keeper_jumping_on is True

def test_zweikampf_timing_constraints():
    """zweikampfImpl.cfg timings se parsean correctamente para constraints de duelo."""
    cfgs = load_all(CFG_ROOT)
    # no_kick_min_time: cooldown entre kicks en duelo
    assert cfgs.zweikampf.timings.no_kick_min_time == 200
    # no_kick_steal_min_time: cooldown para steal kicks
    assert cfgs.zweikampf.timings.no_kick_steal_min_time == 250
    # ignore_skill_request_time: tiempo que ignora peticiones externas durante duelo
    assert cfgs.zweikampf.timings.ignore_skill_request_time == 2000

def test_traceability_contract_14_rows():
    """Validar que las 14 filas del contrato de trazabilidad (Seccion 9.5) estan cubiertas."""
    cfgs = load_all(CFG_ROOT)
    # Fila 1: field_symbols
    assert len(cfgs.field_symbols) >= 30
    # Fila 2: strategy
    assert len(cfgs.strategy.tactics) == 2
    # Filas 3-4: tactics
    assert "t121_4v4_scaled" in cfgs.tactics
    assert "t13_4v4_scaled" in cfgs.tactics
    # Fila 5: kick_at_goal
    assert cfgs.kick_at_goal.min_opening_angle == pytest.approx(6.0)
    # Fila 6: pass_to_teammate (3 params aprendibles)
    assert cfgs.pass_to_teammate.rating_threshold == pytest.approx(0.7)
    assert hasattr(cfgs.pass_to_teammate, 'pass_ahead_distance')
    assert hasattr(cfgs.pass_to_teammate, 'min_angle_offset')
    # Fila 7: zweikampf (6 subsistemas)
    assert cfgs.zweikampf.allowed_kicks is not None
    assert cfgs.zweikampf.ratings.rating_goal_shot == pytest.approx(-1.5)
    # Fila 8: zweikampf timings (reemplaza demo_go_to_ball eliminado)
    assert cfgs.zweikampf.timings.no_kick_min_time > 0
    assert cfgs.zweikampf.timings.deactive_after_doing_nothing > 0
    # Fila 9: smash_or_pass (7 params aprendibles)
    assert cfgs.smash_or_pass.shoot_threshold == pytest.approx(0.8)
    assert cfgs.smash_or_pass.sample_distance == pytest.approx(500)
    # Fila 10: kicks
    assert len(cfgs.kicks) >= 10
    # Fila 11: walk
    assert cfgs.walk.max_forward > 0
    assert cfgs.walk.field_border_expansion > 0
    # Fila 12: behavior
    assert cfgs.behavior.jump_radius > 0
    # Fila 13: strategy_behavior_control
    assert cfgs.strategy_behavior_control.strategy == "s4v4_scaled"
    # Fila 14: skill_behavior_control
    assert len(cfgs.skill_behavior_control.playing_options) >= 1

def test_fallback_matches_parsed():
    """Fallback produce mismos valores que parsing real."""
    parsed = load_all(CFG_ROOT)
    fallback = load_all_fallback()
    for tactic_name in parsed.tactics:
        for i, pos in enumerate(parsed.tactics[tactic_name].positions):
            fb_pos = fallback.tactics[tactic_name].positions[i]
            assert pos.role_type == fb_pos.role_type
            assert pos.pose.x == pytest.approx(fb_pos.pose.x, abs=5)
            assert pos.pose.y == pytest.approx(fb_pos.pose.y, abs=5)
```

---

<a name="17-checklist"></a>
## 17. Checklist de Implementacion

### Fase 0: Preparacion

- [ ] Obtener acceso a los 12 archivos .cfg listados en seccion 13.1
- [ ] Verificar que FIELD_SYMBOLS coincide con `fieldDimensions.cfg` real

### Fase 1: Parser Base

- [ ] Implementar `cfg_tokenizer.py` con soporte para `//`, `/* */`, `#` comentarios, deg, arrays, structs, named maps
- [ ] Implementar `cfg_parser.py` que usa tokenizer + FIELD_SYMBOLS_4V4_SCALED
- [ ] Implementar `strategy_parser.py` -> StrategyConfig (s4v4_scaled.cfg)
- [ ] Implementar `tactic_parser.py` -> TacticConfig (t121/t13_4v4_scaled.cfg)
- [ ] Implementar `walk_parser.py` -> WalkConfig (walkingEngine.cfg efectivo + libWalkProvider.cfg)
- [ ] Implementar `loader.py` con `load_phase1()` -> field_symbols + strategy + tactics + walk
- [ ] Soportar edge cases 1-14 (Seccion 6.4): unary minus, symbol-to-symbol, IEEE floats, named maps
- [ ] Tests unitarios: parsear expresiones, arrays, structs anidados, named maps

### Post-Fase 1: Extensiones Especializadas (delta minimo hacia Fase 2/3)

- [ ] `action_parser.py` → KickAtGoalConfig (kickAtGoalImpl.cfg)
- [ ] `action_parser.py` → PassToTeammateConfig (passToTeammateImpl.cfg — 28 parametros top-level en C++; la dataclass Python los aplana en mas campos por Vector2f/Rangef y head tracking)
- [ ] `action_parser.py` → ZweikampfConfig (zweikampfImpl.cfg — 6 subsistemas: search, ratings, timings, obstacles, stealBall, hysterese)
- ~~`action_parser.py` → DemoGoToBallAndKickConfig~~ — ELIMINADO (demoGoToBallAndKickImpl.cfg no activo en 4v4)
- [ ] `kickinfo_parser.py` → dict[str, KickInfo] (kickInfo.cfg — named map)
- [ ] `behavior_parser.py` → BehaviorParams (behaviorParameters.cfg)
- [ ] `strategy_bc_parser.py` → StrategyBehaviorControlConfig (strategyBehaviorControl.cfg)
- [ ] `skill_bc_parser.py` → SkillBehaviorControlConfig (skillBehaviorControl.cfg)
- [ ] SmashOrPassParams: instanciar desde PlayBall.h defaults (no .cfg)
- [ ] Tests: parsear cada .cfg real y validar valores conocidos

### Post-Fase 1: Integracion Ampliada

- [ ] `loader.py` con `load_all()` → AllConfigs (12 campos parseados)
- [ ] `loader.py` con `load_all_fallback()` → AllConfigs (misma estructura, defaults hardcoded)
- [ ] Assertion: `strategy_behavior_control.strategy == "s4v4_scaled"` (dataclass tipada)
- [ ] `validator.py` con validacion de rangos y consistency checks
- [ ] Actualizar `config.py`: FIELD_SYMBOLS desde parser (no hardcoded 5v5)
- [ ] Actualizar `environment.py`: home poses desde parser
- [ ] Actualizar `environment.py`: FSM de tacticas desde parser
- [ ] Actualizar `environment.py`: TACTIC_NAMES desde strategy (2 valores, no 9)
- [ ] Eliminar `_current_tactic = "t4"` hardcodeado
- [ ] Eliminar home_poses hardcodeadas

### Fase 4: Validacion del Bridge Contract (Seccion 9.4) y Trazabilidad (Seccion 9.5)

- [ ] Verificar que AllConfigs expone TODOS los campos del contrato (Seccion 9.4)
- [ ] Verificar las 14 filas de la tabla de trazabilidad (Seccion 9.5): cada fila tiene parser → AllConfigs.campo → uso RL
- [ ] Verificar que cada punto de inyeccion C++ tiene su campo AllConfigs correspondiente
- [ ] Verificar que `config.py` usa 2813/1875 (no 4500/3000)
- [ ] Verificar que action space usa limites correctos de Categoria 4 (Seccion 14b.2)
- [ ] Verificar que tactic one_hot tiene 2 valores (no 9)
- [ ] Ejecutar todos los tests
- [ ] Hash de CFGs (12 archivos) guardado con modelo entrenado
- [ ] Comparar posiciones parseadas vs fallback (test_fallback_matches_parsed)

### Post-Integracion

- [ ] Eliminar TACTIC_NAMES duplicado de config.py
- [ ] Documentar en que fase se parsean nuevos CFGs
- [ ] Verificar que NO hay tacticas/enums sin sufijo `_4v4_scaled` en el codigo RL
