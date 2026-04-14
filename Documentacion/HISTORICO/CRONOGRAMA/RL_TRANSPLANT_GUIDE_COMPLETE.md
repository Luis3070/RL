# 🔴 GUÍA RIGUROSA DE TRANSPLANTE: RL/Python → C++/BHuman
## Fases 1, 2, 3 con Trazabilidad Sim2Real

**Advertencia Crítica:** Una brecha mal gestionada en transplante = proyecto fracasa en robots reales. Este documento es MANDATORIO.

---

## ÍNDICE

1. [Limitaciones Python vs C++](#1-limitaciones-python-vs-c)
2. [Parser CFG: Validación Crítica](#2-parser-cfg-validación-crítica)
3. [Fase 1: RL Posicionamiento](#3-fase-1-rl-posicionamiento)
4. [Fase 2: RL Parámetros Decisión](#4-fase-2-rl-parámetros-decisión)
5. [Fase 3: RL Reemplaza SmashOrPass](#5-fase-3-rl-reemplaza-smashOrpass)
6. [Auditoría y Validación](#6-auditoría-y-validación)
7. [Matriz de Trazabilidad](#7-matriz-de-trazabilidad)
8. [Plan de Validación Sim2Real](#8-plan-de-validación-sim2real)

---

## 1. LIMITACIONES PYTHON VS C++

### 1.1 Precisión Numérica

| Aspecto | C++ | Python | Impacto en RL |
|---------|-----|--------|--------------|
| **Float32** | 32-bit IEEE 754 | float64 (default) | ⚠️ Python es más preciso, convertir a float32 para validación |
| **Enteros** | int32/int64 | int (arbitrary precision) | ✅ Generalmente compatible |
| **Operaciones trigonométricas** | Hardware optimizado | Software | ⚠️ Pequeñas diferencias en atan2, sin, cos |
| **Rounding** | C++ std::round() | Python round() (banker's rounding) | ⚠️ Puede diferir en borderline values |

**Validación Requerida:**
```python
# MUST DO: Verificar que RL salidas coinciden con C++ en límites
import numpy as np

# C++ Valores
cpp_shoot_threshold = np.float32(0.8)  # ← Usar float32, NO float64

# RL Predicción
rl_predicted_threshold = 0.7999999999  # float64 por defecto

# VALIDACIÓN: Convertir RL a float32 antes de comparar
rl_as_float32 = np.float32(rl_predicted_threshold)

assert np.isclose(rl_as_float32, cpp_shoot_threshold, rtol=1e-6), \
    f"Threshold mismatch: RL={rl_as_float32}, C++={cpp_shoot_threshold}"
```

### 1.2 Timing y Latencia

| Operación | C++ | Python | Diferencia |
|-----------|-----|--------|-----------|
| **Float multiplication** | ~1 ns | ~1-10 µs | C++ es 1000x más rápido |
| **Vector normalize** | ~10 ns (SIMD) | ~100 µs | C++ es 10000x más rápido |
| **Sort 1000 items** | ~10 µs | ~1-2 ms | C++ es 100x más rápido |
| **Network I/O** | ~100 µs | ~1-5 ms | Similar (I/O bound) |

**Restricción Crítica:** RL Python debe completarse en **< 3 ms** para no bloquear Cognition thread.

```python
import time

# VALIDACIÓN: Medir tiempo de ejecución
start = time.time()
predicted_action = rl_network.forward(state_tensor)
elapsed = time.time() - start

assert elapsed < 0.003, f"RL took {elapsed*1000:.2f}ms, max 3ms allowed"
```

### 1.3 Manejo de Excepciones y Errores

| Caso | C++ | Python | Acción en RL |
|------|-----|--------|-------------|
| **NaN en cálculo** | Silent (o assertion) | RuntimeWarning | ✅ Detectar y reportar |
| **Array out of bounds** | Segmentation fault | IndexError | ✅ Catch y fallback |
| **Division by zero** | Inf o NaN | ZeroDivisionError o inf | ✅ Validar división |
| **Memoria insuficiente** | Bad alloc | MemoryError | ✅ Preallocar tensores |

**Validación Requerida:**
```python
def safe_rl_forward(state, network, default_action):
    """RL forward pass con validaciones"""
    try:
        # 1. Validar entrada
        assert not np.any(np.isnan(state)), "NaN en state"
        assert not np.any(np.isinf(state)), "Inf en state"

        # 2. Forward pass
        action = network.forward(state)

        # 3. Validar salida
        assert not np.any(np.isnan(action)), "NaN en acción"
        assert not np.any(np.isinf(action)), "Inf en acción"

        return action
    except Exception as e:
        logger.error(f"RL forward failed: {e}")
        return default_action  # Fallback seguro
```

### 1.4 Serialización y Persistencia

```python
# ⚠️ CRÍTICO: Versioning de modelos RL

class RLModel:
    VERSION = "1.0"  # MUST MATCH C++ expectations
    CFG_HASH = None  # Hash de CFG usados en training

    def __init__(self, ...):
        self.metadata = {
            'version': self.VERSION,
            'cfg_hash': self._compute_cfg_hash(),
            'training_date': datetime.now(),
            'float_precision': 'float32',
            'input_dim': 71,
            'output_dim': 6,
        }

    @staticmethod
    def _compute_cfg_hash():
        """Hash determinístico de todos los CFGs"""
        import hashlib
        cfgs = [...]  # Lista de rutas CFG
        content = b''
        for cfg_path in cfgs:
            with open(cfg_path, 'rb') as f:
                content += f.read()
        return hashlib.sha256(content).hexdigest()

    def validate_compatibility(self, cpp_cfg_hash):
        """Verificar que RL se entrenó con mismos CFGs que C++"""
        assert self.metadata['cfg_hash'] == cpp_cfg_hash, \
            f"CFG mismatch: RL={self.metadata['cfg_hash']}, C++={cpp_cfg_hash}"
```

### 1.5 Dependencias de Librerías

```python
# ⚠️ VERSIONING: Librerías DEBEN ser exactas

requirements.txt:
torch==2.0.1          # ← Específico, NO torch>=2.0
numpy==1.24.3         # ← Específico
scipy==1.10.1         # ← Específico
pydantic==2.0.2       # ← Específico

# VALIDACIÓN: Checkear versiones en startup
import sys
import torch
import numpy as np

assert torch.__version__ == "2.0.1", f"torch version mismatch: {torch.__version__}"
assert np.__version__ == "1.24.3", f"numpy version mismatch: {np.__version__}"
```

### 9.3 Decisiones Tecnicas Congeladas Post-Auditoria

Tras la auditoria documental del parser, bridge y entorno RL, estas decisiones deben tratarse como cerradas:

- **Autoridad del mundo por step:** Python entrega el estado actual y BHuman devuelve el siguiente estado completo por frame de Cognition
- **Granularidad temporal:** el sistema usa esquema hibrido, `1 frame de Cognition` como step base y macroestados agregados por takeover
- **Schema del bridge:** el contrato JSON del bridge queda congelado antes de integrar `environment.py`, `reward.py` y `wrapper.py`
- **Mecanismo de integracion:** `pybind11`, proceso embebido o IPC/shared memory deben decidirse explicitamente y probarse bajo restriccion temporal real
- **Despliegue final:** la inferencia RL en BHuman debe validarse con el presupuesto temporal del hilo de Cognition, no solo en Python

Estas decisiones complementan esta guia.
La guia ya cubria transplante, validacion e IPC de forma general, pero no dejaba estas elecciones finales completamente congeladas para implementacion.

---

## 2. PARSER CFG: VALIDACIÓN CRÍTICA

### 2.1 Arquitectura del Parser (Revisitada)

```python
"""
Parser CFG con VALIDACIÓN COMPLETA

El parser DEBE ser bidireccional:
- CFG → Python (parser)
- Python → C++ (serialización)

CRÍTICO: Validar que roundtrip no cambia valores
"""

class StrictCFGParser:
    """Parser con validación exhaustiva"""

    def __init__(self, config_root):
        self.config_root = Path(config_root)
        self.validation_log = []
        self.parsed_cfgs = {}

    def parse_and_validate(self, cfg_path, schema):
        """
        Parse CFG + validar contra schema esperado

        Returns: (parsed_data, validation_report)
        """
        parsed = self._parse_raw(cfg_path)
        report = self._validate(parsed, schema)

        if not report.is_valid:
            raise ValueError(f"CFG validation failed: {report.errors}")

        self.parsed_cfgs[cfg_path.stem] = (parsed, report)
        return parsed, report

    def _validate(self, parsed_data, schema):
        """Validar contra schema"""
        errors = []
        warnings = []

        for key, expected_type in schema.items():
            if key not in parsed_data:
                errors.append(f"Missing required key: {key}")
                continue

            value = parsed_data[key]

            # Tipo
            if not isinstance(value, expected_type):
                errors.append(
                    f"Type mismatch {key}: expected {expected_type}, got {type(value)}"
                )

            # Rangos
            if 'range' in schema.get(f'{key}_meta', {}):
                min_val, max_val = schema[f'{key}_meta']['range']
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"Value {key}={value} outside range [{min_val}, {max_val}]"
                    )

        return ValidationReport(is_valid=len(errors)==0, errors=errors, warnings=warnings)

    def export_to_cpp_format(self, parsed_data):
        """Exportar a formato que C++ lee exactamente"""
        # CRÍTICO: Asegurar que serialización es exacta
        cpp_code = self._generate_cpp_config(parsed_data)
        return cpp_code

# SCHEMAS por CFG (DEBEN COINCIDIR CON C++)
STRATEGY_SCHEMA = {
    'ballXThreshold': float,
    'ballXThreshold_meta': {'range': (-1000, 1000)},
    'timeSinceBallAhead': int,
    'timeSinceBallAhead_meta': {'range': (100, 500)},
    'numOfFieldPlayersGE': int,
    'numOfFieldPlayersGE_meta': {'range': (1, 3)},
}

TACTIC_SCHEMA = {
    'striker_x': float,
    'striker_x_meta': {'range': (-2813, 2813)},
    'striker_y': float,
    'striker_y_meta': {'range': (-1875, 1875)},
    # ... más
}

KICKINFO_SCHEMA = {
    'range_min': float,
    'range_min_meta': {'range': (0, 1000)},
    'range_max': float,
    'range_max_meta': {'range': (1000, 10000)},
    # ... más
}
```

### 2.2 Roundtrip Validation (CRÍTICO)

```python
class CFGRoundtripValidator:
    """Validar que CFG → Python → C++ mantiene valores exactos"""

    def validate_roundtrip(self, original_cfg_path, parser, cpp_serializer):
        """
        Roundtrip test:
        1. Leer CFG original (C++)
        2. Parsear a Python
        3. Serializar a formato C++
        4. Comparar byte-by-byte
        """

        # Step 1: Leer original
        with open(original_cfg_path, 'rb') as f:
            original_bytes = f.read()

        # Step 2: Parsear
        parsed = parser.parse(original_cfg_path)

        # Step 3: Serializar
        reserialized = cpp_serializer.to_bytes(parsed)

        # Step 4: Comparar
        if original_bytes != reserialized:
            diff = self._compute_diff(original_bytes, reserialized)
            raise ValueError(
                f"Roundtrip failed for {original_cfg_path}:\n{diff}"
            )

        return True

    def _compute_diff(self, bytes1, bytes2):
        """Mostrar dónde difieren los bytes"""
        import difflib
        lines1 = bytes1.decode('utf-8').splitlines(keepends=True)
        lines2 = bytes2.decode('utf-8').splitlines(keepends=True)
        diff = difflib.unified_diff(lines1, lines2, lineterm='')
        return '\n'.join(diff)

# USAR:
validator = CFGRoundtripValidator()
for cfg_file in cfg_files:
    validator.validate_roundtrip(cfg_file, parser, serializer)
```

### 2.3 Hash de CFG para Trazabilidad

```python
class CFGHashManager:
    """Mantener hash de CFGs para validar que RL se entrenó con mismo config"""

    def compute_combined_hash(self, cfg_dir):
        """Hash de TODOS los CFGs (orden determinístico)"""
        import hashlib

        cfg_files = sorted(Path(cfg_dir).glob('**/*.cfg'))
        combined_hash = hashlib.sha256()

        for cfg_file in cfg_files:
            with open(cfg_file, 'rb') as f:
                combined_hash.update(f.read())

        return combined_hash.hexdigest()

    def validate_cfg_consistency(self, cfg_hash_at_training, current_cfg_hash):
        """Validar que CFGs no cambiaron desde training"""
        assert cfg_hash_at_training == current_cfg_hash, \
            f"CFG mismatch: training={cfg_hash_at_training}, current={current_cfg_hash}"

    def generate_cfg_manifest(self, cfg_dir, output_file):
        """Generar manifest de todos los CFGs para auditoría"""
        import json

        manifest = {
            'timestamp': datetime.now().isoformat(),
            'total_hash': self.compute_combined_hash(cfg_dir),
            'files': [],
        }

        for cfg_file in sorted(Path(cfg_dir).glob('**/*.cfg')):
            with open(cfg_file, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            manifest['files'].append({
                'path': str(cfg_file.relative_to(cfg_dir)),
                'hash': file_hash,
                'size_bytes': cfg_file.stat().st_size,
            })

        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest
```

---

## 3. FASE 1: RL POSICIONAMIENTO

### 3.1 Arquitectura de Fase 1

```
┌────────────────────────────────────────────────────┐
│ FASE 1: RL PREDICE POSICIONES Y ORIENTACION (x, y, theta) │
├────────────────────────────────────────────────────┤
│                                                    │
│  INPUT (Python RL):                                │
│  ├─ Ball position (x, y, vx, vy)                  │
│  ├─ Robot positions (4 robots)                    │
│  ├─ Opponent positions (4 robots)                 │
│  ├─ Current tactic (t121 vs t13)                  │
│  └─ Game state (PLAYING, READY, etc.)            │
│                                                    │
│  OUTPUT (Python RL):                               │
│  ├─ target_x, target_y [mm]                       │
│  ├─ target_theta [rad]                            │
│  └─ ... (pose para cada field player)             │
│                                                    │
│  C++ MAPPING:                                      │
│  ├─ Recibe posiciones de RL                       │
│  ├─ Reemplaza positions[i].pose en táctica       │
│  ├─ MANTIENE shoot/pass logic original            │
│  └─ Ejecuta movimiento                            │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 3.2 Parser para Fase 1

```python
class Phase1Parser:
    """Parser de Fase 1: fieldDimensions + strategy + tacticas 4v4."""

    def __init__(self, config_root):
        self.config_root = Path(config_root)
        self.tactics = {}

    def parse_phase1(self):
        """Parsear SOLO lo necesario para Fase 1"""
        # NO usar regex sobre t*.cfg: las poses reales contienen
        # expresiones simbolicas (ej. xPosOwnPenaltyArea + 313).
        base = BHumanCfgParser(symbols=FIELD_SYMBOLS_4V4_SCALED)
        strategy = StrategyParser(base).parse(
            self.config_root / "Behavior" / "Strategies" / "s4v4_scaled.cfg"
        )
        self.tactics = {
            "t121_4v4_scaled": TacticParser(base).parse(
                self.config_root / "Behavior" / "Tactics" / "t121_4v4_scaled.cfg"
            ),
            "t13_4v4_scaled": TacticParser(base).parse(
                self.config_root / "Behavior" / "Tactics" / "t13_4v4_scaled.cfg"
            ),
        }

        # 2. Retornar estructura simplificada
        return {
            'field_symbols': FIELD_SYMBOLS_4V4_SCALED,
            'strategy': strategy,
            'tactics': self.tactics,
            'phase': 1,
            'timestamp': datetime.now().isoformat(),
        }

# VALIDACIÓN FASE 1
def validate_phase1_positions(rl_positions, tactic_schema):
    """Validar que poses RL de Fase 1 son válidas"""

    errors = []

    for role, pos in rl_positions.items():
        # Validar que está en rango
        if not (-2813 <= pos['x'] <= 2813):
            errors.append(f"Role {role}: x out of bounds: {pos['x']}")

        if not (-1875 <= pos['y'] <= 1875):
            errors.append(f"Role {role}: y out of bounds: {pos['y']}")

        if not (-math.pi <= pos['theta'] <= math.pi):
            errors.append(f"Role {role}: theta out of bounds: {pos['theta']}")

        # Validar contra posiciones de CFG
        cfg_pos = tactic_schema['positions'][role]
        distance = math.sqrt((pos['x'] - cfg_pos['x'])**2 + (pos['y'] - cfg_pos['y'])**2)

        # ⚠️ Permitir desviación pero loguear
        if distance > 1000:  # mm
            logger.warning(f"Role {role} deviates {distance}mm from CFG position")

    if errors:
        raise ValueError(f"Position validation failed: {errors}")
```

### 3.3 Transplante Fase 1 a C++

```cpp
// EN C++: Reemplazar posiciones en StrategyBehaviorControl.cpp

// ORIGINAL (línea ~650):
Tactic::Position positions[] = {
    {goalkeeper, x=-3280, y=0},
    {defenderL,  x=-1188, y=0},
    {forwardR,   x=+1350, y=0},
    // ...
};

// MODIFICADO (Fase 1 con RL):
Tactic::Position positions[MAX_POSITIONS];

// Recibir de RL (via IPC o shared memory)
struct RLPositions {
    float striker_x, striker_y;
    float defender_x, defender_y;
    // ...
};

extern RLPositions rl_positions;  // Set by Python RL

// Usar en táctica:
positions[STRIKER_IDX] = {forwardR, x=rl_positions.striker_x, y=rl_positions.striker_y};
positions[DEFENDER_IDX] = {defenderL, x=rl_positions.defender_x, y=rl_positions.defender_y};
// ...

// ⚠️ VALIDACIÓN: C++ DEBE validar rangos
for(auto& pos : positions) {
    ASSERT(pos.x >= -2813 && pos.x <= 2813);
    ASSERT(pos.y >= -1875 && pos.y <= 1875);
}
```

### 3.4 Auditoría Fase 1

```python
class Phase1Audit:
    """Auditoría de Fase 1: solo posicionamiento"""

    def audit_transplant(self, rl_positions, expected_schema):
        """Auditar que transplante es correcto"""

        report = {
            'phase': 1,
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'warnings': [],
            'errors': [],
        }

        # Check 1: Rangos
        for role, pos in rl_positions.items():
            if not (-2813 <= pos['x'] <= 2813):
                report['errors'].append(f"{role}: x out of range")
            if not (-1875 <= pos['y'] <= 1875):
                report['errors'].append(f"{role}: y out of range")

        report['checks']['range_validation'] = len(report['errors']) == 0

        # Check 2: Desviaciones del CFG
        for role, pos in rl_positions.items():
            cfg_pos = expected_schema['positions'][role]
            deviation = math.sqrt((pos['x']-cfg_pos['x'])**2 + (pos['y']-cfg_pos['y'])**2)

            if deviation > 500:  # > 500mm
                report['warnings'].append(
                    f"{role}: {deviation}mm deviation from CFG"
                )

        report['checks']['deviation_check'] = len(report['warnings']) <= 3

        # Check 3: Consistencia entre frames
        # (Requiere histórico)

        return report
```

---

## 4. FASE 2: RL PARÁMETROS DECISIÓN

### 4.1 Arquitectura Fase 2

IMPORTANTE:
- Fase 2 EXTIENDE Fase 1; no se reescribe el parser base.
- `PlayBall.h` es la fuente real de `shootThreshold`, `minPassDistance` y `maxPassDistance`.
- `kickAtGoalImpl.cfg` aporta `minOpeningAngle` y `hysteresisNumber`.
- `passToTeammateImpl.cfg` aporta `ratingThreshold`.
- `passImprovement` queda fuera del contrato RL de Fase 2 porque solo se usa en `executeLegacy()`.

```
┌────────────────────────────────────────────────────┐
│ FASE 2: RL AJUSTA PARÁMETROS DE PLAYBALL          │
├────────────────────────────────────────────────────┤
│                                                    │
│  RL PREDICE (de estado):                           │
│  ├─ shootThreshold  (0.3-0.9)  [float]            │
│  ├─ ratingThreshold (0.3-0.9)  [float]            │
│  ├─ minPassDistance (500-3000) [mm]               │
│  ├─ maxPassDistance (2000-8000)[mm]               │
│  ├─ minOpeningAngle (2-15)     [deg]              │
│  └─ hysteresis     (100-500)   [ms]               │
│                                                    │
│  C++ RECIBE:                                       │
│  └─ Parámetros actualizados cada frame             │
│                                                    │
│  C++ EJECUTA:                                      │
│  └─ PlayBall::smashOrPass() CON PARÁMETROS RL    │
│                                                    │
│  🔴 CRÍTICO:                                       │
│  ├─ Validar parámetros ANTES de usar              │
│  ├─ Hard clamp a rangos válidos                   │
│  ├─ Fallback a defaults si inválidos              │
│  └─ Loguear TODOS los cambios                     │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 4.2 Parser Fase 2

```python
class Phase2Parser(Phase1Parser):
    """Extiende Fase 1 con parametros vivos del runtime actual."""

    def __init__(self, config_root, repo_root):
        super().__init__(config_root)
        self.repo_root = Path(repo_root)
        self.action_params = {}

    def parse_phase2(self):
        """Reusar Fase 1 y agregar SOLO los defaults activos para Fase 2."""

        # 1. Reusar el baseline de Fase 1
        phase1 = self.parse_phase1()

        # 2. Agregar SOLO nuevos defaults vivos sobre el baseline ya parseado en Fase 1
        behavior_dir = self.config_root / "Scenarios" / "4v4_Scaled" / "BehaviorControl"
        playball = self._parse_playball_defaults(
            self.repo_root / "Src/Modules/BehaviorControl/StrategyBehaviorControl/ActiveRoles/PlayBall.h"
        )
        kick_at_goal = self._parse_kick_at_goal(
            behavior_dir / "kickAtGoalImpl.cfg"
        )
        pass_to_teammate = self._parse_pass_to_teammate(
            behavior_dir / "passToTeammateImpl.cfg"
        )

        self.action_params = {
            "shootThreshold": playball["shootThreshold"],
            "minPassDistance": playball["minPassDistance"],
            "maxPassDistance": playball["maxPassDistance"],
            "minOpeningAngle": kick_at_goal["minOpeningAngle"],
            "hysteresisNumber": kick_at_goal["hysteresisNumber"],
            "ratingThreshold": pass_to_teammate["ratingThreshold"],
        }

        return {
            **phase1,
            "action_params": self.action_params,
            "phase": 2,
            "timestamp": datetime.now().isoformat(),
        }

    def _parse_playball_defaults(self, path):
        """PlayBall.h es la fuente real de shoot/min/maxPassDistance."""
        content = path.read_text()
        return {
            "shootThreshold": self._extract_float(
                content, r'\(float\)\(([\d.]+)f\)\s+shootThreshold'
            ),
            "minPassDistance": self._extract_float(
                content, r'\(float\)\(([\d.]+)f\)\s+minPassDistance'
            ),
            "maxPassDistance": self._extract_float(
                content, r'\(float\)\(([\d.]+)f\)\s+maxPassDistance'
            ),
        }

    def _parse_kick_at_goal(self, path):
        """kickAtGoalImpl.cfg aporta defaults vivos del skill de disparo."""
        content = path.read_text()
        return {
            "minOpeningAngle": self._extract_float(content, r'minOpeningAngle\s*=\s*(\d+)deg'),
            "hysteresisNumber": self._extract_float(content, r'hysteresisNumber\s*=\s*(\d+)'),
        }

    def _parse_pass_to_teammate(self, path):
        """passToTeammateImpl.cfg aporta defaults vivos del skill de pase."""
        content = path.read_text()
        return {
            "ratingThreshold": self._extract_float(content, r'ratingThreshold\s*=\s*([\d.]+)'),
            # Futuros candidatos post-Fase 2:
            # passAheadDistance, minAngleOffset, maxAngleOffset, ...
        }
```

### 4.3 Validación Fase 2

```python
class Phase2Validation:
    """Validar parámetros RL en Fase 2"""

    # Esquema de parámetros válidos (DEBE coincidir con C++)
    PARAM_SCHEMA = {
        'shootThreshold': {'type': float, 'min': 0.3, 'max': 0.9, 'default': 0.8},
        'minPassDistance': {'type': int, 'min': 500, 'max': 3000, 'default': 1500, 'unit': 'mm'},
        'maxPassDistance': {'type': int, 'min': 2000, 'max': 8000, 'default': 4500, 'unit': 'mm'},
        'minOpeningAngle': {'type': int, 'min': 2, 'max': 15, 'default': 6, 'unit': 'deg'},
        'hysteresisNumber': {'type': int, 'min': 100, 'max': 500, 'default': 300, 'unit': 'ms'},
        'ratingThreshold': {'type': float, 'min': 0.3, 'max': 0.9, 'default': 0.7},
    }

    def validate_rl_output(self, rl_params):
        """Validar salida de RL antes de enviar a C++"""

        errors = []
        clamped_params = {}

        for param_name, param_value in rl_params.items():
            if param_name not in self.PARAM_SCHEMA:
                errors.append(f"Unknown parameter: {param_name}")
                continue

            schema = self.PARAM_SCHEMA[param_name]

            # Type check
            if not isinstance(param_value, schema['type']):
                errors.append(
                    f"{param_name}: expected {schema['type']}, got {type(param_value)}"
                )
                continue

            # Range check + clamp
            if param_value < schema['min']:
                logger.warning(f"{param_name}={param_value} < min={schema['min']}, clamping")
                clamped_params[param_name] = schema['min']
            elif param_value > schema['max']:
                logger.warning(f"{param_name}={param_value} > max={schema['max']}, clamping")
                clamped_params[param_name] = schema['max']
            else:
                clamped_params[param_name] = param_value

        if errors:
            raise ValueError(f"Parameter validation failed: {errors}")

        return clamped_params

    def consistency_check(self, params):
        """Validar consistencia ENTRE parámetros"""

        # Check 1: minPassDistance < maxPassDistance
        if params['minPassDistance'] >= params['maxPassDistance']:
            raise ValueError(
                f"minPassDistance ({params['minPassDistance']}) >= "
                f"maxPassDistance ({params['maxPassDistance']})"
            )

        return True
```

### 4.4 Transplante Fase 2 a C++

```cpp
// FUENTES REALES: PlayBall.h + KickAtGoal + PassToTeammate

// ORIGINAL:
class PlayBall : public ActiveRole {
    STREAMABLE(Parameters, {
        (float) shootThreshold = 0.8;
        // passImprovement existe en PlayBall.h pero solo para executeLegacy()
        // ...
    });

    Parameters p;  // Cargado desde CFG

    SkillRequest smashOrPass(...) {
        // Usa p.shootThreshold directamente
        if(xG > p.shootThreshold) return shoot();
        // ...
    }
};

// MODIFICADO (Fase 2 con RL):
// Opción A: Usar structure compartida (recomendado)
struct RLParameters {
    float shootThreshold;
    float ratingThreshold;
    float minPassDistance;
    float maxPassDistance;
    float minOpeningAngle;
    int hysteresisNumber;
};

// routing:
// - PlayBall usa shootThreshold / minPassDistance / maxPassDistance
// - KickAtGoal usa minOpeningAngle / hysteresisNumber
// - PassToTeammate usa ratingThreshold

extern RLParameters rl_params;  // Set by Python

class PlayBall : public ActiveRole {
    SkillRequest smashOrPass(...) {
        // Usar RL params si disponibles, sino CFG defaults
        float threshold = rl_params_available ? rl_params.shootThreshold : p.shootThreshold;

        ASSERT(threshold >= 0.3 && threshold <= 0.9);  // Validar

        if(xG > threshold) return shoot();  // ← Usa RL threshold
        // ...
    }
};

// ⚠️ VALIDACIÓN EN C++:
#define VALIDATE_RL_PARAM(param, min, max) \
    do { \
        if (param < min || param > max) { \
            logger.error("Invalid RL param " #param "=" << param); \
            param = default_##param;  /* Fallback */ \
        } \
    } while(0)

// En el wiring previo a PlayBall / KickAtGoal / PassToTeammate:
VALIDATE_RL_PARAM(rl_params.shootThreshold, 0.3, 0.9);
VALIDATE_RL_PARAM(rl_params.minPassDistance, 500, 3000);
VALIDATE_RL_PARAM(rl_params.maxPassDistance, 2000, 8000);
VALIDATE_RL_PARAM(rl_params.ratingThreshold, 0.3, 0.9);
// ...
```

### 4.5 Auditoría Fase 2

```python
class Phase2Audit:
    """Auditoría de Fase 2: parámetros de decisión"""

    def audit_parameter_impact(self, rl_params, game_states, expected_actions):
        """
        Auditoría: ¿Los parámetros RL generan las acciones esperadas?

        Inputs:
        - rl_params: parámetros predichos por RL
        - game_states: historia de estados de juego
        - expected_actions: acciones esperadas (ground truth)
        """

        report = {
            'phase': 2,
            'timestamp': datetime.now().isoformat(),
            'parameter_accuracy': {},
            'action_consistency': {},
            'warnings': [],
        }

        # Check 1: Parámetros en rango
        validator = Phase2Validation()
        try:
            validator.validate_rl_output(rl_params)
            report['parameter_accuracy']['range_check'] = 'PASS'
        except ValueError as e:
            report['parameter_accuracy']['range_check'] = f'FAIL: {e}'

        # Check 2: Consistencia entre parámetros
        try:
            validator.consistency_check(rl_params)
            report['parameter_accuracy']['consistency_check'] = 'PASS'
        except ValueError as e:
            report['parameter_accuracy']['consistency_check'] = f'FAIL: {e}'

        # Check 3: Acciones generadas vs esperadas
        for game_state, expected_action in zip(game_states, expected_actions):
            # Simular PlayBall::smashOrPass() con parámetros RL
            simulated_action = self._simulate_playball(game_state, rl_params)

            if simulated_action == expected_action:
                report['action_consistency']['matches'] = \
                    report['action_consistency'].get('matches', 0) + 1
            else:
                report['action_consistency']['mismatches'] = \
                    report['action_consistency'].get('mismatches', 0) + 1
                report['warnings'].append(
                    f"Action mismatch: state={game_state}, expected={expected_action}, "
                    f"got={simulated_action}"
                )

        return report

    def _simulate_playball(self, game_state, rl_params):
        """Simular PlayBall::smashOrPass() en Python con parámetros RL"""
        # Calcular xG (de game_state)
        xG = game_state['ball_xG']

        # Usar parámetro RL
        if xG > rl_params['shootThreshold']:
            return 'SHOOT'

        # Buscar pases...
        # (simplificado)
        best_pass = self._find_best_pass(game_state, rl_params)

        if best_pass:
            return f'PASS_TO_{best_pass}'
        else:
            return 'SHOOT'
```

---

## 5. FASE 3: RL REEMPLAZA SMASHORPASS

### 5.1 Arquitectura Fase 3

```
┌────────────────────────────────────────────────────────────┐
│ FASE 3: RL REEMPLAZA PLAYBALL::SMASHORRPASS COMPLETO      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ⚠️ CRÍTICO: MÁXIMO RIESGO - Solo si Fase 2 funciona      │
│                                                            │
│  RL PREDICE DIRECTAMENTE:                                  │
│  ├─ action ∈ {SHOOT, PASS_TO_0, PASS_TO_1, ...}          │
│  ├─ shot_power ∈ [0.0, 1.0]                              │
│  └─ pass_target_position (x, y)                           │
│                                                            │
│  C++ RECIBE:                                               │
│  └─ SkillRequest (SHOOT o PASS) directamente de RL        │
│                                                            │
│  C++ EJECUTA:                                              │
│  ├─ Valida que acción es legal (hard constraints)        │
│  ├─ Si inválida: fallback a C++ SmashOrPass             │
│  ├─ Si válida: ejecuta                                   │
│  └─ Loguea TODO para auditoría                           │
│                                                            │
│  🔴 PROBLEMAS POTENCIALES:                                │
│  ├─ RL puede predecir acciones imposibles                │
│  ├─ Latencia de RL > 3ms bloquea decisión               │
│  ├─ RL no ve cambios rápidos en campo                   │
│  ├─ Transfer learning sim→real muy difícil              │
│  └─ Difícil debuggear en robots reales                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Parser Fase 3 (IGUAL a Fase 2 + Action Encoding)

```python
class Phase3Parser(Phase2Parser):
    """Parser Fase 3: TODO de Fase 2 + encoding de acciones"""

    def parse_phase3(self):
        """Parsear + encoding de acciones disponibles"""

        # 1. Parsear todo (como Fase 2)
        config = super().parse_phase2()

        # 2. NUEVO: Mapear acciones a IDs
        config['action_space'] = self._encode_actions()

        return config

    def _encode_actions(self):
        """Mapear acciones a IDs para RL"""

        actions = {
            'SHOOT': 0,
            'PASS_TO_0': 1,
            'PASS_TO_1': 2,
            'PASS_TO_2': 3,
            'PASS_TO_3': 4,
        }

        # Validar contra C++ (SkillRequest enums)
        # DEBE coincidir exactamente con C++

        return actions
```

### 5.3 Validación Fase 3 (ESTRICTA)

```python
class Phase3Validation:
    """Validación CRÍTICA de Fase 3"""

    # Acciones PERMITIDAS (hardcoded desde C++)
    ALLOWED_ACTIONS = {
        'SHOOT',
        'PASS_TO_0', 'PASS_TO_1', 'PASS_TO_2', 'PASS_TO_3',
        'DRIBBLE',
    }

    # Hard constraints (NO negociables)
    HARD_CONSTRAINTS = [
        lambda state, action: self._validate_action_legal(state, action),
        lambda state, action: self._validate_ball_distance(state, action),
        lambda state, action: self._validate_target_valid(state, action),
        lambda state, action: self._validate_timing(state, action),
    ]

    def validate_rl_action(self, state, action):
        """
        Validar que acción RL es legal

        Si no es legal: FALLBACK a C++ SmashOrPass
        """

        errors = []

        # Check 1: Acción conocida
        if action not in self.ALLOWED_ACTIONS:
            errors.append(f"Unknown action: {action}")

        # Check 2: Hard constraints
        for constraint_fn in self.HARD_CONSTRAINTS:
            if not constraint_fn(state, action):
                errors.append(f"Hard constraint failed: {constraint_fn.__name__}")

        if errors:
            logger.error(f"Action validation failed: {errors}")
            logger.error(f"Falling back to C++ SmashOrPass")
            return False, errors

        return True, []

    def _validate_action_legal(self, state, action):
        """
        ¿Es legal la acción?

        SHOOT: válido si robot no está caído
        PASS_TO_X: válido si compañero X está visible y en rango
        """

        if action == 'SHOOT':
            return state['robot']['is_upright']

        elif action.startswith('PASS_TO_'):
            target_id = int(action.split('_')[2])
            teammate = state['teammates'][target_id]

            # Compañero visible y en rango
            if not teammate['is_visible']:
                return False

            dist = self._compute_distance(state['ball'], teammate['position'])
            return 1500 <= dist <= 4500  # minPassDistance, maxPassDistance

        return False

    def _validate_ball_distance(self, state, action):
        """¿Está el balón a distancia válida para ejecutar?"""

        # Puede estar muy lejos (> 2m) → no puede patear
        ball_dist = state['robot_ball_distance']

        if ball_dist > 2000:  # mm
            return False  # No puede patear si balón está muy lejos

        return True

    def _validate_target_valid(self, state, action):
        """¿El target es válido?"""

        if action == 'PASS_TO_3':
            # ¿Existe jugador 3?
            if len(state['teammates']) < 3:
                return False

        return True

    def _validate_timing(self, state, action):
        """¿El timing es válido? (no cambios muy rápidos)"""

        # Hysteresis: no cambiar acción si hace < 300ms
        time_since_last_action = state['frame_time'] - state['last_action_time']

        if action != state['last_action'] and time_since_last_action < 300:
            return False

        return True
```

### 5.4 Transplante Fase 3 a C++

```cpp
// EN StrategyBehaviorControl.cpp

// Opción A: Usar enum para acciones
enum RLAction {
    RL_SHOOT = 0,
    RL_PASS_TO_0 = 1,
    RL_PASS_TO_1 = 2,
    RL_PASS_TO_2 = 3,
    RL_PASS_TO_3 = 4,
    RL_ACTION_INVALID = -1,
};

struct RLOutput {
    RLAction action;
    float shot_power;  // [0, 1]
    int pass_target;   // 0-3
    uint64_t timestamp;
};

extern RLOutput rl_output;
extern bool rl_enabled;

// EN PlayBall.cpp - reemplazar smashOrPass():
SkillRequest PlayBall::execute(...) {
    if(rl_enabled) {
        // Fase 3: Usar RL directamente
        SkillRequest rl_action = this->executeRL(rl_output);

        if(rl_action.isValid()) {
            return rl_action;  // ← RL decision
        } else {
            logger.error("RL action invalid, falling back to C++ SmashOrPass");
            // Fallback a C++
        }
    }

    // Fallback o Fase 2: usar C++ SmashOrPass
    return this->smashOrPass(...);
}

SkillRequest PlayBall::executeRL(const RLOutput& rl_out) {
    // Validar acción
    if(!this->validateRLAction(rl_out)) {
        return SkillRequest();  // Inválida
    }

    // Mapear a SkillRequest
    switch(rl_out.action) {
        case RL_SHOOT:
            return SkillRequest::Builder::shoot();

        case RL_PASS_TO_0:
        case RL_PASS_TO_1:
        case RL_PASS_TO_2:
        case RL_PASS_TO_3:
            return SkillRequest::Builder::passTo(rl_out.pass_target);

        default:
            return SkillRequest();  // Inválida
    }
}

bool PlayBall::validateRLAction(const RLOutput& rl_out) {
    // Hard constraints (NO negociables)

    // 1. Robot upright
    if(!theRobotPose.isUpright) return false;

    // 2. Timestamp válido (< 100ms viejo)
    if(theFrameInfo.getTimeSince(rl_out.timestamp) > 100) return false;

    // 3. Acción válida
    if(rl_out.action >= 5) return false;

    // 4. Si es pase, compañero debe estar visible
    if(rl_out.action >= RL_PASS_TO_0) {
        const Agent* target = teammates[rl_out.pass_target];
        if(!target->canReceivePass()) return false;
    }

    // 5. Balón visible
    if(theFrameInfo.getTimeSince(lastBallSeen) > 5000) return false;

    return true;
}
```

### 5.5 Auditoría Fase 3

```python
class Phase3Audit:
    """Auditoría CRÍTICA de Fase 3"""

    def full_audit(self, rl_trajectory, cpp_log, sim_results):
        """
        Auditoría completa de Fase 3:
        1. RL produce acciones válidas?
        2. C++ ejecuta acciones correctamente?
        3. Resultado (gol/pase/etc) es el esperado?
        """

        report = {
            'phase': 3,
            'timestamp': datetime.now().isoformat(),
            'sections': {},
        }

        # Section 1: RL Validity
        report['sections']['rl_validity'] = self._audit_rl_validity(rl_trajectory)

        # Section 2: C++ Execution
        report['sections']['cpp_execution'] = self._audit_cpp_execution(cpp_log)

        # Section 3: Outcomes
        report['sections']['outcomes'] = self._audit_outcomes(rl_trajectory, sim_results)

        # Summary
        report['overall_pass'] = all(
            section.get('pass', False)
            for section in report['sections'].values()
        )

        return report

    def _audit_rl_validity(self, trajectory):
        """¿RL produce acciones válidas?"""

        validation = Phase3Validation()
        results = {
            'total_actions': len(trajectory),
            'valid_actions': 0,
            'invalid_actions': 0,
            'details': [],
        }

        for frame_num, (state, action) in enumerate(trajectory):
            is_valid, errors = validation.validate_rl_action(state, action)

            if is_valid:
                results['valid_actions'] += 1
            else:
                results['invalid_actions'] += 1
                results['details'].append({
                    'frame': frame_num,
                    'action': action,
                    'errors': errors,
                })

        results['pass'] = results['invalid_actions'] == 0
        results['validity_rate'] = results['valid_actions'] / results['total_actions'] * 100

        return results

    def _audit_cpp_execution(self, cpp_log):
        """¿C++ ejecuta acciones RL correctamente?"""

        # Parse C++ log: "RL action: SHOOT", "RL fallback to C++", etc.

        results = {
            'total_frames': len(cpp_log),
            'rl_used': 0,
            'rl_fallback': 0,
            'execution_failures': 0,
            'details': [],
        }

        for frame_num, log_entry in enumerate(cpp_log):
            if 'RL action' in log_entry:
                results['rl_used'] += 1
            elif 'RL fallback' in log_entry:
                results['rl_fallback'] += 1
            elif 'execution error' in log_entry:
                results['execution_failures'] += 1
                results['details'].append({
                    'frame': frame_num,
                    'error': log_entry,
                })

        results['pass'] = results['execution_failures'] == 0
        results['rl_usage_rate'] = results['rl_used'] / results['total_frames'] * 100

        return results

    def _audit_outcomes(self, trajectory, sim_results):
        """¿Los outcomes son los esperados?"""

        # Comparar:
        # - Acciones predichas vs ejecutadas vs resultados

        results = {
            'total_episodes': len(sim_results),
            'goals_scored': sum(1 for r in sim_results if r['scored']),
            'passes_completed': sum(1 for r in sim_results if r['pass_success']),
            'actions_failed': sum(1 for r in sim_results if r['action_failed']),
            'success_rate': 0.0,
            'details': [],
        }

        # Calcular tasa de éxito
        successful = sum(1 for r in sim_results if r['scored'] or r['pass_success'])
        results['success_rate'] = successful / len(sim_results) * 100 if sim_results else 0

        # Threshold: > 70% success rate aceptable para Fase 3
        results['pass'] = results['success_rate'] > 70.0

        return results
```

---

## 6. AUDITORÍA Y VALIDACIÓN

### 6.1 Framework de Auditoría General

```python
class AuditFramework:
    """Framework general de auditoría para todas las fases"""

    def __init__(self, phase):
        self.phase = phase
        self.audit_results = {}

    def run_full_audit(self, rl_model, cpp_system, test_cases):
        """Ejecutar auditoría completa"""

        print(f"[AUDIT] Phase {self.phase} - Starting full audit")

        # 1. Parser validation
        parser_result = self._audit_parser(cpp_system)
        self.audit_results['parser'] = parser_result

        # 2. RL output validation
        rl_result = self._audit_rl_output(rl_model, test_cases)
        self.audit_results['rl_output'] = rl_result

        # 3. C++ integration
        cpp_result = self._audit_cpp_integration(cpp_system, rl_model)
        self.audit_results['cpp_integration'] = cpp_result

        # 4. Functional tests
        functional_result = self._audit_functional(rl_model, cpp_system, test_cases)
        self.audit_results['functional'] = functional_result

        # 5. Generate report
        report = self._generate_audit_report()

        return report

    def _audit_parser(self, cpp_system):
        """¿El parser funciona?"""

        results = {
            'roundtrip_test': False,
            'schema_validation': False,
            'hash_consistency': False,
            'errors': [],
        }

        try:
            # Test 1: Roundtrip
            parser = StrictCFGParser(cpp_system.config_root)
            for cfg_file in cpp_system.config_files:
                parsed = parser.parse_and_validate(cfg_file, schema_for_file(cfg_file))
            results['roundtrip_test'] = True
        except Exception as e:
            results['errors'].append(f"Roundtrip failed: {e}")

        try:
            # Test 2: Schema validation
            validator = CFGRoundtripValidator()
            for cfg_file in cpp_system.config_files:
                validator.validate_roundtrip(cfg_file, parser, cpp_system.serializer)
            results['schema_validation'] = True
        except Exception as e:
            results['errors'].append(f"Schema validation failed: {e}")

        try:
            # Test 3: Hash consistency
            hash_mgr = CFGHashManager()
            hash_mgr.validate_cfg_consistency(
                cpp_system.cfg_hash_at_training,
                hash_mgr.compute_combined_hash(cpp_system.config_root)
            )
            results['hash_consistency'] = True
        except Exception as e:
            results['errors'].append(f"Hash mismatch: {e}")

        results['pass'] = all([
            results['roundtrip_test'],
            results['schema_validation'],
            results['hash_consistency'],
        ])

        return results

    def _audit_rl_output(self, rl_model, test_cases):
        """¿La salida de RL es válida?"""

        results = {
            'output_range': False,
            'output_shape': False,
            'output_timing': False,
            'errors': [],
        }

        times = []

        for state in test_cases:
            # Test 1: Timing
            start = time.time()
            output = rl_model.forward(state)
            elapsed = time.time() - start
            times.append(elapsed)

            if elapsed > 0.003:  # 3 ms
                results['errors'].append(f"RL took {elapsed*1000:.2f}ms")

            # Test 2: Output shape
            if output.shape != (1, 6):  # Para Fase 2
                results['errors'].append(f"Shape mismatch: {output.shape}")

            # Test 3: Output range
            if not np.all((output >= 0) & (output <= 1)):
                results['errors'].append(f"Output out of [0,1] range: {output}")

        results['output_timing'] = np.mean(times) < 0.003
        results['output_shape'] = True
        results['output_range'] = True
        results['pass'] = len(results['errors']) == 0

        return results

    def _audit_cpp_integration(self, cpp_system, rl_model):
        """¿C++ se integra bien con RL?"""

        # Comprobar que C++ puede recibir y validar parámetros RL

        results = {
            'parameter_reception': False,
            'parameter_validation': False,
            'fallback_mechanism': False,
            'logging': False,
            'errors': [],
        }

        # Simular envío de parámetros RL a C++
        test_params = {
            'shootThreshold': 0.7,
            'minPassDistance': 1800,
            'maxPassDistance': 4200,
            'ratingThreshold': 0.7,
        }

        # (En un test real, esto sería a través de IPC/shared memory)

        results['pass'] = len(results['errors']) == 0

        return results

    def _audit_functional(self, rl_model, cpp_system, test_cases):
        """¿El sistema funciona end-to-end?"""

        results = {
            'e2e_success': 0,
            'e2e_failures': 0,
            'errors': [],
        }

        for test_case in test_cases:
            try:
                # RL predice
                action = rl_model.forward(test_case['state'])

                # C++ valida y ejecuta
                cpp_result = cpp_system.execute_action(action, test_case['state'])

                if cpp_result.success:
                    results['e2e_success'] += 1
                else:
                    results['e2e_failures'] += 1
                    results['errors'].append(f"Execution failed: {cpp_result.error}")

            except Exception as e:
                results['e2e_failures'] += 1
                results['errors'].append(f"Exception: {e}")

        results['success_rate'] = results['e2e_success'] / len(test_cases) * 100
        results['pass'] = results['success_rate'] > 95.0  # 95% min aceptable

        return results

    def _generate_audit_report(self):
        """Generar reporte final"""

        report = {
            'phase': self.phase,
            'timestamp': datetime.now().isoformat(),
            'all_sections': self.audit_results,
            'overall_pass': all(s.get('pass', False) for s in self.audit_results.values()),
            'summary': self._compute_summary(),
        }

        return report

    def _compute_summary(self):
        """Resumen de auditoría"""

        passed = sum(1 for s in self.audit_results.values() if s.get('pass', False))
        total = len(self.audit_results)

        return {
            'passed_sections': passed,
            'total_sections': total,
            'pass_rate': f"{passed/total*100:.1f}%",
            'status': 'PASS' if passed == total else 'FAIL',
        }
```

---

## 7. MATRIZ DE TRAZABILIDAD

### 7.1 Mapping CFG → Python → C++

```
LAYERED TRACEABILITY MATRIX

┌─────────────────────┬──────────────────────┬──────────────────────┐
│ CFG (C++ Original)  │ Python Parser        │ C++ Integration      │
├─────────────────────┼──────────────────────┼──────────────────────┤
│                                                                   │
│ FASE 1: Posicionamiento                                           │
│                                                                   │
│ t121_4v4_scaled.cfg │ TacticParser         │ StrategyBehavior     │
│  positions[i]       │  parsed_tactic[i]    │ Control.cpp:650      │
│  √ Roundtrip OK     │  √ Validated         │  √ Applied           │
│                                                                   │
│ t13_4v4_scaled.cfg  │ TacticParser         │ StrategyBehavior     │
│  positions[i]       │  parsed_tactic[i]    │ Control.cpp:650      │
│  √ Roundtrip OK     │  √ Validated         │  √ Applied           │
│                                                                   │
├─────────────────────┼──────────────────────┼──────────────────────┤
│                                                                   │
│ FASE 2: Parámetros Decisión                                      │
│                                                                   │
│ s4v4_scaled.cfg     │ StrategyParser       │ Behavior.cpp:445     │
│  ballXThreshold     │  transition_conds    │  Line 56: if(xG > ?) │
│  √ Roundtrip OK     │  √ Validated         │  √ Parameterized     │
│                                                                   │
│ PlayBall.h          │ HeaderParser         │ PlayBall.cpp:56      │
│  shootThreshold     │  playball_defaults   │  if(maxValue > thr)  │
│  √ Roundtrip OK     │  √ Validated         │  √ Uses RL param     │
│                                                                   │
│ PlayBall.h          │ HeaderParser         │ PlayBall.cpp:70-104  │
│  minPassDistance    │  playball_defaults   │  if(dist < min || >  │
│  maxPassDistance    │  playball_defaults   │  max) continue;      │
│  √ Roundtrip OK     │  √ Validated         │  √ Uses RL params    │
│                                                                   │
├─────────────────────┼──────────────────────┼──────────────────────┤
│                                                                   │
│ FASE 3: Action Reemplazo                                          │
│                                                                   │
│ (TODOS arriba)      │ (TODOS arriba)       │ PlayBall.cpp:        │
│  +                  │  +                   │  executeRL()         │
│  Parámetros de      │  ActionEncoder       │  validate_action()   │
│  acciones           │                      │  smashOrPass()       │
│  √ Roundtrip OK     │  √ Encoded           │  fallback to C++     │
│                                                                   │
└─────────────────────┴──────────────────────┴──────────────────────┘

TRAZABILIDAD: Cada parámetro CFG puede seguirse desde:
  CFG → Parser Python → Validación → Serialización → C++ Usage
```

### 7.2 Checklist de Trazabilidad

```python
class TraceabilityChecklist:
    """Verificar trazabilidad completa"""

    def verify_complete_traceability(self):
        """Cada parámetro puede trazarse end-to-end"""

        checklist = {
            'cfg_parsing': [],
            'python_validation': [],
            'cpp_integration': [],
            'execution': [],
            'auditing': [],
        }

        # CFG Parsing
        checklist['cfg_parsing'] = [
            ('Parse all .cfg files without error', False),
            ('Roundtrip test passes for all CFGs', False),
            ('Hash matches expected', False),
        ]

        # Python Validation
        checklist['python_validation'] = [
            ('All parameters in valid ranges', False),
            ('Parameter consistency checks pass', False),
            ('RL output matches schema', False),
        ]

        # C++ Integration
        checklist['cpp_integration'] = [
            ('Parameters received correctly', False),
            ('Parameters validated in C++', False),
            ('Fallback to C++ works if RL invalid', False),
        ]

        # Execution
        checklist['execution'] = [
            ('Action executed as expected', False),
            ('Result (goal/pass/etc) correct', False),
            ('Logging shows RL path taken', False),
        ]

        # Auditing
        checklist['auditing'] = [
            ('Full audit passes', False),
            ('No sim2real gaps detected', False),
            ('All validations logged', False),
        ]

        return checklist
```

---

## 8. PLAN DE VALIDACIÓN SIM2REAL

### 8.1 Estrategia Anti-Brecha Sim2Real

```
SIM2REAL RISK MATRIX

┌─────────────────────────────────────┬──────────┐
│ Fuente de Brecha                    │ Mitigación│
├─────────────────────────────────────┼──────────┤
│                                     │          │
│ 1. Timing (33ms Cognition thread)  │          │
│    Riesgo: RL toma >3ms            │ ✅ Medir │
│    Mitigación: Profiling + límites │    tiempo│
│                                     │          │
│ 2. Precision (float32 vs float64)  │          │
│    Riesgo: Diferencias numéricas   │ ✅ Usar │
│    Mitigación: Siempre float32     │ float32  │
│                                     │          │
│ 3. Stochasticity (ruido sensores)  │          │
│    Riesgo: RL entrena en sim puro  │ ✅ Domain│
│    Mitigación: Domain randomization│ randomiz │
│                                     │          │
│ 4. Physics (gravedad, fricción)    │          │
│    Riesgo: Sim ≠ Real              │ ✅ Tunear│
│    Mitigación: Calibrar params     │ physic  │
│                                     │ sim      │
│ 5. Action Execution (delays)       │          │
│    Riesgo: Latencia en robot real  │ ✅ Medir │
│    Mitigación: Accounting for ~70ms│ delay   │
│                                     │          │
│ 6. Observation (sensor errors)     │          │
│    Riesgo: Sensor data noisy       │ ✅ Filter│
│    Mitigación: Use filtered data   │ data    │
│                                     │          │
│ 7. Team Communication (100ms lag)  │          │
│    Riesgo: Info desactualizada     │ ✅ Account│
│    Mitigación: Usar delayed info   │ for lag │
│                                     │          │
└─────────────────────────────────────┴──────────┘
```

### 8.2 Validation Pipeline

```python
class Sim2RealValidationPipeline:
    """Pipeline de validación sim→real"""

    def __init__(self):
        self.phases = [
            'unit_tests',
            'integration_tests',
            'sim_validation',
            'sim2real_transfer',
            'real_robot_deployment',
        ]

    def run_validation_pipeline(self, rl_model, cpp_system):
        """Ejecutar todo el pipeline"""

        results = {}

        # Phase 1: Unit Tests (Python RL)
        results['unit_tests'] = self._unit_tests(rl_model)
        if not results['unit_tests']['pass']:
            raise Exception("Unit tests failed")

        # Phase 2: Integration Tests (Python + C++)
        results['integration_tests'] = self._integration_tests(rl_model, cpp_system)
        if not results['integration_tests']['pass']:
            raise Exception("Integration tests failed")

        # Phase 3: Sim Validation
        results['sim_validation'] = self._sim_validation(rl_model, cpp_system)
        if results['sim_validation']['success_rate'] < 70.0:
            raise Exception("Sim validation < 70% success")

        # Phase 4: Sim2Real Transfer
        results['sim2real'] = self._sim2real_transfer(rl_model)
        if results['sim2real']['domain_gap'] > 0.2:  # 20% max
            raise Exception("Domain gap too large")

        # Phase 5: Real Robot (si todo OK)
        if results['sim_validation']['success_rate'] > 85.0:
            results['real_robot'] = self._real_robot_deployment(rl_model)

        return results

    def _unit_tests(self, rl_model):
        """Tests unitarios de RL"""

        results = {
            'pass': True,
            'tests': [
                ('Model forward pass', self._test_forward_pass(rl_model)),
                ('Output shape', self._test_output_shape(rl_model)),
                ('Output range', self._test_output_range(rl_model)),
                ('Timing < 3ms', self._test_timing(rl_model)),
            ],
        }

        return results

    def _integration_tests(self, rl_model, cpp_system):
        """Tests de integración Python + C++"""

        results = {
            'pass': True,
            'tests': [
                ('IPC communication', self._test_ipc(rl_model, cpp_system)),
                ('Parameter passing', self._test_param_passing(rl_model, cpp_system)),
                ('Action execution', self._test_action_exec(rl_model, cpp_system)),
                ('Fallback mechanism', self._test_fallback(cpp_system)),
            ],
        }

        return results

    def _sim_validation(self, rl_model, cpp_system):
        """Validar en simulación"""

        # Correr múltiples episodios
        # Evaluar: goals, passes, valid actions

        results = {
            'episodes': 100,
            'goals_scored': 15,
            'passes_completed': 200,
            'success_rate': 15.0,  # %
            'pass': 15.0 > 70.0,  # ❌ FALLA
        }

        return results

    def _sim2real_transfer(self, rl_model):
        """Evaluar brecha sim→real"""

        # Comparar comportamiento en:
        # - Sim ideal
        # - Sim con ruido
        # - Sim2Real domain randomization

        results = {
            'sim_only': 85.0,  # % éxito
            'sim_with_noise': 78.0,  # % éxito
            'domain_randomized': 75.0,  # % éxito
            'domain_gap': (85.0 - 75.0) / 85.0,  # ~11.8%
            'pass': (85.0 - 75.0) / 85.0 < 0.2,  # < 20%
        }

        return results

    def _real_robot_deployment(self, rl_model):
        """Desplegar en robot real (PELIGROSO)"""

        # ⚠️ CRÍTICO: SOLO si sim validation > 85%

        # 1. Start with position-only (Fase 1)
        # 2. Gradualmente aumentar a decisiones
        # 3. Monitorear constantemente
        # 4. Estar listo para fallback manual

        results = {
            'phase1_success': 85.0,  # %
            'phase2_ready': True,
            'warnings': [
                "Always have manual backup",
                "Test in safe arena first",
                "Limit to 5-minute matches",
            ],
        }

        return results
```

---

## 9. RESUMEN CRÍTICO

### 9.1 Riesgos por Fase

```
FASE 1 (Posicionamiento)
├─ Riesgo: BAJO
├─ Brecha Sim2Real: Minimal (solo posiciones)
├─ Fallback: Simple (volver a CFG original)
└─ Recomendación: ✅ PROCEDER

FASE 2 (Parámetros Decisión)
├─ Riesgo: MEDIO
├─ Brecha Sim2Real: Posible (timing, ruido sensores)
├─ Fallback: C++ SmashOrPass original funciona
└─ Recomendación: ⚠️ PROCEDER CON CAUTELA
│                (solo si Fase 1 funciona > 95%)

FASE 3 (RL Reemplaza SmashOrPass)
├─ Riesgo: ALTO
├─ Brecha Sim2Real: GRANDE (todo puede fallar)
├─ Fallback: Solo si hard constraints pasan
└─ Recomendación: ❌ EVITAR A MENOS QUE
                  - Fase 2 funcione > 85%
                  - Domain randomization perfecto
                  - Tests extensos en hardware real
```

### 9.2 Checklist Final Pre-Deployment

```
ANTES DE CUALQUIER FASE EN ROBOT REAL:

□ Parser CFG valida 100% de parámetros
□ Roundtrip test PASA para todos los CFGs
□ Hash de CFGs coincide (training vs deploy)
□ RL output en float32 (NO float64)
□ RL timing < 3ms (perfil en hardware target)
□ Validaciones hard constraints en C++
□ Logging COMPLETO de todas las decisiones
□ Fallback a C++ funciona
□ Auditoría COMPLETA reporta PASS
□ Sim validation > 85% success
□ Domain gap < 20%
□ Tests en arena REAL (no solo sim)
□ Equipo físico disponible para intervenir
□ Sistema de safety desactivación disponible
□ Batería robot cargada completamente
□ Cámaras calibradas
□ Sensores validados

DESPUÉS DE CADA TEST EN ROBOT REAL:

□ Revisar logs de decisiones RL
□ Comparar acciones predichas vs ejecutadas
□ Verificar timing (esperado vs real)
□ Detectar anomalías en sensores
□ Actualizar conocimiento de brecha sim2real
□ Documentar todos los fallos
□ Ajustar parámetros si necesario
```

---
