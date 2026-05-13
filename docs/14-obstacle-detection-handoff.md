# Handoff: Obstacle Detection en SimRobot 3D

## Estado al cierre de sesión

El robot RL **no detecta obstáculos** (`cam=0`, `oframe=0`, `hyp=0`, `pub=0` en todos los logs).
Se hizo una investigación completa del pipeline. Aquí está el diagnóstico y el punto exacto donde quedó.

---

## Lo que descubrimos (en orden)

### 1. `cam=0` no es falla de cámara

La pelota sí se detecta (color-based `BallPerceptor`). El problema es específico del
detector de robots (`RobotDetector` upper camera, `PlayersDeeptectorFeatBOPLower` lower camera).

### 2. Las escenas `Fast.con` y `PerceptOracle.con` usan oracle — no son comparables

- `GameFast` → `OracledWorldModelProvider` directamente para `ObstacleModel`, apaga cámaras
- `BHPerceptOracle` → `OracledPerceptsProvider` para percepts, apaga cámaras
- `Normal.con` (BH, Game) → mismo `RobotDetector` que el RL, cámaras encendidas

### 3. El detector SÍ funciona en simulación — el problema era geométrico

En `Normal.con` con 7v7 funciona porque hay robots cerca en múltiples direcciones.
En 1v0 RL, el obstáculo estaba diagonal y nunca entraba al frame de la upper camera.

### 4. El tilt de la cabeza es el cuello de botella

`LibLookActiveProvider::calculateTilt` apunta la cabeza **a la pelota** cuando `forceBall=true`.
`forceBall=true` casi siempre porque el robot RL es siempre el "striker" (único robot).

En 7v7, los robots que no son striker tienen `forceBall=false` → la cabeza escanea libremente →
la upper camera puede ver obstáculos.

### 5. Cuando el obstáculo está DIRECTO AL FRENTE sí se detecta intermitentemente

Con la nueva escena `RL1v0_3D_extra_dummy_front.ros2` (mismo eje y=0 que el robot),
el SimRobot UI muestra detecciones en:
- `ObstaclesImagePercept` ✓
- `ObstaclesFieldPercept` (tipo `unknown`) ✓
- `ObstaclesPerceptorData` ✓
- **`ObstacleModel` → vacío** ✗

### 6. Por qué `ObstacleModel` sigue vacío

El pipeline en `ObstacleModelProvider`:
```
clearAndFinish?  → NO (FallDownState = upright ✓)
ObstaclesFieldPercept en Cognition vacío? → A veces, intermitente
```

`PerceptionObstaclesFieldPerceptProvider` es un `ALIAS_MODULE` que selecciona:
```cpp
if(Cognition::isUpper)
    theObstaclesFieldPercept = theUpperObstaclesFieldPercept;
else
    theObstaclesFieldPercept = theLowerObstaclesFieldPercept;
```

Los datos llegan intermitentes a Cognition. Con `minPercepts=1` debería bastar una detección
para crear hipótesis. Se llegó al log step=64–105 sin ver ningún `cam>0`.

**Nota**: En los logs del nuevo test el robot está en `searchBallAt` (perdió la pelota),
lo que significa que la cabeza sí hace scan libre con `minTilt` más horizontal.
Aún así `cam=0` — la detección intermitente que se veía en el UI no está llegando
a crear hipótesis persistentes, o la bola se perdió y el robot no está pasando frente al dummy.

### 7. El `type=unknown` no es bloqueante

`ObstacleModelProvider` lo mapea a `Obstacle::someRobot` (línea 241) y lo procesa normalmente.

---

## Archivos clave

### Escenas de prueba creadas

| Archivo | Descripción |
|---------|-------------|
| `Config/Scenes/RL1v0_3D_extra_dummy_front.ros2` | Robot en (2.438, 0), dummy en (1.638, 0), pelota en (0,0). Alineados en y=0 |
| `Config/Scenes/RL1v0_3D_extra_dummy_front.con` | Usa `RLNatural`, `gc playing` |
| `Config/Scenes/RL1v0_3D_extra_naoextras_diag.ros2` | Intento anterior: Nao en extras diagonal (no funciona) |
| `Config/Scenes/RL1v0_3D_extra_nao_diag.ros2` | Nao en robots (se relocaliza, no funciona) |

### Config de obstáculos

```
Config/Locations/field_e_obstacle_diag/obstacleModelProvider.cfg
```
Parámetros relevantes actuales:
```
minPercepts = 1
notSeenThreshold = 24
maxDistance = 4000
debug = false
```

### Código relevante en SabanaHerons

| Archivo | Qué hace |
|---------|----------|
| `Src/Modules/Modeling/ObstacleModelProvider/ObstacleModelProvider.cpp` | Pipeline completo de obstáculos |
| `Src/Modules/BehaviorControl/Libraries/LibLookActiveProvider.cpp` | Controla dónde mira la cabeza |
| `Src/Modules/Infrastructure/InterThreadProviders/PerceptionProviders.cpp` | ALIAS_MODULE: puente Upper/Lower → Cognition |
| `Src/Modules/Perception/PlayersPerceptors/RobotDetector.cpp` | Detector neural upper camera |
| `Src/Modules/Perception/PlayersPerceptors/PlayersDeeptectorFeatBOPLower.cpp` | Detector BOP lower camera |
| `Config/Scenarios/4v4_RL3D/threads.cfg` | Mapeo de módulos por thread |

---

## Hipótesis pendiente de confirmar

El robot en los logs del último test (`step=64–105`) está en `searchBallAt` con
`ball.seen=0, since=3396ms+` — lleva ~4 segundos sin ver la pelota. En este estado
la cabeza debería estar escaneando libremente (sin `forceBall`), lo que permitiría
detectar el dummy. Sin embargo `cam=0` sigue.

**Posibles causas restantes:**
1. El dummy aún no está en el camino del robot en este episodio (el robot arrancó
   en este episodio ya en `searchBallAt`, quizás venía de otro lado)
2. La detección intermitente del UI ocurre pero crea hipótesis que se borran en el
   mismo frame por `shouldBeSeen` incrementando `notSeenButShouldSeenCount` agresivamente
3. El `ObstaclesFieldPercept` en Cognition llega vacío justo cuando `ObstacleModelProvider`
   corre (race condition inter-thread)

---

## Próximos pasos recomendados

### Opción A: Confirmar que el pipeline funciona end-to-end con obstáculo muy cerca

Modificar la escena para poner el dummy a 0.3m (casi tocando al robot):

```xml
<Translation x="2.138" y="0" z="320mm"/>
```

Si con eso aparece `cam>0`, el problema es que 0.8m sigue siendo demasiado lejos
para detección consistente desde esa trayectoria.

### Opción B: Aumentar `notSeenThreshold` para hipótesis más persistentes

En `Config/Locations/field_e_obstacle_diag/obstacleModelProvider.cfg`:
```
notSeenThreshold = 100;  # era 24
```

Esto da más tiempo para que la hipótesis sobreviva entre detecciones intermitentes.

### Opción C: Activar `debug` en obstacleModelProvider

```
debug = true;
```

Con `debug=true`, `isObstacle()` retorna `true` siempre (bypasea el check de seenCount).
Sirve para saber si las hipótesis se crean pero no pasan el filtro de publicación.

### Opción D: Modificar LibLookActive para scan periódico de obstáculos

En `LibLookActiveProvider::calculateTilt`, agregar lógica para que cada N frames
fuerce `defaultTilt` aunque `forceBall=true`. Esto requiere recompilar.

### Opción E (last resort): OracledPerceptsProvider solo para obstáculos

En `Config/Scenarios/4v4_RL3D/threads.cfg`, cambiar en Upper y Lower:
```
{representation = ObstaclesFieldPercept; provider = OracledPerceptsProvider;}
{representation = ObstaclesImagePercept; provider = OracledPerceptsProvider;}
{representation = ObstaclesPerceptorData; provider = OracledPerceptsProvider;}
```

El oracle tiene noise realista (`playerRecognitionRate=0.6`) — no es percepción perfecta.
Ya existía antes y fue removido; re-activarlo es el camino más rápido garantizado.

---

## Comandos

### Abrir escena de prueba directamente (sin reset de Python)

```bash
/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Build/Linux/SimRobot/Develop/SimRobot \
  /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenes/RL1v0_3D_extra_dummy_front.ros2
```

### Live probe heurístico con escena de prueba

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 live_simrobot_probe.py \
  --launch \
  --simrobot-profile 3d \
  --scene /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Config/Scenes/RL1v0_3D_extra_dummy_front.ros2
```

### Mandar stand sin reset (para diagnóstico estático)

```bash
cd /home/rafael/Documentos/Robocup2026/RL
PYTHONPATH=/home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python \
python3 stand_probe.py
```

### Recompilar SimulatedNao (necesario si se toca C++ de SabanaHerons)

```bash
cmake --build /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Build/Linux/CMake/Develop \
  --target SimulatedNao
```

### Recompilar pybh (necesario si se toca RLSharedState o Module.cpp)

```bash
cd /home/rafael/Documentos/Robocup2026/SabanaHerons2026/Make/Python
python3 setup.py build_ext --inplace
```

---

## Lo que NO es el problema (descartado)

- ~~Redes neuronales no funcionan en imágenes sintéticas~~ → Sí funcionan, la detección ocurre
- ~~Robots en `extras` no se renderizan~~ → Sí se renderizan
- ~~FallDownState = fallen borra el ObstacleModel~~ → Estado = upright
- ~~`type=unknown` bloquea la hipótesis~~ → Se mapea a `someRobot`, se procesa
- ~~El oracle es la única solución~~ → La percepción natural sí detecta cuando el obstáculo está en frente
- ~~Diferencia de config entre Normal.con y RLNatural.con~~ → Son equivalentes para obstacle detection
