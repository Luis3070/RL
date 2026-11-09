# RL1v0_3D: percepcion natural y diagnostico

## Resumen

El sistema empezo a funcionar cuando `RL1v0_3D` dejo de usar una escena pensada para world model oracular rapido y paso a usar una escena que alimenta la percepcion igual que las escenas normales de B-Human.

La correccion real no fue un ajuste de PPO ni un cambio cosmetico del probe. Fue alinear el flujo de la escena RL con el flujo de percepcion natural de B-Human.

## Problema real

`Config/Scenes/RL1v0_3D.con` estaba llamando:

- `Includes/Fast`

Ese include esta pensado para escenas con world model oracular:

- apaga `CameraImage` con `ci off`,
- no conecta `CameraImage` desde `LogDataProvider`,
- desactiva representaciones de percepcion como `BallPercept`, `BallSpots` y `FilteredBallPercepts`,
- y entrega varias representaciones principales desde `OracledWorldModelProvider`.

Con esa base, el escenario RL podia caminar con el pipeline natural de skills, pero la percepcion visual de pelota no estaba alimentada como en una escena normal.

## Correccion aplicada

Se hicieron dos cambios estructurales en B-Human:

1. `Config/Scenes/RL1v0_3D.con` paso de `Includes/Fast` a `Includes/RLNatural`.
2. `Config/Scenes/Includes/RLNatural.con` se alineo con `Config/Scenes/Includes/Normal.con`.

En concreto, `RLNatural.con` ahora conecta:

- `for Lower Upper mr CameraImage LogDataProvider`
- `for Lower Upper mr CameraInfo LogDataProvider`
- `for Lower Upper mr FrameInfo LogDataProvider`
- los sensores del motion thread
- `for Cognition mr GroundTruthWorldState LogDataProvider`

Eso hace que los threads `Upper` y `Lower` vuelvan a recibir imagen, metadata de camara y frame timing como en una escena 3D normal.

## Cambio en providers RL

Ademas, los escenarios RL dedicados dejaron de usar `OracledWorldModelProvider` para varias representaciones principales:

- `BallModel -> BallStateEstimator`
- `GlobalOpponentsModel -> GlobalOpponentsTracker`
- `ObstacleModel -> ObstacleModelProvider`
- `RobotPose -> SelfLocator`
- `TeammatesBallModel -> TeammatesBallModelProvider`

Eso mueve el observation space hacia estado estimado y no estado perfecto.

## Por que ahora si funciona

Porque ahora existe una cadena coherente de datos:

1. SimRobot entrega imagenes y datos de camara a `Upper` y `Lower`.
2. Los perceptores naturales vuelven a correr.
3. `BallPercept` y `FilteredBallPercepts` vuelven a producir datos reales.
4. `BallStateEstimator` alimenta `BallModel`.
5. `SkillBehaviorControl` y el bridge Python ya pueden observar pelota real.

La evidencia en los logs fue:

- `ballpipe=(percept=1,filtered=1,...)`
- `vision=(up=1, ur=3, ...)`

Interpretacion:

- `percept=1`: existio `BallPercept`.
- `filtered=1`: existio al menos un `FilteredBallPercept`.
- `up=1`: la camara upper acepto una pelota.
- `ur=3`: el candidato fue aceptado por `BallPerceptor`.

Antes de la correccion, el patron era:

- `percept=0`
- `filtered=0`
- `up=0`
- `ur=5`

Eso significaba que habia spots o candidatos parciales, pero el pipeline no estaba produciendo una pelota valida para el modelo.

## Rol del probe

`live_simrobot_probe.py` no arreglo la percepcion por si mismo. Su rol fue:

- dejar de disparar basado en pelota stale,
- entrar en `searchBallAt(...)` cuando la pelota no es confiable,
- y exponer telemetria suficiente para distinguir:
  - falta de spots,
  - rechazo del perceptor,
  - falta de `CameraMatrix`,
  - salida real del `BOPPerceptor`.

Ese probe fue el instrumento de diagnostico. La correccion del bug estuvo en la escena y en el wiring de providers.

## Estado final

El sistema ahora queda asi:

- movimiento: natural (`SkillRequest -> SkillBehaviorControl -> MotionRequest -> MotionEngine`)
- percepcion de pelota: natural
- world model principal usado por RL: natural/estimado en vez de oracular
- observaciones Python: enriquecidas con telemetria de depuracion para verificar la cadena completa

## Leccion tecnica

Si una escena RL quiere usar percepcion natural, no puede heredar de un include optimizado para world model oracular rapido.

La escena tiene que parecerse a `Normal`, no a `Fast`, al menos en:

- `CameraImage`
- `CameraInfo`
- `FrameInfo`
- representaciones activas de percepcion
- providers principales de localizacion y pelota
