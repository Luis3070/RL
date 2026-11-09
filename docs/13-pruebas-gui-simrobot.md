# Comandos GUI 3D

## Agente heuristico

```bash
cd 
PYTHONPATH=/Make/Python \
python3 heuristic_agent_experiment.py --episodes 1 --no-plot --simrobot-real
```

## Probe visible heuristico

```bash
cd 
PYTHONPATH=/Make/Python \
python3 live_simrobot_probe.py --launch --policy heuristic --simrobot-profile 3d
```

## Probe visible fixed

```bash
cd 
PYTHONPATH=/Make/Python \
python3 live_simrobot_probe.py --launch --policy fixed --simrobot-profile 3d
```

## Visualizador de skills

```bash
cd 
PYTHONPATH=/Make/Python \
python3 visualize_simrobot_skills.py --launch --simrobot-profile 3d
```

## Visualizador de skills mas lento

```bash
cd 
PYTHONPATH=/Make/Python \
python3 visualize_simrobot_skills.py --launch --simrobot-profile 3d --steps-per-skill 30 --hold-seconds 2
```
