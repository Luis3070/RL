# Entorno De Validacion

## Objetivo

Dejar un entorno reutilizable para correr:

- `diag_n6_slots.py`
- `probe_protocol_experiment.py`
- `train.py` en smoke o train

sin tener que recordar a mano `PYTHONPATH`, `Xvfb` ni el binding `pybh`.

## Lo Unico Que Si Debe Estar Instalado En El Sistema

En esta maquina falto esto:

```bash
sudo apt install python3.12-venv xvfb
```

Eso resuelve dos bloqueos reales:

- `python3 -m venv` fallaba porque no estaba `ensurepip`
- `Xvfb` no estaba en `PATH`, y el runtime visible N=6 lo necesita

No hace falta instalar paquetes Python globalmente.

## Preparacion Persistente

Desde `RL/`:

```bash
./scripts/setup_validation_env.sh
```

Ese script:

- crea `.venv/` si no existe
- actualiza `pip`, `setuptools`, `wheel`
- instala:
  - `torch`
  - `skrl`
  - `pettingzoo`
  - `gymnasium`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `psutil`
  - `tqdm`

## Como Correr Comandos De Validacion

Usa siempre:

```bash
./scripts/run_with_validation_env.sh <comando> [args...]
```

Ejemplos:

```bash
./scripts/run_with_validation_env.sh python3 diag_n6_slots.py
./scripts/run_with_validation_env.sh python3 probe_protocol_experiment.py --case visible_search
./scripts/run_with_validation_env.sh env RL_RUN_MODE=smoke python3 train.py
```

## Fallback Actual

Mientras `.venv` no exista, `run_with_validation_env.sh` puede usar `.deps/` si ya esta poblado localmente.

Eso sirve para pruebas inmediatas, pero no es el entorno persistente recomendado. El entorno estable para futuras validaciones debe ser `.venv/`.
