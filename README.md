# Houdini Reinforcement Learning Race Car

A self-driving race car trained inside SideFX Houdini. A PPO agent with continuous control (steer, throttle, brake) learns in a SOP training environment, with PyTorch handling the policy network. The agent observes the track through a lidar-based 18-dimensional state and is trained across 5 tracks using a curriculum.

Training runs headless through `hython` across multiple worker processes. Inference runs inside the Houdini GUI from a Python Script node.

---

## Contents

| File | Description |
|---|---|
| `common.py` | Paths, PPO hyperparameters, network definition, GAE, curriculum, atomic file I/O |
| `worker.py` | Rollout collector — one headless Houdini instance per worker |
| `trainer.py` | Distributes assignments, gathers rollouts, runs the PPO update |
| `houdini files.zip` | `worker_1.hip` (training environment) and `inference.hip` (inference environment) |
| `inference cache.zip` | Recorded inference runs for the 5 default tracks |
| `model/` | Trained weights — `Self_Driving_Agent_Example_01.pth` and periodic checkpoints |

---

## Requirements

- **Houdini 20.0.724** — the version the project was built and tested in
- **Windows 10/11** — the scripts use Windows paths and `os.replace()` for atomic writes
- **PyTorch** (see setup below)

Houdini 20.0 ships **Python 3.10**, so the virtual environment must be built from Houdini's own interpreter. The networks are small and run on CPU — CUDA is not required.

**Keep the versions aligned.** The Houdini build, the Python version and the PyTorch wheel all have to match: build the venv from the `python3XX` interpreter inside your own Houdini install, and install a PyTorch wheel that supports that Python version. If you are on a different Houdini build, adjust every path in this README accordingly — a venv built from the wrong Python will import fine in a terminal but fail inside Houdini.

**Note on the `.hip` files.** They were last saved in Houdini 21.0.631, so opening them in 20.0.724 shows a version warning. They open and run normally — the warning can be dismissed.

---

## Part 1 — PyTorch setup

### 1. Create a virtual environment from Houdini's Python

```bat
"C:\Program Files\Side Effects Software\Houdini 20.0.724\python310\python.exe" -m venv houdini_pytorch_env
houdini_pytorch_env\Scripts\activate
python --version
```

Should report `Python 3.10.x`. If `venv` fails with an `ensurepip` error, create it with `--without-pip` and then run `houdini_pytorch_env\Scripts\python.exe -m ensurepip --upgrade`.

### 2. Install PyTorch

CPU build (sufficient for this project):

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.4 build, if you want GPU support:

```bat
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install `torch` only. `torchvision` and `torchaudio` pull in NumPy 2.x, which conflicts with the NumPy that Houdini bundles. If you need NumPy in the venv, pin it with `pip install "numpy<2"`.

Verify:

```bat
python -c "import torch; print(torch.__version__)"
```

### 3. Point Houdini at the virtual environment

Open (or create) `C:\Users\<YourUsername>\Documents\houdini20.0\houdini.env` and add:

```
PYTHONPATH = C:/Users/<YourUsername>/houdini_pytorch_env/Lib/site-packages;$PYTHONPATH
PATH = C:/Users/<YourUsername>/houdini_pytorch_env/Scripts;$PATH
```

Use forward slashes and `;` separators. Houdini only reads this file at launch, so restart afterwards.

### 4. Confirm it works

In Houdini, add a Python SOP containing `import torch; print(torch.__version__)`. Then confirm the headless interpreter too, since training depends on it:

```bat
"C:\Program Files\Side Effects Software\Houdini 20.0.724\bin\hython.exe" -c "import torch; print(torch.__version__)"
```

If the GUI works but `hython` doesn't, `houdini.env` wasn't picked up.

---

## Part 2 — Training

### 1. Set the paths in `common.py`

```python
base_dir    = r"D:\your\project\folder"   # working directory
num_workers = 5                           # must match the number of workers you start
```

`comms/` and `model/` are created automatically under `base_dir`. `comms/` holds assignments, rollouts, per-round weights and done flags; old rounds are cleaned up automatically.

### 2. Create the worker scene files

Only `worker_1.hip` ships in the repo. Each worker loads its own copy so that instances don't contend over the same file. Copy and rename it in `base_dir`:

```
base_dir\worker_1.hip
base_dir\worker_2.hip
base_dir\worker_3.hip
base_dir\worker_4.hip
base_dir\worker_5.hip
```

The count must match `num_workers`. If a worker's `.hip` is missing, or fewer workers are started than `num_workers`, the trainer waits forever for done flags that never arrive.

### 3. Place the scripts

Keep `common.py`, `worker.py` and `trainer.py` together in one folder. Both scripts import `common`, so they need to sit side by side.

### 4. Start the workers

The venv does **not** need to be activated — `hython` picks up PyTorch through `houdini.env`. Open one terminal per worker and pass the worker ID as an argument:

```bat
cd C:\path\to\scripts
"C:\Program Files\Side Effects Software\Houdini 20.0.724\bin\hython.exe" worker.py 1
```

Repeat with `worker.py 2` through `worker.py 5` in separate terminals. Each worker prints `=== Worker N started, waiting for assignments ===` and then waits.

To launch all five at once, save this as `start_workers.bat` next to the scripts:

```bat
@echo off
set HYTHON="C:\Program Files\Side Effects Software\Houdini 20.0.724\bin\hython.exe"
for /L %%i in (1,1,5) do start "Worker %%i" %HYTHON% worker.py %%i
```

Each worker is a full Houdini session, so check you have the licences and the RAM for the number you start.

### 5. Start the trainer

In one more terminal:

```bat
cd C:\path\to\scripts
"C:\Program Files\Side Effects Software\Houdini 20.0.724\bin\hython.exe" trainer.py
```

Each round it writes the current weights, assigns a track per worker, waits for all rollouts, runs the PPO update, and prints a summary line:

```
Round  120 | Episodes  600 | T0:287 | T2:412 | ... | Avg100  1843.21 | LR 4.98e-04 | Ent 0.0495 | LogStd -0.912
```

`T<track>:<steps>` shows the track and step count per worker. Watch `Avg100` for progress and `LogStd` for entropy pressure — climbing towards positive values means the entropy coefficient is too strong.

### 6. Output

- Checkpoints are written to `model/checkpoint_<episodes>.pth` every 100 rounds.
- On completion, the best model from the final 10% of training is saved to `model/Self_Driving_Agent_Example_01.pth`.

Since the final save only happens at the end of `num_episodes`, use the checkpoints if you stop early.

Track selection follows a curriculum in `common.py` — track 0 only at first, widening to all five by 32% through training.

---

## Part 3 — Inference

Inference runs in the **Houdini GUI**, not headless. Nothing needs to be launched from a terminal.

### 1. Unpack the files

- Extract `houdini files.zip` and open `inference.hip`.
- Extract `inference cache.zip` and place the five folders inside a `Recording` folder next to `inference.hip`, so the paths become `<folder containing inference.hip>/Recording/260906_Inference_Monza_01/`, and so on.

The filecache nodes read from `$HIP/Recording/`, so this resolves automatically — with one exception: the Spa folder is named `260903_Inference_Spa_01` in the zip while the node points at `260906_Inference_Spa_01`. Rename the folder or repoint that node.

### 2. Check the settings in `/obj/Inference_Script`

Open the Python Script node `Inference_Script` at object level and edit the config block at the top:

```python
model_path = r"C:\path\to\model\Self_Driving_Agent_Example_01.pth"   # any .pth from model/
track      = 3                                                       # 0-4
max_steps  = 1440
```

`model_path` is the only value that must be changed — it is an absolute path. Point it at `Self_Driving_Agent_Example_01.pth` or any checkpoint in `model/`.

### 3. Set the output location

Inside `/obj/Inference_Environment`, set `Result_Recorder_01` to write somewhere sensible, for example `$HIP/Recording/My_Inference_Run_01`. Otherwise a new run overwrites an existing cache.

### 4. Run

Execute the `Inference_Script` node. Progress prints to the Python Shell.

The script runs in two passes:

1. **Rollout** — the policy runs deterministically and every action is stored in memory. Nothing is written to disk.
2. **Replay** — the environment is reset and the stored actions are replayed step by step, with `Result_Recorder_01` firing on each step.

The split is deliberate: calling the recorder inside the step loop forces the upstream solver to re-evaluate, which breaks simulation continuity. Separating recording from simulation avoids this.

When it finishes, scrub the timeline to play back the recorded run.

---

## Troubleshooting

**`import torch` works in the GUI but not in `hython`.** Check `houdini.env` uses forward slashes, `;` separators, and lives under `Documents\houdini20.0`.

**Trainer hangs at a round.** Fewer workers running than `num_workers`, or a worker crashed. Check each worker terminal.

**Workers run but training doesn't improve.** Confirm each worker was started with a different ID — the seed is derived from it (`base_seed + worker_id * 1000`), and identical IDs produce identical rollouts.

**`PermissionError` on files in `comms/`.** Two processes hit the same file. All writes go through the atomic helpers in `common.py`; check nothing else has the folder open.

**Inference cache doesn't load.** Folder names must match the filecache paths exactly, including the Spa mismatch noted above.
