# QSimPy Setup and Training Guide

This repository contains a complete simulation pipeline for **quantum task scheduling** using **QSimPy** and **Ray RLlib**.
It supports **heuristic baselines**, **A2C**, and **DQN** training.

---

## 1. System Requirements

- Ubuntu 20.04 / 22.04
- Python 3.9 – 3.10
- CPU: 8+ cores recommended
- RAM: 16 GB (32 GB recommended)
- Optional GPU (CUDA-compatible) for faster training

---

## 2. Clone Repository

```bash
git clone <your-repo-url>
cd qsimpy
```

---

## 3. Create Virtual Environment

```bash
python3 -m venv qsimpy-env
source qsimpy-env/bin/activate
```

Verify:
```bash
which python
```

---

## 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Key libraries:
- ray==2.6.3
- ray[rllib]
- gymnasium
- numpy, pandas
- simpy
- qiskit (for later deployment)

---

## 5. QSimPy Environment Structure

Important files:
```
qsimpy/
├── env_creator.py          # registers QSimPyEnv
├── gymenv_qsimpy.py        # core environment logic
├── qdataset/               # quantum task dataset
├── ray_train_dqn.py        # DQN training
├── A2Ctrain.py             # A2C training
├── train_heuristics.py     # heuristic baselines
├── results/                # training outputs
```

---

## 6. Run Heuristic Baselines

```bash
python train_heuristics.py
```

Baselines implemented:
- Greedy
- Round-Robin
- Random

Outputs are logged to CSV for comparison.

---

## 7. Train DQN Agent

```bash
python ray_train_dqn.py \
  --framework torch \
  --stop-iters 100 \
  --stop-timesteps 100000
```

Checkpoints saved to:
```
results/DQN_QCE_1000/
```

---

## 8. Train A2C Agent

```bash
python A2Ctrain.py \
  --framework torch \
  --stop-iters 1000
```

Checkpoints saved to:
```
results/A2C_QCE_1000/
```

---

## 9. Verify Training Completion

Look for:
```
Training completed after XXXX iterations
checkpoint_000100 / checkpoint_001000
```

---

## 10. TensorBoard Visualization (Optional)

Install TensorBoard if missing:
```bash
pip install tensorboard
```

Run:
```bash
tensorboard --logdir ~/ray_results
```

Open browser:
```
http://localhost:6006
```

---

## 11. Notes on Parallelism

- RLlib uses rollout workers (process-based)
- High CPU usage is expected
- If training stalls, reduce:
```python
.rollouts(num_rollout_workers=0)
```

---

## 12. Current Status

✔ Environment recreated  
✔ Heuristic baselines completed  
✔ A2C training completed  
✔ DQN training completed  

---

## 13. Next Steps (Future Work)

- Extract trained policies for inference
- Offline evaluation (no environment recreation)
- Map scheduler decisions to IBM Quantum backends
- Deploy learned policy using Qiskit Runtime
- Compare simulation vs real quantum hardware

---

## Maintainer

Author: *Your Name*  
Project: **RL-based Scheduling for Quantum Cloud Systems**
