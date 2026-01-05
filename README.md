# QSimPy-Simulator-Deep-RL-Based-Task-Placement-in-Heterogeneous-Quantum-Cloud-Systems-
This repository explores quantum task scheduling in heterogeneous quantum cloud systems using simulation, heuristic methods, and Deep Reinforcement Learning (DQN, A2C). Built on QSimPy and Ray RLlib, the goal is to learn adaptive scheduling policies and deploy them on real IBM Quantum hardware to compare simulation and real execution performance.
Below is a **clean, GitHub-ready README.md** that accurately reflects **what you have done**, **what you are trying to achieve**, and **what remains**, based strictly on your actual work so far (QSimPy, heuristics, DQN, A2C, checkpoints, policy extraction issues, IBM Quantum goal).

You can **copy-paste this directly** into `README.md` in your repo.

---

# Task Placement using Deep Reinforcement Learning in Heterogeneous Quantum Cloud Computing

## 📌 Overview

This repository contains the implementation of **quantum task placement policies** for **heterogeneous quantum cloud environments** using **simulation, heuristic scheduling, and Deep Reinforcement Learning (DRL)**.

The work is built on top of **QSimPy**, a SimPy + Gymnasium–based quantum cloud simulator, and uses **Ray RLlib** to train and evaluate **DQN** and **A2C** agents for adaptive task scheduling across heterogeneous IBM quantum backends.

The **ultimate goal** is to **deploy the learned scheduling policies on real IBM Quantum hardware** and study the **simulation-to-real performance gap**.

---

## 🎯 Problem Statement

Given:

* A stream of quantum circuit tasks
* Multiple heterogeneous IBM quantum backends (varying qubits, CLOPS, availability)

**Objective:**
Learn a policy π(s → a) that selects the **best quantum backend** for each task to:

* Minimize waiting time + execution time
* Reduce rescheduling penalties
* Improve backend utilization

---

## 🧩 System Architecture

### Phase 1: Simulation & Training (Completed)

* QSimPy environment with:

  * 5 IBM QNodes (washington, kolkata, hanoi, perth, lagos)
  * 25 tasks per episode
  * SimPy-based discrete-event execution
* Gymnasium-compatible RL environment
* Ray RLlib integration

### Phase 2: Policy Learning (Completed)

* Implemented **heuristic baselines**
* Trained **DQN** and **A2C** agents
* Saved stable checkpoints

### Phase 3: Policy Extraction (In Progress)

* Load trained checkpoints
* Extract learned neural policies
* Convert policies into deployable schedulers

### Phase 4: Real IBM Quantum Deployment (Planned)

* Submit real circuits using Qiskit Runtime
* Collect execution logs
* Compare against simulation results

---

## ✅ What Has Been Completed

### 🔹 Simulation Environment

* Fully implemented QSimPy-based environment
* Realistic modeling of:

  * Task arrival
  * Queueing delay
  * Execution time
  * Rescheduling penalties
* Detailed logging per task

---

### 🔹 Heuristic Scheduling (Completed)

Implemented and evaluated:

* **Round Robin**
* **Greedy**
* **Greedy-Error**

**Observation:**

* Greedy-based heuristics consistently outperform Round Robin
* Random policies show high variance and instability

---

### 🔹 Deep Reinforcement Learning (Completed)

#### ✔ DQN Training

* Distributional DQN (C51)
* Prioritized Replay Buffer
* Checkpoints saved every 10 iterations
* Final training:

  * 100 iterations
  * ~100,000 environment steps
  * Stable convergence observed

#### ✔ A2C Training

* Actor–Critic based scheduler
* Trained up to 1000 iterations
* Checkpoints successfully saved

---

### 🔹 Training Artifacts

Checkpoints available at:

```bash
results/
├── DQN_QCE_1000/
│   └── DQN_QSimPyEnv_.../checkpoint_000100
├── A2C_QCE_1000/
│   └── A2C_QSimPyEnv_.../checkpoint_000100
│   └── A2C_QSimPyEnv_.../checkpoint_001000
```

Each checkpoint contains:

* `algorithm_state.pkl`
* `policies/default_policy/`
* `rllib_checkpoint.json`

---

## ⚠️ Current Issue: Policy Extraction

### What We Tried

* Used `Algorithm.from_checkpoint()`
* Attempted to load A2C and DQN policies
* Tried extracting via `get_policy("default_policy")`

### Problem Encountered

RLlib **recreates rollout workers** during checkpoint loading and expects the environment `QSimPyEnv` to be **registered again**, causing:

```
gymnasium.error.NameNotFound: Environment QSimPyEnv doesn't exist
```

### Key Insight

* **Policy extraction still requires env registration**
* Even for inference-only loading, RLlib internally spins up workers
* The fix is architectural (separate inference-only loader)

👉 This is **expected behavior**, not a bug in your code.

---

## 🎯 What We Are Trying to Achieve (Policy Extraction Goal)

The final extracted policy should:

* Take `(task_state, node_state)` as input
* Output a **QNode selection**
* Work **without Ray / RLlib / Gym**
* Be deployable on:

  * Real IBM Quantum Cloud
  * Heuristic comparison pipelines

This policy will be used as:

* `DQNPolicyScheduler`
* `A2CPolicyScheduler`

---

## 🚀 Planned Next Steps

### 🔜 Short-Term (Next Implementation Step)

1. Create an **inference-only policy loader**
2. Disable rollout workers during checkpoint load
3. Export:

   * Torch model weights
   * Action selection logic

### 🔜 Medium-Term

4. Implement **IBMQuantumExecutor**
5. Submit circuits via `QiskitRuntimeService`
6. Log:

   * Job ID
   * Backend used
   * Execution time
   * Queue delay

### 🔜 Final Evaluation

7. Compare:

   * Heuristics vs DQN vs A2C
   * Simulation vs real hardware
8. Analyze **sim-to-real gap**

---

## 🏁 Final Goal

A **complete, end-to-end quantum task scheduling system** that:

* Trains in simulation
* Extracts learned policies
* Deploys on real IBM quantum hardware
* Scientifically evaluates DRL vs heuristics in real conditions

---

## 📚 References

* DRLQ: Deep Reinforcement Learning for Quantum Cloud Scheduling (IEEE TPDS, 2024)
* QSimPy Framework (Nguyen et al., 2024)
* Rainbow DQN (Hessel et al., AAAI 2018)

---

If you want, next I can:

* Fix the **policy extraction script properly**
* Design a **policy-only inference class**
* Or move directly to **IBM Quantum deployment code**

Just tell me 👍
