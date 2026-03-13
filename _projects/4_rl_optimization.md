---
layout: page
title: Morphology–Control Co-Design via Nested RL
description: Optimizing robot body and brain jointly using evolutionary outer-loops and PPO inner-loop
img: assets/img/codesign/cover.png
importance: 1
category: research
related_publications: false
---

> The right body makes learning easier. The right learning makes the body matter.

This project asks: **can we jointly optimize a robot's physical structure and its controller?**

Rather than designing morphology by hand and training a policy on top, we close the loop —
letting the outer optimizer reshape the body while the inner RL agent learns to use it.

---

# Problem

Robot locomotion performance depends on two coupled factors:

- **Morphology**: link lengths, joint count, stiffness, damping, actuator placement
- **Controller**: the learned policy that maps states to actions

Traditional pipelines treat these sequentially:
fix morphology → train controller → evaluate → repeat manually.

This project automates that loop end-to-end.

---

# System Architecture

The pipeline is a nested optimization loop:

```
Outer loop (morphology optimizer)
    │
    ├── proposes morphology params (DOF, link lengths, stiffness, ...)
    │
    └── Inner loop (RL training)
            │
            ├── builds MuJoCo XML from params via mj-spec
            ├── trains PPO agent (SB3 / Brax)
            └── returns mean eval reward → fitness signal
```

The outer loop treats RL training as a **black-box fitness function**
and iteratively searches for morphology configurations that maximize locomotion reward.

---

# Outer-Loop Optimizers

We implemented and compared three outer-loop strategies:

### Genetic Algorithm (DEAP)

- Population-based search over discrete + continuous morphology parameters
- Genome: `[dof_per_leg (int), link_length (float)]`
- Operators: two-point crossover, Gaussian mutation, tournament selection
- Hall of Fame tracks top-k individuals across generations
- Checkpointing via `pickle` for resumable runs

### Bayesian Optimization

- Gaussian Process surrogate model over morphology parameter space
- Acquisition function (Expected Improvement) guides sampling
- Most sample-efficient: ideal when RL evaluations are expensive
- Particularly effective for continuous parameters like joint stiffness and damping

---

# Inner-Loop RL

Each outer-loop candidate triggers a full RL training run:

- **Algorithm**: PPO with linear learning rate schedule
- **Environment**: custom single-leg MuJoCo environment
- **Reward**: `1000 × forward_velocity` with actuator force and joint rate penalties
- **Control frequency**: 40 Hz
- **Parallelism**: `SubprocVecEnv` with up to 15 parallel workers

---

# Environment & Simulation Backends

We used three simulation backends at different stages:

### MuJoCo + mj-spec (CPU)

- Parametric XML generation via `mujoco.MjSpec`
- Dynamic variation of: link topology, link lengths, joint constraints, actuator parameters
- RobStride 03 motor parameters: peak torque 67 N·m, position gain kp=30

### MJX (GPU-accelerated MuJoCo)

- JAX-based GPU backend for MuJoCo
- Enables massively parallel environment rollouts on GPU
- Used for large-scale multi-seed evaluation of promising morphologies

### Brax (GPU)

- Full JAX/GPU physics engine
- Enables batched simulation of thousands of environments simultaneously
- Used for rapid fitness estimation in the outer loop when exact MuJoCo fidelity is not required

---

# Morphology Parameter Space

| Parameter | Type | Range |
|---|---|---|
| DOF per leg | Integer | 1 – 4 |
| Link length | Continuous | 0.2 – 0.6 m |
| Joint stiffness | Continuous | 10 – 500 N/m |
| Joint damping | Continuous | derived from motor specs |
| Actuator kp | Continuous | 10 – 50 |

---

# Trained Agent Demo

<div class="text-center">
  <video width="100%" controls autoplay muted loop class="rounded z-depth-1">
    <source src="/assets/video/render_L0.400.mp4" type="video/mp4">
  </video>
  <p class="caption mt-2">Best morphology found by GA outer-loop: link length = 0.4m, trained with PPO inner-loop.</p>
</div>

---

# Key Findings

- **DOF matters non-linearly**: 2–3 DOF legs outperform both 1-DOF (too rigid) and 4-DOF (too hard to control)
- **Link length interacts with DOF**: optimal length decreases as DOF increases
- **Bayesian optimization is most sample-efficient** for continuous parameters
- **GA finds diverse high-performing solutions** across the fitness landscape
- **Brax pre-screening** reduces total compute by filtering low-fitness candidates before MuJoCo evaluation

---

# Technical Stack

- **Simulation**: MuJoCo, MJX, Brax
- **XML generation**: mj-spec (`mujoco.MjSpec`)
- **RL**: Stable-Baselines3 (PPO), JAX-based PPO for Brax
- **Optimization**: DEAP (GA/PSO), scikit-optimize / BoTorch (Bayesian)
- **Parallelism**: SubprocVecEnv, JAX `vmap`
- **Tools**: Python, NumPy, TensorBoard, pickle checkpointing

---

# Broader Implication

Body and brain are not independent.

Mechanical structure shapes the learning problem —
it defines the exploration landscape, constrains the policy,
and determines how much the controller needs to compensate.

**Co-designing morphology and control is not just an engineering trick.
It is a more honest model of how biological systems develop.**