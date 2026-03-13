---
layout: page
title: Aim IV — Tendon Elasticity & Learning
description: How tendon stiffness shapes learning dynamics in overdetermined tendon-driven systems
img: assets/img/aim4/cover_leg.jpg
importance: 1
category: research
related_publications: false
---

This project investigates how **mechanical design**, specifically tendon stiffness,
affects reinforcement learning performance in an overdetermined tendon-driven system.

Biological limbs are compliant and redundant.
Can elasticity accelerate skill acquisition?

---

# Problem

Tendon-driven limbs are:

- Overdetermined (more actuators than joints)
- Nonlinear
- Dynamically coupled

Mechanical compliance introduces:

- Energy storage
- Shock absorption
- Smoother force modulation

But also increases control complexity.

The central question:

**Can adaptive tendon stiffness improve learning speed and final performance?**

---

# Environment Design

We built a custom MuJoCo environment:

- 2-joint, 3-tendon leg
- Mass: 0.5m, 3kg
- Task: learn to walk forward
- Control: PPO
- Control frequency: 40 Hz
- 500 episodes per run
- 20+ random seeds

The agent receives reward proportional to forward velocity: `reward = 1000 × forward_velocity`

Additional smoothness penalties are applied:

- Actuator force rate penalty
- Joint angle rate penalty

---

# Stiffness Scheduling

We implemented a flexible stiffness scheduler supporting:

- Constant stiffness
- Linear growth
- Exponential interpolation
- Logarithmic growth
- Sigmoid schedule

This allows controlled experiments comparing:

### Constant stiffness
Tendons remain fixed during training.

### Dynamic stiffness
Tendons adapt gradually during learning.

---

# Experimental Conditions

### Constant stiffness values tested:

- 10 N/m
- 50 N/m
- 100 N/m
- 200 N/m
- 500 N/m

### Dynamic schedule:

- 50 → 100 N/m (linear)
- Other progressive schedules

Each condition evaluated across multiple seeds.

---

# Performance Results

{% include figure.liquid path="assets/img/aim4/displacement_distribution.png" title="Final cumulative displacement across stiffness profiles" class="img-fluid rounded z-depth-1" %}

Dynamic stiffness configurations outperform most constant configurations.

Extremely stiff and extremely compliant systems show:

- Lower mean displacement
- Higher variance across seeds

---

# Gait Pattern Analysis

To understand coordination, we analyzed:

- Hip–knee phase trajectories
- Ellipse area
- Eccentricity
- Dynamic Time Warping (DTW)
- Occupancy entropy

### Observations

Low stiffness:
- High variability
- Less stable gait

High stiffness:
- Rigid, limited coordination

Dynamic stiffness:
- Larger coordination envelope
- Lower eccentricity
- Smoother stabilization

---

# Quantitative Findings

Constant stiffness → higher eccentricity  
Dynamic stiffness → larger ellipse area  

This suggests dynamic adaptation improves joint coordination.

---

# Key Insights

1. Extremely stiff or compliant systems underperform.
2. Dynamic stiffness schedules dominate static configurations.
3. Mechanical adaptivity improves both learning speed and final locomotion quality.
4. Morphology–control co-adaptation matters.

---

# Technical Stack

- MuJoCo + Stable-Baselines3 (PPO)
- Custom stiffness scheduler
- Multi-seed training aggregation
- DTW & entropy-based gait analysis
- NumPy / PyTorch

---

# Broader Implication

Learning is not only shaped by reward and policy.

Mechanical properties influence exploration landscape, stability, convergence path, and variance across learners.

**Mechanical design and learning dynamics are inseparable.**