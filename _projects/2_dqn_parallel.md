---
layout: page
title: Parallelized Q-Learning & DQN
description: Parallel reinforcement learning with lock-free and CUDA-accelerated replay buffers achieving 3× training speedup
img: assets/img/7.jpg
importance: 2
category: research
---

Reinforcement learning training is notoriously slow due to sequential environment interactions and Q-value updates. 
This project explores **parallel strategies for Q-Learning and Deep Q-Network (DQN)** to dramatically reduce 
training time while maintaining learning stability and convergence.

---

## Problem: Why Parallelization Is Hard in RL

Managing shared data across threads in RL introduces two core conflicts:

- **Race conditions**: Simultaneous Q-table or neural network weight updates from multiple threads 
  cause unstable, incorrect learning.
- **Synchronization overhead**: Frequent locking or global model syncing reduces the speedup gains 
  from parallelism.

The goal: find a synchronization strategy that balances **conflict contention** vs **parallel efficiency**.

---

## Three Parallelization Strategies

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="Lock-free parallel" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/6.jpg" title="Lock-based parallel" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/7.jpg" title="CUDA-accelerated" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Three strategies compared: lock-free (max throughput, unstable), lock-based (stable, overhead), 
    and CUDA-accelerated (GPU parallelism for batch updates).
</div>

**1. Lock-Free Parallel Q-Learning**  
Maximum efficiency — no locks, no waiting. Multiple threads update the Q-table simultaneously.  
Trade-off: dependent Q-values may be modified mid-update, causing instability and convergence issues.

**2. Lock-Based Parallel Q-Learning**  
Threads acquire a mutex before writing to shared Q-table. Guarantees correct updates.  
Trade-off: lock contention under high thread counts reduces speedup.

**3. CUDA-Accelerated Replay Buffer**  
Batch Q-value updates offloaded to GPU. Parallelism across thousands of CUDA cores.  
Achieves the best of both: high throughput and correctness via GPU's native parallel architecture.

---

## Key Results

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/9.jpg" title="Training speedup results" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/10.jpg" title="Convergence comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: Training speedup vs. baseline across strategies. Right: Convergence curves for Q-Learning 
    under different parallelization schemes.
</div>

| Strategy | Speedup | Stability |
|---|---|---|
| Baseline (sequential) | 1× | ✅ Stable |
| Lock-free parallel | **3×** | ⚠️ Unstable |
| Lock-based parallel | 1.8× | ✅ Stable |
| CUDA replay buffer | **3× + 58% latency reduction** | ✅ Stable |

The **CUDA-accelerated replay buffer** reduced training latency by **58%** compared to baseline, 
while maintaining convergence quality equivalent to the sequential implementation.

---

## From Q-Learning to DQN

The project also investigated the transition from tabular Q-learning to **neural function approximation** 
(Deep Q-Network), evaluating performance trade-offs in:

- **Discrete action spaces**: CartPole, FrozenLake — tabular Q-learning competitive
- **Continuous/high-dimensional state spaces**: where DQN's generalization capability is essential
- **Asynchronous update instability**: a key failure mode when parallelizing DQN without target network 
  synchronization

---

## Technical Stack

- **Languages**: C++, Python (OpenMP for CPU parallelism)
- **Frameworks**: PyTorch, CUDA
- **Tools**: NumPy, Matplotlib, TensorBoard

---

## Implementation Highlights

The lock-free Q-table update pseudocode:

```
thread i (0 <= i < n):
  for episode = 1 -> E/n:
    initialize s
    for step = 1 -> T:
      choose action a via ε-greedy
      take action a, observe r, s'
      Q(s,a) <- Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]  # no lock
      s <- s'
      if terminal: break
```

Lock-based version wraps the Q update with `mutex.lock()` / `mutex.unlock()`, 
trading throughput for correctness. The CUDA version batches these updates across 
thousands of parallel threads on GPU.