# NETS Drift Simulation

This repo contains a simple Python simulation comparing **standard Langevin dynamics** with a **drift-only NETS-inspired dynamics** in a 1D double-well potential.  

It's meant as a toy project for exploring rare-event sampling and first-passage statistics.

Work is inspired by ["NETS: A Non-Equilibrium Transport Sampler" by Michael S. Albergo, Eric Vanden-Eijnden (2025)](https://arxiv.org/abs/2410.02711)

## Features

- 1D double-well potential $\(V(x) = (x^2 - 1)^2\)$
- Baseline overdamped Langevin dynamics
- Drift-only augmentation $\(u_\alpha(x) = -\alpha \tanh(x)\)$ to accelerate barrier crossings
- Euler–Maruyama integrator
- Sample trajectories and ensemble statistics
- Transition counts and Mean First-Passage Time (MFPT) distributions
- α-sweep: transition rate vs α and histogram distortion vs α

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

Install dependencies:

```bash
pip install numpy scipy matplotlib
