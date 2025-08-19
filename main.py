import os, textwrap, json, numpy as np

os.makedirs("/mnt/data/figs", exist_ok=True)
"""
Toy simulation: Langevin vs. drift-only NETS-ish model
Author: Ethan Furman

Notes:
- Overdamped Langevin + optional drift toward barrier
- Looks at trajectories, histograms, MFPTs
- alpha sweep shows tradeoff between acceleration and histogram distortion
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from typing import Tuple, Dict, List
from scipy.stats import sem, entropy

# Potential and derivatives
def V(x: np.ndarray) -> np.ndarray:
    """Double-well potential: (x^2 - 1)^2"""
    return (x**2 - 1.0)**2

def dVdx(x: float) -> float:
    """d/dx of (x^2 - 1)^2 = 4x(x^2-1)"""
    """derivative of double well"""

    return 4.0 * x * (x**2 - 1.0)

# Single-step functions
def step_langevin(x: float, dt: float, beta: float, rng: np.random.Generator) -> Tuple[float, float]:
    """One Euler-Maruyama step of baseline overdamped Langevin with alpha = sqrt(2/beta). Returns (x_next, xi)."""
    xi = rng.normal()
    sigma = sqrt(2.0 / beta) # diffusion coefficient
    x_next = x - dVdx(x) * dt + sigma * sqrt(dt) * xi
    return x_next, xi

def u_alpha(x: float, alpha: float) -> float:
    """Drift augmentation u(x) = -(alpha) tanh(x). Pushes toward barrier at x=0 from both sides."""
    return -alpha * np.tanh(x)

def step_nets(x: float, dt: float, beta: float, alpha: float, rng: np.random.Generator) -> Tuple[float, float, float]:
    """
    Drift-only augmented dynamics:
      dx = [ -V'(x) + u_alpha(x) ] dt + sqrt(2/β) dW
    Returns (x_next, xi, u) where xi ~ N(0,1), u = u_alpha(x).
    """
    xi = rng.normal()
    sigma = sqrt(2.0 / beta)
    u = u_alpha(x, alpha)
    x_next = x + (-dVdx(x) + u) * dt + sigma * sqrt(dt) * xi
    return x_next, xi, u

# Utilities // helper functs
def theoretical_boltzmann(xgrid: np.ndarray, beta: float) -> np.ndarray:
    """compute normalized Boltzmann pdf"""

    p = np.exp(-beta * V(xgrid))
    # Normalize to integrate to 1
    Z = np.trapz(p, xgrid)
    return p / Z

def count_transitions(traj: np.ndarray, threshold: float = 0.0, buffer: float = 0.1) -> int:
    """
    Count well-to-well transitions with a small buffer around the barrier to avoid flicker.
    We consider a transition only when the trajectory goes from x < -(buffer) to x > +(buffer) or vice-versa.
    """
    left = traj < -buffer
    right = traj > buffer
    # Encode states: -1 (left), +1 (right), 0 (middle)
    state = np.zeros_like(traj, dtype=int)
    state[left] = -1
    state[right] = +1
    # Count signed state changes ignoring zeros
    valid = state != 0
    s = state[valid]
    if len(s) <= 1:
        return 0
    return int(np.sum(np.abs(np.diff(s)) == 2))  # change -1 <-> +1

def first_passage_time(traj: np.ndarray, dt: float, x_hit: float = 0.2) -> float:
    
    """the amount of time it takes to hit right well from left"""
    
    idx = np.where(traj >= x_hit)[0]
    return dt * idx[0] if len(idx) else np.nan

def l1_distance(p: np.ndarray, q: np.ndarray, dx: float) -> float:
    """L1 (integral absolute difference) distance between densities p and q given bin width dx."""
    return float(np.sum(np.abs(p - q)) * dx)

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p||q) for discrete densities (adds small epsilon to avoid zeros)."""
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))  # scipy.stats.entropy computes KL(p||q)

# Core simulation routines
def simulate_trajectory(method: str, T: float, dt: float, beta: float, alpha: float, x0: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a single trajectory for time horizon T with steps dt.
    method: 'langevin' or 'nets'
    """
    steps = int(T / dt)
    x = x0
    traj = np.empty(steps + 1, dtype=float)
    traj[0] = x
    if method == 'langevin':
        for n in range(steps):
            x, _ = step_langevin(x, dt, beta, rng)
            traj[n+1] = x
    elif method == 'nets':
        for n in range(steps):
            x, _, _ = step_nets(x, dt, beta, alpha, rng)
            traj[n+1] = x

    return traj

def ensemble_stats(method: str, N: int, T: float, dt: float, beta: float, alpha: float, x0: float, rng_seed: int = None) -> Dict[str, np.ndarray]:
    """
    Run an ensemble of N independent trajectories and compute:
    - transitions per trajectory
    - MFPT per trajectory (from left to right well)
    - pooled positions after burn-in for histogram
    """
    base_seed = rng_seed if rng_seed is not None else 123  # fallback seed

    print(f"Using base RNG seed = {base_seed}")
    rng = np.random.default_rng(base_seed)
    steps = int(T / dt)
    burn = steps // 2  # burn-in: first half discarded for hist
    trans_counts = []
    mfpts = []
    pooled_positions = []

    for i in range(N):
        # each trajectory uses an independent generator
        # sub_rng = np.random.default_rng(rng.integers(0, 2**63-1))
        sub_rng = np.random.default_rng(base_seed + i)  # deterministic / reproducible

        traj = simulate_trajectory(method, T, dt, beta, alpha, x0, sub_rng)
        trans_counts.append(count_transitions(traj))
        mfpt = first_passage_time(traj, dt, x_hit=0.2)
        if not np.isnan(mfpt):
            mfpts.append(mfpt)
        pooled_positions.append(traj[burn:])

    return {
        "transitions": np.array(trans_counts, dtype=float),
        "mfpt": np.array(mfpts, dtype=float),
        "positions": np.concatenate(pooled_positions)
    }

# Plotting and such
def plot_trajectories(traj_L: np.ndarray, traj_N: np.ndarray, dt: float, outpath: str):
    t = np.arange(len(traj_L)) * dt
    plt.figure(figsize=(8,3))
    plt.plot(t, traj_L, label="Langevin")
    plt.plot(t, traj_N, label="Toy NETS", alpha=0.8)
    plt.axhline(0.0, ls="--", lw=1, color="k")
    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title("Sample trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_histograms_with_boltzmann(positions_L: np.ndarray, positions_N: np.ndarray, beta: float, outpath: str, bins: int = 120, xlim: Tuple[float,float]=(-2.5,2.5)):
    xs = np.linspace(xlim[0], xlim[1], 1001)
    boltz = theoretical_boltzmann(xs, beta)

    plt.figure(figsize=(6,4))
    # histograms
    hL, edges = np.histogram(positions_L, bins=bins, range=xlim, density=True)
    centers = 0.5*(edges[1:]+edges[:-1])
    hN, _ = np.histogram(positions_N, bins=edges, density=True)

    plt.plot(xs, boltz, 'k--', lw=2, label="Boltzmann (theory)")
    plt.plot(centers, hL, label="Langevin")
    plt.plot(centers, hN, label="Toy NETS")
    plt.xlabel("x"); plt.ylabel("density")
    plt.title("Stationary histograms vs theory (after burn-in)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_alpha_sweep(alphas: List[float], trans_means: List[float], trans_se: List[float],
                     distort_L1: List[float], distort_KL: List[float], outpath: str):
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.errorbar(alphas, trans_means, yerr=trans_se, marker='o', label="Mean transitions ± SE")
    ax1.set_xlabel("α")
    ax1.set_ylabel("Transitions per trajectory")
    ax2 = ax1.twinx()
    ax2.plot(alphas, distort_L1, marker='s', linestyle='--', label="L1 distortion")
    ax2.plot(alphas, distort_KL, marker='^', linestyle=':', label="KL distortion")
    ax2.set_ylabel("Histogram distortion")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="best")
    plt.title("Acceleration vs. distortion trade-off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_mfpt_distribution(mfpt_L: np.ndarray, mfpt_N: np.ndarray, outpath: str, bins: int = 60):
    plt.figure(figsize=(6,4))
    plt.hist(mfpt_L, bins=bins, density=True, alpha=0.6, label="Langevin MFPT")
    plt.hist(mfpt_N, bins=bins, density=True, alpha=0.6, label="Toy NETS MFPT")
    plt.xlabel("First passage time")
    plt.ylabel("density")
    plt.title("MFPT distributions (left→right, hit x≥0.2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# Main
def main():
    ap = argparse.ArgumentParser(description="Toy NETS vs Langevin in 1D double well (drift-only augmentation)")
    ap.add_argument("--beta", type=float, default=3.0, help="inverse temperature β")
    ap.add_argument("--dt", type=float, default=0.01, help="time step")
    ap.add_argument("--T", type=float, default=200.0, help="total simulation time per trajectory")
    ap.add_argument("--N", type=int, default=100, help="number of trajectories in ensemble")
    ap.add_argument("--alpha", type=float, default=0.4, help="drift strength for NETS")
    ap.add_argument("--seed", type=int, default=None, help="global RNG seed (optional)")
    ap.add_argument("--outdir", type=str, default="figs", help="output directory for figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Sample trajectories (same seed for fair visual comparison)
    rng_vis = np.random.default_rng(args.seed if args.seed is not None else 123)
    traj_L = simulate_trajectory("langevin", T=args.T, dt=args.dt, beta=args.beta, alpha=0.0, x0=-1.0, rng=rng_vis)
    traj_N = simulate_trajectory("nets",     T=args.T, dt=args.dt, beta=args.beta, alpha=args.alpha, x0=-1.0, rng=rng_vis)
    plot_trajectories(traj_L, traj_N, args.dt, outpath=os.path.join(args.outdir, "trajectories.png"))

    # Ensemble statistics
    stats_L = ensemble_stats("langevin", N=args.N, T=args.T, dt=args.dt, beta=args.beta, alpha=0.0, x0=-1.0, rng_seed=args.seed)
    stats_N = ensemble_stats("nets",     N=args.N, T=args.T, dt=args.dt, beta=args.beta, alpha=args.alpha, x0=-1.0, rng_seed=args.seed)

    # Histograms with Boltzmann
    plot_histograms_with_boltzmann(stats_L["positions"], stats_N["positions"], args.beta, outpath=os.path.join(args.outdir, "hist_vs_boltz.png"))

    # Report transitions and MFPT (mean ± SE)
    mean_trans_L, se_trans_L = np.mean(stats_L["transitions"]), sem(stats_L["transitions"])
    mean_trans_N, se_trans_N = np.mean(stats_N["transitions"]), sem(stats_N["transitions"])
    mean_mfpt_L, se_mfpt_L = np.mean(stats_L["mfpt"]), sem(stats_L["mfpt"]) if len(stats_L["mfpt"])>1 else (np.nan, np.nan)
    mean_mfpt_N, se_mfpt_N = np.mean(stats_N["mfpt"]), sem(stats_N["mfpt"]) if len(stats_N["mfpt"])>1 else (np.nan, np.nan)

    print(f"Transition counts per trajectory:")
    print(f"  Langevin: mean={mean_trans_L:.2f} ± {se_trans_L:.2f} (SE)")
    print(f"  Toy NETS (α={args.alpha}): mean={mean_trans_N:.2f} ± {se_trans_N:.2f} (SE)")
    print(f"MFPT (left→right, hit x≥0.2):")
    print(f"  Langevin: mean={mean_mfpt_L:.2f} ± {se_mfpt_L:.2f} (SE), n={len(stats_L['mfpt'])}")
    print(f"  Toy NETS (α={args.alpha}): mean={mean_mfpt_N:.2f} ± {se_mfpt_N:.2f} (SE), n={len(stats_N['mfpt'])}")

    # alpha sweep: transitions & histogram distortion
    alphas = np.linspace(0.0, 1.0, 6)
    trans_means, trans_se = [], []
    distort_L1, distort_KL = [], []

    # Prepare a fixed binning for histogram-based distortions
    xlim = (-2.5, 2.5)
    bins = 160
    edges = np.linspace(*xlim, bins+1)
    centers = 0.5*(edges[1:]+edges[:-1])
    dx = centers[1]-centers[0]
    boltz = theoretical_boltzmann(centers, args.beta)

    for a in alphas:
        sN = ensemble_stats("nets", N=args.N, T=args.T, dt=args.dt, beta=args.beta, alpha=a, x0=-1.0, rng_seed=(args.seed+int(a*1000) if args.seed is not None else None))
        trans_means.append(np.mean(sN["transitions"]))
        trans_se.append(sem(sN["transitions"]) if len(sN["transitions"])>1 else np.nan)
        hN, _ = np.histogram(sN["positions"], bins=edges, density=True)
        # distortion vs Boltzmann
        distort_L1.append(l1_distance(hN, boltz, dx))
        distort_KL.append(kl_divergence(hN, boltz))

    plot_alpha_sweep(list(alphas), trans_means, trans_se, distort_L1, distort_KL, outpath=os.path.join(args.outdir, "alpha_sweep.png"))

    # MFPT distributions
    plot_mfpt_distribution(stats_L["mfpt"], stats_N["mfpt"], outpath=os.path.join(args.outdir, "mfpt_dist.png"))

if __name__ == "__main__":
    main()
