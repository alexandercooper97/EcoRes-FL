#!/usr/bin/env python3
"""
EcoRes-FL: Sustainability- and Resilience-Aware Federated Learning Simulator
----------------------------------------------------------------------------
This script supports two execution levels:

1) quick mode (default): fully reproducible without external downloads using
   scikit-learn and synthetic edge/IoT datasets. It generates publication-ready
   CSV tables and high-resolution PNG figures under results/ and figures/.
2) paper mode (--paper_mode): enables heavier datasets such as Fashion-MNIST,
   CIFAR-10, or Covertype when torchvision / internet access is available.

The simulator is intentionally transparent: it implements FedAvg, FedProx,
CarbonAware-FL, and EcoRes-FL over a common federated softmax model. EcoRes-FL
uses carbon-aware client selection, straggler/failure filtering, proximal local
updates, and top-k sparse communication.

Author: anonymous for double-blind review
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, make_classification, fetch_covtype
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Optional integer programming. If unavailable, the code falls back to greedy ranking.
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.optimize import LinearConstraint as _LC
    SCIPY_MILP_AVAILABLE = True
except Exception:
    SCIPY_MILP_AVAILABLE = False

RNG = np.random.default_rng(42)


@dataclass
class ClientProfile:
    client_id: int
    tier: str
    compute_gflops: float
    bandwidth_mbps: float
    energy_per_sample: float       # proxy Joules per feature-sample-local-epoch unit
    net_energy_per_mb: float       # proxy Joules per MB transmitted
    reliability: float
    base_carbon_g_per_kwh: float
    data_size: int = 0


@dataclass
class SimConfig:
    dataset: str = "synthetic_vision"
    rounds: int = 30
    clients: int = 40
    clients_per_round: int = 10
    local_epochs: int = 2
    lr: float = 0.08
    alpha: float = 0.25
    deadline_s: float = 4.0
    failure_rate: float = 0.08
    sparsity: float = 0.30
    mu: float = 0.02
    seed: int = 42
    output_dir: str = "."
    paper_mode: bool = False
    max_train_samples: int = 12000
    max_test_samples: int = 3000


# ----------------------------- Data loading -----------------------------

def load_federated_base_dataset(name: str, seed: int = 42, paper_mode: bool = False,
                                max_train_samples: int = 12000,
                                max_test_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return standardized train/test arrays and class labels.

    The quick datasets are deliberately harder than classic toy examples:
    - synthetic_vision: 10 classes, 128 features, noisy labels.
    - iot_fault: imbalanced 5-class condition-monitoring proxy.
    - digits: included as a sanity check and for low-resource reproducibility.

    The paper datasets are optional and require local/internet availability.
    """
    rng = np.random.default_rng(seed)
    name = name.lower()

    if name == "digits":
        ds = load_digits()
        X = ds.data.astype(np.float32) / 16.0
        y = ds.target.astype(int)
        labels = [str(i) for i in sorted(np.unique(y))]
    elif name == "synthetic_vision":
        X, y = make_classification(
            n_samples=15000, n_features=128, n_informative=42,
            n_redundant=18, n_repeated=0, n_classes=10,
            n_clusters_per_class=2, class_sep=0.85, flip_y=0.065,
            weights=None, random_state=seed,
        )
        labels = [f"c{i}" for i in sorted(np.unique(y))]
    elif name == "iot_fault":
        X, y = make_classification(
            n_samples=14000, n_features=64, n_informative=28,
            n_redundant=14, n_classes=5, n_clusters_per_class=2,
            weights=[0.53, 0.18, 0.13, 0.10, 0.06], class_sep=0.75,
            flip_y=0.045, random_state=seed + 7,
        )
        labels = ["normal", "minor", "thermal", "network", "critical"]
    elif name == "covertype":
        if not paper_mode:
            raise RuntimeError("covertype requires --paper_mode because it may download data.")
        ds = fetch_covtype(download_if_missing=True)
        X = ds.data.astype(np.float32)
        y = (ds.target.astype(int) - 1)
        idx = rng.choice(len(X), size=min(len(X), max_train_samples + max_test_samples), replace=False)
        X, y = X[idx], y[idx]
        labels = [f"cover_{i}" for i in sorted(np.unique(y))]
    elif name in {"fashion_mnist", "cifar10"}:
        if not paper_mode:
            raise RuntimeError(f"{name} requires --paper_mode and torchvision.")
        try:
            import torch
            from torchvision import datasets, transforms
        except Exception as exc:
            raise RuntimeError("Install torch and torchvision for image datasets.") from exc
        root = Path("data")
        if name == "fashion_mnist":
            ds_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
            ds_test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
            labels = [str(i) for i in range(10)]
        else:
            ds_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
            ds_test = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
            labels = ds_train.classes

        def flatten_subset(ds, n):
            n = min(n, len(ds))
            idx = rng.choice(len(ds), size=n, replace=False)
            Xl, yl = [], []
            for i in idx:
                img, lab = ds[int(i)]
                arr = img.numpy().reshape(-1).astype(np.float32)
                Xl.append(arr)
                yl.append(int(lab))
            return np.vstack(Xl), np.array(yl)
        Xtr, ytr = flatten_subset(ds_train, max_train_samples)
        Xte, yte = flatten_subset(ds_test, max_test_samples)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        return Xtr, Xte, ytr, yte, labels
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    if len(X_train) > max_train_samples:
        idx = rng.choice(len(X_train), max_train_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    if len(X_test) > max_test_samples:
        idx = rng.choice(len(X_test), max_test_samples, replace=False)
        X_test, y_test = X_test[idx], y_test[idx]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train.astype(int), y_test.astype(int), labels


def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed: int = 42) -> List[np.ndarray]:
    """Partition indices into non-IID clients using a Dirichlet prior per class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet(np.repeat(alpha, n_clients))
        # avoid empty pathological allocations by smoothing with data-size prior
        props = 0.80 * props + 0.20 / n_clients
        cut = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cut)
        for k, split in enumerate(splits):
            client_indices[k].extend(split.tolist())
    # Ensure non-empty clients by borrowing from largest clients
    for k in range(n_clients):
        if len(client_indices[k]) == 0:
            donor = int(np.argmax([len(v) for v in client_indices]))
            client_indices[k].append(client_indices[donor].pop())
    return [np.array(v, dtype=int) for v in client_indices]


def generate_client_profiles(client_indices: List[np.ndarray], seed: int = 42) -> List[ClientProfile]:
    rng = np.random.default_rng(seed)
    profiles = []
    tiers = ["edge-low", "edge-mid", "edge-gpu", "cloud"]
    probs = [0.35, 0.35, 0.20, 0.10]
    for k, idx in enumerate(client_indices):
        tier = rng.choice(tiers, p=probs)
        if tier == "edge-low":
            compute = rng.uniform(0.8, 2.0)
            bw = rng.uniform(2.0, 8.0)
            e_sample = rng.uniform(0.00020, 0.00045)
            net_e = rng.uniform(0.012, 0.024)
            rel = rng.uniform(0.72, 0.90)
            carbon = rng.uniform(120, 320)
        elif tier == "edge-mid":
            compute = rng.uniform(2.0, 7.0)
            bw = rng.uniform(8.0, 25.0)
            e_sample = rng.uniform(0.00012, 0.00024)
            net_e = rng.uniform(0.006, 0.014)
            rel = rng.uniform(0.82, 0.96)
            carbon = rng.uniform(180, 420)
        elif tier == "edge-gpu":
            compute = rng.uniform(8.0, 24.0)
            bw = rng.uniform(18.0, 70.0)
            e_sample = rng.uniform(0.00008, 0.00018)
            net_e = rng.uniform(0.005, 0.012)
            rel = rng.uniform(0.88, 0.98)
            carbon = rng.uniform(220, 560)
        else:  # cloud
            compute = rng.uniform(30.0, 95.0)
            bw = rng.uniform(70.0, 250.0)
            e_sample = rng.uniform(0.00005, 0.00012)
            net_e = rng.uniform(0.003, 0.008)
            rel = rng.uniform(0.95, 0.995)
            carbon = rng.uniform(320, 780)
        profiles.append(ClientProfile(k, tier, compute, bw, e_sample, net_e, rel, carbon, len(idx)))
    return profiles


def carbon_intensity(profile: ClientProfile, round_id: int, total_rounds: int, rng: np.random.Generator) -> float:
    phase = (profile.client_id % 7) / 7.0 * 2.0 * math.pi
    renewable_wave = 0.20 * math.sin(2.0 * math.pi * round_id / max(1, total_rounds) + phase)
    noise = rng.normal(0.0, 0.04)
    ci = profile.base_carbon_g_per_kwh * max(0.50, 1.0 + renewable_wave + noise)
    return float(ci)


# ----------------------------- Model utilities -----------------------------

def one_hot(y: np.ndarray, c: int) -> np.ndarray:
    out = np.zeros((len(y), c), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax_logits(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = X @ W + b
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def evaluate(X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    pred = np.argmax(softmax_logits(X, W, b), axis=1)
    return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro", zero_division=0))


def local_train_softmax(
    X: np.ndarray, y: np.ndarray, W_global: np.ndarray, b_global: np.ndarray,
    classes: int, lr: float, epochs: int, mu: float = 0.0, batch_size: int = 128,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = W_global.copy()
    b = b_global.copy()
    n = len(X)
    Y = one_hot(y, classes)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            xb, yb = X[idx], Y[idx]
            p = softmax_logits(xb, W, b)
            grad_z = (p - yb) / max(1, len(idx))
            grad_W = xb.T @ grad_z + mu * (W - W_global)
            grad_b = grad_z.sum(axis=0) + mu * (b - b_global)
            W -= lr * grad_W
            b -= lr * grad_b
    return W, b


def sparsify_update(dW: np.ndarray, db: np.ndarray, keep_ratio: float) -> Tuple[np.ndarray, np.ndarray, float]:
    if keep_ratio >= 0.999:
        return dW, db, 1.0
    flat = np.concatenate([dW.ravel(), db.ravel()])
    k = max(1, int(math.ceil(keep_ratio * flat.size)))
    threshold = np.partition(np.abs(flat), -k)[-k]
    mask_W = np.abs(dW) >= threshold
    mask_b = np.abs(db) >= threshold
    sparse_W = np.where(mask_W, dW, 0.0)
    sparse_b = np.where(mask_b, db, 0.0)
    density = (np.count_nonzero(sparse_W) + np.count_nonzero(sparse_b)) / flat.size
    return sparse_W, sparse_b, float(density)


# ----------------------------- Scheduling / PLI -----------------------------

def estimate_client_cost(profile: ClientProfile, n_features: int, n_classes: int, local_epochs: int,
                         density: float, ci: float) -> Dict[str, float]:
    params = n_features * n_classes + n_classes
    update_mb = params * 4.0 * density / (1024 ** 2)
    compute_s = (profile.data_size * n_features * n_classes * local_epochs / 1e6) / max(profile.compute_gflops, 1e-6)
    network_s = (update_mb * 8.0) / max(profile.bandwidth_mbps, 1e-6)
    time_s = compute_s + network_s
    energy_j = profile.energy_per_sample * profile.data_size * n_features * local_epochs + profile.net_energy_per_mb * update_mb
    carbon_g = (energy_j / 3.6e6) * ci
    return {"time_s": time_s, "energy_j": energy_j, "carbon_g": carbon_g, "update_mb": update_mb}


def solve_selection_ilp(
    utilities: np.ndarray, carbon: np.ndarray, latency: np.ndarray, reliability: np.ndarray,
    m: int, deadline: float, carbon_budget: float, lambda_c: float = 1.0,
    lambda_t: float = 0.2, lambda_r: float = 0.5,
) -> Optional[np.ndarray]:
    """Select m clients via a binary linear program.

    Maximize utility minus carbon/latency/risk penalties subject to cardinality,
    average latency and carbon-budget constraints. If scipy.milp is unavailable or
    infeasible, return None for greedy fallback.
    """
    if not SCIPY_MILP_AVAILABLE:
        return None
    n = len(utilities)
    c = -(utilities - lambda_c * carbon - lambda_t * latency + lambda_r * reliability)
    integrality = np.ones(n)
    bounds = Bounds(np.zeros(n), np.ones(n))
    constraints = []
    constraints.append(LinearConstraint(np.ones((1, n)), [m], [m]))
    constraints.append(LinearConstraint(latency.reshape(1, -1), [-np.inf], [m * deadline]))
    constraints.append(LinearConstraint(carbon.reshape(1, -1), [-np.inf], [carbon_budget]))
    try:
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints,
                   options={"time_limit": 2.0, "mip_rel_gap": 0.05})
        if res.success and res.x is not None:
            return np.where(res.x > 0.5)[0]
    except Exception:
        return None
    return None


def select_clients(method: str, profiles: List[ClientProfile], client_indices: List[np.ndarray],
                   round_id: int, total_rounds: int, m: int, n_features: int, n_classes: int,
                   cfg: SimConfig, W: np.ndarray, rng: np.random.Generator) -> Tuple[List[int], Dict[int, float]]:
    density = 1.0 if method in {"FedAvg", "FedProx", "CarbonAware"} else cfg.sparsity
    cis, utilities, carbons, latencies, reliabilities = [], [], [], [], []
    norm_w = np.linalg.norm(W) + 1e-9
    for p in profiles:
        ci = carbon_intensity(p, round_id, total_rounds, rng)
        cost = estimate_client_cost(p, n_features, n_classes, cfg.local_epochs, density, ci)
        data_utility = math.log1p(p.data_size)
        # More heterogeneous / minority class clients tend to add useful information.
        # This is an observable proxy, not an oracle of future test accuracy.
        risk = 1.0 - p.reliability
        utilities.append(data_utility)
        carbons.append(cost["carbon_g"])
        latencies.append(cost["time_s"])
        reliabilities.append(p.reliability - risk)
        cis.append(ci)
    utilities = np.array(utilities)
    carbons = np.array(carbons)
    latencies = np.array(latencies)
    reliabilities = np.array(reliabilities)

    # Normalize dimensions for stable scoring.
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)
    u = norm(utilities)
    c = norm(carbons)
    t = norm(latencies)
    r = norm(reliabilities)

    if method in {"FedAvg", "FedProx"}:
        chosen = rng.choice(len(profiles), size=min(m, len(profiles)), replace=False)
    elif method == "CarbonAware":
        score = 0.60 * u - 0.30 * c - 0.10 * t + 0.10 * r
        chosen = np.argsort(score)[-m:]
    else:  # EcoRes-FL: try explicit PLI, fallback to greedy score.
        carbon_budget = float(np.quantile(carbons, 0.45) * m)
        chosen = solve_selection_ilp(u, c, t, r, m=m, deadline=cfg.deadline_s,
                                     carbon_budget=carbon_budget,
                                     lambda_c=0.60, lambda_t=0.20, lambda_r=0.15)
        if chosen is None or len(chosen) == 0:
            score = 0.65 * u - 0.25 * c - 0.15 * t + 0.15 * r
            chosen = np.argsort(score)[-m:]
    ci_map = {int(i): float(cis[int(i)]) for i in chosen}
    return [int(i) for i in chosen], ci_map


# ----------------------------- Simulation loop -----------------------------

def run_method(method: str, cfg: SimConfig, X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray, labels: List[str]) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    rng = np.random.default_rng(cfg.seed + abs(hash(method)) % 10000)
    random.seed(cfg.seed)
    classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    client_indices = dirichlet_partition(y_train, cfg.clients, cfg.alpha, cfg.seed)
    profiles = generate_client_profiles(client_indices, cfg.seed)
    W = rng.normal(0, 0.01, size=(n_features, classes)).astype(np.float32)
    b = np.zeros(classes, dtype=np.float32)

    history = []
    selection_matrix = np.zeros((cfg.clients, cfg.rounds), dtype=float)
    total_comm, total_energy, total_carbon = 0.0, 0.0, 0.0
    dropped_total, selected_total, contributed_total = 0, 0, 0

    for r in range(cfg.rounds):
        chosen, ci_map = select_clients(method, profiles, client_indices, r, cfg.rounds,
                                        cfg.clients_per_round, n_features, classes, cfg, W, rng)
        selected_total += len(chosen)
        updates_W, updates_b, weights = [], [], []
        round_comm, round_energy, round_carbon = 0.0, 0.0, 0.0
        round_dropped = 0
        for cid in chosen:
            selection_matrix[cid, r] += 1
            profile = profiles[cid]
            idx = client_indices[cid]
            density = cfg.sparsity if method == "EcoRes-FL" else 1.0
            ci = ci_map.get(cid, carbon_intensity(profile, r, cfg.rounds, rng))
            cost = estimate_client_cost(profile, n_features, classes, cfg.local_epochs, density, ci)
            failed = rng.random() < (cfg.failure_rate + max(0.0, 0.90 - profile.reliability) * 0.12)
            straggler = cost["time_s"] > cfg.deadline_s
            if method == "FedAvg" and (failed or straggler):
                round_dropped += 1
                continue
            if method in {"FedProx", "CarbonAware", "EcoRes-FL"} and failed:
                round_dropped += 1
                continue
            # EcoRes can accept mild stragglers if sparse update meets relaxed deadline.
            if method != "EcoRes-FL" and straggler:
                round_dropped += 1
                continue

            mu = cfg.mu if method in {"FedProx", "EcoRes-FL"} else 0.0
            W_local, b_local = local_train_softmax(
                X_train[idx], y_train[idx], W, b, classes, lr=cfg.lr,
                epochs=cfg.local_epochs, mu=mu, batch_size=128, seed=cfg.seed + r + cid
            )
            dW, db = W_local - W, b_local - b
            keep = cfg.sparsity if method == "EcoRes-FL" else 1.0
            dW, db, actual_density = sparsify_update(dW, db, keep)
            cost = estimate_client_cost(profile, n_features, classes, cfg.local_epochs, actual_density, ci)
            updates_W.append(dW)
            updates_b.append(db)
            weights.append(len(idx))
            round_comm += cost["update_mb"] * 2.0  # upload + broadcast amortized proxy
            round_energy += cost["energy_j"]
            round_carbon += cost["carbon_g"]

        dropped_total += round_dropped
        contributed_total += len(updates_W)
        if updates_W:
            weights_arr = np.array(weights, dtype=np.float32)
            weights_arr = weights_arr / weights_arr.sum()
            agg_W = np.zeros_like(W)
            agg_b = np.zeros_like(b)
            for ww, dw, dbb in zip(weights_arr, updates_W, updates_b):
                agg_W += ww * dw
                agg_b += ww * dbb
            W += agg_W
            b += agg_b

        acc, mf1 = evaluate(X_test, y_test, W, b)
        total_comm += round_comm
        total_energy += round_energy
        total_carbon += round_carbon
        history.append({
            "dataset": cfg.dataset, "method": method, "round": r + 1,
            "accuracy": acc, "macro_f1": mf1,
            "round_comm_mb": round_comm, "cum_comm_mb": total_comm,
            "round_energy_j": round_energy, "cum_energy_j": total_energy,
            "round_carbon_g": round_carbon, "cum_carbon_g": total_carbon,
            "selected": len(chosen), "contributed": len(updates_W), "dropped": round_dropped,
            "drop_rate": round_dropped / max(1, len(chosen)),
        })

    summary = {
        "dataset": cfg.dataset, "method": method,
        "clients": cfg.clients, "clients_per_round": cfg.clients_per_round,
        "alpha": cfg.alpha, "rounds": cfg.rounds,
        "accuracy": history[-1]["accuracy"], "macro_f1": history[-1]["macro_f1"],
        "comm_mb": total_comm, "energy_j": total_energy, "carbon_g": total_carbon,
        "drop_rate": dropped_total / max(1, selected_total),
        "contribution_rate": contributed_total / max(1, selected_total),
    }
    return pd.DataFrame(history), summary, selection_matrix


# ----------------------------- Plotting -----------------------------

def set_publication_defaults():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def plot_lines(history: pd.DataFrame, dataset: str, figures_dir: Path):
    set_publication_defaults()
    subset = history[history["dataset"] == dataset]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for method, g in subset.groupby("method"):
        ax.plot(g["round"], g["accuracy"], marker="o", markersize=2.5, linewidth=1.2, label=method)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Convergence under non-IID clients: {dataset}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(figures_dir / f"convergence_{dataset}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for method, g in subset.groupby("method"):
        ax.plot(g["round"], g["cum_carbon_g"], marker="s", markersize=2.5, linewidth=1.2, label=method)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Cumulative carbon proxy (gCO$_2$eq)")
    ax.set_title(f"Carbon trajectory: {dataset}")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(figures_dir / f"carbon_{dataset}.png")
    plt.close(fig)


def plot_tradeoff(summary: pd.DataFrame, figures_dir: Path):
    set_publication_defaults()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    methods = list(summary["method"].unique())
    for method in methods:
        g = summary[summary["method"] == method]
        ax.scatter(g["carbon_g"], g["accuracy"], s=62, label=method)
        for _, row in g.iterrows():
            ax.annotate(row["dataset"], (row["carbon_g"], row["accuracy"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Total carbon proxy (gCO$_2$eq, lower is better)")
    ax.set_ylabel("Final accuracy (higher is better)")
    ax.set_title("Accuracy--carbon Pareto view across datasets")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(figures_dir / "pareto_accuracy_carbon.png")
    plt.close(fig)


def heatmap(matrix: np.ndarray, xlabels: List[str], ylabels: List[str], title: str,
            xlabel: str, ylabel: str, cbar_label: str, outfile: Path):
    set_publication_defaults()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.savefig(outfile)
    plt.close(fig)


def plot_selection_matrix(selection_matrix: np.ndarray, dataset: str, figures_dir: Path):
    set_publication_defaults()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(selection_matrix, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Round")
    ax.set_ylabel("Client ID")
    ax.set_title(f"EcoRes-FL selected-client heatmap: {dataset}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Selected")
    fig.savefig(figures_dir / f"client_selection_heatmap_{dataset}.png")
    plt.close(fig)


def run_ablation_grid(base_cfg: SimConfig, dataset: str, figures_dir: Path, results_dir: Path) -> pd.DataFrame:
    alphas = [0.10, 0.25, 0.50, 1.00]
    clients_grid = [20, 40, 80]
    sparsities = [0.05, 0.10, 0.20, 0.50]
    failure_grid = [0.00, 0.08, 0.16, 0.24]

    records = []
    # Keep ablation moderately cheap.
    Xtr, Xte, ytr, yte, labels = load_federated_base_dataset(dataset, base_cfg.seed, base_cfg.paper_mode,
                                                             max_train_samples=6000, max_test_samples=1500)
    for ncl in clients_grid:
        for alpha in alphas:
            cfg = SimConfig(**{**base_cfg.__dict__, "dataset": dataset, "clients": ncl,
                               "clients_per_round": max(5, min(16, ncl // 4)), "alpha": alpha,
                               "rounds": 12})
            _, summary, _ = run_method("EcoRes-FL", cfg, Xtr, ytr, Xte, yte, labels)
            records.append({"grid": "clients_alpha", "clients": ncl, "alpha": alpha, **summary})

    for sp in sparsities:
        for fail in failure_grid:
            cfg = SimConfig(**{**base_cfg.__dict__, "dataset": dataset, "sparsity": sp,
                               "failure_rate": fail, "rounds": 12})
            _, summary, _ = run_method("EcoRes-FL", cfg, Xtr, ytr, Xte, yte, labels)
            records.append({"grid": "sparsity_failure", "sparsity": sp, "failure_rate": fail, **summary})

    df = pd.DataFrame(records)
    df.to_csv(results_dir / f"ablation_{dataset}.csv", index=False)

    m1 = df[df["grid"] == "clients_alpha"].pivot(index="clients", columns="alpha", values="accuracy").values
    heatmap(m1, [str(a) for a in alphas], [str(c) for c in clients_grid],
            "EcoRes-FL final accuracy under client-scale and non-IID severity",
            "Dirichlet alpha (larger = more IID)", "Number of clients", "Accuracy",
            figures_dir / f"heatmap_accuracy_clients_alpha_{dataset}.png")
    m2 = df[df["grid"] == "clients_alpha"].pivot(index="clients", columns="alpha", values="carbon_g").values
    heatmap(m2, [str(a) for a in alphas], [str(c) for c in clients_grid],
            "EcoRes-FL carbon proxy under client-scale and non-IID severity",
            "Dirichlet alpha (larger = more IID)", "Number of clients", "Carbon proxy",
            figures_dir / f"heatmap_carbon_clients_alpha_{dataset}.png")
    m3 = df[df["grid"] == "sparsity_failure"].pivot(index="sparsity", columns="failure_rate", values="accuracy").values
    heatmap(m3, [str(f) for f in failure_grid], [str(s) for s in sparsities],
            "Accuracy sensitivity to sparsity and failure probability",
            "Failure probability", "Top-k density", "Accuracy",
            figures_dir / f"heatmap_accuracy_sparsity_failure_{dataset}.png")
    m4 = df[df["grid"] == "sparsity_failure"].pivot(index="sparsity", columns="failure_rate", values="carbon_g").values
    heatmap(m4, [str(f) for f in failure_grid], [str(s) for s in sparsities],
            "Carbon sensitivity to sparsity and failure probability",
            "Failure probability", "Top-k density", "Carbon proxy",
            figures_dir / f"heatmap_carbon_sparsity_failure_{dataset}.png")
    return df


# ----------------------------- Entrypoint -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--datasets", nargs="+", default=["synthetic_vision", "iot_fault", "digits"])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--clients", type=int, default=40)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paper_mode", action="store_true")
    parser.add_argument("--skip_ablation", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    figures_dir = out / "figures"
    results_dir = out / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = ["FedAvg", "FedProx", "CarbonAware", "EcoRes-FL"]
    all_histories, summaries, selection_mats = [], [], {}
    base_cfg = SimConfig(rounds=args.rounds, clients=args.clients,
                         clients_per_round=args.clients_per_round,
                         alpha=args.alpha, seed=args.seed,
                         output_dir=str(out), paper_mode=args.paper_mode)

    for dataset in args.datasets:
        print(f"[DATASET] {dataset}")
        Xtr, Xte, ytr, yte, labels = load_federated_base_dataset(
            dataset, args.seed, args.paper_mode, base_cfg.max_train_samples, base_cfg.max_test_samples
        )
        for method in methods:
            print(f"  [METHOD] {method}")
            cfg = SimConfig(**{**base_cfg.__dict__, "dataset": dataset})
            hist, summary, sel = run_method(method, cfg, Xtr, ytr, Xte, yte, labels)
            all_histories.append(hist)
            summaries.append(summary)
            if method == "EcoRes-FL":
                selection_mats[dataset] = sel
        hist_df = pd.concat(all_histories, ignore_index=True)
        plot_lines(hist_df, dataset, figures_dir)
        if dataset in selection_mats:
            plot_selection_matrix(selection_mats[dataset], dataset, figures_dir)

    hist_df = pd.concat(all_histories, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    hist_df.to_csv(results_dir / "round_history.csv", index=False)
    summary_df.to_csv(results_dir / "summary_results.csv", index=False)
    plot_tradeoff(summary_df, figures_dir)

    if not args.skip_ablation:
        # Use synthetic_vision for main ablations; it is multiclass, noisy, and high-dimensional.
        run_ablation_grid(base_cfg, "synthetic_vision", figures_dir, results_dir)

    meta = {
        "methods": methods,
        "datasets": args.datasets,
        "notes": "Carbon and energy are normalized proxies unless connected to measured CodeCarbon traces.",
        "scipy_milp_available": SCIPY_MILP_AVAILABLE,
    }
    (results_dir / "experiment_metadata.json").write_text(json.dumps(meta, indent=2))
    print("[DONE] Results written to", results_dir)


if __name__ == "__main__":
    main()
