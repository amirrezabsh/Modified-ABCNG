import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.abcng import ABCNG, BENCHES
from models.abc import ABC


def run_single(name: str, dim: int, budget: int, seed: int, pop_size: int = 40, algo: str = "abcng") -> Tuple[List[float], List[float], float, float]:
    f, (lo, hi) = BENCHES[name]
    if algo.lower() == "abc":
        opt = ABC(func=f, dim=dim, bounds=(lo, hi), pop_size=pop_size, max_evals=budget, seed=seed)
    else:
        opt = ABCNG(func=f, dim=dim, bounds=(lo, hi), pop_size=pop_size, max_evals=budget, seed=seed)
    t0 = time.time()
    _, gval, hist = opt.run()
    elapsed = time.time() - t0
    k_hist = list(getattr(opt, "k_hist", []))
    return hist, k_hist, float(gval), float(elapsed)


def pad_histories(hists: List[List[float]], mode: str = "truncate") -> np.ndarray:
    if not hists:
        return np.empty((0,))
    lengths = [len(h) for h in hists]
    if mode == "truncate":
        L = min(lengths)
        return np.array([h[:L] for h in hists], dtype=float)
    # pad with last value
    L = max(lengths)
    arr = np.zeros((len(hists), L), dtype=float)
    for i, h in enumerate(hists):
        arr[i, : len(h)] = h
        if len(h) < L:
            arr[i, len(h) :] = h[-1]
    return arr


def summarize_runs(name: str, dim: int, budget: int, runs: int, seeds: List[int], algo: str = "abcng") -> Dict[str, object]:
    hists: List[List[float]] = []
    k_hists: List[List[float]] = []
    finals: List[float] = []
    times: List[float] = []
    for s in seeds[:runs]:
        hist, k_hist, gval, elapsed = run_single(name, dim, budget, seed=s, algo=algo)
        hists.append(hist)
        k_hists.append(k_hist)
        finals.append(gval)
        times.append(elapsed)
    H = pad_histories(hists, mode="truncate")
    K = pad_histories(k_hists, mode="truncate") if k_hists else np.empty((0,))
    median = np.median(H, axis=0)
    q1 = np.percentile(H, 25, axis=0)
    q3 = np.percentile(H, 75, axis=0)
    if K.size:
        k_median = np.median(K, axis=0)
        k_q1 = np.percentile(K, 25, axis=0)
        k_q3 = np.percentile(K, 75, axis=0)
    else:
        k_median = np.array([])
        k_q1 = np.array([])
        k_q3 = np.array([])
    return {
        "name": name,
        "histories": H,
        "k_histories": K,
        "median": median,
        "q1": q1,
        "q3": q3,
        "k_median": k_median,
        "k_q1": k_q1,
        "k_q3": k_q3,
        "finals": np.array(finals, dtype=float),
        "times": np.array(times, dtype=float),
        "algo": algo,
    }


def plot_convergence(summary: Dict[str, object], out_dir: str) -> str:
    name = summary["name"]  # type: ignore
    median = summary["median"]  # type: ignore
    q1 = summary["q1"]  # type: ignore
    q3 = summary["q3"]  # type: ignore
    x = np.arange(1, len(median) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(x, median, label="median best-f")
    plt.fill_between(x, q1, q3, alpha=0.25, label="IQR")
    plt.yscale("log")
    plt.xlabel("Iteration (employed+onlooker cycles)")
    plt.ylabel("Best objective (log scale)")
    plt.title(f"ABCNG convergence on {name}")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"conv_{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_k_dynamics(summary: Dict[str, object], out_dir: str) -> str:
    name = summary["name"]  # type: ignore
    k_median = summary["k_median"]  # type: ignore
    k_q1 = summary["k_q1"]  # type: ignore
    k_q3 = summary["k_q3"]  # type: ignore
    x = np.arange(1, len(k_median) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(x, k_median, label="median mean k")
    plt.fill_between(x, k_q1, k_q3, alpha=0.25, label="IQR")
    plt.xlabel("Iteration (employed+onlooker cycles)")
    plt.ylabel("Mean neighborhood radius k")
    plt.title(f"ABCNG k dynamics on {name}")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"kdyn_{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_convergence_compare(summary_a: Dict[str, object], summary_b: Dict[str, object], label_a: str, label_b: str, out_dir: str) -> str:
    name = summary_a["name"]  # type: ignore
    ya = np.asarray(summary_a["median"])  # type: ignore
    yb = np.asarray(summary_b["median"])  # type: ignore
    L = int(min(len(ya), len(yb)))
    plt.figure(figsize=(7, 4))
    if L > 0:
        x = np.arange(1, L + 1)
        plt.plot(x, ya[:L], label=f"{label_a} median")
        plt.plot(x, yb[:L], label=f"{label_b} median")
    else:
        plt.text(0.5, 0.5, "No convergence data", ha="center", va="center", transform=plt.gca().transAxes)
    plt.yscale("log")
    plt.xlabel("Iteration (employed+onlooker cycles)")
    plt.ylabel("Best objective (log scale)")
    plt.title(f"Convergence on {name}: {label_a} vs {label_b}")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"conv_compare_{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_box_compare(finals_map: Dict[str, Dict[str, np.ndarray]], out_dir: str, labels: Tuple[str, str]) -> str:
    names = []
    data = []
    for func, results in finals_map.items():
        for algo in labels:
            names.append(f"{func}:{algo}")
            data.append(results[algo])
    plt.figure(figsize=(max(8, len(names) * 0.9), 4))
    plt.boxplot(data, labels=names, showmeans=True)
    plt.yscale("log")
    plt.ylabel("Final best objective (log)")
    plt.title("Final performance: comparison per function")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "box_compare.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_box(finals_map: Dict[str, np.ndarray], out_dir: str) -> str:
    names = list(finals_map.keys())
    data = [finals_map[n] for n in names]
    plt.figure(figsize=(8, 4))
    plt.boxplot(data, labels=names, showmeans=True)
    plt.yscale("log")
    plt.ylabel("Final best objective (log)")
    plt.title("ABCNG final performance across functions")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "box_finals.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main():
    # Benchmark grid
    suite = [
        ("sphere", 30, 10000),
        ("rastrigin", 30, 10000),
        ("rosenbrock", 30, 10000),
        ("ackley", 30, 10000),
    ]
    runs = 20
    seeds = list(range(1000, 1000 + runs))
    out_dir = os.path.join("figs")
    summary_rows = []
    finals_map: Dict[str, Dict[str, np.ndarray]] = {}

    for name, dim, budget in suite:
        sum_ng = summarize_runs(name, dim, budget, runs=runs, seeds=seeds, algo="abcng")
        sum_abc = summarize_runs(name, dim, budget, runs=runs, seeds=seeds, algo="abc")
        conv_path = plot_convergence(sum_ng, out_dir)
        k_path = plot_k_dynamics(sum_ng, out_dir)
        conv_cmp = plot_convergence_compare(sum_ng, sum_abc, "ABCNG", "ABC", out_dir)
        finals_map[name] = {
            "ABCNG": sum_ng["finals"],  # type: ignore
            "ABC": sum_abc["finals"],   # type: ignore
        }
        summary_rows.append({
            "function": name,
            "dim": dim,
            "budget": budget,
            "runs": runs,
            "abcng_median_final": float(np.median(sum_ng["finals"])),  # type: ignore
            "abc_median_final": float(np.median(sum_abc["finals"])),   # type: ignore
            "abcng_conv_plot": conv_path,
            "abcng_k_plot": k_path,
            "compare_conv_plot": conv_cmp,
        })

    # Save aggregated summary and final boxplot
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    box_path = plot_box_compare(finals_map, out_dir, labels=("ABC", "ABCNG"))
    print("Saved:")
    print(os.path.join(out_dir, "summary.csv"))
    for row in summary_rows:
        print(row["abcng_conv_plot"])  # type: ignore
        print(row["abcng_k_plot"])  # type: ignore
        print(row["compare_conv_plot"])  # type: ignore
    print(box_path)


if __name__ == "__main__":
    main()


def notebook_quick_demo(runs: int = 5, budget: int = 5000, functions: List[str] | None = None):
    """Small demo for notebooks: fewer runs/budget and inline return values.

    Returns a tuple (df_summary, paths) where paths is a dict with lists of
    saved plot paths per function name.
    """
    if functions is None:
        functions = ["sphere", "rastrigin", "rosenbrock", "ackley"]
    suite = [(name, 30, budget) for name in functions]
    seeds = list(range(1000, 1000 + runs))
    out_dir = os.path.join("figs")
    os.makedirs(out_dir, exist_ok=True)
    finals_map: Dict[str, Dict[str, np.ndarray]] = {}
    rows = []
    paths = {"conv": [], "k": [], "box": None, "compare": []}
    for name, dim, budget in suite:
        sum_ng = summarize_runs(name, dim, budget, runs=runs, seeds=seeds, algo="abcng")
        sum_abc = summarize_runs(name, dim, budget, runs=runs, seeds=seeds, algo="abc")
        conv = plot_convergence(sum_ng, out_dir)
        kplt = plot_k_dynamics(sum_ng, out_dir) if sum_ng["k_median"].size else None  # type: ignore
        conv_cmp = plot_convergence_compare(sum_ng, sum_abc, "ABCNG", "ABC", out_dir)
        finals_map[name] = {
            "ABCNG": sum_ng["finals"],  # type: ignore
            "ABC": sum_abc["finals"],   # type: ignore
        }
        rows.append({
            "function": name,
            "dim": dim,
            "budget": budget,
            "runs": runs,
            "abcng_median_final": float(np.median(sum_ng["finals"])),  # type: ignore
            "abcng_mean_final": float(np.mean(sum_ng["finals"])),      # type: ignore
            "abcng_std_final": float(np.std(sum_ng["finals"])),        # type: ignore
            "abcng_median_time_s": float(np.median(sum_ng["times"])),  # type: ignore
            "abc_median_final": float(np.median(sum_abc["finals"])),   # type: ignore
            "abc_mean_final": float(np.mean(sum_abc["finals"])),       # type: ignore
            "abc_std_final": float(np.std(sum_abc["finals"])),         # type: ignore
            "abc_median_time_s": float(np.median(sum_abc["times"])),   # type: ignore
            "abcng_conv_plot": conv,
            "abcng_k_plot": kplt,
            "compare_conv_plot": conv_cmp,
        })
        paths["conv"].append(conv)
        if kplt:
            paths["k"].append(kplt)
        paths["compare"].append(conv_cmp)
    box = plot_box_compare(finals_map, out_dir, labels=("ABC", "ABCNG"))
    paths["box"] = box
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "summary_quick.csv"), index=False)
    return df, paths
