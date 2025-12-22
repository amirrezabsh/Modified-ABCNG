import os
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.abcng import ABCNG, BENCHES
from models.abc import ABC
from tools.benchmark_plots import pad_histories
from tools.env_config import load_env, get_abcng_kwargs


def _run_single(
    name: str,
    dim: int,
    budget: int,
    seed: int,
    pop_size: int,
    algo: str,
    abcng_kwargs: Optional[Dict[str, object]],
) -> Tuple[List[float], List[float], float, float]:
    f, (lo, hi) = BENCHES[name]
    if algo.lower() == "abc":
        opt = ABC(func=f, dim=dim, bounds=(lo, hi), pop_size=pop_size, max_evals=budget, seed=seed)
    else:
        kwargs = abcng_kwargs or {}
        opt = ABCNG(func=f, dim=dim, bounds=(lo, hi), pop_size=pop_size, max_evals=budget, seed=seed, **kwargs)
    t0 = time.time()
    _, gval, hist = opt.run()
    elapsed = time.time() - t0
    k_hist = list(getattr(opt, "k_hist", []))
    return hist, k_hist, float(gval), float(elapsed)


def _plot_convergence_multi(summaries: Dict[str, Dict[str, object]], out_dir: str, name: str) -> str:
    plt.figure(figsize=(7, 4))
    for label, summary in summaries.items():
        y = np.asarray(summary["median"])  # type: ignore
        if len(y) == 0:
            continue
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=label)
    plt.yscale("log")
    plt.xlabel("Iteration (employed+onlooker cycles)")
    plt.ylabel("Best objective (log scale)")
    plt.title(f"Convergence on {name}")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"conv_multi_{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_k_multi(summaries: Dict[str, Dict[str, object]], out_dir: str, name: str) -> Optional[str]:
    has_k = any(len(np.asarray(s["k_median"])) > 0 for s in summaries.values())  # type: ignore
    if not has_k:
        return None
    plt.figure(figsize=(7, 4))
    for label, summary in summaries.items():
        y = np.asarray(summary["k_median"])  # type: ignore
        if len(y) == 0:
            continue
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=label)
    plt.xlabel("Iteration (employed+onlooker cycles)")
    plt.ylabel("Mean neighborhood radius k")
    plt.title(f"k dynamics on {name}")
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"kdyn_multi_{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_box_multi(finals_map: Dict[str, np.ndarray], out_dir: str, name: str) -> str:
    labels = list(finals_map.keys())
    data = [finals_map[k] for k in labels]
    plt.figure(figsize=(max(8, len(labels) * 0.9), 4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.yscale("log")
    plt.ylabel("Final best objective (log)")
    plt.title(f"Final performance on {name}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"box_{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def run_ablation_suite(
    suite: List[Tuple[str, int, int]],
    variants: List[Dict[str, object]],
    runs: int = 25,
    pop_size: int = 50,
    seeds: Optional[List[int]] = None,
    out_dir: str = "figs/ablation",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    if seeds is None:
        seeds = list(range(1000, 1000 + runs))

    os.makedirs(out_dir, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    paths: Dict[str, List[str]] = {"conv": [], "k": [], "box": []}

    for name, dim, budget in suite:
        summaries: Dict[str, Dict[str, object]] = {}
        finals_map: Dict[str, np.ndarray] = {}

        for v in variants:
            label = str(v["name"])
            algo = str(v.get("algo", "abcng"))
            abcng_kwargs = v.get("abcng_kwargs")

            hists: List[List[float]] = []
            k_hists: List[List[float]] = []
            finals: List[float] = []
            times: List[float] = []

            for s in seeds[:runs]:
                hist, k_hist, gval, elapsed = _run_single(
                    name, dim, budget, seed=s, pop_size=pop_size, algo=algo, abcng_kwargs=abcng_kwargs
                )
                hists.append(hist)
                k_hists.append(k_hist)
                finals.append(gval)
                times.append(elapsed)
                run_rows.append({
                    "function": name,
                    "dim": dim,
                    "budget": budget,
                    "variant": label,
                    "seed": s,
                    "final_best": gval,
                    "time_s": elapsed,
                })

            H = pad_histories(hists, mode="truncate")
            K = pad_histories(k_hists, mode="truncate") if k_hists else np.empty((0,))
            summary = {
                "name": name,
                "median": np.median(H, axis=0),
                "k_median": np.median(K, axis=0) if K.size else np.array([]),
                "finals": np.array(finals, dtype=float),
                "times": np.array(times, dtype=float),
            }
            summaries[label] = summary
            finals_map[label] = summary["finals"]  # type: ignore

            summary_rows.append({
                "function": name,
                "dim": dim,
                "budget": budget,
                "runs": runs,
                "variant": label,
                "median_final": float(np.median(summary["finals"])),  # type: ignore
                "mean_final": float(np.mean(summary["finals"])),      # type: ignore
                "std_final": float(np.std(summary["finals"])),        # type: ignore
                "median_time_s": float(np.median(summary["times"])),  # type: ignore
            })

        conv_path = _plot_convergence_multi(summaries, out_dir, name)
        k_path = _plot_k_multi(summaries, out_dir, name)
        box_path = _plot_box_multi(finals_map, out_dir, name)
        paths["conv"].append(conv_path)
        if k_path:
            paths["k"].append(k_path)
        paths["box"].append(box_path)

    df_summary = pd.DataFrame(summary_rows)
    df_runs = pd.DataFrame(run_rows)
    df_summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    df_runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    return df_summary, paths


def default_variants() -> List[Dict[str, object]]:
    return [
        {"name": "ABC", "algo": "abc"},
        {"name": "ABCNG", "algo": "abcng", "abcng_kwargs": {"paper_mode": True}},
        {"name": "no_gbest", "algo": "abcng", "abcng_kwargs": {"paper_mode": True, "use_gbest": False}},
        {"name": "no_adaptive_k", "algo": "abcng", "abcng_kwargs": {"paper_mode": True, "use_adaptive_k": False}},
        {"name": "no_gaussian", "algo": "abcng", "abcng_kwargs": {"paper_mode": True, "use_gaussian": False}},
        {"name": "self_in_eq7", "algo": "abcng", "abcng_kwargs": {"paper_mode": True, "neighbor_mode": "self"}},
        {"name": "single_dim_update", "algo": "abcng", "abcng_kwargs": {"paper_mode": True, "update_dim_mode": "single"}},
        {"name": "noise_cauchy", "algo": "abcng", "abcng_kwargs": {"paper_mode": True, "noise_model": "cauchy"}},
        {"name": "noise_uniform", "algo": "abcng", "abcng_kwargs": {"paper_mode": True, "noise_model": "uniform"}},
    ]


def env_variant(env_path: str = ".env") -> Dict[str, object]:
    env = load_env(env_path)
    kwargs = get_abcng_kwargs(env)
    kwargs["paper_mode"] = True
    return {"name": "ABCNG_ENV", "algo": "abcng", "abcng_kwargs": kwargs}


def build_paper_suite(functions: List[str]) -> List[Tuple[str, int, int]]:
    dims = [30, 50, 100]
    suite: List[Tuple[str, int, int]] = []
    for fn in functions:
        if fn not in BENCHES:
            raise KeyError(f"Unknown benchmark '{fn}'. Available: {sorted(BENCHES.keys())}")
        for d in dims:
            suite.append((fn, d, 5000 * d))
    return suite
