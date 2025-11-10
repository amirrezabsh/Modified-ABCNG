# ABCNG implementation and quick demo
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Dict, Any
import math
import time
import pandas as pd

# -----------------------------
# Benchmark functions
# -----------------------------

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A*len(x) + np.sum(x**2 - A*np.cos(2*np.pi*x)))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0))

def griewank(x: np.ndarray) -> float:
    sum_part = np.sum(x**2)/4000.0
    prod_part = np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1))))
    return float(sum_part - prod_part + 1)

def ackley(x: np.ndarray) -> float:
    d = len(x)
    return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/d)) - np.exp(np.sum(np.cos(2*np.pi*x))/d) + 20 + np.e)

def schwefel(x: np.ndarray) -> float:
    return float(418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x)))))

BENCHES: Dict[str, Tuple[Callable[[np.ndarray], float], Tuple[float,float]]] = {
    "sphere": (sphere, (-100, 100)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-30, 30)),
    "griewank": (griewank, (-600, 600)),
    "ackley": (ackley, (-32.768, 32.768)),
    "schwefel": (schwefel, (-500, 500)),
}

# -----------------------------
# ABCNG Optimizer
# -----------------------------

@dataclass
class ABCNG:
    func: Callable[[np.ndarray], float]
    dim: int
    bounds: Tuple[float, float]
    pop_size: int = 50  # SN
    limit: int = None   # default SN * D
    max_evals: int = None  # default 5000 * D
    seed: int = None

    # internal state (filled in __post_init__)
    rng: np.random.Generator = field(init=False)
    lower: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    fitness: np.ndarray = field(init=False)
    trials: np.ndarray = field(init=False)
    k: int = field(init=False)             # neighborhood radius
    evals: int = field(init=False, default=0)
    gbest: np.ndarray = field(init=False)
    gbest_val: float = field(init=False)
    hist: List[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.lower = np.full(self.dim, self.bounds[0], dtype=float)
        self.upper = np.full(self.dim, self.bounds[1], dtype=float)
        if self.limit is None:
            self.limit = self.pop_size * self.dim
        if self.max_evals is None:
            self.max_evals = 5000 * self.dim

        # init population
        self.X = self.lower + self.rng.random((self.pop_size, self.dim)) * (self.upper - self.lower)
        self.fitness = np.array([self._evaluate(x) for x in self.X])
        self.trials = np.zeros(self.pop_size, dtype=int)

        self.gbest_idx = int(np.argmin(self.fitness))
        self.gbest = self.X[self.gbest_idx].copy()
        self.gbest_val = float(self.fitness[self.gbest_idx])

        # start with the smallest legal neighborhood radius
        self.k = 1
        self.k_min = 1
        self.k_max = (self.pop_size - 1) // 2

        self.hist = [self.gbest_val]

    # -------------------------
    # Utility methods
    # -------------------------
    def _evaluate(self, x: np.ndarray) -> float:
        self.evals += 1
        return self.func(x)

    def _greedy(self, i: int, cand: np.ndarray, cand_val: float) -> bool:
        """Greedy selection; return True if improved (strictly better)."""
        if cand_val < self.fitness[i]:
            self.X[i] = cand
            self.fitness[i] = cand_val
            self.trials[i] = 0
            if cand_val < self.gbest_val:
                self.gbest = cand.copy()
                self.gbest_val = float(cand_val)
            return True
        else:
            self.trials[i] += 1
            return False

    def _ring_index(self, idx: int) -> int:
        return idx % self.pop_size

    def _neighbors_indices(self, i: int) -> List[int]:
        """Return indices in the k-neighborhood (2k+1 ring topology)."""
        return [self._ring_index(i + offset) for offset in range(-self.k, self.k + 1)]

    def _outside_indices(self, i: int) -> List[int]:
        """Return indices outside the k-neighborhood (exclude i)."""
        neigh = set(self._neighbors_indices(i))
        return [idx for idx in range(self.pop_size) if idx not in neigh]

    def _search_eq(self, i: int) -> np.ndarray:
        """Equation (7): neighborhood + gbest guidance."""
        # choose x_ni from neighborhood (excluding i if you want; paper uses dynamic neighborhood of Xi randomly (i != ni))
        neigh = self._neighbors_indices(i)
        if len(neigh) > 1 and i in neigh:
            neigh_no_i = [idx for idx in neigh if idx != i]
        else:
            neigh_no_i = neigh

        if not neigh_no_i:  # degenerate, fallback random different index
            choices = [idx for idx in range(self.pop_size) if idx != i]
            ni = self.rng.choice(choices)
        else:
            ni = int(self.rng.choice(neigh_no_i))

        # choose x_no from outside neighborhood
        outside = self._outside_indices(i)
        if not outside:  # if k is too big (shouldn't happen due to bounds), fallback random different index
            outside = [idx for idx in range(self.pop_size) if idx != ni]
        no = int(self.rng.choice(outside))

        phi = self.rng.uniform(-1.0, 1.0, size=self.dim)  # φ in [-1,1]
        varphi = self.rng.uniform(0.0, 1.5, size=self.dim)  # ϕ in [0,1.5]

        xni = self.X[ni]
        xno = self.X[no]
        v = xni + phi * (xni - xno) + varphi * (self.gbest - xni)

        # bound handling (simple clip)
        return np.clip(v, self.lower, self.upper)

    def _gaussian_perturb(self, xi: np.ndarray, delta_i: float, delta_a: float) -> np.ndarray:
        """Equation (10): v = x * (1 + N(mean=δi, std=δa)) (ABCNG-ia)."""
        if delta_a < 1e-12:
            # avoid zero std -> use a tiny noise
            delta_a = 1e-12
        noise = self.rng.normal(loc=delta_i, scale=delta_a, size=self.dim)
        v = xi * (1.0 + noise)
        return np.clip(v, self.lower, self.upper)

    def _update_k(self, improved: bool):
        if improved:
            self.k = min(self.k + 1, self.k_max)
        else:
            self.k = max(self.k - 1, self.k_min)

    def _scout_if_needed(self, i: int):
        if self.trials[i] >= self.limit:
            self.X[i] = self.lower + self.rng.random(self.dim) * (self.upper - self.lower)
            self.fitness[i] = self._evaluate(self.X[i])
            self.trials[i] = 0
            if self.fitness[i] < self.gbest_val:
                self.gbest = self.X[i].copy()
                self.gbest_val = float(self.fitness[i])

    # -------------------------
    # Main optimize loop
    # -------------------------
    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        # Loop until evaluation budget is exhausted
        while self.evals < self.max_evals:
            # --- Employed bee phase ---
            for i in range(self.pop_size):
                if self.evals >= self.max_evals:
                    break
                v = self._search_eq(i)
                fv = self._evaluate(v)
                improved = self._greedy(i, v, fv)
                self._update_k(improved)
                self._scout_if_needed(i)

            # Record previous fitness for evolutionary rates (onlooker phase uses last iteration values)
            prev_fit = self.fitness.copy()

            # --- Onlooker bee phase (no roulette; visit each solution again) ---
            for i in range(self.pop_size):
                if self.evals >= self.max_evals:
                    break
                # attempt neighborhood search again
                v = self._search_eq(i)
                fv = self._evaluate(v)
                improved = self._greedy(i, v, fv)
                self._update_k(improved)

                if not improved and self.evals < self.max_evals:
                    # Compute evolutionary rates δi and δa (based on prev_fit vs current)
                    # Avoid division by zero by adding tiny epsilon
                    eps = 1e-12
                    delta_i = (self.fitness[i] - prev_fit[i]) / (self.fitness[i] + eps)
                    delta_all = (self.fitness - prev_fit) / (self.fitness + eps)
                    delta_a = float(np.mean(delta_all))

                    # Gaussian perturbation
                    gp = self._gaussian_perturb(self.X[i], delta_i, delta_a)
                    fgp = self._evaluate(gp)
                    improved2 = self._greedy(i, gp, fgp)
                    self._update_k(improved2)

                self._scout_if_needed(i)

            # --- Scout phase enforcement already handled inside loops ---

            # store best
            self.hist.append(self.gbest_val)

        return self.gbest.copy(), float(self.gbest_val), list(self.hist)


# -----------------------------
# Helper: run a quick demo on a few benchmarks
# -----------------------------

def demo_run():
    results = []
    start_time = time.time()

    tests = [
        ("sphere", 30, 10000),
        ("rastrigin", 30, 10000),
        ("rosenbrock", 30, 10000),
        ("ackley", 30, 10000),
    ]

    for name, dim, budget in tests:
        f, (lo, hi) = BENCHES[name]
        opt = ABCNG(func=f, dim=dim, bounds=(lo, hi),
                    pop_size=40, max_evals=budget, seed=42)
        gbest, gval, hist = opt.run()
        results.append({
            "function": name,
            "dim": dim,
            "budget": budget,
            "best_f": gval,
            "k_final": opt.k,
            "evals": opt.evals
        })

    df = pd.DataFrame(results)
    print("\n=== ABCNG quick results ===")
    print(df.to_string(index=False))
    df.to_csv("abcng_results.csv", index=False)
    print("\nResults saved to abcng_results.csv")

    elapsed = time.time() - start_time
    return df, elapsed

