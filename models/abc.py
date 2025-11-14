import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple, List


@dataclass
class ABC:
    func: Callable[[np.ndarray], float]
    dim: int
    bounds: Tuple[float, float]
    pop_size: int = 50
    limit: int = None  # default SN * D
    max_evals: int = None  # default 5000 * D
    seed: int = None

    rng: np.random.Generator = field(init=False)
    lower: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    fitness: np.ndarray = field(init=False)
    trials: np.ndarray = field(init=False)
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

        self.X = self.lower + self.rng.random((self.pop_size, self.dim)) * (self.upper - self.lower)
        self.fitness = np.array([self._evaluate(x) for x in self.X])
        self.trials = np.zeros(self.pop_size, dtype=int)

        gidx = int(np.argmin(self.fitness))
        self.gbest = self.X[gidx].copy()
        self.gbest_val = float(self.fitness[gidx])
        self.hist = [self.gbest_val]

    def _evaluate(self, x: np.ndarray) -> float:
        self.evals += 1
        return self.func(x)

    def _greedy(self, i: int, cand: np.ndarray, cand_val: float) -> bool:
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

    def _scout_if_needed(self, i: int):
        if self.trials[i] >= self.limit:
            self.X[i] = self.lower + self.rng.random(self.dim) * (self.upper - self.lower)
            self.fitness[i] = self._evaluate(self.X[i])
            self.trials[i] = 0
            if self.fitness[i] < self.gbest_val:
                self.gbest = self.X[i].copy()
                self.gbest_val = float(self.fitness[i])

    def _search_eq(self, i: int) -> np.ndarray:
        # Standard ABC perturbation: v = x_i + phi*(x_i - x_k)
        choices = [idx for idx in range(self.pop_size) if idx != i]
        k = int(self.rng.choice(choices))
        phi = self.rng.uniform(-1.0, 1.0, size=self.dim)
        v = self.X[i] + phi * (self.X[i] - self.X[k])
        return np.clip(v, self.lower, self.upper)

    def _probabilities(self) -> np.ndarray:
        fit_values = np.empty(self.pop_size, dtype=float)
        for idx, fval in enumerate(self.fitness):
            if fval >= 0:
                fit_values[idx] = 1.0 / (1.0 + fval)
            else:
                fit_values[idx] = 1.0 + abs(fval)
        total = float(np.sum(fit_values))
        if total <= 0.0:
            return np.full(self.pop_size, 1.0 / self.pop_size)
        return fit_values / total

    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        while self.evals < self.max_evals:
            # Employed bees
            for i in range(self.pop_size):
                if self.evals >= self.max_evals:
                    break
                v = self._search_eq(i)
                fv = self._evaluate(v)
                self._greedy(i, v, fv)
                self._scout_if_needed(i)

            # Onlookers via roulette
            probs = self._probabilities()
            onlookers = 0
            idx = 0
            while onlookers < self.pop_size:
                if self.evals >= self.max_evals:
                    break
                i = idx % self.pop_size
                idx += 1
                if self.rng.random() > probs[i]:
                    continue
                onlookers += 1
                v = self._search_eq(i)
                fv = self._evaluate(v)
                self._greedy(i, v, fv)
                self._scout_if_needed(i)

            self.hist.append(self.gbest_val)

        return self.gbest.copy(), float(self.gbest_val), list(self.hist)

