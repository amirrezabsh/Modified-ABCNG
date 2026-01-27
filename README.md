# 🐝 Artificial Bee Colony with Adaptive Neighborhood & Gaussian Perturbation (ABCNG)

This repository implements the **Artificial Bee Colony algorithm based on Adaptive Neighborhood Search and Gaussian Perturbation (ABCNG)** — proposed by *Songyi Xiao et al., 2021* in *Applied Soft Computing Journal* ([DOI: 10.1016/j.asoc.2020.106955](https://doi.org/10.1016/j.asoc.2020.106955)).

ABCNG enhances the standard Artificial Bee Colony (ABC) by:

1. **Adaptive Neighborhood Search** — dynamically adjusts the neighborhood size `k` during the optimization process to balance exploration and exploitation.
2. **Gaussian Perturbation Based on Evolutionary Rate** — introduces stochastic perturbations guided by the population’s improvement rate to escape local optima and maintain diversity.

---

## 📘 Paper Summary

> **Reference:**
> *Songyi Xiao, Hui Wang, Wenjun Wang, Zhikai Huang, Xinyu Zhou, Minyang Xu.*
> *Artificial bee colony algorithm based on adaptive neighborhood search and Gaussian perturbation.*
> *Applied Soft Computing Journal, Vol. 100, 2021, 106955.*

### Core Concepts

* **Neighborhood Radius (`k`)** adapts dynamically:
  [
  k =
  \begin{cases}
  k + 1, & \text{if } f(V_i) < f(X_i)\
  k - 1, & \text{otherwise}
  \end{cases}
  ]
* **Modified search equation:**
  [
  v_{i,j} = x_{n_i,j} + \phi_{i,j}(x_{n_i,j} - x_{n_o,j}) + \varphi_{i,j}(G_{best,j} - x_{n_i,j})
  ]
* **Gaussian Perturbation (ABCNG-ia variant):**
  [
  v_{i,j} = x_{i,j} \cdot (1 + \mathcal{N}(\delta_i, \delta_a))
  ]
  where
  (\delta_i): individual evolutionary rate,
  (\delta_a): population average evolutionary rate.

---

## ⚙️ Installation

```bash
git clone https://github.com/amirrezabsh/Modified-ABCNG.git
cd Modified-ABCNG
pip install -r requirements.txt
```

### `requirements.txt`

```txt
numpy>=1.26.0
pandas>=2.2.0
# Optional (for visualization and progress bars)
matplotlib>=3.8.0
tqdm>=4.66.0
```

---

## 🚀 Quick Start

### Run the demo notebook

Open and run:

- `main.ipynb` for a quick benchmark demo and plots (includes notes on settings).
- `ablation_runs.ipynb` for ablation studies and comparison plots (paper protocol notes included).

### Use ABCNG in your project

```python
from models.abcng import ABCNG
import numpy as np

# Example: Sphere function
def sphere(x): 
    return float(np.sum(x**2))

optimizer = ABCNG(func=sphere, dim=30, bounds=(-100, 100), pop_size=50, max_evals=150000)
best_x, best_val, history = optimizer.run()

print("Best solution:", best_val)
```

### Paper settings

To match the paper’s protocol, set `paper_mode=True` (forces SN=50, limit=SN·D, max_evals=5000·D, and the onlooker loop behavior).

---

## 🧪 Features

* ✅ Fully functional **ABCNG (Adaptive + Gaussian Perturbation)** variant
* ✅ Modular, extendable Python class (`models/abcng.py`)
* ✅ Supports **any continuous optimization problem**
* ✅ Includes classic benchmarks (Sphere, Rastrigin, Ackley, etc.)
* ✅ Ready for **further optimization and hybridization** (e.g., with DE, PSO, or RL)

---

## 📊 Future Work

In the next phase, we plan to:

* 🔧 Optimize hyperparameters and test adaptive strategies for `ϕ` and `φ`.
* 📈 Add support for **multi-objective and constrained** problems.
* 🧠 Experiment with **RL-based parameter adaptation**.
* ⚡ Integrate GPU-based acceleration for large-scale search.
* 🧩 Benchmark on **real-world optimization tasks** (e.g., feature selection, neural tuning).

---

## 📚 Citation

If you use or modify this implementation in research, please cite:

```
@article{Xiao2021ABCNG,
  title={Artificial bee colony algorithm based on adaptive neighborhood search and Gaussian perturbation},
  author={Xiao, Songyi and Wang, Hui and Wang, Wenjun and Huang, Zhikai and Zhou, Xinyu and Xu, Minyang},
  journal={Applied Soft Computing},
  volume={100},
  pages={106955},
  year={2021},
  publisher={Elsevier}
}
```

---

## 👨‍💻 Author Notes

This repository was created as part of an ongoing **meta-heuristic optimization research project**.
The next milestone will focus on **improving ABCNG’s convergence speed and dynamic parameter control**.

> Maintained by: [Amirreza Behmanesh](https://github.com/amirrezabsh)
