import sys, os
import numpy as np
import time

# --- mealpy paketini görünür yap ---
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from mealpy import FloatVar
from mealpy.physics_based.CEO import OriginalCEO
from mealpy.physics_based.EO import OriginalEO
from mealpy.physics_based.HGSO import OriginalHGSO


# ---------------- Problem tanımı ----------------
def sphere(x):
    return np.sum(x**2)

problem = {
    "bounds": FloatVar(lb=(-10.,)*20, ub=(10.,)*20),
    "obj_func": sphere,
    "minmax": "min",
}


# ---------------- Benchmark fonksiyonu ----------------
def run_algo(name, model_class, runs=5):
    results = []
    times = []

    for seed in range(runs):
        np.random.seed(seed)
        model = model_class(epoch=100, pop_size=30)

        start = time.time()
        best = model.solve(problem)
        elapsed = time.time() - start

        results.append(best.target.fitness)
        times.append(elapsed)

    return {
        "name": name,
        "mean": np.mean(results),
        "std": np.std(results),
        "best": np.min(results),
        "time": np.mean(times)
    }


# ---------------- ÇALIŞTIR ----------------
if __name__ == "__main__":
    print("\n=== MEALPY BENCHMARK (tek komut) ===\n")

    algos = [
        ("CEO", OriginalCEO),
        ("EO", OriginalEO),
        ("HGSO", OriginalHGSO),
    ]

    table = []
    for name, algo in algos:
        print(f"Running {name} ...")
        res = run_algo(name, algo)
        table.append(res)

    # ---------------- Sonuç Tablosu ----------------
    print("\nRESULTS (lower is better)\n")
    print(f"{'Algo':<6} | {'Best':<12} | {'Mean':<12} | {'Std':<12} | {'Time (s)':<10}")
    print("-"*65)

    for r in table:
        print(f"{r['name']:<6} | {r['best']:<12.4e} | {r['mean']:<12.4e} | {r['std']:<12.4e} | {r['time']:<10.3f}")

    print("\nDONE.")
