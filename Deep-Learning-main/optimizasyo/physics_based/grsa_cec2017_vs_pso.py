import os
import sys
import numpy as np

# ==== 0) ROOT'u path'e ekle (pip -e . uğraşmadan import için) ====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))  # .../mealpy-master
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mealpy import FloatVar
from mealpy.physics_based.GRSA import OriginalGRSA
from mealpy.swarm_based import PSO
from opfunu.cec_based import cec2017


# ==== 1) Ortak yardımcı fonksiyonlar ====

def run_algorithm_many_times(model_cls, model_kwargs, problem_dict, n_runs=10, seed_base=0):
    """
    Aynı problemi aynı algoritmayla n_runs kez çalıştırır,
    her seferinde en iyi fitness'i kaydeder ve istatistik döndürür.
    """
    print(f" -> Running {model_cls.__name__} for {n_runs} runs ...", end="", flush=True)

    best_vals = []

    for run_id in range(n_runs):
        np.random.seed(seed_base + run_id)
        problem = dict(problem_dict)  # olası yan etkilere karşı kopya

        model = model_cls(**model_kwargs)
        g_best = model.solve(problem)
        best_vals.append(g_best.target.fitness)

    best_vals = np.array(best_vals, dtype=float)

    print(" done ✔")

    return {
        "mean": float(np.mean(best_vals)),
        "std": float(np.std(best_vals)),
        "best": float(np.min(best_vals)),
        "worst": float(np.max(best_vals)),
        "all_runs": best_vals,
    }


def print_stats(title, stats):
    print(f"\n[{title}]")
    print(f"  Mean  : {stats['mean']:.6e}")
    print(f"  Std   : {stats['std']:.6e}")
    print(f"  Best  : {stats['best']:.6e}")
    print(f"  Worst : {stats['worst']:.6e}")


# ==== 2) Makaledeki test fonksiyonu (şimdilik placeholder) ====

def run_paper_test():
    """
    Buraya GRSA makalesinde kullanılan ana test fonksiyonunu koyman gerekiyor.
    Şu anlık kontrol amaçlı Sphere kullanıyoruz.
    İstersen burayı Rastrigin/Rosenbrock vs. ile değiştirirsin.
    """
    print("\n========== PAPER TEST (placeholder) ==========")

    def paper_objective(x):
        # TODO: Makaledeki gerçek test fonksiyonuyla değiştir.
        return np.sum(x ** 2)

    dim = 30
    lb = [-100.0] * dim
    ub = [100.0] * dim

    problem = {
        "obj_func": paper_objective,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "min",
        "verbose": True,     # burada logları görebilirsin
    }

    common_paras = {"epoch": 1000, "pop_size": 50}

    print("▶ GRSA vs PSO on paper-test function")

    grsa_stats = run_algorithm_many_times(
        OriginalGRSA,
        dict(common_paras),
        problem,
        n_runs=5,
        seed_base=100,
    )
    print_stats("GRSA (paper test)", grsa_stats)

    pso_stats = run_algorithm_many_times(
        PSO.OriginalPSO,
        dict(common_paras),
        problem,
        n_runs=5,
        seed_base=200,
    )
    print_stats("PSO (paper test)", pso_stats)

    print("✔ PAPER TEST block completed.\n")


# ==== 3) CEC 2017: F15, F19, F20 testleri ====

def run_cec2017_tests():
    """
    CEC 2017 F15, F19, F20 fonksiyonlarında
    GRSA ve PSO karşılaştırması.
    """
    dim = 30
    n_runs = 10
    common_paras = {"epoch": 1000, "pop_size": 50}

    cec_funcs = [
        ("F15", cec2017.F152017, "Hybrid Function 6"),
        ("F19", cec2017.F192017, "Hybrid Function 10"),
        ("F20", cec2017.F202017, "Composition Function 1"),
    ]

    print("\n===== COMPARISON TABLE (GRSA vs PSO on CEC2017) =====")
    print("Func | Alg  |   Mean        |   Std         |   Best        |   Worst")
    print("--------------------------------------------------------------------------")

    for fid, f_class, desc in cec_funcs:
        print(f"\n========== CEC2017 {fid} ({desc}, D={dim}) ==========")
        print(f"▶ Starting test: {fid} ({desc})")

        # opfunu fonksiyon nesnesi
        f = f_class(ndim=dim)

        problem = {
            "obj_func": f.evaluate,
            "bounds": FloatVar(lb=f.lb, ub=f.ub),
            "minmax": "min",
            "verbose": False,
        }

        # ---- GRSA ----
        grsa_stats = run_algorithm_many_times(
            OriginalGRSA,
            dict(common_paras),
            problem,
            n_runs=n_runs,
            seed_base=1000,
        )

        # ---- PSO ----
        pso_stats = run_algorithm_many_times(
            PSO.OriginalPSO,
            dict(common_paras),
            problem,
            n_runs=n_runs,
            seed_base=2000,
        )

        # Detay yaz (istersen burada tut)
        print_stats("GRSA", grsa_stats)
        print_stats("PSO", pso_stats)

        # Özet tablo satırlarını yaz (karşılaştırma için asıl işine yarayan kısım)
        print("\n--- Summary row for table ---")
        print(
            f"{fid}  | GRSA | {grsa_stats['mean']:.3e} | {grsa_stats['std']:.3e} | "
            f"{grsa_stats['best']:.3e} | {grsa_stats['worst']:.3e}"
        )
        print(
            f"{fid}  | PSO  | {pso_stats['mean']:.3e} | {pso_stats['std']:.3e} | "
            f"{pso_stats['best']:.3e} | {pso_stats['worst']:.3e}"
        )

        print(f"\n✔ {fid} completed.\n")

def main():
    print("=== GRSA vs PSO Benchmark Runner ===")

    # 1) Makale fonksiyonu testi
    run_paper_test()

    # 2) CEC 2017 testleri
    run_cec2017_tests()

    print("\n🎉 All GRSA vs PSO tests completed successfully.")


if __name__ == "__main__":
    main()
