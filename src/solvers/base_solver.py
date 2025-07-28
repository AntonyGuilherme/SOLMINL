from src.generators.combinatorial.instance_generator import Permutation
import numpy as np
import copy
import matplotlib.pyplot as plt
from .utils import plot_optimization_histories, Solution, solve, plot_samples

np.random.seed(91)

def next_swap(f, x, n):
    y = copy.deepcopy(x)

    for i in range(n - 1):
        for j in range(i+1, n):
            y.solution[i], y.solution[j] = y.solution[j], y.solution[i]
            y.single_objective_value = f.evaluate(y.solution)

            if y.single_objective_value > x.single_objective_value:

                return y
            else:
                # undoing the change to no copy the solution again
                y.solution[i], y.solution[j] = y.solution[j], y.solution[i]

    return y

def change(f, x):
    x.solution = np.random.permutation(x.solution)
    x.single_objective_value = f.evaluate(x.solution)





for size in [10]:
    for distance in ["C", "K"]:
        base = Solution()
        base.solution = list(range(1, size + 1))
        np.random.shuffle(base.solution)

        permutation = Permutation(len(base.solution), 2, distance=distance) 

        permutation.calc_parameters_difficult()

        base.single_objective_value = permutation.evaluate(base.solution)

        consensus = permutation.consensus[0]

        number_of_samples = 10
        historic0, samples0 = solve(permutation, base, change_nbg=change, next=next_swap, maxeval=number_of_samples)
        
        best = permutation.evaluate(permutation.consensus[0])

        plot_samples(samples0, best_possible=best, title="SWAP", output=f"swap_{distance}_{size}.png")

        plot_optimization_histories(
             [historic0], 
             ["SWAP", "ADJ. SWAP", "ADJ. INVERTION"], 
             [best for _ in range(3)],
             output_path=f"historic_{distance}_{size}.png")





