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

def next_swap_close(f, x, n):
    y = copy.deepcopy(x)

    for i in range(1, n):
        y.solution[i], y.solution[i-1] = y.solution[i-1], y.solution[i]
        y.single_objective_value = f.evaluate(y.solution)

        if y.single_objective_value > x.single_objective_value:
            return y
        else:
            # undoing the change to no copy the solution again
            y.solution[i], y.solution[i-1] = y.solution[i-1], y.solution[i]

    return y

def next_swap_invertion(f, x, n):
    y = copy.deepcopy(x)
    # same computational cost as copy the invertion
    invertion = np.argsort(y.solution)

    for i in range(1, n):
        j, k = invertion[i-1], invertion[i]
        invertion[i], invertion[i-1] = invertion[i-1], invertion[i]

        y.solution[j], y.solution[k] = y.solution[k], y.solution[j]
        y.single_objective_value = f.evaluate(y.solution)

        if y.single_objective_value > x.single_objective_value:
            return y
        else:
            # undoing the change to no copy the solution again
            y.solution[i], y.solution[i-1] = y.solution[i-1], y.solution[i]
            invertion[i], invertion[i-1] = invertion[i-1], invertion[i]

    return y

def change(f, x):
    x.solution = np.random.permutation(x.solution)
    x.single_objective_value = f.evaluate(x.solution)


for size in [5, 10, 20]:
    for distance in ["C", "K"]:
        base = Solution()
        base.solution = list(range(1, size + 1))
        np.random.shuffle(base.solution)

        permutation = Permutation(len(base.solution), len(base.solution), distance=distance) 

        permutation.calc_parameters_difficult()

        base.single_objective_value = permutation.evaluate(base.solution)

        consensus = permutation.consensus[0]

        number_of_samples = 10
        historic0, samples0 = solve(permutation, base, change_nbg=change, next=next_swap, maxeval=number_of_samples)
        historic1, samples1 = solve(permutation, base, change_nbg=change, next=next_swap_close, maxeval=number_of_samples)
        historic2, samples2 = solve(permutation, base, change_nbg=change, next=next_swap_invertion, maxeval=number_of_samples)
        
        best = permutation.evaluate(permutation.consensus[0])

        plot_samples(samples0, best_possible=best, title="SWAP", output=f"swap_{distance}_{size}.png")
        plot_samples(samples1, best_possible=best, title="ADJ. SWAP", output=f"adj_swap_{distance}_{size}.png")
        plot_samples(samples2, best_possible=best, title="ADJ. INVERTION", output=f"adj_inv_{distance}_{size}.png")

        plot_optimization_histories(
             [historic0, historic1, historic2], 
             ["SWAP", "ADJ. SWAP", "ADJ. INVERTION"], 
             [best for _ in range(3)],
             output_path=f"historic_{distance}_{size}.png")





