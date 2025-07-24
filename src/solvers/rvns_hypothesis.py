from src.generators.combinatorial.instance_generator import Permutation
import numpy as np
import copy
import matplotlib.pyplot as plt
from .utils import plot_optimization_histories, Solution, solve, plot_samples

np.random.seed(91)


size = 10
base = Solution()
base.solution = list(range(1, size + 1))
np.random.shuffle(base.solution)

permutation = Permutation(len(base.solution), len(base.solution)) 

permutation.calc_parameters_difficult()

base.single_objective_value = permutation.evaluate(base.solution)

consensus = permutation.consensus[0]

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

number_of_samples = 30
historic0, samples = solve(permutation, base, change_nbg=change, next=next_swap, maxeval=number_of_samples)


# historic1 = solve(permutation, base, change_nbg=change, next=next_swap_close, maxeval=200)
# historic2 = solve(permutation, base, change_nbg=change, next=next_swap_invertion, maxeval=200)
best = permutation.evaluate(permutation.consensus[0])
plot_samples(samples, best_possible=best)

# plot_optimization_histories(
#     [historic0, historic1, historic2], 
#     ["SWAP", "SWAP CLOSE", "SWAP INVERTION"], 
#     [best for _ in range(3)])

plot_optimization_histories(
    [historic0], 
    ["SWAP"], 
    [best],
    "swap.png")

# plot_optimization_histories(
#     [historic1], 
#     ["SWAP CLOSE"], 
#     [best],
#     "swap_close.png")

# plot_optimization_histories(
#     [historic2], 
#     ["SWAP INVERTION"], 
#     [best],
#     "swap_invertion")





