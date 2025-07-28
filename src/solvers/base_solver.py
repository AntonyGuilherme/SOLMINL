from src.generators.mixed.mixed import MixedFunction, Solution
import numpy as np
import copy
from .utils import plot_optimization_histories, plot_samples


np.random.seed(91)

def next_swap(f : MixedFunction, x : Solution):
    y = copy.deepcopy(x)
    n = len(y.permutation)

    for i in range(n - 1):
        for j in range(i+1, n):
            y.permutation[i], y.permutation[j] = y.permutation[j], y.permutation[i]
            y.value, y.c_value, y.p_value = f.evaluate(y, c_value=y.c_value)

            if y.value < x.value:
                return y
            else:
                # undoing the change to no copy the solution again
                y.permutation[i], y.permutation[j] = y.permutation[j], y.permutation[i]

    return y

def change_permutation(x: Solution):
    return np.random.permutation(x.permutation)

def numerical_gradient(fobj: MixedFunction, x: Solution, epsilon=1e-6):
    grad = np.zeros_like(x.continuos)
    y = copy.deepcopy(x)
    for i in range(len(x.continuos)):
        x1 = x.continuos.copy()
        x2 = x.continuos.copy()
        x1[i] += epsilon
        x2[i] -= epsilon
        y.continuos = x1
        ev_x1, _, _ = fobj.evaluate(y, p_value=y.p_value)
        y.continuos = x2
        ev_x2, _, _ = fobj.evaluate(y, p_value=y.p_value)
        grad[i] = (ev_x1 - ev_x2) / (2 * epsilon)
    return grad

def continuos_step(objective: MixedFunction, x: Solution, direction = -1):
    y = copy.deepcopy(x)
    grad = numerical_gradient(objective, y)
    step_size = 0.05
    y.continuos = y.continuos + direction * step_size * grad
    y.continuos = np.clip(y.continuos, 0.0, 1.0)
    y.value, y.c_value, y.p_value = objective.evaluate(y, p_value=y.p_value)

    return y

def random_continuos_reposition(x:Solution, epslon=1e-6):
    """
    Generate a point in [0,1]^D that is at least `min_dist` away from `x`,
    by adding/subtracting noise dimension-wise until the norm is sufficient.
    """
    D = len(x.continuos)
    delta = np.zeros_like(x.continuos)

    while np.linalg.norm(delta) < epslon:
        direction = np.random.choice([-1, 1], size=D)
        step = np.random.uniform(0.05, 0.2, size=D)
        delta = direction * step
        candidate = x.continuos + delta

        # Project to [0,1]
        candidate = np.clip(candidate, 0.0, 1.0)
        delta = candidate - x.continuos

    return candidate


def step(objective: MixedFunction , x: Solution):
    y = continuos_step(objective, x)
    
    if y.value > x.value:
        y = continuos_step(objective, x, direction = 1)
    
    p = next_swap(objective, y)

    return p

def change(objective: MixedFunction, x: Solution):
    x.permutation = change_permutation(x)
    x.continuos = random_continuos_reposition(x)
    x.value, x.c_value, x.p_value = objective.evaluate(x)
    pass


def solve(fobj: MixedFunction, x: Solution, maxeval=50):
        """
        Args:
            *change_nbg*: It is a callback function that will be call whenever a better solution is not found.
            *next*: It is a callback function that will be call when the next possible solution need to be constructed
        """
                
        num_evals = 0
        history = []
        samples = [[]]

        # Initial evaluation
        num_evals += 1
        history.append(x.value)
        samples[-1].append(x.value)

        while num_evals < maxeval:
            y = step(fobj, x)

            if y.value < x.value:
                x = y
                history.append(x.value)
                samples[-1].append(x.value)

            else:
                num_evals += 1
                if num_evals >= maxeval:
                    break
                change(fobj, x)
                samples.append([])
                history.append(x.value)
                samples[-1].append(x.value)

        return history, samples


objective_function = MixedFunction()
objective_function.calculate_parameters()
x = Solution(dimension=objective_function.continuos.dimension, permutation_size=objective_function.permutation.permutation_size)
x.value, x.c_value, x.p_value = objective_function.evaluate(x)

historic, samples = solve(objective_function, x)

print(objective_function.continuos.minimas)

plot_optimization_histories(
             [historic], 
             ["QUADRATIC"],
             best_possible=objective_function.continuos.minimas,
             output_path=f"historic.png")

plot_samples(samples, output="mixed.png", best_possible=objective_function.continuos.minimas)