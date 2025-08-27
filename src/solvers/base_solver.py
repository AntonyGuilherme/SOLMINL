from src.generators.mixed.mixed import MixedFunction, Solution, MixIndependentFunction
import numpy as np
import copy
from .utils import plot_optimization_histories, plot_samples, plot_samples_with_ci
import os


np.random.seed(91)

def next_swap_close(f: MixedFunction, x: Solution):
    y = copy.deepcopy(x)
    n = len(y.permutation)

    for i in range(1, n):
        y.permutation[i], y.permutation[i-1] = y.permutation[i-1], y.permutation[i]
        y.value, y.c_value, y.p_value = f.evaluate(y, c_value=y.c_value)

        if y.value < x.value:
            return y
        else:
            # undoing the change to no copy the solution again
            y.permutation[i], y.permutation[i-1] = y.permutation[i-1], y.permutation[i]

    return y

def next_swap_invertion(f: MixedFunction, x: Solution):
    y = copy.deepcopy(x)
    n = len(y.permutation)
    # same computational cost as copy the invertion
    invertion = np.argsort(y.permutation)

    for i in range(1, n):
        j, k = invertion[i-1], invertion[i]
        invertion[i], invertion[i-1] = invertion[i-1], invertion[i]

        y.permutation[j], y.permutation[k] = y.permutation[k], y.permutation[j]
        y.value, y.c_value, y.p_value = f.evaluate(y, c_value=y.c_value)

        if y.value < x.value:
            return y
        else:
            # undoing the change to no copy the solution again
            y.permutation[j], y.permutation[k] = y.permutation[k], y.permutation[j]
            invertion[i], invertion[i-1] = invertion[i-1], invertion[i]

            assert (all(invertion == np.argsort(y.permutation)))

    return y

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
    steps = 20
    i = 0
    while y.value <= x.value and steps > i:
        i += 1
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


def step(objective: MixedFunction , x: Solution, next):
    # solve until rech a local minimum
    y = continuos_step(objective, x)
    
    p = next(objective, y)

    return p

def change(objective: MixedFunction, x: Solution):
    x.permutation = change_permutation(x)
    x.continuos = random_continuos_reposition(x)
    x.value, x.c_value, x.p_value = objective.evaluate(x)
    pass


def solve(fobj: MixedFunction, x: Solution, next, maxeval=50):
        """
        Args:
            *change_nbg*: It is a callback function that will be call whenever a better solution is not found.
            *next*: It is a callback function that will be call when the next possible solution need to be constructed
        """
                
        num_evals = 0
        history = []
        samples = [[]]
        samples_q = [[]]
        samples_p = [[]]

        # Initial evaluation
        num_evals += 1
        history.append(x.value)
        samples[-1].append(x.value)
        samples_q[-1].append(x.c_value)
        samples_p[-1].append(x.p_value)

        while num_evals <= maxeval:
            y = step(fobj, x, next)

            if y.value <= np.multiply(x.value, 0.999):
                x = y
                history.append(x.value)
                samples[-1].append(x.value)
                samples_q[-1].append(x.c_value)
                samples_p[-1].append(x.p_value)
                print([num_evals,x.continuos, x.permutation, x.c_value, x.p_value, x.value])

            else:
                num_evals += 1
                if num_evals > maxeval:
                    break
                change(fobj, x)
                samples.append([])
                samples_q.append([])
                samples_p.append([])
                history.append(x.value)
                samples[-1].append(x.value)
                samples_q[-1].append(x.c_value)
                samples_p[-1].append(x.p_value)

        return history, samples, samples_p, samples_q


dimensions = [2]
sizes = [5]
distances = ["K"]
nexts = [next_swap, next_swap_close, next_swap_invertion]
objectives = [MixIndependentFunction()]
number_of_evaluations_for_each_experiment = 3
number_of_continuos_minima = 2
number_of_permutation_minima = 2

for dimension in dimensions:
    for permutation_size in sizes:
            for distance in distances:
                for next in nexts:
                    for objective_function in objectives:
                        objective_function.calculate_parameters(continuos_dimension=dimension, 
                                                                permutation_size=permutation_size, 
                                                                continuos_minima=number_of_continuos_minima, 
                                                                permutation_minima=number_of_permutation_minima,
                                                                distance=distance)
                        
                        x = Solution(dimension=dimension, permutation_size=permutation_size)
                        x.value, x.c_value, x.p_value = objective_function.evaluate(x)

                        historic, samples, samples_p, samples_q = solve(objective_function, x, next=next, maxeval=number_of_evaluations_for_each_experiment)

                        # Create a folder for the current configuration
                        folder_name = f"{objective_function.name}_{dimension}_{permutation_size}_{next.__name__}"
                        os.makedirs(folder_name, exist_ok=True)

                        plot_optimization_histories(
                            [historic], 
                            ["QUADRATIC"],
                            best_possible=objective_function.minimas,
                            output_path=os.path.join(folder_name, f"historic.png"),
                            log=objective_function.log
                        )

                        plot_samples(
                            samples, 
                            output=os.path.join(folder_name, f"samples.png"), 
                            best_possible=objective_function.minimas,
                            log=objective_function.log
                        )

                        plot_samples_with_ci(
                            [samples_p, samples_q], 
                            "Quadratic and Permutation Evolution", 
                            subtitle=["Permutation", "Quadratic"],
                            log=objective_function.log,
                            output=os.path.join(folder_name, f"ie.png")
                        )