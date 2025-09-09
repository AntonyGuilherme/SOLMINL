from src.generators.mixed.mixed import MixedFunction, Solution, MixIndependentFunction, QuadraticLandscapeByMallows
import numpy as np
import copy
from .utils import plot_optimization_histories, plot_samples, plot_samples_with_ci
from decimal import Decimal, getcontext
import os

getcontext().prec = 100

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

def next_swap(f : MixedFunction, x : Solution, num_evals:int, log = True):
    y = copy.deepcopy(x)
    k = x
    n = len(y.permutation)

    for i in range(n - 1):
        for j in range(i+1, n):
            y.permutation[i], y.permutation[j] = y.permutation[j], y.permutation[i]
            y.value, y.c_value, y.p_value, y.comp_p_value = f.evaluate(y, c_value=y.c_value)
            
            if y.comp_p_value > k.comp_p_value:
                k = copy.deepcopy(y)
                if log:
                    k.print(num_evals, "P")
            else:
                # undoing the change to no copy the solution again
                y.permutation[i], y.permutation[j] = y.permutation[j], y.permutation[i]
    
    return k

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
        ev_x1, _, _,_ = fobj.evaluate(y, p_value=y.p_value)
        y.continuos = x2
        ev_x2, _, _, _ = fobj.evaluate(y, p_value=y.p_value)
        grad[i] = (ev_x1 - ev_x2) / Decimal(2 * epsilon)
    return grad

def continuos_step(objective: MixedFunction, x: Solution, num_evals: int, step =1e-3, num_steps = 100, log = True):
    y = copy.deepcopy(x)
    k = copy.deepcopy(x)
    n_steps = 0
    while n_steps < num_steps:
        n_steps += 1
        grad = numerical_gradient(objective, k)
        k.continuos = k.continuos - step * grad
        k.continuos = np.clip(k.continuos, 0.0, 1.0)
        k.value, k.c_value, k.p_value, k.comp_p_value = objective.evaluate(k, p_value=k.p_value)
        
        if (k.value + Decimal(1e-5)) > y.value:
            break
        else:
            y = copy.deepcopy(k)
    
    if log:
        y.print(num_evals, "C")

    return y

def random_continuos_reposition(x:Solution):
    """
    Generate a point in [0,1]^D that is at least `min_dist` away from `x`,
    by adding/subtracting noise dimension-wise until the norm is sufficient.
    """
    D = len(x.continuos)
    delta = np.zeros_like(x.continuos)
    dist = np.sqrt(D) * 0.50

    while np.linalg.norm(delta) < dist:
        candidate = np.random.random(D)
        delta = candidate - x.continuos

    return candidate

def step(objective: MixedFunction , x: Solution, next, num_evals, strategy, log:bool) -> Solution:
    
    if log:
        x.print(num_evals, objective.first_step)

    if strategy == "C":
        y = continuos_step(objective, x, num_evals= num_evals, log= log)
        p = next(objective, x, num_evals = num_evals, log = log)
    else:
        y = next(objective, x, num_evals, log = log)
        p = continuos_step(objective, y, num_evals = num_evals, log = log)

    return p

def change(objective: MixedFunction, x: Solution):
    x.permutation = change_permutation(x)
    x.continuos = random_continuos_reposition(x)
    x.value, x.c_value, x.p_value, x.comp_p_value = objective.evaluate(x)
    pass

def select_solver_strategy(f: MixedFunction, x: Solution, next) -> str:
    _, samplesC, _, _ = solve(f, x, next, maxeval= 10, strategy= "C", log = False)
    _, samplesP, _, _ = solve(f, x, next,  maxeval= 10, strategy= "P", log = False)

    results = []
    for i, sc in enumerate(samplesC):
        results.append((sc[-1],"C"))
        results.append((samplesP[i][-1],"P"))

    results.sort(key=lambda x: x[0])

    continuos_strategy = 0
    discret_strategy = 0
    for r in results:
        if r[1] == "P":
            discret_strategy += 1
        else:
            continuos_strategy += 1
        
        if continuos_strategy >= 10:
            return "C"
        if discret_strategy >= 10:
            return "P"

def solve(fobj: MixedFunction, x: Solution, next, strategy = "C", maxeval=50, log = True):
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
            y = step(fobj, x, next, num_evals, strategy= strategy, log=log)

            if y.value < x.value or y.comp_p_value > x.comp_p_value:
                x = y
                history.append(x.value)
                samples[-1].append(x.value)
                samples_q[-1].append(x.c_value)
                samples_p[-1].append(x.p_value)

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

def define_strategy_and_solve(fobj: MixedFunction, x: Solution, next, maxeval=50):
    strategy = select_solver_strategy(fobj, x, next)

    return solve(fobj, x, next, strategy, maxeval, log = True)

objective_functions = {
    'mif': MixIndependentFunction(),
    'qlm': QuadraticLandscapeByMallows()
}

strategies = {
    'n': next_swap, 
    'nc': next_swap_close,
    'nsi': next_swap_invertion
}

def run(continuos_dimension: int, permutation_size: int, difficulty: str, distance: str, continuos_minima: int, next:str, objective:str, attempts: int = 30):
    next_str = strategies[next]
    objective_function = objective_functions[objective]

    objective_function.calculate_parameters(continuos_dimension=continuos_dimension, 
                                                        permutation_size=permutation_size, 
                                                        continuos_minima=continuos_minima, 
                                                        permutation_minima=permutation_size,
                                                        distance=distance,
                                                        difficult=difficulty)
    
    objective_function.log_info()
                        
    x = Solution(dimension=continuos_dimension, permutation_size=permutation_size)
    x.value, x.c_value, x.p_value, x.comp_p_value = objective_function.evaluate(x)

    define_strategy_and_solve(objective_function, x, next=next_str, maxeval=attempts)
    pass

dimensions = [2]
sizes = [5]
distances = ["K"]
nexts = [next_swap]
objectives = [QuadraticLandscapeByMallows()]
number_of_evaluations_for_each_experiment = 100
number_of_continuos_minima = 2
number_of_permutation_minima = sizes[0]

for dimension in dimensions:
    for permutation_size in sizes:
            for distance in distances:
                for next in nexts:
                    for objective_function in objectives:
                        objective_function.calculate_parameters(continuos_dimension=dimension, 
                                                                permutation_size=permutation_size, 
                                                                continuos_minima=number_of_continuos_minima, 
                                                                permutation_minima=number_of_permutation_minima,
                                                                distance=distance,
                                                                difficult="H")
                        
                        objective_function.log_info()
                        x = Solution(dimension=dimension, permutation_size=permutation_size)
                        x.value, x.c_value, x.p_value, x.comp_p_value = objective_function.evaluate(x)
                        
                        historic, samples, samples_p, samples_q = define_strategy_and_solve(objective_function, x, next=next, maxeval=number_of_evaluations_for_each_experiment)

                        #Create a folder for the current configuration
                        folder_name = f"{objective_function.name}_{dimension}_{permutation_size}_{next.__name__}{distance}"
                        os.makedirs(folder_name, exist_ok=True)



                        plot_optimization_histories(
                            [historic], 
                            ["QUADRATIC"],
                            best_possible=objective_function.minimas,
                            output_path=os.path.join(folder_name, f"historic.png"),
                            log=False
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