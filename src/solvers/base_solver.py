from src.generators.mixed import Solution, MixOfIndependentSpaces, SingleDiscretMultipleContinuos, SingleContinuosMultipleDiscret, ObjectiveFunction
import numpy as np
import copy
from .utils import plot_optimization_histories, plot_samples, plot_samples_with_ci
from decimal import Decimal, getcontext
from typing import List, Dict
import os

getcontext().prec = 100

np.random.seed(91)

class Logger:
    historic: List[float]
    samples: List[List[float]]
    keepHistory: bool

    def __init__(self, keepHistory = True):
        self.keepHistory = keepHistory
        self.historic = []
        self.samples = []
        self.temp_historic = []
        self.temp_samples = []
        self.solutions = []
        pass

    def print(self, x: Solution, numberOfEvaluations: int, owner: str):
        self.temp_historic.append(x.value)
        self.temp_samples.append(x.value)
        self.solutions.append([copy.deepcopy(x), owner, numberOfEvaluations])
        pass

    def flush(self):
        if self.keepHistory:
            self.samples.append(self.temp_samples)
            self.historic.extend(self.temp_historic)
        
        for x,o,n in self.solutions:
            x.print(n, o)

        self.clean()
        pass

    def clean(self):
        self.temp_historic = []
        self.temp_samples = []
        self.solutions = []
        pass

def mostImprovedSwap(f: ObjectiveFunction, x : Solution, num_evals:int, logger: Logger = None):
    y = copy.deepcopy(x)
    k = x
    n = len(y.permutation)
    improved = True

    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i+1, n):
                y.permutation[i], y.permutation[j] = y.permutation[j], y.permutation[i]
                f.evaluate(y, fixContinuosValue=True)
                
                if  y.value < k.value or (y.value == k.value and y.comp_p_value > k.comp_p_value):
                    k = copy.deepcopy(y)
                    improved = True
                    if logger is not None:
                        logger.print(k, num_evals, "P")
                else:
                    # undoing the change to no copy the solution again
                    y.permutation[i], y.permutation[j] = y.permutation[j], y.permutation[i]
    
    return k

def change_permutation(x: Solution):
    return np.random.permutation(x.permutation)

def numerical_gradient(fobj: ObjectiveFunction, x: Solution, epsilon=1e-6):
    grad = np.zeros_like(x.continuos)
    y = copy.deepcopy(x)
    for i in range(len(x.continuos)):
        x1 = x.continuos.copy()
        x2 = x.continuos.copy()
        x1[i] += epsilon
        x2[i] -= epsilon
        y.continuos = x1
        fobj.evaluate(y, fixDiscretValue=True)
        ev_x1 = y.value
        y.continuos = x2
        fobj.evaluate(y, fixDiscretValue=True)
        ev_x2 = y.value
        grad[i] = (ev_x1 - ev_x2) / Decimal(2 * epsilon)
    return grad

def continuosGradientStep(objective: ObjectiveFunction, x: Solution, num_evals: int, step =1e-4, logger: Logger = None):
    y = copy.deepcopy(x)
    k = copy.deepcopy(x)
    n_steps = 0
    plot_space = 10**3
    while True:
        n_steps += 1
        grad = numerical_gradient(objective, k)
        k.continuos = k.continuos - step * grad
        k.continuos = np.clip(k.continuos, 0.0, 1.0)
        objective.evaluate(k, fixDiscretValue= True)
        
        if (k.value + Decimal(1e-7)) > y.value:
            break
        else:
            y = copy.deepcopy(k)
    
        if logger is not None and n_steps == plot_space:
            logger.print(y, num_evals, "C")
            n_steps = 0

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

def step(objective: ObjectiveFunction, x: Solution, next, num_evals, logger: Logger = None) -> Solution:  
    if logger is not None:
            logger.print(x, num_evals, "C")

    y = continuosGradientStep(objective, x, num_evals= num_evals, logger = logger)
    p = next(objective, y, num_evals = num_evals, logger = logger)

    return p

def change(objective: ObjectiveFunction, x: Solution):
    x.permutation = change_permutation(x)
    x.continuos = random_continuos_reposition(x)
    objective.evaluate(x)
    pass

def solve(fobj: ObjectiveFunction, x: Solution, next, maxeval=50, logger: Logger = None):
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

        while num_evals < maxeval:
            y = step(fobj, x, next, num_evals, logger = logger)

            if y.value < x.value or (y.value == x.value and y.comp_p_value > x.comp_p_value):
                x = y

                if logger is not None:
                    logger.flush()

            else:
                num_evals += 1
                if num_evals > maxeval:
                    break
                change(fobj, x)
                logger.clean()

objective_functions: Dict[str,ObjectiveFunction] = {
    'mif': MixOfIndependentSpaces(),
    'scmd': SingleContinuosMultipleDiscret(),
    'sdmc': SingleDiscretMultipleContinuos()
}

strategies = {
    'n': mostImprovedSwap
}

def run(continuos_dimension: int, permutation_size: int, difficulty: str, distance: str, continuos_minima: int, next:str, objective:str, attempts: int = 30):
    next_str = strategies[next]
    objective_function = objective_functions[objective]

    objective_function.defineDomains(   continuosDimension=continuos_dimension, 
                                        discretDimension=permutation_size, 
                                        numberOfContinuosMinima=continuos_minima, 
                                        numberOfDiscretMinima=permutation_size,
                                        distance=distance,
                                        difficult=difficulty)

    objective_function.log()
                        
    x = Solution(dimension=continuos_dimension, permutation_size=permutation_size)
    objective_function.evaluate(x)

    solve(objective_function, x, next=next_str, maxeval=attempts, logger = Logger())
    pass


# dimensions = [2]
# sizes = [5, 10]
# distances = ["K", "H", "C"]
# nexts = [mostImprovedSwap]
# objectives: List[ObjectiveFunction] = [
#     SingleDiscretMultipleContinuos(), 
#     SingleContinuosMultipleDiscret(), 
#     MixOfIndependentSpaces()
#     ]
# number_of_evaluations_for_each_experiment = 5

# for dimension in dimensions:
#     for permutation_size in sizes:
#             for distance in distances:
#                 for next in nexts:
#                     for objective_function in objectives:
#                         objective_function.defineDomains(continuosDimension=dimension, 
#                                                                 discretDimension=permutation_size, 
#                                                                 numberOfContinuosMinima=dimension, 
#                                                                 numberOfDiscretMinima=permutation_size,
#                                                                 distance=distance,
#                                                                 difficult="E")
                        
#                         objective_function.log()
#                         x = Solution(dimension=dimension, permutation_size=permutation_size)
#                         objective_function.evaluate(x)


#                         logger = Logger(True)
#                         solve(objective_function, x, next=next, logger=logger, maxeval=number_of_evaluations_for_each_experiment)

#                         #Create a folder for the current configuration
#                         folder_name = f"{objective_function.name}_{dimension}_{permutation_size}_{next.__name__}{distance}"
#                         os.makedirs(folder_name, exist_ok=True)


#                         plot_optimization_histories(
#                             [logger.historic], 
#                             ["QUADRATIC"],
#                             best_possible=objective_function.optima,
#                             output_path=os.path.join(folder_name, f"historic.png"),
#                             log=True
#                         )

#                         plot_samples(
#                             logger.samples, 
#                             output=os.path.join(folder_name, f"samples.png"), 
#                             best_possible=objective_function.optima,
#                             log=True
#                         )