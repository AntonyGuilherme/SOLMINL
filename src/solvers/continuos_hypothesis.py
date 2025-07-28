from src.generators.continuos.instance_generator import QuadraticFunction
from .utils import Solution, solve, plot_optimization_histories, plot_samples
import numpy as np
import copy

np.random.seed(78)

def numerical_gradient(fobj, x_vec, epsilon=1e-6):
    grad = np.zeros_like(x_vec)
    for i in range(len(x_vec)):
        x1 = x_vec.copy()
        x2 = x_vec.copy()
        x1[i] += epsilon
        x2[i] -= epsilon
        grad[i] = (fobj.evaluate(x1) - fobj.evaluate(x2)) / (2 * epsilon)
    return grad


def continuos_step(objective, x, d):
    y = copy.deepcopy(x)
    grad = numerical_gradient(objective, y.solution)
    step_size = 0.05
    y.solution = y.solution - step_size * grad
    y.solution = np.clip(y.solution, 0.0, 1.0)
    y.single_objective_value = objective.evaluate(y.solution)

    return y

def random_continuos_reposition(f, x, epslon=1e-6):
    """
    Generate a point in [0,1]^D that is at least `min_dist` away from `x`,
    by adding/subtracting noise dimension-wise until the norm is sufficient.
    """
    D = len(x.solution)
    delta = np.zeros_like(x.solution)

    while np.linalg.norm(delta) < epslon:
        direction = np.random.choice([-1, 1], size=D)
        step = np.random.uniform(0.05, 0.2, size=D)
        delta = direction * step
        candidate = x.solution + delta

        # Project to [0,1]
        candidate = np.clip(candidate, 0.0, 1.0)
        delta = candidate - x.solution

    x.solution = candidate
    x.single_objective_value = f.evaluate(x.solution)


objective = QuadraticFunction(numberOfLocalMinima=4)

base = Solution()
base.solution = np.zeros(objective.dimension)
base.single_objective_value = objective.evaluate(base.solution)
historic, samples = solve(objective, base, change_nbg=random_continuos_reposition, next = continuos_step)

print(objective.minimas)

plot_optimization_histories(
             [historic], 
             ["QUADRATIC"],
             best_possible=[min(objective.minimas)],
             output_path=f"historic_q.png")

plot_samples(samples)

