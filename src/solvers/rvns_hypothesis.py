from src.generators.combinatorial.instance_generator import Permutation
import numpy as np
import copy
import matplotlib.pyplot as plt

def plot_optimization_histories(histories, titles=None, best_possible=None, output_path="historic.png"):
    """
    Plot multiple optimization histories side by side using lines.

    Args:
        histories (list of list): Each element is a list of objective values (history).
        titles (list of str, optional): Titles for each subplot.
        best_possible (list of float, optional): Best possible values for each history.
    """
    n = len(histories)
    plt.figure(figsize=(6 * n, 5))

    for i, history in enumerate(histories):
        plt.subplot(1, n, i + 1)
        plt.plot(range(len(history)), history, label='Objective Value', color=f'C{i}', marker='o', markersize=4)
        plt.xlabel('Evaluations')
        plt.ylabel('Objective Value')
        best_found = np.max(history)
        title = f'Optimization {i+1}\nBest Found: {best_found:.4f}'
        if titles and i < len(titles):
            title = f'{titles[i]}\nBest Found: {best_found:.4f}'
        plt.title(title, fontsize=12, fontweight='bold')
        plt.grid(True)

        # Target line if provided
        if best_possible and i < len(best_possible) and best_possible[i] is not None:
            plt.axhline(y=best_possible[i], linestyle='--', color='gray', linewidth=1)
            plt.text(0, best_possible[i] + 0.01, f'Target: {best_possible[i]:.4f}',
                     fontsize=9, color='gray')

        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)


def solve(fobj, x, change_nbg, next, maxeval=50):
        """
        Args:
            *change_nbg*: It is a callback function that will be call whenever a better solution is not found.
            *next*: It is a callback function that will be call when the next possible solution need to be constructed
        """
                
        num_evals = 0
        kmax = int(np.abs(len(x.solution) / 2)) + 1
        history = []

        # Initial evaluation
        x.single_objective_value = fobj.evaluate(x.solution)
        num_evals += 1
        history.append(x.single_objective_value)

        best = Solution()
        best.single_objective_value = x.single_objective_value
        best.solution = np.array(x.solution)

        while num_evals < maxeval:
            k = 1
            improved = False

            while k <= kmax and num_evals < maxeval:
                # Generate a neighbor by swapping k pairs
                for _ in range(k):
                    y = next(fobj, x)
        
                num_evals += 1

                if y.single_objective_value > x.single_objective_value:
                    x = copy.deepcopy(y)
                    k = 1
                    improved = True

                    if x.single_objective_value > best.single_objective_value:
                        best.single_objective_value = x.single_objective_value
                        best.solution = np.array(x.solution)
                else:
                    k += 1

                history.append(x.single_objective_value)

            if not improved:
                change_nbg(fobj, x)
                num_evals += 1
                history.append(x.single_objective_value)

        return best, history

class Solution:
    def __init__(self):
        self.single_objective_value = 0
        self.solution = np.array([])
        pass


np.random.seed(91)


size = 7
base = Solution()
base.solution = np.random.permutation(size)

print(base.solution)

permutation = Permutation(len(base.solution), len(base.solution)**2) 

permutation.calc_parameters_difficult()

consensus = permutation.consensus[0]
print(f"{consensus} {permutation.evaluate(consensus)}")

def next_swap(f, x):
    y = copy.deepcopy(x)
    idx = np.random.choice(len(x.solution), size=2, replace=False)
    y.solution[idx[0]], y.solution[idx[1]] = y.solution[idx[1]], y.solution[idx[0]]
    y.single_objective_value = f.evaluate(y.solution)

    return y

def next_swap_close(f, x):
    y = copy.deepcopy(x)
    idx = np.random.randint(1, len(x.solution)-1)
    y.solution[idx], y.solution[idx+1] = y.solution[idx + 1], y.solution[idx]
    y.single_objective_value = f.evaluate(y.solution)

    return y

def next_ivertion(f, x):
    y = copy.deepcopy(x)
    # i represents a value from 1 to the permutation size
    i = np.random.randint(1, len(x.solution) + 1)
    # j represents the position that matches with this element i
    inverse = np.argsort(y.solution)
    j = inverse[i-1]
    k = inverse[j]

    y.solution[j], y.solution[k] = y.solution[k], y.solution[j]

    y.single_objective_value = f.evaluate(y.solution)

    return y

def change(f, x):
    x.solution = np.random.permutation(x.solution)
    x.single_objective_value = f.evaluate(x.solution)

x0, historic0 = solve(permutation, base, change_nbg=change, next=next_swap, maxeval=500)
x1, historic1 = solve(permutation, base, change_nbg=change, next=next_swap_close, maxeval=500)
x2, historic2 = solve(permutation, base, change_nbg=change, next=next_ivertion, maxeval=500)
best = permutation.evaluate(permutation.consensus[0])

plot_optimization_histories([historic0, historic1, historic2], ["SWAP", "SWAP CLOSE", "SWAP INVERTION"], [best for _ in range(3)])
