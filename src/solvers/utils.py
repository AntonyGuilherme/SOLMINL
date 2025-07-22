import numpy as np
import copy
import matplotlib.pyplot as plt

def solve(fobj, x, change_nbg, next, maxeval=50):
        """
        Args:
            *change_nbg*: It is a callback function that will be call whenever a better solution is not found.
            *next*: It is a callback function that will be call when the next possible solution need to be constructed
        """
                
        num_evals = 0
        history = []

        # Initial evaluation
        num_evals += 1
        history.append(x.single_objective_value)

        best = Solution()
        best.single_objective_value = x.single_objective_value
        best.solution = np.array(x.solution)
        size = len(best.solution)

        while num_evals < maxeval:
            y = next(fobj, x, size)
        
            num_evals += 1

            if y.single_objective_value > x.single_objective_value:
                x = copy.deepcopy(y)
                history.append(x.single_objective_value)

            else:
                change_nbg(fobj, x)
                num_evals += 1
                history.append(x.single_objective_value)

        return history

def plot_optimization_histories(histories, titles=None, best_possible=None, output_path="historic.png"):
    """
    Plot multiple optimization histories on the same graph using lines (log scale on y-axis).

    Args:
        histories (list of list): Each element is a list of objective values (history).
        titles (list of str, optional): Labels for each history.
        best_possible (list of float, optional): Best possible values for each history.
    """
    plt.figure(figsize=(16, 9))

    for i, history in enumerate(histories):
        label = f'Optimization {i+1}'
        if titles and i < len(titles):
            label = titles[i]
        plt.plot(range(len(history)), history, label=f'{label} (Best: {np.max(history):.4f})', marker='o', markersize=4)

    plt.xlabel('Evaluations')
    plt.ylabel('Objective Value (log scale)')
    plt.yscale('log')
    plt.title('Optimization Histories', fontsize=13, fontweight='bold')
    plt.grid(True, which='both', axis='y')

    # Target lines if provided
    if best_possible:
        for i, target in enumerate(best_possible):
            if target is not None:
                plt.axhline(y=target, linestyle='--', color=f'C{i}', linewidth=1, alpha=0.6)
                plt.text(0, target * 1.01, f'Target {i+1}: {target:.4f}', fontsize=9, color=f'C{i}')

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)

class Solution:
    def __init__(self):
        self.single_objective_value = 0
        self.solution = np.array([])
        pass