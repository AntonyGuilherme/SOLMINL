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
        kmax = int(np.abs(len(x.solution) / 2)) + 1
        history = []

        # Initial evaluation
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

class Solution:
    def __init__(self):
        self.single_objective_value = 0
        self.solution = np.array([])
        pass