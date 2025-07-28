import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def solve(fobj, x, change_nbg, next, maxeval=50):
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
        history.append(x.single_objective_value)

        best = Solution()
        best.single_objective_value = x.single_objective_value
        best.solution = np.array(x.solution)
        size = len(best.solution)
        samples[-1].append(x.single_objective_value)

        while num_evals < maxeval:
            y = next(fobj, x, size)

            if y.single_objective_value < x.single_objective_value:
                x = copy.deepcopy(y)
                history.append(x.single_objective_value)
                samples[-1].append(x.single_objective_value)

            else:
                num_evals += 1
                if num_evals >= maxeval:
                    break
                change_nbg(fobj, x)
                samples.append([])
                history.append(x.single_objective_value)
                samples[-1].append(x.single_objective_value)

        return history, samples


def plot_samples(samples, confidence_interval=95, title="None", output="sample.png", best_possible=None):
    # Pad samples to the max length
    max_len = max(len(s) for s in samples)
    padded_samples = []
    for s in samples:
        pad_size = max_len - len(s)
        if pad_size > 0:
            s = np.concatenate([s, np.full(pad_size, np.nan)])
        padded_samples.append(s)

    # Convert to DataFrame: rows=timepoints, columns=samples
    samples_matrix = np.array(padded_samples).T  # shape: (max_len, n_samples)
    df = pd.DataFrame(samples_matrix, columns=[f'sample_{i}' for i in range(len(padded_samples))])
    df['timepoint'] = np.arange(max_len)
    df_long = df.melt(id_vars='timepoint', var_name='sample', value_name='value')

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 9))
    sns.lineplot(data=df_long, x='timepoint', y='value', errorbar=("ci", confidence_interval))
    plt.xlabel('Function Step')
    plt.ylabel('Objective Function Value (log scale)')
    plt.yscale('log')
    plt.title(f'Mean and Dispersion Across Samples of {title}')

    # Add best possible line if provided
    if best_possible is not None:
        plt.axhline(y=best_possible, linestyle='--', color='red', linewidth=1.5, alpha=0.7)
        plt.text(0, best_possible, f'Best Possible: {best_possible:.4f}', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

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
        plt.plot(range(len(history)), history, label=f'{label} (Best: {np.min(history):.4f})', marker='o', markersize=4)

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
                plt.text(0, target, f'Target {i+1}: {target:.4f}', fontsize=9, color=f'C{i}')

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

class Solution:
    def __init__(self):
        self.single_objective_value = 0
        self.solution = np.array([])
        pass