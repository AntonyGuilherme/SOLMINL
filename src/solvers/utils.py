import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import t

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

def plot_samples_with_ci(samples_list, title="None", subtitle = [],  output="sample_ci.png", best_possible=None, ci=95):
    """
    Plot the average and confidence interval of multiple sets of samples using seaborn.

    Args:
        samples_list (list of list of lists): Each element is a list of samples (each sample is a list of values).
        title (str): Plot title.
        output (str): Output file path.
        best_possible (list of float, optional): Best possible values for each group.
        ci (float): Confidence interval percentage (default 95).
    """
    plt.figure(figsize=(16, 9))
    sns.set(style="whitegrid")

    all_df = []
    for idx, samples in enumerate(samples_list):
        max_len = max(len(s) for s in samples)
        for sample_id, s in enumerate(samples):
            pad_size = max_len - len(s)
            if pad_size > 0:
                s = np.concatenate([s, np.full(pad_size, np.nan)])
            df = pd.DataFrame({
                "step": np.arange(max_len),
                "value": s,
                "group": subtitle[idx]
            })
            all_df.append(df)
    df_long = pd.concat(all_df, ignore_index=True)

    sns.lineplot(
        data=df_long,
        x="step",
        y="value",
        hue="group",
        errorbar=("ci", ci),
        estimator="mean",
        err_style="band"
    )

    plt.xlabel('Function Step')
    plt.ylabel('Objective Function Value (log scale)')
    #plt.yscale('log')
    plt.title(f'Average Samples with {ci}% CI: {title}')

    if best_possible is not None:
        for i, bp in enumerate(best_possible):
            if bp is not None:
                plt.axhline(y=bp, linestyle='--', color=f'C{i}', linewidth=1.5, alpha=0.7)
                plt.text(0, bp, f'Best Possible {i+1}: {bp:.4f}', fontsize=10, color=f'C{i}')

    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def plot_samples(samples, title="None", output="sample.png", best_possible=None):
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
    sns.lineplot(data=df_long, x='timepoint', y='value', hue='sample', legend=False, alpha=0.7)
    plt.xlabel('Function Step')
    plt.ylabel('Objective Function Value (log scale)')
    #plt.yscale('log')
    plt.title(f'All Samples of {title}')

    # Add best possible lines if provided as a list
    if best_possible is not None:
        for i, bp in enumerate(best_possible):
            if bp is not None:
                plt.axhline(y=bp, linestyle='--', color=f'C{i}', linewidth=1.5, alpha=0.7)
                plt.text(0, bp, f'Best Possible {i+1}: {bp:.4f}', fontsize=10, color=f'C{i}')

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
    #plt.yscale('log')
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