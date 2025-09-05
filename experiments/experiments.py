import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import re

def plot_optimization_histories(histories, titles=None, best_possible=None, output_path="historic.png", log=False, colors=None):
    """
    Plot each optimization history in a separate subplot (not in the same graph, but in the same figure).
    Each subplot will have its corresponding minimas line(s) from best_possible.

    Args:
        histories (list of list): Each element is a list of objective values (history).
        titles (list of str, optional): Titles for each history.
        best_possible (list of list or list of float, optional): List of minimas for each history.
    """

    n_hist = len(histories)
    fig, axes = plt.subplots(n_hist, 1, figsize=(16, 5 * n_hist), sharex=True)
    if n_hist == 1:
        axes = [axes]

    # Generate random colors for each history
    random.seed(42)
    color_list = []
    for _ in range(n_hist):
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        color_list.append(color)

    for i, (history, ax) in enumerate(zip(histories, axes)):
        label = f'Optimization {i+1}'
        if titles and i < len(titles):
            label = titles[i]
        color = color_list[i]
        ax.plot(range(len(history)), history, label=f'{label} (Best: {np.min(history):.4f})', marker='o', markersize=4, color=color)
        # Draw minimas lines for this plot
        if best_possible:
            minimas_for_plot = best_possible[i] if isinstance(best_possible[i], (list, np.ndarray)) else [best_possible[i]]
            for j, target in enumerate(minimas_for_plot):
                if target is not None:
                    ax.axhline(y=target, linestyle='--', color=f'C{j}', linewidth=1, alpha=0.6)
                    ax.text(0, target, f'Target {j+1}: {target:.4f}', fontsize=9, color=f'C{j}')
        ax.set_xlabel('Evaluations')
        ax.set_ylabel('Objective Value (log scale)')
        if log:
            ax.set_yscale('log')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, which='both', axis='y')
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_samples(samples_list, titles=None, output_path="sample.png", best_possible=None, log=False):

    """
    Plot each set of samples in a separate subplot (not in the same graph, but in the same figure).
    Each subplot will have its corresponding best_possible line(s).

    Args:
        samples_list (list of list): Each element is a list of samples (each sample is a list of objective values).
        titles (list of str, optional): Titles for each subplot.
        best_possible (list of list or list of float, optional): List of best possible values for each subplot.
    """
    n_samples = len(samples_list)
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 5 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    for i, (samples, ax) in enumerate(zip(samples_list, axes)):
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
        df = pd.DataFrame(samples_matrix, columns=[f'sample_{j}' for j in range(len(padded_samples))])
        df['timepoint'] = np.arange(max_len)
        df_long = df.melt(id_vars='timepoint', var_name='sample', value_name='value')

        sns.set(style="whitegrid")
        sns.lineplot(data=df_long, x='timepoint', y='value', hue='sample', legend=False, alpha=0.7, ax=ax)
        ax.set_xlabel('Function Step')
        ax.set_ylabel('Objective Function Value (log scale)')
        if log:
            ax.set_yscale('log')
        label = f'Samples {i+1}'
        if titles and i < len(titles):
            label = titles[i]
        ax.set_title(f'All Samples of {label}', fontsize=13, fontweight='bold')

        # Add best possible lines if provided as a list
        if best_possible is not None:
            bp_for_plot = best_possible[i] if isinstance(best_possible[i], (list, np.ndarray)) else [best_possible[i]]
            # Sort and select up to 10 equally distributed values
            bp_for_plot = sorted([bp for bp in bp_for_plot if bp is not None])
            n_bp = len(bp_for_plot)
            if n_bp > 10:
                indices = np.linspace(0, n_bp - 1, 10, dtype=int)
                bp_for_plot = [bp_for_plot[idx] for idx in indices]
            for j, bp in enumerate(bp_for_plot):
                ax.axhline(y=bp, linestyle='--', color=f'C{j}', linewidth=1.5, alpha=0.7)
                ax.text(0, bp, f'Best Possible {j+1}: {bp:.4f}', fontsize=10, color=f'C{j}')

        ax.grid(True, which='both', axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

files_names = os.listdir("../new_results")
print(files_names)
ran = {}
for file_name in files_names:
    with open("../new_results/" + file_name) as result:
        historic = []
        continuos_historic = []
        permutation_historic = []
        samples = {}
        continuos_samples = {}
        permutations_samples = {}
        
        params = ''
        minimas = []
        continuos_minima = []
        permutation_minima = []
        p_minima = {}
        c_minima = {}

        content = None

        should_proccess = True
        for line in result:
            if not params:
                params = line.split(',')
                number_of_continuos_minima = int(params[1])
                number_of_permutation_minima = int(params[3])
                
                if params[5].strip() != 'n' or params[4].strip() != 'K' or line in ran:
                    should_proccess = False
                    break

                ran[line] = True

            elif len(minimas) < (number_of_continuos_minima * number_of_permutation_minima):
                line_split = line.split(' ')
                minimas.append(float(line_split[-1].strip()))
                
                if len(permutation_minima) < number_of_permutation_minima and line_split[-2].strip() not in p_minima:
                    permutation_minima.append(float(line_split[-2].strip()))
                    p_minima[line_split[-2].strip()] = True

                if len(continuos_minima) < number_of_continuos_minima and line_split[-3].strip() not in c_minima:
                    continuos_minima.append(float(line_split[-3].strip()))
                    c_minima[line_split[-3].strip()] = True
            else:
                
                if re.match(r'^\d+\s*\[', line) is not None:
                    if  content is not  None:
                        match = content.split(" ")
                        if match[0] not in samples:
                            samples[match[0]] = []
                            continuos_samples[match[0]] = []
                            permutations_samples[match[0]] = []
                        
                            
                        historic.append(float(match[-1].strip()))
                        permutation_historic.append(float(match[-2].strip()))
                        continuos_historic.append(float(match[-3].strip()))
                        samples[match[0]].append(float(match[-1].strip()))
                        continuos_samples[match[0]].append(float(match[-3].strip()))
                        permutations_samples[match[0]].append(float(match[-2].strip()))

                    content = ''
         
                
                content += line.replace('\n', '').replace('\r', '')
                #print(content)
                
                

        if should_proccess:
            # plot_optimization_histories(
            #     [historic, continuos_historic, permutation_historic], 
            #     titles=[f"[{params[0]} {params[3]}] {params[1]} x {params[3]} Mixed", 
            #             f"[{params[0]}] {params[1]} Continuos",
            #             f"[{params[3]}] {params[3]} Permutation"
            #             ],
            #             output_path=f"{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[5]}.png",
            #             best_possible=[minimas, continuos_minima, permutation_minima], 
            #             log=True)
            
            plot_samples(
                [list(samples.values()), list(continuos_samples.values()), list(permutations_samples.values())], 
                titles=[f"[{params[0]} {params[3]}] {params[1]} x {params[3]} Mixed", 
                        f"[{params[0]}] {params[1]} Continuos",
                        f"[{params[3]}] {params[3]} Permutation"
                        ],
                        output_path=f"samples_{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[5]}.png",
                        best_possible=[minimas, continuos_minima, permutation_minima], 
                        log=True)
            
            print(len(continuos_samples))

        
                

                