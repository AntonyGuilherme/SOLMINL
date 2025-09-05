import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_optimization_histories(histories, titles=None, best_possible=None, output_path="historic.png", log=False):
    """
    Plot each optimization history in a separate subplot (not in the same graph, but in the same figure).
    Each subplot will have its corresponding minimas line(s) from best_possible.
    Each point is colored by its label: 'C' (blue) or 'P' (orange).

    Args:
        histories (list of list): Each element is a list of tuples (label, value), e.g., ('C', value) or ('P', value).
        titles (list of str, optional): Titles for each history.
        best_possible (list of list or list of float, optional): List of minimas for each history.
    """
    color_map = {'C': 'blue', 'P': 'orange'}
    n_hist = len(histories)
    fig, axes = plt.subplots(n_hist, 1, figsize=(16, 5 * n_hist), sharex=True)
    if n_hist == 1:
        axes = [axes]

    for i, (history, ax) in enumerate(zip(histories, axes)):
        x = list(range(len(history)))
        y = [val for _, val in history]
        labels = [lbl for lbl, _ in history]
        # Draw lines between points, colored by the label of the next point
        for j in range(1, len(history)):
            ax.plot([x[j-1], x[j]], [y[j-1], y[j]], color=color_map.get(labels[j], 'gray'), alpha=0.7, zorder=1)
        # Plot points
        c_x = [xi for xi, lbl in zip(x, labels) if lbl == 'C']
        c_y = [yi for yi, lbl in zip(y, labels) if lbl == 'C']
        p_x = [xi for xi, lbl in zip(x, labels) if lbl == 'P']
        p_y = [yi for yi, lbl in zip(y, labels) if lbl == 'P']
        ax.scatter(c_x, c_y, color='blue', marker='o', s=40, label='C', zorder=2)
        ax.scatter(p_x, p_y, color='orange', marker='o', s=40, label='P', zorder=2)

        label = f'Optimization {i+1}'
        if titles and i < len(titles):
            label = titles[i]
        # Draw only best and worst minima lines for this plot, both in light gray, just put the value
        if best_possible:
            minimas_for_plot = best_possible[i] if isinstance(best_possible[i], (list, np.ndarray)) else [best_possible[i]]
            minimas_for_plot = [m for m in minimas_for_plot if m is not None]
            if minimas_for_plot:
                best = min(minimas_for_plot)
                worst = max(minimas_for_plot)
                # Plot best minima
                ax.axhline(y=best, linestyle='--', color='lightgray', linewidth=2, alpha=0.8, zorder=0)
                ax.text(0, best, f'{best:.4f}', fontsize=10, color='gray', va='bottom', ha='left', alpha=0.8)
                # Plot worst minima (only if different from best)
                if worst != best:
                    ax.axhline(y=worst, linestyle=':', color='lightgray', linewidth=2, alpha=0.8, zorder=0)
                    ax.text(0, worst, f'{worst:.4f}', fontsize=10, color='gray', va='top', ha='left', alpha=0.8)
        ax.set_xlabel('Evaluations')
        ax.set_ylabel('Objective Value (log scale)')
        if log:
            ax.set_yscale('log')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, which='both', axis='y')
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_samples(samples_list, titles=None, output_path="sample.png", best_possible=None, log=False):
    """
    Plot each set of samples in a separate subplot (not in the same graph, but in the same figure).
    Each subplot will have its corresponding best_possible line(s).
    Lines connecting two points have the color of the next point ('C': blue, 'P': orange).

    Args:
        samples_list (list of list): Each element is a list of samples (each sample is a list of tuples ('C' or 'P', value)).
        titles (list of str, optional): Titles for each subplot.
        best_possible (list of list or list of float, optional): List of best possible values for each subplot.
    """
    color_map = {'C': 'blue', 'P': 'orange'}
    n_samples = len(samples_list)
    fig, axes = plt.subplots(n_samples, 1, figsize=(16, 5 * n_samples), sharex=True)
    if n_samples == 1:
        axes = [axes]

    for i, (samples, ax) in enumerate(zip(samples_list, axes)):
        for sample in samples:
            x = list(range(len(sample)))
            y = [val for _, val in sample]
            labels = [lbl for lbl, _ in sample]
            # Draw lines between points, colored by the label of the next point
            for j in range(1, len(sample)):
                ax.plot([x[j-1], x[j]], [y[j-1], y[j]], color=color_map.get(labels[j], 'gray'), alpha=0.7, zorder=1)
            # Plot points
            c_x = [xi for xi, lbl in zip(x, labels) if lbl == 'C']
            c_y = [yi for yi, lbl in zip(y, labels) if lbl == 'C']
            p_x = [xi for xi, lbl in zip(x, labels) if lbl == 'P']
            p_y = [yi for yi, lbl in zip(y, labels) if lbl == 'P']
            ax.scatter(c_x, c_y, color='blue', marker='o', s=40, label='C', zorder=2)
            ax.scatter(p_x, p_y, color='orange', marker='o', s=40, label='P', zorder=2)

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
            bp_for_plot = sorted([bp for bp in bp_for_plot if bp is not None])
            n_bp = len(bp_for_plot)
            if n_bp > 10:
                indices = np.linspace(0, n_bp - 1, 10, dtype=int)
                bp_for_plot = [bp_for_plot[idx] for idx in indices]
            for j, bp in enumerate(bp_for_plot):
                ax.axhline(y=bp, linestyle='--', color=f'C{j}', linewidth=1.5, alpha=0.7)
                ax.text(0, bp, f'Best Possible {j+1}: {bp:.4f}', fontsize=10, color=f'C{j}')

        ax.grid(True, which='both', axis='y')
        # Only show legend once per subplot
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

files_names = os.listdir("./test")
ran = {}
for file_name in files_names:
    with open("./test/" + file_name) as result:
        
        all_text = result.read()
        lines = all_text.split('+')

        historic = []
        continuos_historic = []
        permutation_historic = []
        samples = {}
        continuos_samples = {}
        permutations_samples = {}
        
        params = None
        minimas = []
        continuos_minima = []
        permutation_minima = []
        p_minima = {}
        c_minima = {}

        content = None

        should_proccess = True
        for line in lines:
            if params is None:
                params = line.split('&')
                number_of_continuos_minima = int(params[1])
                number_of_permutation_minima = int(params[3])

            elif len(minimas) < (number_of_continuos_minima * number_of_permutation_minima):
                line_split = line.split('&')
                minimas.append(float(line_split[-1].strip()))
                
                if len(permutation_minima) < number_of_permutation_minima and line_split[-2].strip() not in p_minima:
                    permutation_minima.append(float(line_split[-2].strip()))
                    p_minima[line_split[-2].strip()] = True

                if len(continuos_minima) < number_of_continuos_minima and line_split[-3].strip() not in c_minima:
                    continuos_minima.append(float(line_split[-3].strip()))
                    c_minima[line_split[-3].strip()] = True
            elif line.strip():
                content = line.replace('\n', '').replace('\r', '').split('&')

                if content[0] not in samples:
                        samples[content[0]] = []
                        continuos_samples[content[0]] = []
                        permutations_samples[content[0]] = []
                        
                historic.append((content[1], float(content[-1].strip())))
                continuos_historic.append(('C',float(content[-3].strip())))
                permutation_historic.append(('P',float(content[-2].strip())))
                samples[content[0]].append((content[1], float(content[-1].strip())))
                continuos_samples[content[0]].append(('C',float(content[-3].strip())))
                permutations_samples[content[0]].append(('P',float(content[-2].strip())))

        plot_optimization_histories(
                [historic, continuos_historic, permutation_historic], 
                titles=[f"[{params[0]} {params[3]}] {params[1]} x {params[3]} Mixed", 
                        f"[{params[0]}] {params[1]} Continuos",
                        f"[{params[3]}] {params[3]} Permutation"
                        ],
                        output_path=f"{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[5]}.png",
                        best_possible=[minimas, continuos_minima, permutation_minima], 
                        log=True)

            
        plot_samples(
                [list(samples.values())[:5], list(continuos_samples.values())[:1], list(permutations_samples.values())[:1]], 
                titles=[f"[{params[0]} {params[3]}] {params[1]} x {params[3]} Mixed", 
                        f"[{params[0]}] {params[1]} Continuos",
                        f"[{params[3]}] {params[3]} Permutation"
                        ],
                        output_path=f"samples_{params[0]}_{params[1]}_{params[2]}_{params[3]}_{params[5]}.png",
                        best_possible=None, 
                        log=False)




        
                

                