from src.generators.combinatorial.instance_generator import Permutation
import numpy as np
import copy
import matplotlib.pyplot as plt
from .utils import plot_optimization_histories, Solution, solve
import pandas as pd


np.random.seed(91)


size = 9
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

# x0, historic0 = solve(permutation, base, change_nbg=change, next=next_swap, maxeval=500)
# x1, historic1 = solve(permutation, base, change_nbg=change, next=next_swap_close, maxeval=500)
# x2, historic2 = solve(permutation, base, change_nbg=change, next=next_ivertion, maxeval=500)
best = permutation.evaluate(permutation.consensus[0])

# plot_optimization_histories(
#     [historic0, historic1, historic2], 
#     ["SWAP", "SWAP CLOSE", "SWAP INVERTION"], 
#     [best for _ in range(3)])



def results_table(results, best):
    # Calculate the difference from the best for each run and each method
    diffs = np.array([[abs(x.single_objective_value - best) for x in run] for run in results])
    avg = np.mean(diffs, axis=0)
    std = np.std(diffs, axis=0)

    # Prepare DataFrame for display, rounding to 4 decimals
    methods = ["SWAP", "SWAP CLOSE", "SWAP INVERTION"]
    df = pd.DataFrame({
        "Method": methods,
        "Avg. Dis. from Best": np.round(avg, 4),
        "Std Dev": np.round(std, 4)
    })

    # Plot as a table with improved style
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#40466e']*len(df.columns),
        cellColours=[['#f1f1f2']*len(df.columns) for _ in range(len(df))]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.2, 1.3)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f1f1f2')
            cell.set_edgecolor('#bbbbbb')

    plt.tight_layout()
    plt.savefig(f"hypothesis_{len(results)}_{len(results[0][0].solution)}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)

results = []

for i in range(20):
    x0, _ = solve(permutation, base, change_nbg=change, next=next_swap, maxeval=500)
    x1, _ = solve(permutation, base, change_nbg=change, next=next_swap_close, maxeval=500)
    x2, _ = solve(permutation, base, change_nbg=change, next=next_ivertion, maxeval=500)

    results.append([x0, x1, x2])

results_table(results, best)





