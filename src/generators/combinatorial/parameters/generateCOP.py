import numpy as np
from .Zvalue import Zvalue
from .LinearProg import LinearProg

# Zvalue, zi, _create_glpk_pyomo_model, MaxGO, MinGO, SimAB, select_func, LinearProg
# are defined above or in the same script.

def generateCOP(n, m, FileSigma, FileDistances, FileTheta, G, typeDistance, FileOut):
    # R's set.seed(a) using Sys.time() is for reproducibility if the same timestamp.
    # In Python, we can set seed for random if needed, but for actual timestamp use, it's fine.
    # Python's `random` module (used for `runif` equivalent) has its own seeding.
    # `np.random` has its own as well.

    # 1. Read input files
    # R's scan reads space-separated or newline-separated values.
    # For matrix, it reads into a vector then reshapes.

    # ConsensusPerms.txt
    try:
        # Assuming space-separated values, read into a list, then reshape.
        with open(FileSigma, 'r') as f:
            data = [int(val) for line in f for val in line.split()]
        ConsensusPerms = np.array(data).reshape(m, n) # n columns in R, so reshape to (m,n)
    except FileNotFoundError:
        print(f"Error: File not found: {FileSigma}")
        return
    except ValueError:
        print(f"Error: Could not parse {FileSigma} into integers. Check file format.")
        return

    # distancesKendall.txt
    try:
        with open(FileDistances, 'r') as f:
            distances = np.array([float(val) for val in f.read().split()])
    except FileNotFoundError:
        print(f"Error: File not found: {FileDistances}")
        return
    except ValueError:
        print(f"Error: Could not parse {FileDistances} into floats. Check file format.")
        return

    # Thetas.txt
    try:
        with open(FileTheta, 'r') as f:
            data2 = [float(val) for line in f for val in line.split()]
        # R's `thetas=matrix(data2,byrow=T,nrow=m)` means `m` rows and `n-1` columns.
        thetas = np.array(data2).reshape(m, n - 1)
    except FileNotFoundError:
        print(f"Error: File not found: {FileTheta}")
        return
    except ValueError:
        print(f"Error: Could not parse {FileTheta} into floats. Check file format.")
        return

    # 2. Calculate normalization terms (zeta)
    zeta = Zvalue(n, m, thetas, typeDistance)

    # 3. Solve the Linear Programming problem
    # LinearProg returns the solved Pyomo model
    res_LinearProg_model = LinearProg(n, m, thetas, distances, G, zeta)
    
    # 4. Extract results (equivalent to R's getColsPrimIptGLPK(res_LinearProg))
    # In Pyomo, we iterate over the variables to get their values.
    # R's `x` corresponds to the primal solution values for the columns (decision variables).
    x_solution_values = np.array([res_LinearProg_model.x[i].value for i in range(m)])

    # 5. Write results to output file
    # R's `sink(FileOut)` redirects output, `cat` writes formatted strings.
    with open(FileOut, 'w') as f_out:
        for i in range(m):
            # Format ConsensusPerms: remove brackets, handle spacing
            perms_str = np.array_str(ConsensusPerms[i, :])
            formatted_perms = ' '.join(perms_str[1:-1].strip().split())
                
            # Format x[i]/zeta[i]
            objective_value_str = f"{x_solution_values[i] / zeta[i]:.6f}"
            # FIX: Directly format thetas[i,:] without using np.array_str first
            # This avoids potential unexpected newlines from np.array_str
            formatted_thetas = ' '.join(f"{val:.6f}" for val in thetas[i, :])
                
            # Write the complete line
            f_out.write(f"{formatted_perms}   {objective_value_str}  {formatted_thetas} \n")

    print(f"Results written to {FileOut}")

    return True