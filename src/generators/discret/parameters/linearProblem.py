import math
import numpy as np
import random 
from pyomo.environ import *

# This functions defines the constraints
def _create_glpk_pyomo_model(m, zeta, thetas_min_val):
    model = ConcreteModel()

    # Decision Variables (m variables, 0-indexed)
    model.x = Var(range(m), domain=NonNegativeReals, bounds=(0, None))
    for i in range(m):
        # defines the upper bound
        model.x[i].setub(50 * zeta[i])

    # Constraints (m constraints, 0-indexed)

    py_ia = [0, m-1, 0]
    for i in range(1, m-1):
        py_ia.extend([i, i])

    if m > 1: # Add the last 'm' from R's ia if m > 1
        py_ia.append(m-1)

    py_ja = []
    for i in range(m): # R's 1:m
        py_ja.extend([i, i]) # Each 'i' is repeated twice for column index

    py_ar = []
    py_ar.append(1/zeta[0]) # R: 1/zeta[1]
    py_ar.append(-1/zeta[0]) # R: -1/zeta[1] (This is for the term in the last constraint involving x[0])
    
    # This loop generates coefficients for rows 2 to m-1 (R indexing)
    for i in range(1, m-1): # Python indices 1 to m-2 correspond to R's 2 to m-1
        py_ar.append(-1/zeta[i])
        py_ar.append(1/zeta[i])

    py_ar.append(-1/zeta[m-1]) # R: -1/zeta[m] (This is a coefficient for the last constraint involving x[m-1])
    py_ar.append((2 - math.exp(-thetas_min_val)) / zeta[m-1]) # R: (2-exp(-min(thetas)))/zeta[m]

    # Now, set up the constraints dynamically based on the (py_ia, py_ja, py_ar) triplets
    # Initialize expressions for each row
    row_terms = [[] for _ in range(m)]
    for idx in range(len(py_ar)):
        row_idx = py_ia[idx]
        col_idx = py_ja[idx]
        coefficient = py_ar[idx]
        row_terms[row_idx].append(coefficient * model.x[col_idx])

    # Define bounds for each row
    rlower_values = [random.uniform(0.0001, 0.0002) for _ in range(m)]
    rupper_values = [100.0] * m

    model.constraints = Constraint(range(m))
    for i in range(m):
        expr = sum(row_terms[i]) if row_terms[i] else 0
        model.constraints[i] = (rlower_values[i], expr, rupper_values[i])

    return model

# define the maximation objective function
# this problem defines the easy set of problems
def MaxGO(m, zeta, thetas):
    model = _create_glpk_pyomo_model(m, zeta, np.min(thetas)) # Pass the min value from the entire thetas matrix

    obj_expr = (1/zeta[0]) * model.x[0]
    if m >= 2:
        obj_expr -= (1/zeta[1]) * model.x[1]

    model.objective = Objective(expr=obj_expr, sense=maximize)
    return model

# define the minimization objective function
# this problem defines the difficulty set of problem for the discret domain
def MinGO(distances, m, zeta, thetas):
    model = _create_glpk_pyomo_model(m, zeta, np.min(thetas))

    obj_expr = (1/zeta[0]) * model.x[0]
    if m > 1:
      for rest_ind in range(1, m): # Python indices 1 to m-1
          max_theta_row = np.max(thetas[rest_ind,:])
          obj_coeff = -math.exp(-max_theta_row * (distances[rest_ind] + 1)) / ((m - 1) * zeta[rest_ind])
          obj_expr += obj_coeff * model.x[rest_ind]

    model.objective = Objective(expr=obj_expr, sense=minimize)
    return model


# defining objective function
def select_func(G, distances, m, zeta, thetas):
    if G == 'max':
        return MaxGO(m, zeta, thetas)
    elif G == 'min':
        return MinGO(distances, m, zeta, thetas)
    else:
        raise ValueError("Invalid G parameter. Must be 'max', 'min', or 'sim'.")


def defineDiscretFunctionParameters(n, m, thetas, distances, G, zeta):
    linearProblem = select_func(G, distances, m, zeta, thetas)

    # Solve the model using GLPK
    solver = SolverFactory('glpk')
    results = solver.solve(linearProblem,  tee=False)

    if (results.solver.status != SolverStatus.ok) or \
       (results.solver.termination_condition != TerminationCondition.optimal):
        print(f"Warning: Solver did not find an optimal solution. Status: {results.solver.status}, Termination: {results.solver.termination_condition}")

    return linearProblem