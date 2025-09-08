from src.solvers.base_solver import run
import sys

script_name = sys.argv[0]

continuos_dimension = int(sys.argv[1])
continuos_minima = int(sys.argv[2])
permutation_size = int(sys.argv[3])
distance = sys.argv[4]
difficulty = sys.argv[5]
strategie = sys.argv[6]
objective_function = sys.argv[7]

run(continuos_dimension=continuos_dimension, 
    continuos_minima = continuos_minima, 
    permutation_size=permutation_size, 
    distance=distance,
    difficulty=difficulty,
    next=strategie,
    objective=objective_function,
    attempts=2)