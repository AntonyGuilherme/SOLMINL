import numpy as np
import sys

class QuadraticFunction:

    def __init__(self, dimension: int = 2, numberOfLocalMinima: int = 10, minimas = None):
        self.dimension = dimension
        self.numberOfLocalMinima = numberOfLocalMinima

        self.p_list = [np.random.rand(dimension) for _ in range(numberOfLocalMinima)]
        self.B_inv_list = [np.linalg.inv(self.generate_positive_definite_matrix(dimension)) for _ in range(numberOfLocalMinima)]
        
        if minimas == None:
            self.minimas = [np.random.rand() for _ in range(self.numberOfLocalMinima)]
        else:
            self.minimas = minimas
        pass

    # Generate symmetric positive definite matrices B_i
    def generate_positive_definite_matrix(self, d):
        A = np.random.randn(d, d)
        return A @ A.T + np.eye(d) * 0.1  # Ensure positive definiteness

    # Define the function f_quad(x)
    def evaluate(self, x):
        value: float = sys.float_info.max
        for i in range(self.numberOfLocalMinima):
            diff = x - self.p_list[i]
            partial_value: float = diff.T @ self.B_inv_list[i] @ diff + self.minimas[i]
            if value > partial_value:
                value = partial_value
            
        return value
