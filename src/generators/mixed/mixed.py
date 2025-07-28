from src.generators.combinatorial.instance_generator import Permutation
from src.generators.continuos.instance_generator import QuadraticFunction
import numpy as np

class Solution:
    def __init__(self, dimension = 2, permutation_size = 5):
        self.continuos = [np.random.rand() for _ in range(dimension)]
        self.permutation = list(range(1, permutation_size + 1))
        self.value = 0.0
        self.c_value = 0.0
        self.p_value = 0.0
        pass

class MixedFunction:
    permutation: Permutation
    continuos: QuadraticFunction

    def calculate_parameters(self):
        self.permutation = Permutation(5, 5)
        self.permutation.calc_parameters_difficult()
        
        minimas = [np.divide(self.permutation.weights[i], self.permutation.zetas[i])  for i in range(len(self.permutation.consensus))]
        self.continuos = QuadraticFunction(numberOfLocalMinima=len(minimas), minimas=minimas)

    def evaluate(self, x: Solution, c_value = None, p_value = None):
        if p_value is None:
            p_value = self.permutation.evaluate(x.permutation)
        
        if c_value is None:
            c_value = self.continuos.evaluate(x.continuos)

        term0 = np.power(p_value - c_value, 2)

        return term0 + c_value, c_value, p_value


