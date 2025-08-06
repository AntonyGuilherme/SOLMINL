from src.generators.combinatorial.instance_generator import Permutation
from src.generators.continuos.instance_generator import QuadraticFunction
import numpy as np
from typing import Dict, List

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
    name = "mf"
    log = True

    def calculate_parameters(self, continuos_dimension = 2, permutation_size = 5, number_of_minimas = 5, distance = "K"):
        self.permutation = Permutation(permutation_size, number_of_minimas, distance)
        self.permutation.calc_parameters_difficult()
        
        self.minimas = [self.permutation.evaluate(consensus)  for consensus in self.permutation.consensus]
        self.continuos = QuadraticFunction(dimension=continuos_dimension, numberOfLocalMinima=len(self.minimas), minimas=self.minimas)

    def evaluate(self, x: Solution, c_value = None, p_value = None):
        if p_value is None:
            p_value = self.permutation.evaluate(x.permutation)
        
        if c_value is None:
            c_value = self.continuos.evaluate(x.continuos)

        term0 = np.power(p_value - c_value, 2)

        return term0 + c_value, c_value, p_value

class QuadraticLandscapeByMallows:
    name = "qlm"
    log = False
    permutation: Permutation
    continuos: Dict[int, QuadraticFunction]

    def calculate_parameters(self, continuos_dimension = 2, permutation_size = 5, number_of_minimas = 5, distance = "K"):
        self.permutation = Permutation(permutation_size, number_of_minimas, distance)
        self.permutation.calc_parameters_easy()

        self.continuos = {}
        self.minimas = []
        for consensus in self.permutation.consensus:
            value, i = self.permutation.evaluate_and_get_index(consensus)
            minimas_ = [value + 2 * np.random.rand()  for _ in range(number_of_minimas-1)]
            minimas_.append(value)
            self.continuos[i] = QuadraticFunction(dimension=continuos_dimension, numberOfLocalMinima=len(minimas_), minimas=minimas_)
            self.minimas.append(value**2)

    def evaluate(self, x: Solution, c_value = None, p_value = None):
        p_value, i = self.permutation.evaluate_and_get_index(x.permutation)
        
        c_value = self.continuos[i].evaluate(x.continuos)

        return np.multiply(c_value, p_value), c_value, p_value