from src.generators.combinatorial.instance_generator import Permutation, ZetaPermutation
from src.generators.continuos.instance_generator import QuadraticFunction
import numpy as np
from typing import Dict
from decimal import Decimal, getcontext

getcontext().prec = 100

class Solution:
    def __init__(self, dimension = 2, permutation_size = 5):
        self.continuos = [np.random.rand() for _ in range(dimension)]
        self.permutation = list(range(1, permutation_size + 1))
        self.value = 0.0
        self.c_value = 0.0
        self.p_value = 0.0
        self.comp_p_value = 0.0
        pass

    def print(self, ref,owner):
        print(f"{ref}&{owner}&{self.continuos}&{self.permutation}&{float(self.c_value):.6f}&{float(self.p_value):.6f}&{float(self.value):.6f}+")
        pass

class MixIndependentFunction:
    permutation: ZetaPermutation
    continuos: QuadraticFunction
    name = "mif"
    log = True
    first_step = "C"

    def calculate_parameters(self, continuos_dimension = 2, permutation_size = 5, continuos_minima = 2, permutation_minima = 2, distance = "K", difficult="E"):
        self.permutation = ZetaPermutation()
        self.permutation.caculate_parameters(permutation_size, permutation_minima, distance, difficult)
        
        permutation_minima = [self.permutation.evaluate(consensus)  for consensus in self.permutation.permutation.consensus]
        self.continuos = QuadraticFunction(dimension=continuos_dimension, numberOfLocalMinima= continuos_minima, minima_proximity=10)

        self.minimas = []

        for i, minimum_i in enumerate(permutation_minima):
            for j , minimum_j in enumerate(self.continuos.minimas):
                self.minimas.append(np.multiply(minimum_i[0], Decimal(minimum_j)))
        pass

    def transform(self, discret, continuos) -> Decimal:
        return Decimal(discret) * Decimal(continuos)

    def evaluate(self, x: Solution, c_value = None, p_value = None):
        comp = x.comp_p_value
        if p_value is None:
            p_value, comp = self.permutation.evaluate(x.permutation)
        
        if c_value is None:
            c_value = self.continuos.evaluate(x.continuos)

        return self.transform(p_value, c_value), c_value, p_value, comp
    
    def log_info(self):
        print(f"{self.continuos.dimension}&{self.continuos.numberOfLocalMinima}&{self.permutation.permutation.permutation_size}&{self.permutation.permutation.number_of_optimas}&{self.permutation.permutation.distance}&{self.permutation.permutation.difficult}+")
        for p, p_optimum in enumerate(self.permutation.optima):
            for c, c_optimum in enumerate(self.continuos.minimas):
                print(f"{self.continuos.p_list[c]}&{self.permutation.permutation.consensus[p]}&{c_optimum:.6}&{p_optimum:.6}&{self.transform(p_optimum,c_optimum):.6}+")
                


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
    permutation: ZetaPermutation
    continuos: Dict[int, QuadraticFunction]
    first_step = "P"

    def calculate_parameters(self, continuos_dimension = 2, permutation_size = 5, continuos_minima = 2, permutation_minima = 2, distance = "K", difficult="E"):
        self.permutation = ZetaPermutation()
        self.permutation.caculate_parameters(permutation_size, permutation_minima, distance, difficult)

        self.continuos = {}
        self.minimas = []
        
        for i, optimum in enumerate(self.permutation.optima):
            self.continuos[i] = QuadraticFunction(dimension=continuos_dimension, 
                                                  numberOfLocalMinima=continuos_minima, 
                                                  minima_proximity=10, 
                                                  global_minimum=float(optimum))
        
        self.minimas = []

        for i, p in enumerate(self.permutation.optima):
            for c in self.continuos[i].minimas:
                self.minimas.append(p * Decimal(c))
        
        pass

    def transform(self, discret, continuos) -> Decimal:
        return Decimal(discret) * Decimal(continuos)

    def evaluate(self, x: Solution, c_value = None, p_value = None):
        p_value, ln_value, i = self.permutation.evaluate_and_get_index(x.permutation)
        
        c_value = self.continuos[i].evaluate(x.continuos)

        return self.transform(p_value, c_value), c_value, p_value, ln_value
    
    
    def log_info(self):
        print(f"{self.continuos[0].dimension}&{self.continuos[0].numberOfLocalMinima}&{self.permutation.permutation.permutation_size}&{self.permutation.permutation.number_of_optimas}&{self.permutation.permutation.distance}&{self.permutation.permutation.difficult}+")
        for p, p_optimum in enumerate(self.permutation.optima):
            for c, c_optimum in enumerate(self.continuos[p].minimas):
                print(f"{self.continuos[p].p_list[c]}&{self.permutation.permutation.consensus[p]}&{c_optimum:.6}&{p_optimum:.6}&{self.transform(p_optimum,c_optimum):.6}+")