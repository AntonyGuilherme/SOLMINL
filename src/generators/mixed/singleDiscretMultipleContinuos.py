from src.generators.combinatorial.instance_generator import ZetaPermutation
from src.generators.continuos.instance_generator import QuadraticFunction
from src.generators.mixed.solution import Solution
from decimal import Decimal, getcontext
from typing import Dict

getcontext().prec = 100

class SingleDiscretMultipleContinuos:
    name = "qlm"
    permutation: ZetaPermutation
    continuos: Dict[int, QuadraticFunction]

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
