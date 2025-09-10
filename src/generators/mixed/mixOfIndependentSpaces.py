from src.generators.combinatorial.instance_generator import ZetaPermutation
from src.generators.continuos.instance_generator import QuadraticFunction
from src.generators.mixed.solution import Solution
from decimal import Decimal, getcontext
import numpy as np

getcontext().prec = 100

class MixOfIndependentSpaces:
    permutation: ZetaPermutation
    continuos: QuadraticFunction
    name = "mif"
    log = True
    first_step = "C"

    def calculate_parameters(self, continuos_dimension = 2, permutation_size = 5, continuos_minima = 2, permutation_minima = 2, distance = "K", difficult="E"):
        self.permutation = ZetaPermutation()
        self.permutation.caculate_parameters(permutation_size, permutation_minima, distance, difficult)
        
        self.continuos = QuadraticFunction(dimension=continuos_dimension, numberOfLocalMinima= continuos_minima, minima_proximity=10)

        self.minimas = []

        for minimum_i in self.permutation.optima:
            for minimum_j in self.continuos.minimas:
                self.minimas.append(np.multiply(minimum_i, Decimal(minimum_j)))
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
                print(f"{self.continuos.p_list[c]}&{self.permutation.permutation.consensus[p]}&{c_optimum:.6}&{p_optimum:.6}&{self.transform(p_optimum,c_optimum):.6}&{np.log(float(p_optimum))}+")
                
