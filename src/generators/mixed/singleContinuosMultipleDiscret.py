from src.generators.combinatorial.instance_generator import ZetaPermutation
from src.generators.continuos.instance_generator import QuadraticFunction
from src.generators.mixed.solution import Solution
from decimal import Decimal, getcontext
from typing import List


getcontext().prec = 100


class SingleContinuosMultipleDiscret:
    name = "zqs"
    discrets: List[ZetaPermutation]
    continuos: QuadraticFunction

    def calculate_parameters(self, continuos_dimension = 2, permutation_size = 5, continuos_minima = 2, permutation_minima = 2, distance = "K", difficult="E"):
        self.discrets: List[ZetaPermutation] = [] 
        
        for i in range(continuos_minima):
            self.discrets.append(ZetaPermutation())
            self.discrets[-1].caculate_parameters(permutation_size=permutation_size, 
                                                  number_of_minimas=permutation_minima, 
                                                  distance = distance, 
                                                  difficult= difficult)
        
        self.discrets.sort(key=lambda x: x.optima[0])

        self.continuos = QuadraticFunction(dimension=continuos_dimension, 
                                           numberOfLocalMinima=continuos_minima, 
                                           minima_proximity=10, 
                                           global_minimum=float(self.discrets[0].optima[0]))

        self.minimas = []

        for i, c in enumerate(self.continuos.minimas):
            for d in self.discrets[i].optima:
                self.minimas.append(d * Decimal(c))
        
        pass

    def transform(self, discret, continuos) -> Decimal:
        return Decimal(discret) * Decimal(continuos)

    def evaluate(self, x: Solution, c_value = None, p_value = None):
        c_value, i = self.continuos.evaluate_and_get_index(x.continuos)
        p_value, ln_value = self.discrets[i].evaluate(x.permutation)

        return self.transform(p_value, c_value), c_value, p_value, ln_value
    
    
    def log_info(self):
        print(f"{self.continuos.dimension}&{self.continuos.numberOfLocalMinima}&{self.discrets[0].permutation.permutation_size}&{self.discrets[0].permutation.number_of_optimas}&{self.discrets[0].permutation.distance}&{self.discrets[0].permutation.difficult}+")
        
        for i, c in enumerate(self.continuos.minimas):
            for j, p in enumerate(self.discrets[i].optima):
                print(f"{self.continuos.p_list[i]}&{self.discrets[i].permutation.consensus[j]}&{c:.6}&{p:.6}&{self.transform(p,c):.6}+")