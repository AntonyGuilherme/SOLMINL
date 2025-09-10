from src.generators.combinatorial.instance_generator import ZetaPermutation
from src.generators.continuos.instance_generator import QuadraticFunction
from src.generators.mixed.solution import Solution
from decimal import Decimal, getcontext
from src.generators.mixed.objectiveFunction import ObjectiveFunction
from typing import List


getcontext().prec = 100


class SingleContinuosMultipleDiscret(ObjectiveFunction):
    _discrets: List[ZetaPermutation]
    _continuos: QuadraticFunction

    def __init__(self):
        super().__init__()
        self.name = "SCMD"

    def defineDomains(self, continuosDimension: int = 2,
                        numberOfContinuosMinima: int = 2, 
                        discretDimension: int = 5, 
                        numberOfDiscretMinima: int = 2, 
                        distance = "K", 
                        difficult="E"):
        self._discrets = [] 
        
        for i in range(numberOfContinuosMinima):
            self._discrets.append(ZetaPermutation())
            self._discrets[-1].caculate_parameters(permutation_size=discretDimension, 
                                                  number_of_minimas=numberOfDiscretMinima, 
                                                  distance = distance, 
                                                  difficult= difficult)
        
        self._discrets.sort(key=lambda x: x.optima[0])

        self._continuos = QuadraticFunction(dimension=continuosDimension, 
                                           numberOfLocalMinima=numberOfContinuosMinima, 
                                           minima_proximity=10, 
                                           global_minimum=float(self._discrets[0].optima[0]))

        self.optima = []

        for i, c in enumerate(self._continuos.minimas):
            for d in self._discrets[i].optima:
                self.optima.append(self.transform(d,c))
        pass

    def transform(self, discret, continuos) -> Decimal:
        return Decimal(discret) * Decimal(continuos)

    def evaluate(self, x: Solution, fixContinuosValue: bool = False, fixDiscretValue: bool = False):
        x.c_value, i = self._continuos.evaluate_and_get_index(x.continuos)
        x.p_value, x.comp_p_value = self._discrets[i].evaluate(x.permutation)
        x.value = self.transform(x.c_value, x.p_value)
        pass
    
    
    def log(self):
        print(f"{self.name}&{self._continuos.dimension}&{self._continuos.numberOfLocalMinima}&{self._discrets[0].permutation.permutation_size}&{self._discrets[0].permutation.number_of_optimas}&{self._discrets[0].permutation.distance}&{self._discrets[0].permutation.difficult}+")
        
        for i, c in enumerate(self._continuos.minimas):
            for j, p in enumerate(self._discrets[i].optima):
                print(f"{self._continuos.p_list[i]}&{self._discrets[i].permutation.consensus[j]}&{c:.6}&{p:.6}&{self.transform(p,c):.6}+")