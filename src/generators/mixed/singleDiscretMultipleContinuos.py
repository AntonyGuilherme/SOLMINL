from src.generators.combinatorial.instance_generator import ZetaPermutation
from src.generators.continuos.multiQuadratic import MultiQuadratic
from src.generators.mixed.solution import Solution
from decimal import Decimal, getcontext
from typing import Dict
from src.generators.mixed.objectiveFunction import ObjectiveFunction

getcontext().prec = 100

class SingleDiscretMultipleContinuos(ObjectiveFunction):
    _discret: ZetaPermutation
    _continuos: Dict[int, MultiQuadratic]

    def __init__(self):
        super().__init__()
        self.name = "SDMC"


    def defineDomains(self, continuosDimension: int = 2,
                        numberOfContinuosMinima: int = 2, 
                        discretDimension: int = 5, 
                        numberOfDiscretMinima: int = 2, 
                        distance = "K", 
                        difficult="E"):
        self._discret = ZetaPermutation()
        self._discret.caculate_parameters(discretDimension, numberOfDiscretMinima, distance, difficult)

        self._continuos = {}
        self.optima = []
        
        for i, optimum in enumerate(self._discret.optima):
            self._continuos[i] = MultiQuadratic(dimension=continuosDimension, 
                                                  numberOfLocalMinima=numberOfContinuosMinima, 
                                                  minimaProximity=10, 
                                                  globalMinimum=float(optimum))

        for i, p in enumerate(self._discret.optima):
            for c in self._continuos[i].minima:
                self.optima.append(self.transform(p,c))
        
        pass

    def evaluate(self, x: Solution, fixContinuosValue: bool = False, fixDiscretValue: bool = False) -> None:
        x.p_value, x.comp_p_value, i = self._discret.evaluate_and_get_index(x.permutation)
        x.c_value = self._continuos[i].evaluate(x.continuos)
        x.value = self.transform(x.p_value, x.c_value)

        pass
    
    
    def log(self) -> None:
        print(f"{self._continuos[0].dimension}&{self._continuos[0].numberOfLocalMinima}&{self._discret.permutation.permutation_size}&{self._discret.permutation.number_of_optimas}&{self._discret.permutation.distance}&{self._discret.permutation.difficult}+")
        for p, p_optimum in enumerate(self._discret.optima):
            for c, c_optimum in enumerate(self._continuos[p].minima):
                print(f"{self._continuos[p].minimaPositions[c]}&{self._discret.permutation.consensus[p]}&{c_optimum:.6}&{p_optimum:.6}&{self.transform(p_optimum,c_optimum):.6}+")
