from src.generators.discret.multiMallowsDiscret import NormalizedDiscret
from src.generators.continuos.multiQuadratic import MultiQuadratic
from src.generators.mixed.solution import Solution
from src.generators.mixed.objectiveFunction import ObjectiveFunction
from decimal import getcontext
import numpy as np

getcontext().prec = 100

class MixOfIndependentSpaces(ObjectiveFunction):
    _discret: NormalizedDiscret
    _continuos: MultiQuadratic
    
    def __init__(self):
        super().__init__()
        self.name = "MIF"

    def defineDomains(self, continuosDimension: int = 2, numberOfContinuosMinima: int = 2, discretDimension: int = 5, numberOfDiscretMinima: int = 2, distance = "K", difficult="E") -> None:
        self._discret = NormalizedDiscret()
        self._discret.createParameters(discretDimension, numberOfDiscretMinima, distance, difficult)
        
        self._continuos = MultiQuadratic(dimension=continuosDimension, 
                                            numberOfLocalMinima= numberOfContinuosMinima, 
                                            minimaProximity=10)

        self.optima = []

        for minimum_i in self._discret.optima:
            for minimum_j in self._continuos.minima:
                self.optima.append(self.transform(minimum_i, minimum_j))
        pass

    def evaluate(self, x: Solution, fixContinuosValue: bool = False, fixDiscretValue: bool = False):
        if not fixDiscretValue:
            x.p_value, x.comp_p_value = self._discret.evaluate(x.permutation)
        
        if not fixContinuosValue:
            x.c_value = self._continuos.evaluate(x.continuos)
        
        x.value = self.transform(x.p_value, x.c_value)

    pass
    
    def log(self):
        print(f"{self.name}&{self._continuos.dimension}&{self._continuos.numberOfLocalMinima}&{self._discret._discret.discretDimension}&{self._discret._discret.numberOfMaxima}&{self._discret._discret.distance}&{self._discret._discret.difficult}+")
        for p, p_optimum in enumerate(self._discret.optima):
            for c, c_optimum in enumerate(self._continuos.minima):
                print(f"{self._continuos.minimaPositions[c]}&{self._discret._discret.consensus[p]}&{c_optimum:.6}&{p_optimum:.6}&{self.transform(p_optimum,c_optimum):.6}+")

        pass
                
