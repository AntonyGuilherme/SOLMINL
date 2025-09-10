from src.generators.mixed import Solution
from typing import List
from decimal import Decimal

class ObjectiveFunction:
    name: str
    optima: List[Decimal]

    def defineDomains(self, 
                        continuosDimension: int = 2,
                        numberOfContinuosMinima: int = 2, 
                        discretDimension: int = 5, 
                        numberOfDiscretMinima: int = 2, 
                        distance = "K", 
                        difficult="E") -> None:
        pass

    def evaluate(self, x: Solution, fixContinuosValue: bool = False, fixDiscretValue: bool = False) -> None:
        pass

    def transform(self, discret: float | Decimal, continuos: float | Decimal) -> Decimal:
        return Decimal(discret) * Decimal(continuos)

    def log(self) -> None:
        pass
