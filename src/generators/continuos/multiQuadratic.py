import numpy as np
import sys
from typing import List
from decimal import Decimal

class MultiQuadratic:
    def __init__(self, dimension: int = 2, numberOfLocalMinima: int = 5, minimaProximity = 2, globalMinimum: float | Decimal = 1):
        self.dimension = dimension
        self.numberOfLocalMinima = numberOfLocalMinima
        self.minimaProximity = minimaProximity
        self.globalMinimum = globalMinimum
        self.generateFormAndFunctionOptima()
        pass

    def generateFormAndFunctionOptima(self):
        self.forms = [np.linalg.inv(self.generatePositiveDefiniteMatrix(self.dimension)) for _ in range(self.numberOfLocalMinima)]

        i = 1
        minimaDifference = np.sqrt(self.dimension)/(self.minimaProximity*(self.numberOfLocalMinima - 1))
        
        self.minima: List[float] = []
        for w in range(self.numberOfLocalMinima):
            self.minima.append(self.globalMinimum + minimaDifference*w)

        globalMinimumPosition = np.random.rand(self.dimension)
        self.minimaPositions = [globalMinimumPosition]

        # minima positions contraints
        # define the difference between two components value at some minima point
        spreadConstant = np.divide(1, self.numberOfLocalMinima)
        minimumDistanceInSpace = np.ones(self.dimension) * np.sqrt(self.dimension) * spreadConstant
        
        trials = 10000
        reshape = 1000

        while i < self.numberOfLocalMinima:
            yIsAValidMinimum = True
            trials = trials - 1

            # tries a number of times to positione a minima
            # if it is not possible, it redefines the forms and restart
            if trials <= 0:
    
                self.forms = [np.linalg.inv(self.generatePositiveDefiniteMatrix(self.dimension)) for _ in range(self.numberOfLocalMinima)]
                i = 1
                
                self.minimaPositions = [globalMinimumPosition]
                trials = 1000
                reshape -= 1

                if reshape <= 0:
                    raise Exception("The minima could not be posicioned using this dimension and number of optima combination.")
            
            y = np.random.rand(self.dimension)
            space: float =  minimumDistanceInSpace.T @ self.forms[i] @ minimumDistanceInSpace
                
            for j, pj in enumerate(self.minimaPositions):
                diff = y - pj

                # checking if the placed minima y is well spread in space
                if np.linalg.norm(diff) < np.sqrt(self.dimension) * spreadConstant:
                    yIsAValidMinimum = False
                    break
                
                # ensuring that the minima value is indeed minima on this position
                # also ensures that the component that defines a minimum in y has a ball
                # to create some attraction to it
                pjValueOnPositionY: float = diff.T @ self.forms[j] @ diff + self.minima[j]
                if pjValueOnPositionY < (self.minima[i] + space) :
                    yIsAValidMinimum = False
                    break
                
            if yIsAValidMinimum:
                self.minimaPositions.append(y)
                i += 1
                trials = 10000

        pass

    
    # generates a elipse matrix form
    def generatePositiveDefiniteMatrix(self, d):
        eigenvalues = np.linspace(0.2, 2.0, d)
        D = np.diag(eigenvalues)
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        pd_matrix = Q @ D @ Q.T 
        return pd_matrix

    def evaluate(self, x):
        value: float = sys.float_info.max

        for i, m in enumerate(self.minima):
            diff = x - self.minimaPositions[i]
            partial_value: float = diff.T @ self.forms[i] @ diff + m

            if value > partial_value:
                value = partial_value
            
        return value
    
    def evaluateAndGetComponentPosition(self, x):
        value: float = sys.float_info.max
        componentPosition = 0
        for i, m in enumerate(self.minima):
            diff = x - self.minimaPositions[i]
            partial_value: float = diff.T @ self.forms[i] @ diff + m
            if value > partial_value:
                value = partial_value
                componentPosition = i
            
        return value, componentPosition