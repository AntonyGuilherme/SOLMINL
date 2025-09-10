import numpy as np
from src.generators.discret.distances import kendall, caylley, hamming
from .parameters import normalizationConstants
from .parameters import linearProblem
from decimal import Decimal, getcontext
from typing import Tuple, List

getcontext().prec = 100

import matplotlib
matplotlib.use('Agg')

def get_ranges_by_distance(distance: str, permutation_size: int):
    small_range = None
    normal_range = None

    if distance == "K":
        small_range = (np.log(permutation_size - .5), 1.5*np.log(permutation_size - .5))
        normal_range = (2 * np.log(permutation_size - 1), 3*np.log(permutation_size - 1))
    else:
        small_range = (2*np.log(permutation_size - 0.33), 2.5*np.log(permutation_size - 1))
        normal_range = (3*np.log(permutation_size - 1), 6*np.log(permutation_size - 1))
    
    return small_range, normal_range

def _generate_difficult_thetas(permutation_size, number_of_optimas, distance):
    small_range, normal_range = get_ranges_by_distance(distance, permutation_size)
    num_cols = permutation_size - 1
    thetas_matrix = np.zeros((number_of_optimas, num_cols))

    for i in range(number_of_optimas):
        if i == 1:
            theta_i = np.random.uniform(*small_range)
        else:
            theta_i = np.random.uniform(*normal_range)
        
        thetas_matrix[i, :] = theta_i

    return thetas_matrix

def _generate_easy_thetas(permutation_size, number_of_optimas, distance):
    small_range, normal_range = get_ranges_by_distance(distance, permutation_size)
    num_cols = permutation_size - 1 
    thetas_matrix = np.zeros((number_of_optimas, num_cols))

    for i in range(number_of_optimas):
        if i == 0:
            theta_i = np.random.uniform(*small_range)
        else:
            theta_i = np.random.uniform(*normal_range)

        thetas_matrix[i,:] = theta_i
        
    return thetas_matrix

def _calculate_kendall_tau_distances(permutations):
        reference_permutation = permutations[0]
        return [kendall(reference_permutation, perm) for perm in permutations]

def _calculate_cayley_distances(permutations):
    reference_permutation = permutations[0]
    return [caylley(reference_permutation, perm) for perm in permutations]

def _calculate_hamming_distances(permutations):
    reference_permutation = permutations[0]
    return [hamming(reference_permutation, perm) for perm in permutations]

def _create_permutations(permutation_size: int, number_of_optimas: int, distance: str):
    consensus_permutations = []
    created_consensus = {}

    dist_cal = None
    min_dist = None
    if distance == "K":
        dist_cal = kendall
        min_dist = 1
    elif distance == "C":
        dist_cal = caylley
        min_dist = 1
    else:
        dist_cal = hamming
        min_dist = 2

    while len(created_consensus) < number_of_optimas:
        elements = list(range(1, permutation_size + 1))
        np.random.shuffle(elements)
        
        if str(elements) not in created_consensus:
            posicioned_dist = True
            for c in consensus_permutations:
                if dist_cal(elements, c) <= min_dist:
                    posicioned_dist = False
                    break
            
            if posicioned_dist:
                created_consensus[str(elements)] = True
                consensus_permutations.append(elements)

    return consensus_permutations

class Instance:
    def __init__(self, consensus_permutations: np.array, weights: np.array, zetas: np.array, thetas: np.array):
        self.consensus_permutations = consensus_permutations
        self.weights = weights
        self.zetas = zetas
        self.thetas = thetas

def _create_instance(permutation_size: int, number_of_optimas: int, distance: str= 'K', typ = "max"):

    consensus_permutations =  _create_permutations(permutation_size, number_of_optimas, distance)

    if distance == "K":
        distances = _calculate_kendall_tau_distances(consensus_permutations)
    elif distance == "C":
        distances = _calculate_cayley_distances(consensus_permutations)
    elif distance == "H":
        distances = _calculate_hamming_distances(consensus_permutations)

    if typ == "max":
        thetas = _generate_easy_thetas(permutation_size, number_of_optimas, distance)
    else:
        thetas = _generate_difficult_thetas(permutation_size, number_of_optimas, distance)

    zeta = normalizationConstants.Zvalue(permutation_size, number_of_optimas, thetas, distance)

    instance_parameters = linearProblem.defineDiscretFunctionParameters(permutation_size, number_of_optimas, thetas, distances, typ, zeta)

    solution = np.array([instance_parameters.x[i].value for i in range(number_of_optimas)])

    return Instance(consensus_permutations, solution, zeta, thetas)


class MultiMallowsDiscret:

    def __init__(self, discretDimension, numberOfMaxima, distance="K", difficult = "E"):
        self.discretDimension = discretDimension
        self.numberOfMaxima = numberOfMaxima
        self.distance = distance
        self.difficult = difficult

        if self.distance == "K":
            self.calcDistance = kendall
        elif self.distance == "C":
            self.calcDistance = caylley
        else:
            self.calcDistance = hamming
        pass

    def create_parameters(self):
        if self.difficult == "E":
            self.calc_parameters_easy()
        elif self.difficult == "H":
            self.calc_parameters_difficult()

    def calc_parameters_easy(self):
        instance_parameters = _create_instance(self.discretDimension, self.numberOfMaxima, self.distance, typ="max")
        self._extract_parameters(instance_parameters)

    def calc_parameters_difficult(self):
        instance_parameters = _create_instance(self.discretDimension, self.numberOfMaxima, self.distance, typ="min")
        self._extract_parameters(instance_parameters)

    def _extract_parameters(self, instance_parameters):
        self.weights: List[float] = instance_parameters.weights
        self.consensus: List[List[int]] = instance_parameters.consensus_permutations
        self.zetas: List[float] = instance_parameters.zetas
        self.thetas: List[float] = [theta[0] for theta in instance_parameters.thetas]
        self.global_optimum: float = np.divide(self.weights[0], self.zetas[0])
        self.optima : List[float] = [self.weights[i] / zeta for i, zeta in enumerate(self.zetas)]

    def evaluate(self, perm: np.ndarray) -> Tuple[float, float]:
        value = Decimal(0)
        comp_value = Decimal(-float('inf'))

        for i in range(self.numberOfMaxima):
            distance = self.calcDistance(self.consensus[i], perm)

            weight_normalized = Decimal(self.weights[i]/self.zetas[i])
            
            mallows_value = Decimal(weight_normalized * Decimal(np.exp(-distance * self.thetas[i])))
            c = np.log(self.weights[i]/self.zetas[i]) - distance*self.thetas[i]
            if  c > comp_value:
                value = mallows_value
                comp_value = c

        return value, comp_value
    
    def evaluateAndGetComponentIndex(self, perm: np.ndarray):
        value = Decimal(0)
        ln_value = Decimal(-float('inf'))
        k = 0

        for i in range(self.numberOfMaxima):
            distance = self.calcDistance(self.consensus[i], perm)
            weight_normalized = Decimal(self.weights[i]/self.zetas[i])
            
            mallows_value = Decimal(weight_normalized * Decimal(np.exp(-distance * self.thetas[i])))
            ln_mallows_value = np.log(self.weights[i]/self.zetas[i]) - distance*self.thetas[i]

            if  ln_mallows_value > ln_value:
                value = mallows_value
                ln_value = ln_mallows_value
                k = i

        return value, ln_value, k

class NormalizedDiscret:
    _discret: MultiMallowsDiscret
    optima: List[Decimal]

    def createParameters(self, permutation_size, number_of_minimas, distance = "K", difficult = "E"):
        self._discret = MultiMallowsDiscret(permutation_size, number_of_minimas, distance, difficult)
        self._discret.create_parameters()
        self.optima = []

        for optimum in self._discret.optima:
            self.optima.append(self.transform(optimum))
        
        pass

    def transform(self, value) -> Decimal:
        valueNormalized = Decimal(value) / Decimal(self._discret.global_optimum)

        return Decimal(2) - Decimal(valueNormalized)

    
    def evaluate(self, perm: np.ndarray) -> Tuple[float, float]:
        value, ln_value = self._discret.evaluate(perm)

        return self.transform(value), ln_value
    
    def evaluateAndGetComponentIndex(self, perm: np.ndarray):
        value, ln_value, i = self._discret.evaluateAndGetComponentIndex(perm)

        return self.transform(value), ln_value, i