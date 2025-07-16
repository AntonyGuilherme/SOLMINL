import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
import itertools
from scipy.stats import kendalltau
from .parameters import Zvalue
from .parameters import LinearProg

import matplotlib
matplotlib.use('Agg')


def _generate_difficult_thetas(permutation_size, number_of_optimas):
    small_range = (np.log(permutation_size - .5), np.log(permutation_size - .5) + 1)
    normal_range = (2 * np.log(permutation_size - 1), 3*np.log(permutation_size - 1))
    num_cols = permutation_size - 1
    thetas_matrix = np.zeros((number_of_optimas, num_cols))

    for i in range(number_of_optimas):
        if i == 1:
            theta_i = np.random.uniform(*small_range)
            thetas_matrix[i, :] = theta_i
        else:
            theta_i = np.random.uniform(*normal_range)
            thetas_matrix[i, :] = theta_i

    return thetas_matrix

def _generate_easy_thetas(permutation_size, number_of_optimas):
    small_range = (np.log(permutation_size - .5), np.log(permutation_size - .5) + 1)
    normal_range = (2 * np.log(permutation_size - 1), 3*np.log(permutation_size - 1))
    num_cols = permutation_size - 1 
    thetas_matrix = np.zeros((number_of_optimas, num_cols))
    for i in range(number_of_optimas):
        if i == 0:
            theta_0 = np.random.uniform(*small_range)
            thetas_matrix[i,:] = theta_0
        else:
            theta_i = np.random.uniform(*normal_range)
            thetas_matrix[i,:] = theta_i
    return thetas_matrix

def _calculate_kendall_tau_distances(permutations):
        reference_permutation = permutations[0]
        distances = []
        n = len(reference_permutation)

        # The maximum possible number of pairs for a permutation of length n
        max_possible_pairs = n * (n - 1) / 2

        # Iterate through the permutations starting from the second one
        for i in range(0, len(permutations)):
            current_permutation = permutations[i]

            # Calculate the Kendall tau correlation coefficient
            tau, _ = kendalltau(reference_permutation, current_permutation)

            # Convert the Kendall tau coefficient to the number of discordant pairs (distance)
            # Formula: discordant_pairs = (1 - tau) * (N * (N - 1) / 2) / 2
            discordant_pairs = (1 - tau) * max_possible_pairs / 2
            distances.append(round(discordant_pairs))  # Round to nearest integer

        return distances

def _create_permutations(permutation_size: int, number_of_optimas: int):
    consensus_permutations = []
    for _ in range(number_of_optimas):
        # Create a list of elements from 1 to permutation_size
        elements = list(range(1, permutation_size + 1))

        # Shuffle the list to get a random permutation
        np.random.shuffle(elements)

        consensus_permutations.append(elements)

    return consensus_permutations


class Instance:
    def __init__(self, consensus_permutations: np.array, weights: np.array, zetas: np.array, thetas: np.array):
        self.consensus_permutations = consensus_permutations
        self.weights = weights
        self.zetas = zetas
        self.thetas = thetas

def _create_instance_max(permutation_size: int, number_of_optimas: int):

    consensus_permutations =  _create_permutations(permutation_size, number_of_optimas)

    distances = _calculate_kendall_tau_distances(consensus_permutations)

    thetas = _generate_easy_thetas(permutation_size, number_of_optimas)

    zeta = Zvalue.Zvalue(permutation_size, number_of_optimas, thetas, "K")

    instance_parameters = LinearProg.LinearProg(permutation_size, number_of_optimas, thetas, distances, "max", zeta)

    solution = np.array([instance_parameters.x[i].value for i in range(number_of_optimas)])

    return Instance(consensus_permutations, solution, zeta, thetas)


def _create_instance_min(permutation_size: int, number_of_optimas: int):

    consensus_permutations =  _create_permutations(permutation_size, number_of_optimas)

    distances = _calculate_kendall_tau_distances(consensus_permutations)

    thetas = _generate_difficult_thetas(permutation_size, number_of_optimas)

    zeta = Zvalue.Zvalue(permutation_size, number_of_optimas, thetas, "K")

    instance_parameters = LinearProg.LinearProg(permutation_size, number_of_optimas, thetas, distances, "min", zeta)

    solution = np.array([instance_parameters.x[i].value for i in range(number_of_optimas)])

    return Instance(consensus_permutations, solution, zeta, thetas)

class Permutation:

    def __init__(self, permutation_size, number_of_optimas):
        self.permutation_size = permutation_size
        self.number_of_optimas = number_of_optimas
        pass

    def calc_parameters_easy(self):
        instance_parameters = _create_instance_max(self.permutation_size, self.number_of_optimas)
        self._extract_parameters(instance_parameters)

    def calc_parameters_difficult(self):
        instance_parameters = _create_instance_min(self.permutation_size, self.number_of_optimas)
        self._extract_parameters(instance_parameters)

    def _extract_parameters(self, instance_parameters):
        self.weights = instance_parameters.weights
        self.consensus = instance_parameters.consensus_permutations
        self.zetas = instance_parameters.zetas
        self.thetas = [theta[0] for theta in instance_parameters.thetas]

    def evaluate(self, perm: np.ndarray) -> float:
        value = list()
        n = len(perm)
        max_possible_pairs = n * (n - 1) / 2

        for i in range(self.number_of_optimas):
            tau, _ = kendalltau(self.consensus[i], perm)
            discordant_pairs = (1 - tau) * max_possible_pairs / 2
            distance = round(discordant_pairs)

            mallows_value = np.divide(
                np.multiply(self.weights[i], np.exp(-distance * self.thetas[i])),
                self.zetas[i])

            value.append(mallows_value)

        return np.max(value)

    pass

    def plot(self, output_path):
        all_perms = list(itertools.permutations(range(1, self.permutation_size + 1)))

        # Evaluate all permutations and store their values and labels
        perm_values = []
        for perm in all_perms:
            value = self.evaluate(np.array(perm))
            perm_values.append((perm, value))

        # Sort permutations by value (descending for max)
        perm_values.sort(key=lambda x: x[1], reverse=True)
        x_vals = list(range(len(perm_values)))
        y_vals = [v for _, v in perm_values]

        # Find indices of consensus permutations in the sorted list
        consensus_indices = []
        consensus_values = []
        consensus_labels = []
        for consensus in self.consensus:
            for idx, (perm, value) in enumerate(perm_values):
                if np.array_equal(np.array(perm), np.array(consensus)):
                    consensus_indices.append(idx)
                    consensus_values.append(value)
                    consensus_labels.append(str(list(consensus)))
                    break

        # Calculate average value
        avg_value = np.mean(y_vals)

        # Plotting
        plt.figure(figsize=(14, 6))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='skyblue', label='All Permutations')
        plt.scatter(consensus_indices, [y_vals[i] for i in consensus_indices], color='red', marker='x', s=100, label='Consensus')

        # Add labels to consensus points
        for idx, label in zip(consensus_indices, consensus_labels):
            plt.annotate(label, (idx, y_vals[idx]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='red')

        plt.axhline(avg_value, color='green', linestyle='--', label=f'Average Value: {avg_value:.4f}')

        plt.xticks([])
        plt.xlabel("Permutation Index (sorted by value, descending)")
        plt.ylabel("Evaluated Value")
        plt.title("Permutation Values (Sorted, Descending) with Consensus Marked")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)