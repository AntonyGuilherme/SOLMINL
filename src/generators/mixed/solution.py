import numpy as np

class Solution:
    def __init__(self, dimension = 2, permutation_size = 5):
        self.continuos = [np.random.rand() for _ in range(dimension)]
        self.permutation = list(range(1, permutation_size + 1))
        np.random.shuffle(self.permutation)
        self.value = 0.0
        self.c_value = 0.0
        self.p_value = 0.0
        self.comp_p_value = 0.0
        pass

    def print(self, ref,owner):
        print(f"{ref}&{owner}&{self.continuos}&{self.permutation}&{float(self.c_value):.6f}&{float(self.p_value):.6f}&{float(self.value):.6f}+")
        pass