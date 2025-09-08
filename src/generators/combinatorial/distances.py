from scipy.stats import kendalltau
from typing import List

def kendall(perm1: List[int], perm2: List[int]) -> int:
    n = len(perm1)
    max_possible_pairs = n * (n - 1) / 2

    tau, _ = kendalltau(perm1, perm2)
    discordant_pairs = (1 - tau) * max_possible_pairs / 2
        
    return int(discordant_pairs)

def caylley(perm1: List[int], perm2: List[int]) -> int:
    n = len(perm1)
    visited = [False] * n
    cycles = 0

    pos2 = {val: idx for idx, val in enumerate(perm2)}
    mapping = [pos2[val] for val in perm1]

    for i in range(n):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = mapping[j]

    return n - cycles

def hamming(perm1: List[int], perm2: List[int]) -> int:
    dist = 0

    for i, p in enumerate(perm1):
        if p != perm2[i]:
            dist +=  1

    return dist