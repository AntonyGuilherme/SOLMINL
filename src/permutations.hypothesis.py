from generators.combinatorial.instance_generator import Permutation

permutation = Permutation(5, 4)

permutation.calc_parameters_easy()

for x in range(len(permutation.consensus)):
    print(f"{permutation.consensus[x]} {permutation.evaluate(permutation.consensus[x])}")

permutation.plot("easy.png")


print("difficult problem")
permutation.calc_parameters_difficult()

for x in range(len(permutation.consensus)):
    print(f"{permutation.consensus[x]} {permutation.evaluate(permutation.consensus[x])}")

permutation.plot("difficult.png")


