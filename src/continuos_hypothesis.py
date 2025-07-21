from src.generators.continuos.instance_generator import QuadraticFunction
from src.generators.continuos.visualize import Visualize


quadratic = QuadraticFunction()
visualizer = Visualize(quadratic.evaluate, quadratic.p_list)


visualizer.spacial("teste.png")