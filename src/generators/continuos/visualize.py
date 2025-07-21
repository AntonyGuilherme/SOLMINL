from numpy import array, linspace, zeros_like, meshgrid
import matplotlib.pyplot as plt
from typing import List

import matplotlib
matplotlib.use('Agg')
# This is expected to be used in function with two dimensions, i.e. f(x,y)
class Visualize:
    def __init__(self, f, localMinimuns: List):
        grid_size = 100    # Resolution of the grid
        self.localMinimuns = localMinimuns
        self.minimuns = [f(p) for p in localMinimuns]
        # Evaluate function on a grid
        x_vals = linspace(0, 1, grid_size)
        y_vals = linspace(0, 1, grid_size)
        self.X, self.Y = meshgrid(x_vals, y_vals)
        self.Z = zeros_like(self.X)

        for i in range(grid_size):
            for j in range(grid_size):
                x = array([self.X[i, j], self.Y[i, j]])
                self.Z[i, j] = f(x)
        pass

    
    def curveLevel(self):
        # Plotting
        plt.figure(figsize=(8, 6))
        contour = plt.contourf(self.X, self.Y, self.Z, levels=50, cmap='viridis')
        plt.colorbar(contour)
        minima = array(self.localMinimuns)
        plt.scatter(minima[:, 0], minima[:, 1], color='red', marker='x', label='Minima (p_i)')
        plt.title('Quadratic Family Function (q = {})'.format(len(self.localMinimuns)))
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        pass

    def spacial(self, output):

        # ==== 3D Plotting ====
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax.set_title('3D Surface of Quadratic Family Function')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f_{quad}(x)$')
        fig.colorbar(surface, shrink=0.5, aspect=10)

        # Plot minima points on the surface
        minima = array(self.localMinimuns)
        ax.scatter(minima[:, 0], minima[:, 1], self.minimuns, color='red', marker='x', s=50, label='Minima ($p_i$)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output)

        pass
