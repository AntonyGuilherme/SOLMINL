import numpy as np
import sys
from typing import List

class QuadraticFunction:

    def __init__(self, dimension: int = 2, numberOfLocalMinima: int = 5, minima_proximity = 2, global_minimum = 1):
        self.dimension = dimension
        self.numberOfLocalMinima = numberOfLocalMinima
        self.minima_proximity = minima_proximity
        self.global_minimum = global_minimum
        self.generate_form()
        pass

    def generate_form(self):
        minima_x = []

        self.B_inv_list = [np.linalg.inv(self.generate_positive_definite_matrix(self.dimension)) for _ in range(self.numberOfLocalMinima)]

        i = 1
        delta = np.sqrt(self.dimension)/(self.minima_proximity*(self.numberOfLocalMinima - 1))
        self.minimas: List[float] = []
        for w in range(self.numberOfLocalMinima):
            self.minimas.append(self.global_minimum + delta*w)

        global_minimum = np.random.rand(self.dimension)

        minima_x.append(global_minimum)
        minima_spread_constant = np.divide(1, self.numberOfLocalMinima)
        delta_space = np.ones(self.dimension) * np.sqrt(self.dimension) * minima_spread_constant
        
        trials = 10000
        reshape = 1000

        while i < self.numberOfLocalMinima:
            distant = True
            trials = trials - 1

            if trials <= 0:
    
                self.B_inv_list = [np.linalg.inv(self.generate_positive_definite_matrix(self.dimension)) for _ in range(self.numberOfLocalMinima)]
                i = 1
                
                minima_x = [global_minimum]
                trials = 1000
                reshape = reshape - 1

                if reshape <= 0:
                    raise Exception("The minima could not be posicioned using this dimension and number of optima combination.")
            
            y = np.random.rand(self.dimension)
            space: float =  delta_space.T @ self.B_inv_list[i] @ delta_space
                
            for j,pj in enumerate(minima_x):
                diff = y - pj

                if np.linalg.norm(diff) < np.sqrt(self.dimension) * minima_spread_constant:
                    distant = False
                    break

                add: float = diff.T @ self.B_inv_list[j] @ diff + self.minimas[j]
                if add < (self.minimas[i] + space) :
                    distant = False
                    break
                
            if distant:
                minima_x.append(y)
                i += 1
                trials = 10000

        self.p_list = minima_x   


    # Generate symmetric positive definite matrices B_i
    def generate_positive_definite_matrix(self, d):
        eigenvalues = np.linspace(0.2, 2.0, d)
        D = np.diag(eigenvalues)
        Q, _ = np.linalg.qr(np.random.randn(d, d))
        pd_matrix = Q @ D @ Q.T 
        return pd_matrix

    # Define the function f_quad(x)
    def evaluate(self, x):
        value: float = sys.float_info.max
        for i, m in enumerate(self.minimas):
            diff = x - self.p_list[i]
            partial_value: float = diff.T @ self.B_inv_list[i] @ diff + m
            if value > partial_value:
                value = partial_value
            
        return value

    def visualize(self, xlim=(0, 1), ylim=(0, 1), grid_points=200, show_minima=True):
        """
        Visualize the quadratic function as a 3D surface plot and contour plot (only for 2D functions).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        if self.dimension != 2:
            raise ValueError("3D visualization only supported for 2D functions.")
        x = np.linspace(xlim[0], xlim[1], grid_points)
        y = np.linspace(ylim[0], ylim[1], grid_points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(grid_points):
            for j in range(grid_points):
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] = self.evaluate(point)

        fig = plt.figure(figsize=(16, 7))

        # 3D surface plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Function value')
        if show_minima:
            p_arr = np.array(self.p_list)
            min_z = np.array([self.evaluate(p_arr[i]) for i in range(len(p_arr))])
            ax1.scatter(p_arr[:, 0], p_arr[:, 1], min_z, color='red', marker='x', s=60, label='Local Minima')
            ax1.legend()
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('f(x1, x2)')
        ax1.set_title('Quadratic Function 3D Visualization')

        # Contour plot (nivel curve)
        ax2 = fig.add_subplot(1, 2, 2)
        contour = ax2.contourf(X, Y, Z, levels=30, cmap='viridis')
        fig.colorbar(contour, ax=ax2, shrink=0.8, aspect=20, label='Function value')
        if show_minima:
            ax2.scatter(p_arr[:, 0], p_arr[:, 1], color='red', marker='x', s=60, label='Local Minima')
            ax2.legend()
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_title('Quadratic Function Contour (Nivel Curve)')

        plt.tight_layout()
        plt.savefig("teste.png")
        plt.close()