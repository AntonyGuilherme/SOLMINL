import numpy as np
import sys

class QuadraticFunction:

    def __init__(self, dimension: int = 2, numberOfLocalMinima: int = 5, minimas = None):
        self.dimension = dimension
        self.numberOfLocalMinima = numberOfLocalMinima
        self.generate_form()

        for i, x in enumerate(self.p_list):
             print(f"{self.minimas[i]} {self.evaluate(x)}")
        pass

    def generate_form(self):
        minima_x = []

        self.B_inv_list = [np.linalg.inv(self.generate_positive_definite_matrix(self.dimension)) for _ in range(self.numberOfLocalMinima)]

        i = 1
        delta = (np.sqrt(self.dimension)- 0.2)/(2*(self.numberOfLocalMinima - 1))
        self.minimas = [delta]
        minima_x.append(np.ones(self.dimension) * 0.1)
        
        trials = 1000 * self.numberOfLocalMinima
        reshape = 1000

        while i < self.numberOfLocalMinima:
            distant = True
            trials = trials - 1

            if trials <= 0:
                self.B_inv_list = [np.linalg.inv(self.generate_positive_definite_matrix(self.dimension)) for _ in range(self.numberOfLocalMinima)]
                i = 1
                minima_x = [np.ones(self.dimension) * 0.1]
                trials = 10000
                reshape = reshape - 1
                if reshape <= 0:
                    raise Exception("The minima could not be posicioned using this dimension and number of optima combination.")
            
            if i >= len(self.minimas):
                self.minimas.append(0.1 + delta*i)
            
            y = np.random.rand(self.dimension)

            for j,pj in enumerate(minima_x):
                diff = y - pj

                if np.linalg.norm(diff) < np.sqrt(self.dimension) * 0.20:
                    distant = False
                    break

                add: float = diff.T @ self.B_inv_list[j] @ diff + self.minimas[j]
                if add < (self.minimas[i] + 0.1) :
                    distant = False
                    break
                
            if distant:
                minima_x.append(y)
                i += 1

        self.p_list = minima_x   


    # Generate symmetric positive definite matrices B_i
    def generate_positive_definite_matrix(self, d):
        A = np.random.randn(d, d)
        return A @ A.T + np.eye(d) * 0.1  # Ensure positive definiteness

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
        plt.show()





# f = QuadraticFunction(dimension= 2, numberOfLocalMinima=7)

# f.visualize()



