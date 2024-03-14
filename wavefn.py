# simulating the propagation of a one-dimensional wave governed by the wave equation
import numpy as np
import matplotlib.pyplot as plt

class WaveFunction:
    def __init__(self, c=1, dx=0.1, dt=0.1):
        """
        Initialize the wave simulation parameters.
        
        :param c: Wave speed.
        :param dx: Spatial step size.
        :param dt: Time step size.
        :param steps: Number of time steps to simulate.
        """
        self.c = c
        self.dx = dx
        self.dt = dt

    def simulate_wave_equation_non_bound(self, initial_profile, steps):
        """
        Simulate the wave equation with non-boundary conditions (first and last step aren't bound to zero).
        
        :param initial_profile: Numpy array representing the initial wave profile (Y coordinates).
        :param steps: Number of time steps to simulate.
        :return: The initial and final wave profiles.
        """
        n_points = len(initial_profile)
        u = np.zeros((steps + 1, n_points))
        u[0, :] = initial_profile
        
        C2 = (self.c * self.dt / self.dx)**2  # Courant number squared
        
        for n in range(0, steps):
            for i in range(1, n_points - 1):
                if n == 0:
                    # For the first step, using a different approach for the edges to simulate non-boundary
                    u[n+1, i] = u[n, i] + 0.5 * C2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                else:
                    u[n+1, i] = 2*u[n, i] - u[n-1, i] + C2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            # Update the first and last points differently to remove boundary constraints
            u[n+1, 0] = u[n+1, 1]
            u[n+1, -1] = u[n+1, -2]
        
        return initial_profile, u[-1, :]

if __name__ == "__main__":
    # Simulation parameters
    c = 1.0  # Wave speed
    dx = 0.1  # Spatial step size
    dt = 0.1  # Time step size
    steps = 100  # Number of time steps
    
    # Generate a "random" initial wave profile
    np.random.seed(42)  # Seed for reproducibility
    initial_random_profile = np.random.rand(100)  # 100 points of random amplitudes
    
    # Create an instance of WaveFunction and run the simulation
    wave_fn = WaveFunction(c, dx, dt)
    initial_profile, final_profile = wave_fn.simulate_wave_equation_non_bound(initial_random_profile, steps)
    
    # Plotting (Optional)
    plt.figure(figsize=(10, 6))
    plt.plot(initial_profile, label='Initial Profile')
    plt.plot(final_profile, label='Final Profile')
    plt.legend()
    plt.title('Wave Propagation Simulation')
    plt.xlabel('Position')
    plt.ylabel('Amplitude')
    plt.show()
