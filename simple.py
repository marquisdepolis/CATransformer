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

    def simulate_wave_equation(self, initial_profile, steps):
        """
        Simulate a simplified wave propagation based on previous values.
        
        :param initial_profile: Numpy array representing the initial wave profile (Y coordinates).
        :param steps: Number of time steps to simulate.
        :return: The initial and final wave profiles.
        """
        n_points = len(initial_profile)
        u = np.zeros((steps + 1, n_points))
        u[0, :] = initial_profile
        
        for n in range(1, steps + 1):
            for i in range(1, n_points - 1):
                # Simple update based on the average of the two neighboring points
                u[n, i] = (u[n-1, i-1] + u[n-1, i+1]) / 2
            # Handling the first and last point to simulate open boundaries
            u[n, 0] = u[n, 1]
            u[n, -1] = u[n, -2]
        
        return initial_profile, u[-1, :].astype(int)

if __name__ == "__main__":
    # Simulation parameters
    c = 1.0  # Wave speed
    dx = 0.1  # Spatial step size
    dt = 0.1  # Time step size
    steps = 10  # Number of time steps
    
    # Generate a "random" initial wave profile
    np.random.seed(42)  # Seed for reproducibility
    initial_random_profile = (np.random.rand(10)*5+5).astype(int)  # 100 points of random amplitudes
    
    # Create an instance of WaveFunction and run the simulation
    wave_fn = WaveFunction(c, dx, dt)
    initial_profile, final_profile = wave_fn.simulate_wave_equation(initial_random_profile, steps)
    print(initial_profile)
    print(final_profile)
    # Plotting (Optional)
    plt.figure(figsize=(10, 6))
    plt.plot(initial_profile, label='Initial Profile')
    plt.plot(final_profile, label='Final Profile')
    plt.legend()
    plt.title('Wave Propagation Simulation')
    plt.xlabel('Position')
    plt.ylabel('Amplitude')
    plt.show()
