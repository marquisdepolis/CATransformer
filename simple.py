# a simple equation. repurposed from wavefn, so some extra info below
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

    def simulate_wave_equation(self, input_wave, step, frequency=0.5, phase=0, max_val=9, min_val=0):
        """
        Apply sinusoidal modulation to an input wave over a number of steps.
        
        :param input_wave: Initial wave profile as a NumPy array.
        :param steps: Number of transformation steps.
        :param frequency: Frequency of the sinusoidal modulation.
        :param phase: Phase shift of the sinusoidal modulation.
        :param max_val: Maximum value for the wave after modulation.
        :param min_val: Minimum value for the wave after modulation.
        :return: The initial input wave and the modulated wave profile at the specified step.
        """
        # Convert the input wave from strings to integers
        input_wave = np.array([int(num) for num in input_wave])
        
        n_points = len(input_wave)  # Number of points in the wave
        amplitude = (max_val - min_val) / 2  # Calculate the amplitude scaling factor
        offset = amplitude + min_val  # Offset to ensure the wave oscillates around the midpoint
        
        # Calculate the sinusoidal modulation factor for the specified step
        modulation_factor = np.sin(2 * np.pi * frequency * step / n_points + phase)
        
        # Apply the modulation factor to each point in the input wave
        modulated_wave = input_wave + modulation_factor * amplitude
        
        # Ensure all values are within the specified range and convert to integers
        modulated_wave = np.clip(modulated_wave, min_val, max_val).astype(int)
        
        return input_wave, modulated_wave

if __name__ == "__main__":
    # Simulation parameters
    c = 1.0  # Wave speed
    dx = 0.1  # Spatial step size
    dt = 0.1  # Time step size
    steps = 1  # Number of time steps
    
    # Generate a "random" initial wave profile
    # np.random.seed(42)  # Seed for reproducibility
    initial_random_profile = (np.random.rand(10)*10).astype(int)  # 10 points of random amplitudes
    
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
