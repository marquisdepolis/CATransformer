# Underwater Acoustic Communication
import random
import numpy as np
import matplotlib.pyplot as plt

class WaveFunction:
    def __init__(self, c=1, dx=0.1, dt=0.1):
        """
        Initialize the wave transformation parameters.
        
        :param c: Not used in the current context but could be for wave speed in future extensions.
        :param dx: Not used in the current context but could represent spatial step size for future extensions.
        :param dt: Not used in the current context but could represent time step size for future extensions.
        """
        self.c = c
        self.dx = dx
        self.dt = dt

    def F1(self, sequence, dispersion_factor=1):
        """
        Apply dispersion to the wave sequence, simulating spreading.
        
        :param sequence: The wave sequence as a NumPy array.
        :param dispersion_factor: Controls the spreading effect.
        :return: The wave sequence after dispersion.
        """
        dispersed_sequence = []
        for i in range(len(sequence)):
            if i == 0:
                avg = (sequence[i] + sequence[i+1]) // 2
            elif i == len(sequence) - 1:
                avg = (sequence[i] + sequence[i-1]) // 2
            else:
                avg = (sequence[i-1] + sequence[i] + sequence[i+1]) // 3
            dispersed_sequence.append(max(0, avg - dispersion_factor))
        return np.array(dispersed_sequence)

    def F2(self, sequence, non_linear_factor=1):
        """
        Apply non-linear effects to the wave sequence, modifying based on amplitude.
        
        :param sequence: The wave sequence as a NumPy array.
        :param non_linear_factor: Controls the strength of the non-linear effect.
        :return: The wave sequence after non-linear effects are applied.
        """
        non_linear_sequence = [max(0, int(amplitude + non_linear_factor * amplitude**2)) for amplitude in sequence]
        return np.array(non_linear_sequence)

    def F3(self, sequence, attenuation_factor=0.1):
        """
        Apply attenuation to the wave sequence, simulating energy loss over distance.
        
        :param sequence: The wave sequence as a NumPy array.
        :param attenuation_factor: Controls the rate of attenuation.
        :return: The wave sequence after attenuation.
        """
        attenuated_sequence = [max(0, int(amplitude - attenuation_factor * amplitude)) for amplitude in sequence]
        return np.array(attenuated_sequence)
 
    def F4(self, sequence):
        """
        Simulates a wave interference pattern with a shift, 
        akin to transposing musical notes.

        :param sequence: The original wave sequence as a NumPy array.
        :param shift_amount: The number of positions to shift the wave (positive for up, negative for down).
        :return: The resulting wave sequence after interference.
        """
        shifted_sequence = np.roll(sequence, 3)  # Shifted version of the wave
        interference_factor = 0.5  # Adjust the strength of the interference

        return interference_factor * shifted_sequence

    def simulate_wave_equation(self, input_wave, steps=1):
        """
        Simulate the transformation of a wave sequence through dispersion,
        non-linear effects, attenuation, and interference over a specified number of steps.
        
        :param input_wave: The initial wave sequence as a list of integers.
        :param steps: The number of steps over which to apply the transformations.
        :return: The transformed wave sequence as a NumPy array of integers.
        """
        current_wave = np.array(input_wave, dtype=int)

        for _ in range(steps):
            wave_after_dispersion = self.F1(current_wave)
            wave_after_non_linear = self.F2(wave_after_dispersion)
            wave_after_attenuation = self.F3(wave_after_non_linear)
            
            # wave_for_interference = np.flip(wave_after_attenuation)
            current_wave = self.F4(wave_after_attenuation)

            # Optionally clip the wave to a specific range after each step
            current_wave = np.clip(current_wave, 0, 9).astype(int)
        
        return input_wave, current_wave
    
    def normalize_profile(self, profile, desired_max=9):
        profile_max = np.max(profile)
        if profile_max > 0:
            scale_factor = desired_max / profile_max
            profile = profile * scale_factor
        return profile.astype(int)
    
if __name__ == "__main__":
    length = 32
    initial_profile = (np.random.rand(length) * 10).astype(int)
    
    wave_transformation = WaveFunction()
    current_profile = initial_profile

    # Apply transformations sequentially
    for _ in range(1):  # Number of full transformation cycles
        current_profile = wave_transformation.F1(current_profile)
        current_profile = wave_transformation.F2(current_profile)
        current_profile = wave_transformation.F3(current_profile)
        current_profile = wave_transformation.F4(current_profile)
        current_profile = wave_transformation.normalize_profile(current_profile, desired_max=9)  # Normalize after each cycle

    final_profile = current_profile

    print("Initial Profile:", initial_profile)
    print("Final Profile:", final_profile)
    
    # Plotting the transformation
    plt.figure(figsize=(10, 6))
    plt.plot(initial_profile, label='Initial Profile')
    plt.plot(final_profile, label='Final Profile')
    plt.legend()
    plt.title('Wave Sequence Transformation')
    plt.xlabel('Position')
    plt.ylabel('Amplitude')
    plt.show()

