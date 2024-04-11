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

    def simulate_wave_equation(self, input_wave, function_sequence=None):
        """
        Simulate the transformation of a wave sequence by applying a sequence of functions
        in the order specified by the user.
        
        :param input_wave: The initial wave sequence as a list of integers.
        :param function_sequence: A list of function names as strings to be applied in order.
        :return: The transformed wave sequence as a NumPy array of integers.
        """
        if function_sequence is None:
            function_sequence = ['F1', 'F2', 'F3', 'F4']  # Default sequence

        current_wave = np.array(input_wave, dtype=int)
        function_map = {
            'F1': self.F1,
            'F2': self.F2,
            'F3': self.F3,
            'F4': self.F4
        }

        for func_name in function_sequence:
            if func_name in function_map:
                current_wave = function_map[func_name](current_wave)

        # Optionally clip the wave to a specific range after processing
        current_wave = np.clip(current_wave, 0, 9).astype(int)
        
        return current_wave
    
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

    # Specify the sequence of transformations
    transformation_sequence = ['F1', 'F3', 'F2', 'F4']  # Example order, can be any combination of F1, F2, F3, F4

    # Apply transformations in the specified order
    current_profile = wave_transformation.simulate_wave_equation(current_profile, transformation_sequence)
    current_profile = wave_transformation.normalize_profile(current_profile, desired_max=9)  # Normalize after transformations

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
