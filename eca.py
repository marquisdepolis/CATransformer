import matplotlib
import numpy as np

class ECA: #Elementary Cellular Automata
    def __init__(self, width):
        self.rule_number = 110
        self.width = width
        self.generations = []

    def rule(self, left, center, right):
        """Applies the automaton rule to the cells."""
        return (self.rule_number >> (left * 4 + center * 2 + right)) & 1

    def generate_next_generation(self, current_gen):
        """Generates the next generation for a given array of cells."""
        next_gen = []
        for i in range(len(current_gen)):
            left = current_gen[i - 1] if i > 0 else 0
            center = current_gen[i]
            right = current_gen[i + 1] if i < len(current_gen) - 1 else 0
            next_gen.append(self.rule(left, center, right))
        return next_gen

    def set_initial_state(self, initial_state):
        """Sets the initial state for the automaton."""
        initial_state = [int(c) for c in initial_state]  # Convert string to list of integers
        if len(initial_state) == self.width:
            self.generations = [initial_state]
        else:
            raise ValueError("Initial state width does not match the automaton width.")

    def run_simulation(self, num_generations):
        """Simulates the automaton for a given number of generations."""
        # Only set the default initial state if no initial state has been set
        if not self.generations:
            self.generations = [[0] * self.width]
            self.generations[0][self.width // 2] = 1  # Start with a single alive cell in the middle
        for _ in range(1, num_generations):
            self.generations.append(self.generate_next_generation(self.generations[-1]))

    def visualize(self, external_generations=None):
        import matplotlib.pyplot as plt
        # Determine which generations to visualize: external or internal
        generations_to_visualize = external_generations if external_generations is not None else self.generations
        
        # Visualize the generations using a single plot
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        plt.imshow(generations_to_visualize, cmap="Greys", interpolation="nearest")
        plt.title(f"Rule {self.rule_number} Evolution")
        plt.xlabel('Cell Position')
        plt.ylabel('Generation')
        plt.colorbar(label='State', orientation='vertical')
        plt.show()

    def visualize_comparison(self, ca_sequence, transformer_sequence, title1="CA Sequence", title2="Transformer Sequence"):
        """
        Visualizes two sequences side by side for comparison.
        """
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # Adjust figure size as needed

        # Visualize CA sequence
        axs[0].imshow(ca_sequence, cmap="Greys", interpolation="nearest")
        axs[0].set_title(title1)
        axs[0].set_xlabel('Cell Position')
        axs[0].set_ylabel('Generation')

        # Visualize Transformer sequence
        axs[1].imshow(transformer_sequence, cmap="Greys", interpolation="nearest")
        axs[1].set_title(title2)
        axs[1].set_xlabel('Cell Position')
        axs[1].set_ylabel('Generation')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    eca = ECA(width=32)
    eca.set_initial_state(''.join(np.random.choice(['0','1'], 32)))
    eca.run_simulation(num_generations=2)
    eca.visualize()
    final_state = eca.generations[-1]
    print("Final State:", ''.join(map(str, final_state))) # to get a simple list
