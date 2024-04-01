import matplotlib

class ElementaryCellularAutomaton:
    def __init__(self, rule_number, width):
        self.rule_number = rule_number
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

    def run_simulation(self, num_generations):
        """Simulates the automaton for a given number of generations."""
        self.generations = [[0] * self.width]
        self.generations[0][self.width // 2] = 1  # Start with a single alive cell in the middle
        for _ in range(1, num_generations):
            self.generations.append(self.generate_next_generation(self.generations[-1]))

    def visualize(self):
        """Visualizes the automaton's evolution in 2D and compares initial & final states in 1D."""
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
        axs[0].imshow(self.generations, cmap="Greys", interpolation="nearest")
        axs[0].set_title(f"Rule {self.rule_number} Evolution")
        axs[0].axis("off")
        axs[1].imshow([self.generations[0], self.generations[-1]], cmap="Greys", interpolation="nearest")
        axs[1].set_title("Initial & Final State")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    eca = ElementaryCellularAutomaton(rule_number=30, width=31)
    eca.run_simulation(num_generations=15)
    eca.visualize()
