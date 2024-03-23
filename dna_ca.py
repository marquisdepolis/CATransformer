import numpy as np
import matplotlib.pyplot as plt

class DNA_CA:
    def __init__(self, grid_length, grid_width, generations, program=['C', 'M', 'X', 'I']): # Not including 'R' for Recursion since that seems to break for some reason. And it's slow!
        self.grid_length = grid_length
        self.grid_width = grid_width
        self.generations = generations
        self.program = program
        self.BASES = ['A', 'T', 'C', 'G']
        self.MODIFIERS = [0, 1]
        self.OPERATIONS = ['C', 'M', 'X', 'I', 'R']  # Copy, Move, Modify, Conditional, Recursive

    def initialize_grid_with_modifiers(self):
        dna_sequence = np.random.choice(self.BASES, size=(self.grid_length, self.grid_width))
        modifiers = np.random.choice(self.MODIFIERS, size=(self.grid_length, self.grid_width, 2))
        grid = np.dstack((dna_sequence, modifiers))
        return grid

    def apply_rules_with_modifiers(self, grid, iterations, program = None, depth=2):
        if program is None:
            program = self.program
        if depth <= 0:
            return grid  # Base case: stop recursion when depth is 0 or less
        new_grid = grid.copy()
        for _ in range(iterations):
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    base, mod1, mod2 = new_grid[i, j]
                    for op in self.program:
                        if op == 'C':
                            # Copy operation
                            neighbor_i, neighbor_j = self.get_neighbor(i, j, grid.shape)
                            new_base, new_mod1, new_mod2 = new_grid[neighbor_i, neighbor_j]
                        elif op == 'M':
                            # Move operation
                            neighbor_i, neighbor_j = self.get_neighbor(i, j, grid.shape)
                            # Swap the current base and modifiers with the neighbor's
                            current_base, current_mod1, current_mod2 = new_grid[i, j]
                            neighbor_base, neighbor_mod1, neighbor_mod2 = new_grid[neighbor_i, neighbor_j]
                            new_grid[i, j] = (neighbor_base, neighbor_mod1, neighbor_mod2)
                            new_grid[neighbor_i, neighbor_j] = (current_base, current_mod1, current_mod2)
                        elif op == 'X':
                            # Modify operation
                            if np.random.rand() < 0.2:
                                new_mod2 = np.random.choice(self.MODIFIERS)
                            else:
                                new_mod2 = mod2
                            new_base = base
                            if new_mod2 == 1:
                                new_base = np.random.choice(self.BASES)
                            elif base == 'C' and mod1 == 1 and np.random.rand() > 0.9:
                                new_base = 'C'
                        elif op == 'I':
                            # Conditional operation
                            neighbors = self.get_neighbors(i, j, grid.shape)
                            neighbor_bases = [grid[n_i, n_j, 0] for n_i, n_j in neighbors]
                            if 'A' in neighbor_bases and 'T' in neighbor_bases:
                                new_base = 'G'
                        elif op == 'R':
                            # Recursive operation
                            sub_grid = grid[max(0, i-1):i+2, max(0, j-1):j+2]
                            sub_program = self.program[:3]  # Apply the first few operations recursively
                            sub_grid = self.apply_rules_with_modifiers(sub_grid, 1, sub_program, depth-1)
                            new_grid[max(0, i-1):i+2, max(0, j-1):j+2] = sub_grid
                        # Update the grid with the new values
                        new_grid[i, j] = (new_base, new_mod1, new_mod2)
        return new_grid

    def get_neighbor(self, i, j, shape):
        # Helper function to get a random neighbor
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < shape[0] - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < shape[1] - 1:
            neighbors.append((i, j + 1))
        return neighbors[np.random.randint(len(neighbors))]

    def get_neighbors(self, i, j, shape):
        # Helper function to get all neighbors
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, j))
        if i < shape[0] - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < shape[1] - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def generate_output_with_modifiers(self, grid):
        for _ in range(self.generations):
            iterations = np.random.randint(2, 5)
            grid = self.apply_rules_with_modifiers(grid, iterations)
            if np.random.rand() < 0.1:
                additional_iterations = np.random.randint(1, 3)
                grid = self.apply_rules_with_modifiers(grid, additional_iterations)
        return grid

    def flatten_grid(self, grid):
        """Change the grid to a text file with a flattened representation."""
        flattened = []
        for row in grid:
            for base, mod1, mod2 in row:
                flattened.append(f"{base}{mod1}{mod2}")
        text = ''.join(flattened)
        return text

    def unflatten_string(self, flat_grid):
        """Convert a flattened grid string back into a 3D grid structure."""
        # Ensure the flat grid has a length that is a multiple of 3
        assert len(flat_grid) % 3 == 0, "Invalid flat grid length."
        
        # Calculate the total number of cells in the grid
        num_cells = len(flat_grid) // 3
        
        # Ensure the provided dimensions match the number of cells
        assert num_cells == self.grid_length * self.grid_width, "Provided dimensions do not match the flat grid."
        
        # Split the flat grid into chunks of 3 characters (base + 2 modifiers)
        cells = [flat_grid[i:i+3] for i in range(0, len(flat_grid), 3)]
        
        # Convert each chunk into the desired format and reshape into the grid
        grid = np.array([(cell[0], int(cell[1]), int(cell[2])) for cell in cells])
        grid = grid.reshape((self.grid_length, self.grid_width, 3))
        
        return grid

    def visualize_grid_with_modifiers(self, grid, title_prefix="Enhanced DNA Sequence Visualization - "):
        """Visualise the grid."""
        base_colors = {'A': 'red', 'T': 'blue', 'C': 'green', 'G': 'yellow'}
        mod_colors = {'0': 'grey', '1': 'black'}  # Assuming 0 and 1 as the only modifiers for simplicity

        # Determine the number of cells for plotting
        num_cells = grid.shape[0] * grid.shape[1]
        
        fig, axs = plt.subplots(3, 1, figsize=(20, 6))  # 3 rows for bases and each modifier, 1 column
        fig.suptitle(title_prefix + str(type(grid).__name__))

        # Base row
        base_colors_list = [base_colors[base] for row in grid for base, _, _ in row]
        axs[0].bar(range(num_cells), np.ones(num_cells), color=base_colors_list)
        axs[0].set_title('Bases')
        axs[0].axis('off')

        # First modifier row
        mod1_colors_list = [mod_colors[mod1] for row in grid for _, mod1, _ in row]
        axs[1].bar(range(num_cells), np.ones(num_cells), color=mod1_colors_list)
        axs[1].set_title('Modifier 1')
        axs[1].axis('off')

        # Second modifier row
        mod2_colors_list = [mod_colors[mod2] for row in grid for _, _, mod2 in row]
        axs[2].bar(range(num_cells), np.ones(num_cells), color=mod2_colors_list)
        axs[2].set_title('Modifier 2')
        axs[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the main title
        plt.show()

    def run_simulation(self):
        initial_grid_with_modifiers = self.initialize_grid_with_modifiers()
        final_grid_with_modifiers = self.generate_output_with_modifiers(initial_grid_with_modifiers)
        flattened_grid_with_modifiers = self.flatten_grid(final_grid_with_modifiers)
        # self.visualize_grid_with_modifiers(initial_grid_with_modifiers)
        # self.visualize_grid_with_modifiers(final_grid_with_modifiers)
        return flattened_grid_with_modifiers

if __name__ == "__main__":
    simulator = DNA_CA(64,3,2)
    simulator.run_simulation()