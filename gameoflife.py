import numpy as np
import matplotlib.pyplot as plt

class GameOfLife:
    def __init__(self, input_string, generations):
        grid_size = int(len(input_string)**0.5)
        self.grid_size = grid_size
        self.generations = generations
        self.grid = self.string_to_grid(input_string)

    def string_to_grid(self, input_string):
        '''Convert the input string to a numpy array of the correct shape'''
        flat_array = np.array(list(map(int, input_string)))
        return flat_array.reshape((self.grid_size, self.grid_size))

    def apply_rules(self):
        for _ in range(self.generations):
            new_grid = self.grid.copy()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    live_neighbors = self.count_live_neighbors(i, j)
                    if self.grid[i, j] == 1:
                        if live_neighbors < 2 or live_neighbors > 4 or np.random.rand() < 0.05:
                            new_grid[i, j] = 0  # Die
                        elif np.random.rand() < 0.95:
                            new_grid[i, j] = 1
                    elif self.grid[i, j] == 0:
                        if live_neighbors in [2, 3] or np.random.rand() < 0.1:
                            new_grid[i, j] = 1
            self.grid = new_grid

    def count_live_neighbors(self, x, y):
        total = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size
                total += self.grid[nx, ny]
        return total

    def visualize_grid(self, title_prefix = "Game of Life"):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary')
        title = title_prefix  + str(type(self.grid).__name__)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def flatten_grid(self):
        return ''.join(map(str, self.grid.flatten())).strip()

    def run_simulation(self):
        # self.visualize_grid()  # Visualize the initial state
        self.apply_rules()
        # self.visualize_grid()  # Visualize the final state
        text = self.flatten_grid()  # Output the flattened final state
        # print(text)
        return text
    
# Create an instance of the GameOfLife and run the simulation
if __name__ == "__main__":
    input_string = "0010100110100110001010011010011000101001101001100010100110100110"  # Example input string for a 4x4 grid
    game = GameOfLife(input_string=input_string, generations=100)
    generated_text_ca = game.run_simulation()