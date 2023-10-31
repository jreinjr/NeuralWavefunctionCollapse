import numpy as np
import random

class WaveFunctionCollapse:
    def __init__(self, tiles, N=3):
        self.tiles = tiles
        self.N = N
        self.patterns = self.extract_patterns()
        self.weights = {pattern: 0 for pattern in self.patterns}
        self.calculate_weights()

    def extract_patterns(self):
        patterns = set()
        for y in range(self.tiles.shape[0] - self.N + 1):
            for x in range(self.tiles.shape[1] - self.N + 1):
                pattern = tuple(map(tuple, self.tiles[y:y+self.N, x:x+self.N]))
                patterns.add(pattern)
        return patterns

    def calculate_weights(self):
        for y in range(self.tiles.shape[0] - self.N + 1):
            for x in range(self.tiles.shape[1] - self.N + 1):
                pattern = tuple(map(tuple, self.tiles[y:y+self.N, x:x+self.N]))
                self.weights[pattern] += 1

    def run(self, output_shape):
        self.output = np.zeros(output_shape, dtype=int)
        self.output_waves = [[[True for _ in self.patterns] for _ in range(output_shape[1])] for _ in range(output_shape[0])]

        while True:
            y, x = self.find_lowest_entropy_cell()
            if y is None or x is None:
                break

            weights = self.calculate_superimposed(y, x)
            chosen_pattern = random.choices(list(self.patterns), weights=weights)[0]
            self.collapse_cell(y, x, chosen_pattern)

        return self.output

    def find_lowest_entropy_cell(self):
        min_entropy = float('inf')
        min_coord = (None, None)

        for y in range(len(self.output_waves)):
            for x in range(len(self.output_waves[y])):
                entropy = self.calculate_entropy(y, x)
                if entropy < min_entropy:
                    min_entropy = entropy
                    min_coord = (y, x)

        return min_coord

    def calculate_entropy(self, y, x):
        wave = self.output_waves[y][x]
        entropy = sum(self.weights[pattern] for i, pattern in enumerate(self.patterns) if wave[i])
        return entropy

    def calculate_superimposed(self, y, x):
        wave = self.output_waves[y][x]
        return [self.weights[pattern] if wave[i] else 0 for i, pattern in enumerate(self.patterns)]

    def collapse_cell(self, y, x, pattern):
        for dy in range(self.N):
            for dx in range(self.N):
                self.output[y + dy][x + dx] = pattern[dy][dx]
        self.output_waves[y][x] = [pattern == p for p in self.patterns]

# Example usage
tiles = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])
wfc = WaveFunctionCollapse(tiles)
output = wfc.run((8, 8))
print(output)
