import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

# Constants
GRID_SIZE = 32
WHITE_PIXEL = 1

# Initialize grid
def initialize_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[GRID_SIZE//2, GRID_SIZE//2] = WHITE_PIXEL
    return grid

# Move white pixel
def move_white_pixel(grid):
    # Find the white pixel
    y, x = np.where(grid == WHITE_PIXEL)
    
    # Define possible moves
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    random_move = random.choice(moves)
    
    # Clear the current white pixel
    grid[y, x] = 0
    
    # Set the new white pixel while making sure it stays within the boundaries
    new_y = min(max(0, y[0] + random_move[0]), GRID_SIZE-1)
    new_x = min(max(0, x[0] + random_move[1]), GRID_SIZE-1)
    
    grid[new_y, new_x] = WHITE_PIXEL
    return grid

# Simulate for T timesteps
def simulate(T):
    grid = initialize_grid()
    result = [grid.copy()]
    for _ in range(T-1):
        grid = move_white_pixel(grid)
        result.append(grid.copy())
    return np.stack(result)

# Save the results to a file
def save_to_file(data, filename="data.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(data, file)

# Load the results from a file
def load_from_file(filename="data.pkl"):
    with open(filename, "rb") as file:
        return pickle.load(file)


def display_animation(data):
    fig, ax = plt.subplots()
    im = ax.imshow(data[0], animated=True, cmap="gray")
    title_text = fig.suptitle(f"Frame: 0")

    def update(frame):
        im.set_array(data[frame])
        title_text.set_text(f"Frame: {frame}")
        return im, title_text

    ani = FuncAnimation(fig, update, frames=range(data.shape[0]), repeat=True, blit=False)
    
    plt.show()




if __name__ == "__main__":
    T = 32
    data = simulate(T)
    save_to_file(data)
    loaded_data = load_from_file()
    display_animation(loaded_data)
