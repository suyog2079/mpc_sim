import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def f(x):
    return (0.04 * x**3 - 0.8 * x**2 + 3 * x + 30)/6


def generate_path(start, num_points):
    path = np.zeros((num_points, 2))
    x = np.linspace(start, start + 10, num_points)
    y = f(x)

    return np.vstack((x, y)).T

def plot_path(path):

    fig, ax = plt.subplots()

    ax.plot(path[:, 0], path[:, 1], '-b')

    box = patches.Rectangle((0, 0),
                            15, 8,
                            linewidth=4,
                            edgecolor='r',
                            facecolor='none')
    ax.add_patch(box)
    
    pad = 1.0
    ax.set_xlim(-pad, 15 + pad)
    ax.set_ylim(-pad, 8 + pad)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    plt.show()

if __name__ == "__main__":
    start = 2 
    num_points = 100
    path = generate_path(start, num_points)
    plot_path(path)
