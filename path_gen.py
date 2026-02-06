import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def curvature(s):
    return (
        0.05                       
        + 0.02 * np.sin(0.5 * s)*s**2   
        + 0.01 * np.sin(2.0 * s)*s   
    )

def generate_path_curvature(x, y, N=150, ds=0.1):
    path = []
    theta = 0.0

    for i in range(N):
        kappa = curvature(i * ds)
        theta += kappa * ds
        x += np.cos(theta) * ds
        y += np.sin(theta) * ds
        path.append([x, y])

    return np.array(path)


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
    path = generate_path_curvature(1,1)
    plot_path(path)
