import numpy as np
import matplotlib.pyplot as plt

from path_gen import generate_path_curvature, plot_path
from robot import Robot, DT, NUM_STATES
from controller import MPC_Controller

def align_ref(X_log, X_ref):
    T = X_log.shape[0]
    return X_ref[:T]


def build_reference_from_path(path, v_avg=3.0):
    """
    Given path (N,2), returns reference trajectory
    X_ref: (N, 8)
    """
    N = path.shape[0]

    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])

    ds = np.sqrt(dx**2 + dy**2)
    ds[ds < 1e-6] = 1e-6

    # Tangent direction
    theta = np.arctan2(dy, dx)
    np.unwrap(theta)

    # Velocity components
    vx = v_avg * dx / ds
    vy = v_avg * dy / ds

    # Angular velocity
    dtheta = np.gradient(theta)
    omega = dtheta / DT

    # Accelerations (reference)
    ax = np.gradient(vx) / DT
    ay = np.gradient(vy) / DT

    X_ref = np.zeros((N, NUM_STATES))
    X_ref[:, 0] = path[:, 0]
    X_ref[:, 1] = path[:, 1]
    X_ref[:, 2] = theta
    X_ref[:, 3] = vx
    X_ref[:, 4] = vy
    X_ref[:, 5] = omega
    # X_ref[:, 6] = ax
    # X_ref[:, 7] = ay

    return X_ref


def interpolate_reference(X_ref, factor=5):
    """
    Increase resolution to match controller DT
    """
    N, nx = X_ref.shape
    t = np.arange(N)
    t_fine = np.linspace(0, N - 1, N * factor)

    X_ref_fine = np.zeros((len(t_fine), nx))
    for i in range(nx):
        X_ref_fine[:, i] = np.interp(t_fine, t, X_ref[:, i])

    return X_ref_fine


# -----------------------------
# Main simulation
# -----------------------------

if __name__ == "__main__":

    # Generate path
    path = generate_path_curvature(1,1)

    # Reference trajectory
    X_ref = build_reference_from_path(path, v_avg=2.0)
    X_ref = interpolate_reference(X_ref, factor=4)

    # MPC parameters
    p = 10
    m = 10 

    # defining state transition matrix 
    A = np.array([
        [1, 0, 0, DT, 0, 0],
        [0, 1, 0, 0, DT, 0],
        [0, 0, 1, 0, 0, DT],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
        ])

    # defining control input matrix
    B = np.array([
        [ 0, 0, 0],
        [ 0, 0, 0],
        [ 0, 0, 0],
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]
        ])

    # Track x, y, theta, vx, vy, omega
    # C = np.eye(8)
    C = np.eye(NUM_STATES)

    Qu = np.diag([50, 50, 20, 2, 2, 0.5])
    Ru = np.diag([4.0, 4.0, 2.5])

    mpc = MPC_Controller(p, m, A, B, C, Qu, Ru)

    # Robot
    robot = Robot(np.array([1, 1 , 0, 0, 0, 0]))
    U_prev = np.zeros(3)

    # Logs
    X_log = []
    U_log = []
    U_ref_log = []

    # Simulation loop
    for k in range(len(X_ref) - p):

        X_k = robot.observe()

        Y_ref = X_ref[k:k + p].reshape(-1)

        U_k = mpc.control_output(X_k, U_prev, Y_ref)
        # U_k = np.zeros(3)

        robot.update(U_k)

        X_log.append(robot.X.copy())
        U_log.append(U_k.copy())
        U_ref_log.append(X_ref[k, 6:9] if X_ref.shape[1] >= 9 else np.zeros(3))

        U_prev = U_k

    X_log = np.array(X_log)
    U_log = np.array(U_log)

    # -----------------------------
    # Plot path vs trajectory
    # -----------------------------

    plt.figure()
    plt.plot(path[:, 0], path[:, 1], 'k--', label="Reference path")
    plt.plot(X_log[:, 0], X_log[:, 1], 'r', label="Robot trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Path tracking")
    plt.savefig('path_tracking.png', dpi=300)
    plt.show()

    # -----------------------------
    # State plots
    # -----------------------------

    t = np.arange(len(X_log)) * DT

    X_ref = align_ref(X_log, X_ref)

    pos_error = np.linalg.norm(
    X_log[:, 0:2] - X_ref[:, 0:2], axis=1)




    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

    axs[0].plot(t, X_log[:, 0], label="x")
    axs[0].plot(t, X_ref[:len(t), 0], '--', label="x_ref")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t, X_log[:, 1], label="y")
    axs[1].plot(t, X_ref[:len(t), 1], '--', label="y_ref")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t, pos_error, label="error")
    axs[2].legend()
    axs[2].grid()

    axs[2].set_xlabel("Time [s]")
    fig.savefig('position_error.png', dpi=300)
    plt.show()






    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 10))

    axs[0].plot(t, X_log[:, 2], label="θ")
    axs[0].plot(t, X_ref[:len(t), 2], '--', label="θ_ref")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t, X_log[:, 3], label="vx")
    axs[1].plot(t, X_ref[:len(t), 3], '--', label="vx_ref")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t, X_log[:, 4], label="vy")
    axs[2].plot(t, X_ref[:len(t), 4], '--', label="vy_ref")
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(t, X_log[:, 5], label="ω")
    axs[3].plot(t, X_ref[:len(t), 5], '--', label="ω_ref")
    axs[3].legend()
    axs[3].grid()

    axs[3].set_xlabel("Time [s]")
    fig.savefig('state_tracking.png', dpi=300)
    plt.show()

