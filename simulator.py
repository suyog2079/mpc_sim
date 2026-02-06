import numpy as np
import matplotlib.pyplot as plt

from path_gen import generate_path_curvature
from robot import Robot, DT, NUM_STATES
from controller import MPC_Controller
from kalman_filter import KalmanFilter


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def align_ref(X_log, X_ref):
    return X_ref[:X_log.shape[0]]


def build_reference_from_path(path, v_avg=3.0):
    N = path.shape[0]

    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])

    ds = np.sqrt(dx**2 + dy**2)
    ds[ds < 1e-6] = 1e-6

    theta = np.unwrap(np.arctan2(dy, dx))

    vx = v_avg * dx / ds
    vy = v_avg * dy / ds

    omega = np.gradient(theta) / DT

    X_ref = np.zeros((N, NUM_STATES))
    X_ref[:, 0] = path[:, 0]
    X_ref[:, 1] = path[:, 1]
    X_ref[:, 2] = theta
    X_ref[:, 3] = vx
    X_ref[:, 4] = vy
    X_ref[:, 5] = omega

    return X_ref


def interpolate_reference(X_ref, factor=4):
    N, nx = X_ref.shape
    t = np.arange(N)
    t_fine = np.linspace(0, N - 1, N * factor)

    X_ref_fine = np.zeros((len(t_fine), nx))
    for i in range(nx):
        X_ref_fine[:, i] = np.interp(t_fine, t, X_ref[:, i])

    return X_ref_fine


# --------------------------------------------------
# Main simulation
# --------------------------------------------------

if __name__ == "__main__":

    # -------------------------
    # Path + reference
    # -------------------------
    path = generate_path_curvature(1, 1)
    X_ref = build_reference_from_path(path, v_avg=2.0)
    X_ref = interpolate_reference(X_ref, factor=4)

    # -------------------------
    # System matrices
    # -------------------------
    A = np.array([
        [1, 0, 0, DT, 0, 0],
        [0, 1, 0, 0, DT, 0],
        [0, 0, 1, 0, 0, DT],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    C = np.eye(NUM_STATES)

    # -------------------------
    # MPC
    # -------------------------
    p = 10
    m = 5 

    Qu = np.diag([50, 50, 20, 2, 2, 0.5])
    Ru = np.diag([4.0, 4.0, 2.5])

    mpc = MPC_Controller(p, m, A, B, C, Qu, Ru)

    # -------------------------
    # Robot
    # -------------------------
    robot = Robot(
        _X=np.array([1, 1, 0, 0, 0, 0], dtype=float),
        _A=A,
        _B=B,
        _C=C
    )

    # -------------------------
    # Kalman Filter
    # -------------------------
    input_noise_cov = np.diag([0.001, 0.001, 0.00001])
    Q_kf = B @ input_noise_cov @ B.T

    R_kf = np.diag([0.001, 0.001, 0.0001, 0.001, 0.001, 0.0001])

    kf = KalmanFilter(
        _A=A,
        _B=B,
        _C=C,
        _X=robot.X.copy(),
        _P=np.eye(NUM_STATES),
        _Q=Q_kf,
        _R=R_kf
    )

    # -------------------------
    # Logs
    # -------------------------
    X_log = []
    X_hat_log = []
    U_log = []

    U_prev = np.zeros(3)

    # -------------------------
    # Simulation loop
    # -------------------------
    for k in range(len(X_ref) - p):

        # Measurement
        Z_k = robot.observe()

        # Kalman filter
        kf.predict(U_prev)
        kf.update(Z_k)
        X_hat = kf.state_estimate()

        # MPC
        Y_ref = X_ref[k:k + p].reshape(-1)
        U_k = mpc.control_output(X_hat, U_prev, Y_ref)

        # Plant update
        robot.update(U_k)

        # Logs
        X_log.append(robot.X.copy())
        X_hat_log.append(X_hat.copy())
        U_log.append(U_k.copy())

        U_prev = U_k

    X_log = np.array(X_log)
    X_hat_log = np.array(X_hat_log)
    U_log = np.array(U_log)

    # -------------------------
    # Plots
    # -------------------------

    t = np.arange(len(X_log)) * DT
    X_ref = align_ref(X_log, X_ref)

    # ---- Path tracking ----
    plt.figure()
    plt.plot(path[:, 0], path[:, 1], 'k--', label="Reference path")
    plt.plot(X_log[:, 0], X_log[:, 1], 'r', label="Robot trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Path tracking")
    plt.savefig("figures/path_tracking.png", dpi=300)
    plt.show()

    # ---- Position error ----
    pos_error = np.linalg.norm(
        X_log[:, 0:2] - X_ref[:, 0:2], axis=1
    )

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

    axs[0].plot(t, X_log[:, 0], label="x")
    axs[0].plot(t, X_ref[:, 0], '--', label="x_ref")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(t, X_log[:, 1], label="y")
    axs[1].plot(t, X_ref[:, 1], '--', label="y_ref")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(t, pos_error, label="position error")
    axs[2].legend()
    axs[2].grid()

    axs[2].set_xlabel("Time [s]")
    fig.savefig("figures/position_error.png", dpi=300)
    plt.show()

    # ---- State tracking ----
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 10))

    labels = ["θ", "vx", "vy", "ω"]
    idxs = [2, 3, 4, 5]

    for ax, idx, label in zip(axs, idxs, labels):
        ax.plot(t, X_log[:, idx], label=label)
        ax.plot(t, X_ref[:, idx], '--', label=f"{label}_ref")
        ax.legend()
        ax.grid()

    axs[-1].set_xlabel("Time [s]")
    fig.savefig("figures/state_tracking.png", dpi=300)
    plt.show()

    # ---- KF estimation error ----
    plt.figure()
    plt.plot(t, X_log[:, 0] - X_hat_log[:, 0], label="x error")
    plt.plot(t, X_log[:, 1] - X_hat_log[:, 1], label="y error")
    plt.plot(t, X_log[:, 2] - X_hat_log[:, 2], label="θ error")
    plt.legend()
    plt.grid()
    plt.title("Kalman filter estimation error")
    plt.savefig("figures/kf_error.png", dpi=300)
    plt.show()

