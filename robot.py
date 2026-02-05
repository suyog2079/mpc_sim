import numpy as np
import matplotlib.pyplot as plt

MAX_ACC = 2 
MAX_YAW_ACC = 0.1

def acceleration_limit(vel, prev_vel, dt):
    acc = (vel - prev_vel) / dt
    acc = np.clip(acc, -MAX_ACC, MAX_ACC)
    return prev_vel + acc * dt

def yaw_acceleration_limit(yaw_rate, prev_yaw_rate, dt):
    yaw_acc = (yaw_rate - prev_yaw_rate) / dt
    yaw_acc = np.clip(yaw_acc, -MAX_YAW_ACC, MAX_YAW_ACC)
    return prev_yaw_rate + yaw_acc * dt

def angle_wrap(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

class Robot:
    def __init__(self, _state):
        self.state = _state 
        self.prev_state = np.zeros_like(_state)

# simulate robot motion based on inputs.
    def update(self, input, dt):

        # adding noise to the input
        noise = np.random.multivariate_normal(np.zeros(3), np.diag([0.1, 0.1, 0.001]), size=1).flatten()
        input = input + noise


        # defining state transition matrix 
        A = np.array([
            [1, 0, 0, dt, 0, 0, 1/2 * dt**2, 0],
            [0, 1, 0, 0, dt, 0, 0, 1/2 * dt**2],
            [0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 1, 0 ,0, dt],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
                      ])

        # defining control input matrix
        B = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            ])

        self.state = A @ self.state + B @ input


# I have introduced some gaussian noise
    def observe(self):
        noise = np.random.multivariate_normal(np.zeros(8), np.diag([0.01, 0.01, 0.001, 0.01,0.01, 0.001, 0.001, 0.001]), size=1).flatten()
        observation = self.state + noise

        return observation 


xs, ys = [], []
robot = Robot(np.zeros(8))
for i in range(100):
    robot.update(np.array([0.0, 2.0, 0], dtype=float), 0.1)
    observation = robot.observe()

    print(observation)
    # xs.append(observation[0])
    # ys.append(observation[1])
    xs.append(i)
    ys.append(observation[4])
plt.figure()
plt.plot(xs, ys, '-o')
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.show()
