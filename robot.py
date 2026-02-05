import numpy as np
import matplotlib.pyplot as plt

MAX_ACC = 2 
MAX_YAW_ACC = 0.1
DT = 0.1

def acceleration_limit(vel, prev_vel):
    acc = (vel - prev_vel) / DT
    acc = np.clip(acc, -MAX_ACC, MAX_ACC)
    return prev_vel + acc * DT

def yaw_acceleration_limit(yaw_rate, prev_yaw_rate):
    yaw_acc = (yaw_rate - prev_yaw_rate) / DT
    yaw_acc = np.clip(yaw_acc, -MAX_YAW_ACC, MAX_YAW_ACC)
    return prev_yaw_rate + yaw_acc * DT

def angle_wrap(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

class Robot:
    def __init__(self, _X):
        self.X = _X 
        self.prev_X = np.zeros_like(_X)

# simulate robot motion based on inputs.
    def update(self, input):

        # adding noise to the input
        noise = np.random.multivariate_normal(np.zeros(3), np.diag([0.01, 0.01, 0.001]), size=1).flatten()
        input = input + noise


        # defining state transition matrix 
        A = np.array([
            [1, 0, 0, DT, 0, 0, 1/2 * DT**2, 0],
            [0, 1, 0, 0, DT, 0, 0, 1/2 * DT**2],
            [0, 0, 1, 0, 0, DT, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0 ,0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
                      ])

        # defining control input matrix
        B = np.array([
            [ 0, 0, 0],
            [ 0, 0, 0],
            [ 0, 0, 0],
            [ 1, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 1],
            [ 0, 0, 0],
            [ 0, 0, 0],
            ])

        # implement acceleration limits 
        self.X = A @ self.X + B @ input
        # self.X[3] = acceleration_limit(self.X[3], self.prev_X[3])
        # self.X[4] = acceleration_limit(self.X[4], self.prev_X[4])
        # self.X[5] = yaw_acceleration_limit(self.X[5], self.prev_X[5])

        # implement angle wrap for yaw
        self.X[2] = angle_wrap(self.X[2])

        self.prev_X = self.X.copy()


# I have introduced some gaussian noise
    def observe(self):
        noise = np.random.multivariate_normal(np.zeros(8), np.diag([0.01, 0.01, 0.001, 0.01, 0.01, 0.001, 0.001, 0.001]), size=1).flatten()
        observation = self.X + noise

        return observation 


if __name__ == "__main__":
    xs, ys = [], []
    robot = Robot(np.zeros(8))
    for i in range(100):
        robot.update(np.array([2.0, 2.0, 0], dtype=float))
        observation = robot.observe()
    
        print(observation)
        xs.append(observation[0])
        ys.append(observation[1])
        # xs.append(i)
        # ys.append(observation[4])
    plt.figure()
    plt.plot(xs, ys, '-o')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
