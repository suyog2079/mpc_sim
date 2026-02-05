import numpy as np
import matplotlib.pyplot as plt

def angle_wrap(angle):
    while angle > numpy.pi:
        angle -= 2 * numpy.pi
    while angle < -numpy.pi:
        angle += 2 * numpy.pi
    return angle

Class Robot:
    def __init__(self, x, y, theta, vx, vy, omega):
        self.state = np.array([x, y, theta, vx, vy, omega])

# simulate robot motion based on inputs.
    def update(self, input, dt):
        noise = np.random.normal(0, 0.01, (3,3))
        self.state[3] += input[0]
        self.state[4] += input[1]
        self.state[5] += input[2]

        self.state[0] += self.state[3] * dt
        self.state[1] += self.state[4] * dt
        self.state[2] = angle_wrap(self.state[2] + self.state[5] * dt)


# I have introduced some gaussian noise
    def observe(self):
        noise = np.random.normal(0, 0., (6,6))
        observation = noise @ self.state
        return observation 
