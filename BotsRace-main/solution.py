import random

from bots_race.environment_factory import EnvironmentFactory
import time


class Solution:
    def __init__(self):
        self.is_first_action = True
        self.aligning = True
        self.side = False
        self.prev_front = 0
        self.threshold = .0025

        # PID Controller parameters
        self.Kp = 10
        self.Ki = .1
        self.Kd = 0.5
        self.prev_error = 0
        self.integral = 0

        # Logging initialization
        print("Solution Initialized with PID Controller")
        print(f"Kp: {self.Kp}, Ki: {self.Ki}, Kd: {self.Kd}\n")

    def track(self):
        self.prev_error = 0
        self.integral = 0
        print("Robot placed on a new track. PID controller state reset.")

    def get_action(self, robot_observation):
        print("\nNew Action Requested")
        print("---------------------")
        print("Robot Observation:",
              f"Orientation: {robot_observation[0]:.3f}",
              f"Front Sensor: {robot_observation[1]:.3f}",
              f"Left Sensor: {robot_observation[2]:.3f}",
              f"Back Sensor: {robot_observation[3]:.3f}",
              f"Right Sensor: {robot_observation[4]:.3f}")

        if self.is_first_action:
            self.is_first_action = False
            print("First action: No movement.")
            return [0, 0]

        angular_accel = 0
        linear_accel = 0
        sensor_diff = robot_observation[2] - robot_observation[4]

        if self.aligning:
            print("Robot is aligning...")
            if abs(sensor_diff) < self.threshold and robot_observation[1] > robot_observation[3]:
                self.aligning = False
                print("Finished Initial Alignment.")
                self.side = robot_observation[2] <= robot_observation[4]
                print(f"Robot aligned to {'Right' if self.side else 'Left'} side.")
            else:
                # PID controller logic
                error = sensor_diff
                self.integral += error
                derivative = error - self.prev_error
                angular_accel = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
                self.prev_error = error

                print(f"PID Controller Output: P({self.Kp * error:.3f}), I({self.Ki * self.integral:.3f}), D({self.Kd * derivative:.3f})")
                print(f"Angular acceleration set to: {angular_accel:.3f}")
        else:
            linear_accel = .01
            if self.prev_front > 0 and robot_observation[1] == 0:
                self.side = not self.side
                print("Robot crossed over track. Switching side.")
            angular_accel = 0.01 if self.side else -0.01
            print(f"Robot moving forward with linear acceleration: {linear_accel:.3f} and angular acceleration: {angular_accel:.3f}")

        self.prev_front = robot_observation[1]
        return [linear_accel, angular_accel]


# this is example of code to test your solution locally
if __name__ == '__main__':
    solution = Solution()

    # TODO check out the environment_factory.py file to create your own test tracks
    env_factory = EnvironmentFactory(debug=True)
    env = env_factory.get_random_environment()

    done = False
    fitness = 0
    robot_observation = env.reset()
    while not done:
        robot_action = solution.get_action(robot_observation)
        robot_observation, fitness, done = env.step(robot_action)
        time.sleep(.0625)

    print('Solution score:', fitness)
