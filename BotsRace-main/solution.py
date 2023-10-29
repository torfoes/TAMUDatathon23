import random

from bots_race.environment_factory import EnvironmentFactory
import time


class Solution:
    def __init__(self):
        # TODO code to initialize your solution

        pass

    def track(self):
        # TODO fill in code here which initializes your controller
        # whenever your robot is placed on a new track
        pass

    # should return [linear_acceleration, angular_acceleration]
    def get_action(self, robot_observation):
        # Print the robot's observation with corrected labels
        print("Orientation:", robot_observation[0],
              "Front Sensor:", robot_observation[1],
              "Right Sensor:", robot_observation[2],
              "Back Sensor:", robot_observation[3],
              "Left Sensor:", robot_observation[4])

        # Initialize default accelerations
        linear_acceleration = .00
        angular_acceleration = 0

        if robot_observation[1] > 0.5:
            linear_acceleration = 0.5
        # If the left sensor detects the track more than the right sensor, turn left
        elif robot_observation[4] > robot_observation[2]:
            angular_acceleration = -0.1
        # If the right sensor detects the track more than the left sensor, turn right
        elif robot_observation[2] > robot_observation[4]:
            angular_acceleration = 0.1

        return [linear_acceleration, angular_acceleration]

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
        time.sleep(.125)

    print('Solution score:', fitness)