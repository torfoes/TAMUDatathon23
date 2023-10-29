import random

from bots_race.environment_factory import EnvironmentFactory


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
        # TODO replace code here to see robot_observation to compute an action whenever your robot receives an observation

        return [random.random() - .5, random.random() - .5]

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

    print('Solution score:', fitness)