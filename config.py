import math


class LearningRateFunctions:
    @staticmethod
    def constant(x: float) -> float:
        return 1.0

    @staticmethod
    def linear(x: float) -> float:
        return 1 - x

    @staticmethod
    def logarithmic(x: float) -> float:
        return 1 + math.log(1 - ((math.e - 1) / math.e) * x)

    @staticmethod
    def exponential(x: float) -> float:
        return (math.e ** x) * (-1 / (math.e - 1)) + math.e / (math.e - 1)

    @staticmethod
    def cosine(x: float) -> float:
        return math.cos(x)/(1-math.cos(1)) - math.cos(1)/(1-math.cos(1))


# parameters
cf = 2  # caution factor
epsilon = 15  # distance squared units
step_sizes = (2.5, 2.5)
false_grab_penalty = -epsilon  # currently unused
first_rounds = 5000
first_rounds_bonus = epsilon / 3
debug = True
training_start = 500_000
penalize_repetitions = 5  # penalize repetitions of the same action
repetition_penalty = -epsilon
delta = 0.01  # reward units
histogram = [0 for i in range(11)]
learning_rate_function = LearningRateFunctions.cosine
learning_rate_base_rate = 1e-3
