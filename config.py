import math


class LearningRateFunctions:
    @staticmethod
    def constant(x: float) -> float:
        return 1.0

    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def logarithmic(x: float) -> float:
        return -math.log(1 - ((math.e - 1) / math.e) * x)

    @staticmethod
    def exponential(x: float) -> float:  # pretty useless, logarithmic differs from exponential by less than a percent at worst
        return (math.e ** x) * (1 / (math.e - 1)) - 1 / (math.e - 1)

    @staticmethod
    def cosine(x: float) -> float:
        return -math.cos(x)/(1-math.cos(1)) + 1/(1-math.cos(1))


# parameters
cf = 2  # caution factor
epsilon = 15  # distance squared units
step_sizes = (2.5, 2.5)
false_grab_penalty = -epsilon  # currently unused
debug = True
training_start = 500_000
learning_rate_function = LearningRateFunctions.cosine
learning_rate_max_rate = 1e-3
learning_rate_min_rate = 1e-6
max_unsuccessful_grabs = 400
