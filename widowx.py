from typing import Tuple

import numpy as np

"""
This class implements a WidowX interface supporting various simulated (or real) operations.
The implementations of this class are:
WidowXSimulator - A one-cube simulator
WidowXMultiSimulator - A multi-cube simulator
WidowXInterface (will write if time allows, otherwise easy) - A real WidowX interface
"""


class GenericWidowX:
    def step(self, steps: Tuple[float, float]) -> Tuple[float, float]:
        """
        Performs a single step characterized by the given parameter.
        This method should not throw an exception upon receiving an illegal parameter.
        If the step given is illegal, a truncation is attempted (partial step) and if not possible,
        the step is ignored and the old position is returned.
        :param steps: Two floats characterizing fractions in each direction of the maximum step size. Assumed to have
        both components between -1 and 1.
        :return: The current position.
        """
        pass

    def get_pos(self) -> Tuple[float, float]:
        """
        Returns this WidowX's current spatial position.
        This method is assumed to be constant between subsequent calls to GenericWidowX#step.
        This method never throws an exception. The returned position must obey the contract described in
        GenericWidowX#bounds.
        :return: The current position.
        """
        pass

    def eval_pos(self) -> Tuple[int, float]:
        """
        Returns an evaluation of the current state as a tuple, where the first component describes the amount of
        cubes gathered (can be binary for single-cube experiments) and the second component describes the reward
        for a step that brought us to the current state.
        :return: An evaluation of the current state.
        """
        pass

    def reset(self):
        """
        Resets the WidowX environment. Specifically, returns to the neutral position and regenerates the position(s)
        for (the) cube(s).
        This method does not reset evaluative stats kept by the simulator, like scores and success rates.
        :return: Nothing by specification. May or may not return the new image (result of self.get_image() after the
        reset).
        """
        pass

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the bounds of the arena in the following format: ((min_x, min_y), (max_x, max_y))
        The caution factor *should* be taken into consideration in the values returned from this function
        (A cube can appear anywhere within these bounds, and self.get_pos() can return any value within these bounds)
        This method never throws an exception and is assumed to always return a constant (it must support being checked
        only once).
        :return: The bounds of the arena.
        """
        pass

    def get_image(self) -> np.ndarray:
        """
        Returns the current image the WidowX's camera is seeing (or a simulated version thereof). This image is not
        edited. This method's return value must change upon calls to GenericWidowX#reset and may change upon calls to
        GenericWidowX#step, and should remain otherwise constant. This method's return value should change upon
        *successful* calls to GenericWidowX#step but may or may not change upon unsuccessful calls. This is
        implementation-defined.
        :return: The image the WidowX is seeing.
        """
        pass

    def get_image_shape(self) -> (int, int, int):
        """
        Returns the current image's shape as defined by numpy. The default implementation should be sufficient for all
        use cases and this method should not be overriden.
        Like GenericWidowX#bounds, it never throws an exception and is assumed to be checked only once.
        :return: The image's shape
        """
        return self.get_image().shape

    def diag_length_sq(self) -> float:
        """
        Returns the diagonal length squared. This method never throws an exception and should never be overriden.
        :return: The diagonal length of the arena, squared.
        """
        return (self.bounds()[0][1] - self.bounds()[0][0]) ** 2 + (self.bounds()[1][1] - self.bounds()[1][0]) ** 2

    @staticmethod
    def clip(value: float, min_value: float, max_value: float) -> float:
        """
        Returns the given value clipped to be at least min_value and at most max_value. This is a static helper function
        that will never throw an exception.
        :param value: The value to clip.
        :param min_value: The minimum value.
        :param max_value: The maximum value.
        :return: If value is less than min_value, then min_value. If value is more than max_value, then max_value. Else,
        value.
        """
        return max(min(value, max_value), min_value)
