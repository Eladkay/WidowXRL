from typing import Tuple


class GenericWidowX:
    def step(self, steps: Tuple[float, float]) -> Tuple[float, float]:
        pass

    def get_pos(self) -> Tuple[float, float]:
        pass

    def eval_pos(self) -> Tuple[int, float]:
        pass

    def reset(self):
        pass

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pass

    def get_image(self):
        pass

    # Assumed to be constant
    def get_image_shape(self) -> (int, int, int):
        pass
