from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

from miniverse.plt import make_plt


def array_to_str(vector: np.ndarray) -> str:
    str_vector = ['+' if i==1 else '-' for i in vector]
    return ''.join(str_vector)


@dataclass
class Animal:
    pattern: str = "0110"
    segment_len: int = 1
    repeat: int = 1
    frequency: float = 0.01
    def __post_init__(self):
        segment_vector = [int(c) for c in self.pattern]
        self.vector = np.tile(np.repeat(segment_vector, self.segment_len), self.repeat)
        self.len = len(self.pattern) * self.segment_len * self.repeat

    def __str__(self) -> str:
        return array_to_str(self.vector)

    def to_plt(self, animal_name: str = '') -> plt:
        return make_plt(matrix=[self.vector], matrix_names=[animal_name])


lion = Animal(pattern='0110', segment_len=8, frequency=.2)
zebra = Animal(pattern='01', repeat=16, frequency=.1)
snake = Animal(pattern='0101', segment_len=8, frequency=.2)

