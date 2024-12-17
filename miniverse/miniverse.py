import itertools
import logging
from dataclasses import dataclass, field
import numpy as np
from typing import Literal, Optional

import pandas as pd
from matplotlib import pyplot as plt

from miniverse.bin_converter import BinConverter, prob_to_char
from miniverse.animal import Animal
from miniverse.plt import make_plt
from miniverse.timer import timed

logger = logging.getLogger(__name__)


@dataclass
class Miniverse:
    zoo: dict[str, Animal] = field(default_factory=dict)
    verse_len: int = 100000
    record_input_len: int = 40
    record_output_len: int = 1
    animal_position: Literal['random', 'equidistant'] = 'random'
    random_seed: int = None

    def __post_init__(self):
        self._prediction = None
        self.set_up(timed_log_str=f'setting up miniverse of size {self.verse_len}.')

    @timed(logger=logger)
    def set_up(self):
        self.rng = np.random.default_rng(self.random_seed)
        animal_keys = list(self.zoo.keys())
        animal_values = [self.zoo[key] for key in animal_keys]
        total_frequency = sum([animal.frequency for animal in animal_values])
        assert total_frequency <= 1.0
        float_animal_num_list: list[float] = [animal.frequency * self.verse_len / animal.len for animal in animal_values]
        float_total_animal_num = sum(float_animal_num_list) - 1e-11
        self.total_animal_num = int(float_total_animal_num + 1e-10)
        animal_num_list = np.diff(np.digitize(
            np.cumsum(float_animal_num_list),
            np.linspace(start=0, stop=float_total_animal_num, num=self.total_animal_num)), prepend=0)
        self.animal_num: dict = {animal_keys[i]: animal_num_list[i] for i in range(len(animal_num_list))}

        if len(self.animal_num) > 0:
            pass

        num_animals = int(np.sum([self.animal_num[key] for key, animal in self.zoo.items()]) + .5)
        if num_animals == 0:
            total_animal_len = 0
            all_animals = []
        else:
            total_animal_len = np.sum([animal.len * self.animal_num[key] for key, animal in self.zoo.items()])
            all_animals = np.concatenate(
                [np.repeat(key, num) for key, num in self.animal_num.items()]) if total_animal_len > 0 else []

        random_len = self.verse_len - total_animal_len
        pure_noise = self.rng.choice([0, 1], size=random_len)
        self.rng.shuffle(all_animals)
        if self.animal_position == 'random':
            self.insertion_points = np.sort(self.rng.choice(random_len, size=num_animals))
        elif self.animal_position == 'equidistant':
            self.insertion_points = ((np.arange(num_animals) + 0.5) * (random_len / num_animals)).astype('int')
        insertion_points = np.concatenate([np.array([0]), self.insertion_points])
        concat_list = list(itertools.chain(
            np.concatenate([pure_noise[insertion_points[i]: insertion_points[i + 1]],
                                            self.zoo[all_animals[i]].vector]) for i in range(num_animals)))
        concat_list.append(pure_noise[insertion_points[num_animals]:])
        self.verse = np.concatenate(concat_list)

    def get_record_input(self, i) -> np.ndarray:
        return self.verse[i - self.record_input_len: i]

    def get_record_output(self, i) -> np.ndarray:
        return self.verse[i: i + self.record_output_len]

    @timed(logger=logger)
    def get_training_data(self, ordered: bool = False) -> tuple[np.ndarray, np.ndarray]:
        r = np.arange(self.record_input_len, self.verse_len - self.record_output_len, dtype=int)
        if not ordered:
            self.rng.shuffle(r)
        input = np.vstack([self.get_record_input(i) for i in r])
        output = np.vstack([self.get_record_output(i) for i in r])
        return input, output

    @timed(logger=logger)
    def get_input_data(self) -> np.ndarray:
        r = np.arange(self.record_input_len, self.verse_len, dtype=int)
        input = np.vstack([self.get_record_input(i) for i in r])
        return input

    def to_plt(self, to_index:int = None) -> plt:
        return make_plt(matrix=[self.verse[:to_index]], matrix_names='')

    def prediction_to_strs(self, to_index:int = None) -> list[str]:
        rjust_len = self.record_output_len + 1
        matrix = self.prediction_to_matrix(to_index=to_index)
        str_list = [f'{self.matrix_names[0].rjust(rjust_len)}: {"".join(prob_to_char(matrix[0, :]))}']
        str_list += [f'{self.matrix_names[i].rjust(rjust_len)}: {"".join(prob_to_char(matrix[i, :]))}'
                     for i in range(1, 2 ** self.record_output_len + 1)]
        return str_list

    @property
    def matrix_names(self) -> list[str]:
        bc = BinConverter(num_digits=self.record_output_len)
        return ['m'] + [f'o{bc.to_str(i)}' for i in range(2 ** self.record_output_len)]

    def prediction_to_matrix(self, to_index:int = None) -> np.ndarray:
        matrix = np.vstack([[self.verse[self.record_input_len -1: -1]], self.prediction.transpose()])
        if to_index:
            matrix = matrix[:, :to_index]
        return matrix

    def prediction_to_plt(self, to_index:int = None) -> plt:
        return make_plt(matrix=self.prediction_to_matrix(to_index=to_index), matrix_names=self.matrix_names)

    def prediction_to_data_frame(self, to_index:int = None) -> pd.DataFrame:
        df = pd.DataFrame(self.prediction_to_matrix(to_index=to_index).transpose())
        df.columns = self.matrix_names
        return df

    @property
    def prediction(self) -> Optional[np.ndarray]:
        assert self._prediction is not None, 'No prediction was set.'
        return self._prediction

    @prediction.setter
    def prediction(self, val: np.ndarray):
        assert isinstance(val, np.ndarray)
        expected_shape = (self.verse_len - self.record_input_len, 2 ** self.record_output_len)
        assert val.shape == expected_shape, f"{expected_shape = }, received_shape = {val.shape}"
        self._prediction = val

    def __str__(self) -> str:
        bc = BinConverter(num_digits=1)
        return "".join(bc.to_str(self.verse))
