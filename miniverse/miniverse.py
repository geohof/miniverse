import logging
from dataclasses import dataclass
import numpy as np
from typing import Literal, Optional

import pandas as pd
from matplotlib import pyplot as plt

from miniverse.bin_converter import BinConverter, prob_to_char
from miniverse.bug import Bug
from miniverse.timer import timed

logger = logging.getLogger(__name__)


@dataclass
class Miniverse:
    bug_spec: dict[str, Bug]
    verse_len: int = 100000
    record_input_len: int = 40
    record_output_len: int = 1
    bug_position: Literal['random', 'equidistant'] = 'random'
    random_seed: int = None

    def __post_init__(self):
        self._prediction = None
        self.set_up(timed_log_str=f'setting up miniverse of size {self.verse_len}.')


    @timed(logger=logger)
    def set_up(self):
        self.rng = np.random.default_rng(self.random_seed)
        bug_keys = list(self.bug_spec.keys())
        bug_values = [self.bug_spec[key] for key in bug_keys]
        total_frequency = sum([bug.frequency for bug in bug_values])
        assert total_frequency <= 1.0
        float_bug_num_list: list[float] = [bug.frequency * self.verse_len / bug.len for bug in bug_values]
        float_total_bug_num = sum(float_bug_num_list) - 1e-11
        self.total_bug_num = int(float_total_bug_num + 1e-10)
        bug_num_list = np.diff(np.digitize(
            np.cumsum(float_bug_num_list),
            np.linspace(start=0, stop=float_total_bug_num, num=self.total_bug_num)), prepend=0)
        self.bug_num: dict = {bug_keys[i]: bug_num_list[i] for i in range(len(bug_num_list))}

        total_bug_len = np.sum([bug.len * self.bug_num[key] for key, bug in self.bug_spec.items()])
        random_len = self.verse_len - total_bug_len
        pure_noise = self.rng.choice([0, 1], size=random_len)
        all_bugs = np.concatenate([np.repeat(key, num) for key, num in self.bug_num.items()])
        self.rng.shuffle(all_bugs)
        num_bugs = len(all_bugs)
        if self.bug_position == 'random':
            self.insertion_points = np.sort(self.rng.choice(random_len, size=num_bugs))
        elif self.bug_position == 'equidistant':
            self.insertion_points = ((np.arange(num_bugs) + 0.5) * (random_len / num_bugs)).astype('int')
        insertion_points = np.concatenate([np.array([0]), self.insertion_points])
        verse: np.ndarray[int] = (
            np.concatenate([np.concatenate([pure_noise[insertion_points[i]: insertion_points[i + 1]],
                                            self.bug_spec[all_bugs[i]].vector]) for i in range(num_bugs)]))
        self.verse = np.concatenate([verse, pure_noise[insertion_points[num_bugs]:]])
        # # for key, bug in self.bug_spec.items():
        # #     insertion_points =
        # bug_segment_vector = [int(c) for c in self.bug_pattern]
        # bug_vector = np.repeat(bug_segment_vector, self.bug_segment_len)
        # bug_len = self.bug_segment_len * len(self.bug_pattern)
        #
        # num_bugs = int(self.verse_len / self.bug_interval_len)
        #
        # random_vector = self.rng.randint(0, self.bug_interval_len - bug_len, size=num_bugs)
        # linear_sequence = np.arange(0, num_bugs * self.bug_interval_len, self.bug_interval_len)
        #
        # bug_positions = linear_sequence + random_vector
        # for i in range(num_bugs):
        #     pure_noise[bug_positions[i]:bug_positions[i] + bug_len] = bug_vector
        #
        # self.verse = pure_noise

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


    def prediction_to_strs(self, to_index:int = None) -> list[str]:
        rjust_len = self.record_output_len + 1
        matrix = self.to_matrix(to_index=to_index)
        str_list = [f'{self.matrix_names[0].rjust(rjust_len)}: {"".join(prob_to_char(matrix[0, :]))}']
        str_list += [f'{self.matrix_names[i].rjust(rjust_len)}: {"".join(prob_to_char(matrix[i, :]))}'
                     for i in range(1, 2 ** self.record_output_len + 1)]
        return str_list

    @property
    def matrix_names(self) -> list[str]:
        bc = BinConverter(num_digits=self.record_output_len)
        return ['m'] + [f'o{bc.to_str(i)}' for i in range(2 ** self.record_output_len)]

    def to_matrix(self, to_index:int = None) -> np.ndarray:
        matrix = np.vstack([[self.verse[self.record_input_len -1: -1]], self.prediction.transpose()])
        if to_index:
            matrix = matrix[:, :to_index]
        return matrix


    def to_plt(self, to_index:int = None) -> plt:
        matrix = self.to_matrix(to_index=to_index)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.matshow(matrix, cmap='viridis')
        ax.set_yticks(np.arange(len(matrix)))
        ax.set_yticklabels(self.matrix_names)
        def tmp_format(x):
            integer = int(val * 100 + .5)
            if integer == 100:
                return '**'
            else:
                return str(integer).rjust(2, '0')
        for (i, j), val in np.ndenumerate(matrix):
            txt = int(val + 0.5) if i == 0 else tmp_format(val)
            ax.text(j, i, txt, ha='center', va='center', color='white')
        return plt

    def to_data_frame(self, to_index:int = None) -> pd.DataFrame:
        df = pd.DataFrame(self.to_matrix(to_index=to_index).transpose())
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


def get_data_from_miniverse():
    pass
