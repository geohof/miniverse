import logging
from dataclasses import dataclass
import numpy as np
from typing import Literal

from miniverse.bin_converter import BinConverter
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

    @timed(logger=logger)
    def __post_init__(self):
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
    def get_data(self, ordered: bool = False):
        r = np.arange(self.record_input_len, self.verse_len - self.record_output_len, dtype=int)
        if not ordered:
            self.rng.shuffle(r)
        input = np.vstack([self.get_record_input(i) for i in r])
        output = np.vstack([self.get_record_output(i) for i in r])
        return input, output

    def __str__(self) -> str:
        bc = BinConverter(num_digits=1)
        return "".join(bc.to_str(self.verse))


def get_data_from_miniverse():
    pass
