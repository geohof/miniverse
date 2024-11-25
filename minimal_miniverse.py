from dataclasses import dataclass

import numpy as np
from numpy import floor



@dataclass
class Miniverse:
    verse_len: int = 1000
    bug_len: int = 30
    bug_pattern: str = "010"
    bug_interval_len: int = 100
    record_input_len: int = 50
    record_output_len: int = 1


    def __post_init__(self):

        pure_noise = np.random.choice([0, 1], size=self.verse_len)
        bug_vector = [int(c) for c in self.bug_pattern]
        bug_segment_len = int(self.bug_len / len(self.bug_pattern))

        num_bugs = int(self.verse_len / self.bug_interval_len)

        random_vector = np.random.randint(0, self.bug_interval_len - self.bug_len, size=num_bugs)
        linear_sequence = np.arange(0, num_bugs * self.bug_interval_len, self.bug_interval_len)

        bug_positions = linear_sequence + random_vector
        for i in range(num_bugs):
            pure_noise[bug_positions[i]:bug_positions[i] + self.bug_len] = np.repeat(bug_vector, bug_segment_len)

        self.verse = pure_noise


    def get_record(self, i):
        input = self.verse[i - self.record_input_len: i]
        output = self.verse[i: i + self.record_output_len]
        return input, output
    
    def get_data(self):
        data = []
        for i in range(self.record_input_len, self.verse_len - self.record_output_len):
            input, output = self.get_record(i)
            data.append((input, output))
        return zip(*data)





def get_data_from_miniverse():
    pass


