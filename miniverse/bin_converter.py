from dataclasses import dataclass

import numpy as np

@np.vectorize
def to_str(integer: int, num_digits: int) -> str:
    assert 0 <= integer <= 2 ** num_digits - 1, \
        f"Parameter integer needs to be in the interval [0, {2 ** num_digits})"
    return bin(integer)[2:].rjust(num_digits, '0')

@np.vectorize
def to_int(string: str, num_digits: int) -> int:
    integer = int(string, 2)
    assert 0 <= integer <= 2 ** num_digits - 1, \
        f"Resulting integer needs to be in the interval [0, {2 ** num_digits})"
    return integer



@np.vectorize
def prob_to_char(prob: float, num_digits: int = 1) -> str:
    integer = int(prob * (10 ** num_digits) + 0.5)
    char = str(integer).rjust(num_digits, '0')
    if integer == 10 ** num_digits:
        char = "*" * num_digits
    return char




@dataclass
class BinConverter:
    num_digits: int = 3

    def to_str(self, integer: int) -> str:
        return to_str(integer=integer, num_digits=self.num_digits)

    def to_int(self, string: str) -> int:
        return to_int(string=string, num_digits=self.num_digits)
