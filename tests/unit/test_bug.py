import numpy as np

from miniverse.bug import array_to_str, Bug


def test_array_to_str():
    s = array_to_str(np.array([0, 1, 1, 0]))
    assert s=='-++-'

def test_bug():
    bug = Bug(pattern='0110', segment_len=8)
    assert str(bug)=='--------++++++++++++++++--------'
    bug = Bug(pattern='01', repeat=16)
    assert str(bug)=='-+' * 16


def test_test():
    print(bin(9))
    print(int('1111', 2))
    print(type(bin(7)))
