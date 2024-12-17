import numpy as np
from miniverse.animal import array_to_str, Animal

def test_array_to_str():
    s = array_to_str(np.array([0, 1, 1, 0]))
    assert s=='-++-'

def test_animal():
    lion = Animal(pattern='0110', segment_len=8)
    assert str(lion)=='--------++++++++++++++++--------'
    zebra = Animal(pattern='01', repeat=16)
    assert str(zebra) == '-+' * 16


def test_animal_plot():
    lion = Animal(pattern='0110', segment_len=8)
    lion.to_plt().show()
