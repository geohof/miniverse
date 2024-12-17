import numpy as np
import pytest

from miniverse.bin_converter import BinConverter, prob_to_char


def test_bin_converter_to_str():
    bc = BinConverter()
    assert bc.to_str(integer=np.array([5]))=="101"
    assert bc.to_str(3)=="011"
    assert np.array_equal(bc.to_str(np.array([5, 6])), np.array(['101', '110']))
    assert np.array_equal(bc.to_str([5, 6]), np.array(['101', '110']))
    with pytest.raises(AssertionError) as e:
        _ = bc.to_str(9)
    assert "Parameter integer needs to be in the interval [0, 8)" in str(e)

def test_bin_converter_to_int():
    bc = BinConverter()
    assert bc.to_int("101") == 5
    assert bc.to_int("011") == 3
    assert np.array_equal(bc.to_int(["011", "1"]), [3, 1])
    with pytest.raises(AssertionError) as e:
        _ = bc.to_int("1000")
    assert "Resulting integer needs to be in the interval [0, 8)" in str(e)



def test_prob_to_char():
    assert prob_to_char(.6) == '6'
    assert prob_to_char(.34) == '3'
    assert prob_to_char(.19) == '2'
    assert prob_to_char(.98) == '*'
    assert prob_to_char(.98, num_digits=2) == '98'
    assert prob_to_char(.02, num_digits=2) == '02'
    assert prob_to_char(.002, num_digits=2) == '00'
