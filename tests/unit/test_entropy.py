from miniverse.animal import lion, snake
from miniverse.entropy import compression_entropy
from miniverse.miniverse import Miniverse


def test_entropy():
    mv = Miniverse(verse_len = 100_000)
    assert compression_entropy(mv.verse) > 0.05
    mv = Miniverse(zoo={'lion1':lion, 'lion2':lion, 'snake':snake}, verse_len = 100_000)
    assert compression_entropy(mv.verse) < 0.035
    sequence = [0, 1] * 50_000  # Repeating pattern
    assert compression_entropy(sequence) < 0.002

