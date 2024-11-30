
from miniverse.minimal_miniverse import Miniverse
from miniverse.bug import Bug, array_to_str


def insert_newlines(text: str, every: int = 100) -> str:
    return '\n'.join(text[i:i + every] for i in range(0, len(text), every))


def test_create_miniverse():
    mv = Miniverse(
        bug_spec={
            'zebra': Bug(pattern='01', repeat=16, frequency=.2),
            'snake': Bug(pattern='0101', segment_len=8, frequency=.1)
        },
        verse_len=1000
    )
    mvv = mv.verse
    assert len(mvv) == mv.verse_len
    assert set(mvv).issubset({0, 1})
    assert mv.bug_num == {'zebra': 6, 'snake': 3}
    print(insert_newlines(array_to_str(mv.verse), 100))
