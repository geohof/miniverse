
from miniverse.miniverse import Miniverse
from miniverse.bug import Bug, array_to_str


def insert_newlines(text: str, every: int = 100) -> str:
    return '\n'.join(text[i:i + every] for i in range(0, len(text), every))

def test_miniverse_bug_num():
    bug_spec = {
        'zebra': Bug(pattern='01', repeat=16, frequency=.1),
        'snake': Bug(pattern='0101', segment_len=8, frequency=.2),
        'lion': Bug(pattern='0110', segment_len=8, frequency=.2)
    }

    mv = Miniverse(bug_spec=bug_spec, verse_len=64)
    assert mv.bug_num == {'zebra': 1, 'snake': 0, 'lion': 0}

    mv = Miniverse(bug_spec=bug_spec, verse_len=64 * 2 - 1)
    assert mv.bug_num == {'zebra': 1, 'snake': 0, 'lion': 0}

    mv = Miniverse(bug_spec=bug_spec, verse_len=64 * 2)
    assert mv.bug_num == {'zebra': 1, 'snake': 0, 'lion': 1}

    mv = Miniverse(bug_spec=bug_spec, verse_len=64 * 3)
    assert mv.bug_num == {'zebra': 1, 'snake': 1, 'lion': 1}

    mv = Miniverse(bug_spec=bug_spec, verse_len=64 * 5_000 - 1)
    assert mv.bug_num == {'zebra': 1_000, 'snake': 1_999, 'lion': 2_000}

    mv = Miniverse(bug_spec=bug_spec, verse_len=64 * 5_000)
    assert mv.bug_num == {'zebra': 1_000, 'snake': 2000, 'lion': 2_000}

def test_miniverse_insertion_point():
    mv = Miniverse(bug_spec={'snake': Bug(pattern='0101', segment_len=8, frequency=.7)}, verse_len=50,
                   bug_position='equidistant')
    assert list(mv.insertion_points) == [9]
    mv = Miniverse(bug_spec={'snake': Bug(pattern='0101', segment_len=8, frequency=.7)}, verse_len=100,
                   bug_position='equidistant')
    assert list(mv.insertion_points) == [9, 27]
    print(str(mv))



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
