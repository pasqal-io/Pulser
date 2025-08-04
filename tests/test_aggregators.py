from collections import Counter

import numpy as np
import pytest

from pulser.backend.aggregators import _bag_union_aggregator, _mean_aggregator


def test_bag_union():
    counter1 = Counter({"1010": 5, "0101": 7, "0000": 2})
    counter2 = Counter({"1010": 3, "0101": 9, "1111": 4})

    union = _bag_union_aggregator([counter1, counter2])
    print(union)
    assert union == {"1010": 8, "0101": 16, "0000": 2, "1111": 4}


@pytest.mark.parametrize(
    "test_torch",
    [True, False],
)
def test__mean_aggregator(test_torch: bool):
    input = [1.0, 2.0, 3.0, 4.0]
    assert _mean_aggregator(input) == 2.5

    input2 = [1.0j, 2.0j, 3.0j, 4.0j]
    assert _mean_aggregator(input2) == 2.5j

    input3 = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 4.0]),
        np.array([3.0, 4.0, 5.0]),
    ]
    assert np.all(_mean_aggregator(input3) == np.array([2.0, 3.0, 4.0]))

    input4 = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    assert _mean_aggregator(input4) == [2.0, 3.0, 4.0]

    input5 = [[[1.0, 2.0, 3.0]], [[2.0, 3.0, 4.0]], [[3.0, 4.0, 5.0]]]
    assert _mean_aggregator(input5) == [[2.0, 3.0, 4.0]]
    if test_torch:
        torch = pytest.importorskip("torch")
        input6 = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([2.0, 3.0, 4.0]),
            torch.tensor([3.0, 4.0, 5.0]),
        ]
        assert torch.allclose(
            _mean_aggregator(input6), torch.tensor([2.0, 3.0, 4.0])
        )


def test__mean_aggregator_errors():
    with pytest.raises(ValueError) as ex:
        _mean_aggregator([])
    assert str(ex.value) == "Cannot average 0 samples"

    with pytest.raises(ValueError) as ex:
        _mean_aggregator([[], []])
    assert str(ex.value) == "Cannot average list of empty lists"

    with pytest.raises(AssertionError) as ex:
        _mean_aggregator("abcd")
    assert str(ex.value) == "Need to supply a list of values to average."

    with pytest.raises(ValueError) as ex:
        _mean_aggregator([{}, {}])
    assert str(ex.value) == "Cannot average this type of data"

    with pytest.raises(ValueError) as ex:
        _mean_aggregator([[{}], [{}]])
    assert str(ex.value) == f"Cannot average list of lists of {type({})}"

    with pytest.raises(ValueError) as ex:
        _mean_aggregator([[["abcd"]], [["efgh"]]])
    assert str(ex.value) == f"Cannot average list of matrices of {type('a')}"

    with pytest.raises(ValueError) as ex:
        _mean_aggregator([[[]], [[]]])
    assert (
        str(ex.value) == "Cannot average list of matrices with empty columns"
    )
