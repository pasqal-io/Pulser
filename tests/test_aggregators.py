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
    with pytest.raises(ValueError, match="Cannot average 0 samples."):
        _mean_aggregator([])

    with pytest.raises(
        ValueError, match="Cannot average list of empty lists."
    ):
        _mean_aggregator([[], []])

    with pytest.raises(
        ValueError, match="Need to supply a list of values to average."
    ):
        _mean_aggregator("abcd")

    with pytest.raises(ValueError, match="Cannot average this type of data."):
        _mean_aggregator([{}, {}])

    with pytest.raises(
        ValueError, match=f"Cannot average list of lists of {type({})}."
    ):
        _mean_aggregator([[{}], [{}]])

    with pytest.raises(
        ValueError, match=f"Cannot average list of matrices of {type('a')}."
    ):
        _mean_aggregator([[["abcd"]], [["efgh"]]])

    with pytest.raises(
        ValueError, match="Cannot average list of matrices with empty columns."
    ):
        _mean_aggregator([[[]], [[]]])
