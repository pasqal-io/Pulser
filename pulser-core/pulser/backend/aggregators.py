import collections
import statistics
from typing import Any, Callable

from numpy.typing import ArrayLike

import pulser.math as pm
from pulser.backend.observable import AggregationType

_NUMERIC_TYPES = {int, float, complex}


def mean_aggregator(
    values: list[Any],
) -> (
    complex
    | float
    | list[complex]
    | list[float]
    | list[list[complex]]
    | list[list[float]]
    | ArrayLike
):  # FIXME: support tuples?
    if values == []:
        raise ValueError("Cannot average 0 samples")

    element_type = type(values[0])

    if element_type in _NUMERIC_TYPES:
        return statistics.fmean(values)

    if pm.AbstractArray.has_torch() and element_type == pm.torch.Tensor:
        acc = pm.torch.zeros_like(values[0])
        for ten in values:
            acc += ten
        return acc / len(values)

    if element_type != list:
        raise NotImplementedError("Cannot average this type of data")

    if values[0] == []:
        raise ValueError("Cannot average list of empty lists")

    sub_element_type = type(values[0][0])

    if sub_element_type in _NUMERIC_TYPES:
        dim = len(values[0])
        return [
            statistics.fmean(value[i] for value in values) for i in range(dim)
        ]

    if sub_element_type != list:  # FIXME: ABC.Iterable? Collection? subclass?
        raise ValueError(f"Cannot average list of lists of {sub_element_type}")

    if values[0][0] == []:
        raise ValueError("Cannot average list of matrices with no columns")

    if (sub_sub_element_type := type(values[0][0][0])) not in _NUMERIC_TYPES:
        raise ValueError(
            f"Cannot average list of matrices of {sub_sub_element_type}"
        )

    dim1 = len(values[0])
    dim2 = len(values[0][0])
    return [
        [
            statistics.fmean(value[i][j] for value in values)
            for j in range(dim2)
        ]
        for i in range(dim1)
    ]


def bag_union_aggregator(
    values: list[collections.Counter],
) -> collections.Counter:
    return sum(values, start=collections.Counter())


aggregation_type_definitions: dict[AggregationType, Callable] = {
    AggregationType.MEAN: mean_aggregator,
    AggregationType.BAG_UNION: bag_union_aggregator,
}
