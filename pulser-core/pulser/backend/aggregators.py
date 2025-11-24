# Copyright 2025 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines aggregation functions for use in `Results.aggregate`."""

from collections import Counter
from typing import Callable, Sequence, TypeVar, cast

import numpy as np

import pulser.math as pm
from pulser.backend.observable import AggregationMethod

T = TypeVar(
    "T",
    float,
    list[float],
    list[list[float]],
    complex,
    list[complex],
    list[list[complex]],
    "pm.torch.Tensor",
    np.ndarray,
)


def _mean_aggregator(
    values: list[T],
) -> T:
    """Take the mean of the given results.

    Argument:
        values: The results to average. Supported are lists of:
            numeric values, lists of numeric values,
            lists of lists of numeric values, torch Tensors and numpy arrays.

    Returns:
        The average over the first dimension of the provided results.
    """
    if not isinstance(values, list):
        raise ValueError("Need to supply a list of values to average.")
    if values == []:
        raise ValueError("Cannot average 0 samples.")

    elt = values[0]

    if pm.AbstractArray.has_torch() and isinstance(elt, pm.torch.Tensor):
        return pm.torch.stack(values).mean(dim=0)

    if isinstance(elt, np.ndarray):
        return cast(np.ndarray, np.stack(values).mean(axis=0))

    if isinstance(elt, float):
        return cast(float, np.mean(values))  # this would have type np.floating
    if isinstance(elt, complex):
        return cast(
            complex, np.mean(values)
        )  # this would have type np.complexfloating

    if not isinstance(elt, Sequence):
        raise ValueError("Cannot average this type of data.")

    if values[0] == []:
        raise ValueError("Cannot average list of empty lists.")

    if isinstance(elt[0], (float, complex)):
        return list(np.mean(values, axis=0).tolist())

    if not isinstance(elt[0], list):
        raise ValueError(f"Cannot average list of lists of {type(elt[0])}.")

    if len(elt[0]) == 0:
        raise ValueError("Cannot average list of matrices with empty columns.")

    if not isinstance(elt[0][0], (float, complex)):
        raise ValueError(
            f"Cannot average list of matrices of {type(elt[0][0])}."
        )
    return list(np.mean(values, axis=0).tolist())


def _bag_union_aggregator(
    values: list[Counter],
) -> Counter:
    """Join a list of Counter objects."""
    return sum(map(Counter, values), start=Counter())


AGGREGATOR_MAPPING: dict[AggregationMethod, Callable] = {
    AggregationMethod.MEAN: _mean_aggregator,
    AggregationMethod.BAG_UNION: _bag_union_aggregator,
}
