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

import collections
from numbers import Number
from typing import Any, Callable, List, Union

import numpy as np
from numpy.typing import ArrayLike

import pulser.math as pm
from pulser.backend.observable import AggregationType


def mean_aggregator(
    values: list[Any],
) -> Union[Number, List[Number], List[List[Number]], ArrayLike]:
    """Take the mean of the given Results.

    Supported are numeric values, lists of numeric values
    lists of lists of numeric values, torch Tensors and numpy arrays.
    """
    if values == []:
        raise ValueError("Cannot average 0 samples")

    elt = values[0]

    if pm.AbstractArray.has_torch() and isinstance(elt, pm.torch.Tensor):
        return pm.torch.stack(values).mean(dim=0)

    if isinstance(elt, np.ndarray):
        return np.stack(values).mean(axis=0)  # type: ignore[no-any-return]

    if isinstance(elt, Number):
        return np.mean(values)  # type: ignore[no-any-return]

    if not isinstance(elt, list):
        raise ValueError("Cannot average this type of data")

    if values[0] == []:
        raise ValueError("Cannot average list of empty lists")

    sub_element_type = type(elt[0])

    if isinstance(elt[0], Number):
        dim = len(elt)
        return [np.mean([value[i] for value in values]) for i in range(dim)]

    if sub_element_type != list:  # FIXME: ABC.Iterable? Collection? subclass?
        raise ValueError(f"Cannot average list of lists of {sub_element_type}")

    if elt[0] == []:
        raise ValueError("Cannot average list of matrices with empty columns")

    if not isinstance(elt[0][0], Number):
        raise ValueError(
            f"Cannot average list of matrices of {type(elt[0][0])}"
        )

    dim1 = len(elt)
    dim2 = len(elt[0])
    return [
        [np.mean([value[i][j] for value in values]) for j in range(dim2)]
        for i in range(dim1)
    ]


def bag_union_aggregator(
    values: list[collections.Counter],
) -> collections.Counter:
    """Join a list of Counter objects."""
    return sum(values, start=collections.Counter())


aggregation_type_definitions: dict[AggregationType, Callable] = {
    AggregationType.MEAN: mean_aggregator,
    AggregationType.BAG_UNION: bag_union_aggregator,
}
