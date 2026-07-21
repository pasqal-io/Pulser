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


def _validate_values(values: list[T]) -> None:
    """Validate that ``values`` is a non-empty list.

    Args:
        values: The list of values to validate.
        action: A verb describing the operation (used in error messages).

    Raises:
        ValueError: If ``values`` is not a list or is empty.
    """
    if not isinstance(values, list):
        raise ValueError("Need to supply a list of values to process.")
    if values == []:
        raise ValueError("Cannot process 0 samples.")


def _validate_sequence_elements(elt: Sequence) -> None:
    """Validate the nested structure of a sequence element.

    Args:
        elt: The first element of the values list (must be a ``Sequence``).

    Raises:
        ValueError: If the nested structure contains bad types.
    """
    if elt == []:
        raise ValueError("Cannot process list of empty lists.")

    if not isinstance(elt[0], (float, complex, list)):
        raise ValueError(f"Cannot process list of lists of {type(elt[0])}.")

    if isinstance(elt[0], list):
        if len(elt[0]) == 0:
            raise ValueError(
                "Cannot process list of matrices with empty columns."
            )
        if not isinstance(elt[0][0], (float, complex)):
            raise ValueError(
                f"Cannot process list of matrices of {type(elt[0][0])}."
            )


def _std_aggregator(
    values: list[T],
) -> T:
    """Get the standard deviation of the given results.

    Argument:
        values: The results to use. Supported are lists of:
            numeric values, lists of numeric values,
            lists of lists of numeric values, torch Tensors and numpy arrays.

    Returns:
        The standard deviation over the first dimension of the given values.
    """
    _validate_values(values)

    elt = values[0]

    if pm.AbstractArray.has_torch() and isinstance(elt, pm.torch.Tensor):
        return pm.torch.stack(values).std(dim=0)

    if isinstance(elt, np.ndarray):
        return cast(np.ndarray, np.stack(values).std(axis=0, ddof=1))

    if isinstance(elt, float):
        return cast(
            float, np.std(values, ddof=1)
        )  # this would have type np.floating

    if isinstance(elt, complex):
        return cast(
            complex, np.std(values, ddof=1)
        )  # this would have type np.complexfloating

    if not isinstance(elt, Sequence):
        raise ValueError("Cannot process this type of data.")

    _validate_sequence_elements(elt)

    return list(np.std(values, axis=0, ddof=1).tolist())


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
    _validate_values(values)

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
        raise ValueError("Cannot process this type of data.")

    _validate_sequence_elements(elt)

    return list(np.mean(values, axis=0).tolist())


def _mean_std_aggregator(
    values: list[T],
) -> tuple[T, T]:
    """Get the mean and standard deviation of the given results.

    Argument:
        values: The results to use. Supported are lists of:
            numeric values, lists of numeric values,
            lists of lists of numeric values, torch Tensors and numpy arrays.

    Returns:
        A tuple (mean, standard deviation)
            over the first dimension of the provided results.
    """
    mean = _mean_aggregator(values)
    std = _std_aggregator(values)
    return (mean, std)


def _bag_union_aggregator(
    values: list[Counter],
) -> Counter:
    """Join a list of Counter objects."""
    return sum(map(Counter, values), start=Counter())


AGGREGATOR_MAPPING: dict[AggregationMethod, Callable] = {
    AggregationMethod.MEAN: _mean_aggregator,
    AggregationMethod.BAG_UNION: _bag_union_aggregator,
    AggregationMethod.STD: _std_aggregator,
    AggregationMethod.MEANSTD: _mean_std_aggregator,
}
