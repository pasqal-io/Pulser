# Copyright 2024 Pulser Development Team
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

"""Custom implementation of math and array functions."""
from __future__ import annotations

from collections.abc import Sequence
from typing import cast, Protocol, TypeVar

import numpy as np
import scipy.fft

from pulser.math.abstract_array import (
    AbstractArray as AbstractArray,
    AbstractArrayLike,
)

try:
    import torch
except ImportError:  # pragma: no cover
    pass


T = TypeVar("T", covariant=True)


class TensorLike(Protocol[T]):
    """A type hint to signal that a parameter behaves like a torch Tensor."""

    def detach(self: T) -> T: ...  # noqa: D102

    def __array__(self) -> np.ndarray: ...


# Custom function definitions


def exp(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.exp(a.as_tensor()))
    return AbstractArray(np.exp(a.as_array()))


def sqrt(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.sqrt(a.as_tensor()))
    return AbstractArray(np.sqrt(a.as_array()))


def log2(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.log2(a.as_tensor()))
    return AbstractArray(np.log2(a.as_array()))


def log(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.log(a.as_tensor()))
    return AbstractArray(np.log(a.as_array()))


def sin(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.sin(a.as_tensor()))
    return AbstractArray(np.sin(a.as_array()))


def cos(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.cos(a.as_tensor()))
    return AbstractArray(np.cos(a.as_array()))


def tan(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.tan(a.as_tensor()))
    return AbstractArray(np.tan(a.as_array()))


def pad(
    a: AbstractArrayLike,
    pad_width: tuple | int,
    mode: str = "constant",
    constant_values: tuple | int | float = 0,
) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        t = cast(torch.Tensor, a._array)
        if isinstance(pad_width, (int, float)):
            pad_width = (pad_width, pad_width)
        if mode == "constant":
            if isinstance(constant_values, (int, float)):
                out = torch.nn.functional.pad(
                    t, pad_width, "constant", constant_values
                )
            else:
                out = torch.nn.functional.pad(
                    t, (pad_width[0], 0), "constant", constant_values[0]
                )
                out = torch.nn.functional.pad(
                    out, (0, pad_width[1]), "constant", constant_values[1]
                )
        elif mode == "edge":
            out = torch.nn.functional.pad(
                t, (pad_width[0], 0), "constant", float(t[0])
            )
            out = torch.nn.functional.pad(
                out, (0, pad_width[1]), "constant", float(t[-1])
            )
        return AbstractArray(out)

    arr = cast(np.ndarray, a._array)
    kwargs = (
        dict(constant_values=constant_values) if mode == "constant" else {}
    )
    return AbstractArray(
        np.pad(arr, pad_width, mode, **kwargs),  # type: ignore[call-overload]
    )


def fft(a: AbstractArrayLike) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.fft.fft(a.as_tensor()))
    return AbstractArray(scipy.fft.fft(a.as_array()))


def ifft(a: AbstractArrayLike) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.fft.ifft(a.as_tensor()))
    return AbstractArray(scipy.fft.ifft(a.as_array()))


def fftfreq(n: int) -> AbstractArray:
    return AbstractArray(scipy.fft.fftfreq(n))


def round(a: AbstractArrayLike, decimals: int = 0) -> AbstractArray:
    return AbstractArray(a).__round__(decimals)


def ceil(a: AbstractArrayLike) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.ceil(a.as_tensor()))
    return AbstractArray(np.ceil(a.as_array()))


def floor(a: AbstractArrayLike) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.floor(a.as_tensor()))
    return AbstractArray(np.floor(a.as_array()))


def mean(a: AbstractArrayLike, axis: int | None = None) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.mean(a.as_tensor(), dim=axis))
    return AbstractArray(np.mean(a.as_array(), axis=axis))


def sum(a: AbstractArrayLike) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.sum(a.as_tensor()))
    return AbstractArray(np.sum(a.as_array()))


def cumsum(a: AbstractArrayLike, axis: int = 0) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.cumsum(a.as_tensor(), dim=axis))
    return AbstractArray(np.cumsum(a.as_array(), axis=axis))


def diff(a: AbstractArrayLike) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.diff(a.as_tensor()))
    return AbstractArray(np.diff(a.as_array()))


def dot(a: AbstractArrayLike, b: AbstractArrayLike) -> AbstractArray:
    a, b = map(AbstractArray, (a, b))
    if a.is_tensor or b.is_tensor:
        return AbstractArray(torch.dot(a.as_tensor(), b.as_tensor()))
    return AbstractArray(np.dot(a.as_array(), b.as_array()))


def pdist(a: AbstractArrayLike) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.nn.functional.pdist(a.as_tensor()))
    return AbstractArray(scipy.spatial.distance.pdist(a.as_array()))


def concatenate(arrs: Sequence[AbstractArrayLike]) -> AbstractArray:
    abst_arrs = tuple(map(AbstractArray, arrs))
    if any(a.is_tensor for a in abst_arrs):
        return AbstractArray(torch.cat([a.as_tensor() for a in abst_arrs]))
    return AbstractArray(np.concatenate([a.as_array() for a in abst_arrs]))


def vstack(arrs: Sequence[AbstractArrayLike]) -> AbstractArray:
    abst_arrs = tuple(map(AbstractArray, arrs))
    if any(a.is_tensor for a in abst_arrs):
        return AbstractArray(torch.vstack([a.as_tensor() for a in abst_arrs]))
    return AbstractArray(np.vstack([a.as_array() for a in abst_arrs]))


def hstack(arrs: Sequence[AbstractArrayLike]) -> AbstractArray:
    abst_arrs = tuple(map(AbstractArray, arrs))
    if any(a.is_tensor for a in abst_arrs):
        return AbstractArray(torch.hstack([a.as_tensor() for a in abst_arrs]))
    return AbstractArray(np.hstack([a.as_array() for a in abst_arrs]))


def flatten(a: AbstractArrayLike) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.flatten(a.as_tensor()))
    return AbstractArray(a.as_array().flatten())
