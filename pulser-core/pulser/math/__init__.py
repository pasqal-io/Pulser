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

import numpy as np
import scipy.fft

from pulser.math.abstract_array import (
    AbstractArray as AbstractArray,
    AbstractArrayLike,
)

try:
    import torch
except ImportError:
    pass

# Custom function definitions


def sin(a: AbstractArrayLike, /) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        return AbstractArray(torch.sin(a.as_tensor()))
    return AbstractArray(np.sin(a.as_array()))


def pad(
    a: AbstractArrayLike,
    pad_width: tuple | int,
    mode: str = "constant",
    constant_values: tuple | int | float = 0,
) -> AbstractArray:
    a = AbstractArray(a)
    if a.is_tensor:
        if mode == "constant":
            if isinstance(pad_width, int) and isinstance(
                constant_values, (int, float)
            ):
                out = torch.nn.functional.pad(
                    a._array,
                    (pad_width, pad_width),
                    "constant",
                    constant_values,
                )
            elif isinstance(pad_width, tuple) and isinstance(
                constant_values, (int, float)
            ):
                out = torch.nn.functional.pad(
                    a._array, pad_width, "constant", constant_values
                )
            elif isinstance(pad_width, int) and isinstance(
                constant_values, tuple
            ):
                out = torch.nn.functional.pad(
                    a._array, (pad_width, 0), "constant", constant_values[0]
                )
                out = torch.nn.functional.pad(
                    out, (0, pad_width), "constant", constant_values[1]
                )
            else:
                out = torch.nn.functional.pad(
                    a._array,
                    (pad_width[0], 0),
                    "constant",
                    constant_values[0],
                )
                out = torch.nn.functional.pad(
                    out,
                    (0, pad_width[1]),
                    "constant",
                    constant_values[1],
                )
        elif mode == "edge":
            if isinstance(pad_width, (int, float)):
                out = torch.nn.functional.pad(
                    a._array, (pad_width, 0), "constant", a._array[0]
                )
                out = torch.nn.functional.pad(
                    out, (0, pad_width), "constant", a._array[-1]
                )
            else:
                out = torch.nn.functional.pad(
                    a._array, (pad_width[0], 0), "constant", a._array[0]
                )
                out = torch.nn.functional.pad(
                    out, (0, pad_width[1]), "constant", a._array[-1]
                )
        return AbstractArray(out)
    if mode == "constant":
        return AbstractArray(
            np.pad(a._array, pad_width, mode, constant_values=constant_values)
        )
    elif mode == "edge":
        return AbstractArray(
            np.pad(a._array, pad_width, mode, constant_values=constant_values)
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
