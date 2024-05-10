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
"""Defines the AbstractArray class."""
from __future__ import annotations

import functools
import importlib.util
import operator
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

try:
    import torch
except ImportError:
    pass


class AbstractArray:
    """An abstract array containing an array or tensor.

    Args:
        array: The array to store.
    """

    def __init__(self, array: AbstractArrayLike):
        """Initializes a new AbstractArray."""
        self._array: np.ndarray | torch.Tensor
        if isinstance(array, AbstractArray):
            self._array = array._array
        elif self.has_torch() and type(array) is torch.Tensor:
            self._array = array
        else:
            self._array = np.asarray(array, dtype=float)

    @staticmethod
    @functools.lru_cache
    def has_torch() -> bool:
        """Checks whether torch is installed."""
        return importlib.util.find_spec("torch") is not None

    @functools.cached_property
    def is_tensor(self) -> bool:
        """Whether the stored array is a tensor."""
        return self.has_torch() and type(self._array) is torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        """Converts the stored array to a torch Tensor."""
        if not self.has_torch():
            raise RuntimeError("`torch` is not installed.")
        return torch.as_tensor(self._array)

    def as_array(self, detach: bool = False) -> np.ndarray:
        """Converts the stored array to a Numpy array.

        Args:
            detach: Whether to detach before converting.
        """
        if detach:
            return cast(
                np.ndarray,
                (
                    cast(torch.Tensor, self._array).detach().numpy()
                    if self.is_tensor
                    else np.array(self._array)
                ),
            )
        return np.asarray(self._array)

    def copy(self) -> AbstractArray:
        """Makes a copy itself."""
        return AbstractArray(
            cast(torch.Tensor, self._array).clone()
            if self.is_tensor
            else cast(np.ndarray, self._array).copy()
        )

    @property
    def size(self) -> int:
        """The number of elements in the array."""
        return int(np.prod(self._array.shape))

    @property
    def real(self) -> AbstractArray:
        """The real part of each element in the array."""
        return AbstractArray(self._array.real)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return self._array.__array__(dtype)

    def __array_wrap__(self, array: np.ndarray) -> AbstractArray:
        return AbstractArray(self._array.__array_wrap__(array))

    def __repr__(self) -> str:
        return str(self._array.__repr__())

    # Unary operators
    def __neg__(self) -> AbstractArray:
        return AbstractArray(-self._array)

    def __abs__(self) -> AbstractArray:
        return AbstractArray(cast(ArrayLike, abs(self._array)))

    def __round__(self, decimals: int = 0) -> AbstractArray:
        return AbstractArray(
            torch.round(cast(torch.Tensor, self._array), decimals)
            if self.is_tensor
            else np.round(cast(np.ndarray, self._array), decimals)
        )

    def __int__(self) -> int:
        return int(self._array)

    def __float__(self) -> int:
        return float(self._array)

    def _binary_operands(
        self, other: AbstractArrayLike
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        other = AbstractArray(other)
        if self.is_tensor or other.is_tensor:
            return self.as_tensor(), other.as_tensor()
        return self.as_array(), other.as_array()

    # Binary operators
    def __add__(self, other: AbstractArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.add(*self._binary_operands(other)))

    def __radd__(self, other: ArrayLike, /) -> AbstractArray:
        return self.__add__(other)

    def __mul__(self, other: AbstractArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.mul(*self._binary_operands(other)))

    def __rmul__(self, other: ArrayLike, /) -> AbstractArray:
        return self.__mul__(other)

    def __sub__(self, other: AbstractArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.sub(*self._binary_operands(other)))

    def __rsub__(self, other: ArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.sub(*self._binary_operands(other)[::-1]))

    def __truediv__(self, other: AbstractArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.truediv(*self._binary_operands(other)))

    def __rtruediv__(self, other: ArrayLike, /) -> AbstractArray:
        return AbstractArray(
            operator.truediv(*self._binary_operands(other)[::-1])
        )

    def __floordiv__(self, other: AbstractArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.floordiv(*self._binary_operands(other)))

    def __rfloordiv__(self, other: ArrayLike, /) -> AbstractArray:
        return AbstractArray(
            operator.floordiv(*self._binary_operands(other)[::-1])
        )

    def __pow__(self, other: AbstractArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.pow(*self._binary_operands(other)))

    def __rpow__(self, other: ArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.pow(*self._binary_operands(other)[::-1]))

    def __mod__(self, other: AbstractArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.mod(*self._binary_operands(other)))

    def __rmod__(self, other: ArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.mod(*self._binary_operands(other)[::-1]))

    def __matmul__(self, other: AbstractArrayLike, /) -> AbstractArray:
        return AbstractArray(operator.matmul(*self._binary_operands(other)))

    def __rmatmul__(self, other: ArrayLike, /) -> AbstractArray:
        return AbstractArray(
            operator.matmul(*self._binary_operands(other)[::-1])
        )

    def __getitem__(self, indices: Any) -> AbstractArray:
        return AbstractArray(self._array[indices])

    def __setitem__(self, indices: Any, values: AbstractArrayLike) -> None:
        array, values = self._binary_operands(values)
        array[indices] = values  # type: ignore[assignment]
        self._array = array
        del self.is_tensor  # Clears cache

    def __len__(self) -> int:
        return len(self._array)


AbstractArrayLike = ArrayLike | AbstractArray
