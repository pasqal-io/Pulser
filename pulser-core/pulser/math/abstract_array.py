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
from typing import Any, Generator, Union, cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from pulser.json.utils import obj_to_dict

try:
    import torch
except ImportError:  # pragma: no cover
    pass


class AbstractArray:
    """An abstract array containing an array or tensor.

    Args:
        array: The array to store.
        dtype: The data type of the array.
        force_array: Forces the array to be at least 1D.
    """

    def __init__(
        self,
        array: AbstractArrayLike,
        dtype: DTypeLike = None,
        force_array: bool = False,
    ):
        """Initializes a new AbstractArray."""
        self._array: np.ndarray | torch.Tensor
        if isinstance(array, AbstractArray):
            self._array = array._array
        elif self.has_torch() and isinstance(array, torch.Tensor):
            self._array = torch.as_tensor(
                array,
                dtype=dtype,  # type: ignore[arg-type]
            )
        else:
            self._array = np.asarray(array, dtype=dtype)

        if force_array and self._array.ndim == 0:
            self._array = self._array[None]

    @staticmethod
    @functools.lru_cache
    def has_torch() -> bool:
        """Checks whether torch is installed."""
        return importlib.util.find_spec("torch") is not None

    @functools.cached_property
    def is_tensor(self) -> bool:
        """Whether the stored array is a tensor."""
        return self.has_torch() and isinstance(self._array, torch.Tensor)

    @property
    def requires_grad(self) -> bool:
        """Whether the stored array is a tensor that needs a gradient."""
        return self.is_tensor and cast(torch.Tensor, self._array).requires_grad

    def astype(self, dtype: DTypeLike) -> AbstractArray:
        """Casts the data type of the array contents."""
        if self.is_tensor:
            return AbstractArray(
                cast(torch.Tensor, self._array).to(
                    dtype=dtype  # type: ignore[arg-type]
                )
            )
        return AbstractArray(cast(np.ndarray, self._array).astype(dtype))

    def as_tensor(self) -> torch.Tensor:
        """Converts the stored array to a torch Tensor."""
        if not self.has_torch():
            raise RuntimeError("`torch` is not installed.")
        return torch.as_tensor(self._array)

    def as_array(self, *, detach: bool = False) -> np.ndarray:
        """Converts the stored array to a Numpy array.

        Args:
            detach: Whether to detach before converting.
        """
        if detach and self.is_tensor:
            return cast(torch.Tensor, self._array).detach().numpy()
        return np.asarray(self._array)

    def tolist(self) -> list:
        """Converts the stored array to a Python list."""
        return self._array.tolist()

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
    def ndim(self) -> int:
        """The number of dimensions in the array."""
        return self._array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        return self._array.shape

    @property
    def real(self) -> AbstractArray:
        """The real part of each element in the array."""
        return AbstractArray(self._array.real)

    @property
    def dtype(self) -> Any:
        """The data type of the array elements."""
        return self._array.dtype

    def detach(self) -> AbstractArray:
        """Detaches the data from the computational graph.

        Analogous to torch.Tensor.detach().
        """
        if self.is_tensor:
            return AbstractArray(cast(torch.Tensor, self._array).detach())
        return self

    def __array__(
        self,
        dtype: None = None,
        copy: np.bool_ | None = None,
    ) -> np.ndarray:
        if self.is_tensor or np.lib.NumpyVersion(np.__version__) < "2.0.0":
            array: np.ndarray = self._array.__array__(dtype)
            if copy:
                return np.copy(array)
            else:
                return array
        else:  # pragma: no cover
            return self._array.__array__(dtype, copy=copy)  # type: ignore

    def __repr__(self) -> str:
        return str(self._array.__repr__())

    def __int__(self) -> int:
        return int(self._array)

    def __float__(self) -> float:
        return float(self._array)

    def __bool__(self) -> bool:
        return bool(self._array)

    # Unary operators
    def __neg__(self) -> AbstractArray:
        return AbstractArray(-self._array)

    def __abs__(self) -> AbstractArray:
        return AbstractArray(cast(ArrayLike, abs(self._array)))

    def __round__(self, decimals: int = 0, /) -> AbstractArray:
        return AbstractArray(
            torch.round(cast(torch.Tensor, self._array), decimals=decimals)
            if self.is_tensor
            else np.round(cast(np.ndarray, self._array), decimals=decimals)
        )

    def _binary_operands(
        self, other: AbstractArrayLike
    ) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
        other = AbstractArray(other)
        if self.is_tensor or other.is_tensor:
            return self.as_tensor(), other.as_tensor()
        return self.as_array(), other.as_array()

    # Comparison operators

    def __lt__(self, other: AbstractArrayLike) -> AbstractArray:
        return AbstractArray(operator.lt(*self._binary_operands(other)))

    def __le__(self, other: AbstractArrayLike) -> AbstractArray:
        return AbstractArray(operator.le(*self._binary_operands(other)))

    def __gt__(self, other: AbstractArrayLike) -> AbstractArray:
        return AbstractArray(operator.gt(*self._binary_operands(other)))

    def __ge__(self, other: AbstractArrayLike) -> AbstractArray:
        return AbstractArray(operator.ge(*self._binary_operands(other)))

    def __eq__(self, other: Any) -> AbstractArray:  # type: ignore[override]
        return AbstractArray(operator.eq(*self._binary_operands(other)))

    def __ne__(self, other: Any) -> AbstractArray:  # type: ignore[override]
        return AbstractArray(operator.ne(*self._binary_operands(other)))

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

    def _process_indices(self, indices: Any) -> Any:
        try:
            return indices.tolist()
        except Exception:
            return indices

    def __getitem__(self, indices: Any) -> AbstractArray:
        return AbstractArray(self._array[self._process_indices(indices)])

    def __setitem__(self, indices: Any, values: AbstractArrayLike) -> None:
        array, values = self._binary_operands(values)
        try:
            array[
                self._process_indices(indices)
            ] = values  # type: ignore[assignment]
        except RuntimeError as e:
            if self.requires_grad:
                raise RuntimeError(
                    "Failed to modify a tensor that requires grad in place."
                ) from e
            else:  # pragma: no cover
                raise e
        self._array = array
        del self.is_tensor  # Clears cache

    def __iter__(self) -> Generator[AbstractArray, None, None]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self) -> int:
        return len(self._array)

    def _to_dict(self) -> dict[str, Any]:
        try:
            return obj_to_dict(self, self.as_array())
        except RuntimeError as e:
            raise NotImplementedError(
                "A tensor that requires grad can't be serialized without"
                " losing the computational graph information."
            ) from e

    def _to_abstract_repr(self) -> Any:
        try:
            return self.as_array().tolist()
        except RuntimeError as e:
            raise NotImplementedError(
                "A tensor that requires grad can't be serialized without"
                " losing the computational graph information."
            ) from e


AbstractArrayLike = Union[AbstractArray, ArrayLike]
