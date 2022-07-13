# Copyright 2020 Pulser Development Team
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
"""Contains the Variable and auxiliary classes."""

from __future__ import annotations

import collections.abc  # To use collections.abc.Sequence
import dataclasses
from typing import Any, Iterator, Optional, Union, cast

import numpy as np
from numpy.typing import ArrayLike

from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized
from pulser.parametrized.paramobj import OpSupport


@dataclasses.dataclass(frozen=True, eq=False)
class Variable(Parametrized, OpSupport):
    """A variable for parametrized sequence building.

    Args:
        name: Unique name for the variable.
        dtype: Type of the variable's content. Supports `float` and
            `int`.
        size: The number of values stored. Defaults to a single value.
    """

    name: str
    dtype: Union[type[float], type[int]]
    size: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Variable's 'name' has to be of type 'str'.")
        if self.dtype not in [int, float]:
            raise TypeError(f"Invalid data type '{self.dtype}' for Variable.")
        if not isinstance(self.size, int):
            raise TypeError("Given variable 'size' is not of type 'int'.")
        elif self.size < 1:
            raise ValueError("Variables must be of size 1 or larger.")

        self._count: int
        object.__setattr__(self, "_count", -1)  # Counts the updates
        self._clear()

    @property
    def variables(self) -> dict[str, Variable]:
        """Returns a dictionary with the only variable involved (itself)."""
        return {self.name: self}

    def _clear(self) -> None:
        object.__setattr__(self, "value", None)  # TODO rename _value?
        object.__setattr__(self, "_count", self._count + 1)

    def _assign(self, value: Union[ArrayLike, float, int]) -> None:
        val = self._validate_value(value)
        object.__setattr__(self, "value", val)
        object.__setattr__(self, "_count", self._count + 1)

    def _validate_value(
        self, value: Union[ArrayLike, float, int]
    ) -> np.ndarray:
        val = np.array(value, dtype=self.dtype, ndmin=1)
        if val.size != self.size:
            raise ValueError(
                f"Can't assign array of size {val.size} to "
                + f"variable of size {self.size}."
            )
        return val

    def build(self) -> ArrayLike:
        """Returns the variable's current value."""
        self.value: Optional[ArrayLike]
        if self.value is None:
            raise ValueError(f"No value assigned to variable '{self.name}'.")
        return self.value

    def _to_dict(self) -> dict[str, Any]:
        d = obj_to_dict(self, _build=False)
        d.update(dataclasses.asdict(self))
        return d

    def _to_abstract_repr(self) -> dict[str, str]:
        return {"variable": self.name}

    def __str__(self) -> str:
        return self.name

    def __getitem__(self, key: Union[int, slice]) -> VariableItem:
        if not isinstance(key, (int, slice)):
            raise TypeError(f"Invalid key type {type(key)} for '{self.name}'.")
        if isinstance(key, int):
            if not -self.size <= key < self.size:
                raise IndexError(f"{key} outside of range for '{self.name}'.")

        return VariableItem(self, key)

    def __iter__(self) -> Iterator[VariableItem]:
        for i in range(self.size):
            yield self[i]


@dataclasses.dataclass(frozen=True)
class VariableItem(Parametrized, OpSupport):
    """Stores access to items of a variable with multiple values."""

    var: Variable
    key: Union[int, slice]

    @property
    def variables(self) -> dict[str, Variable]:
        """All the variables involved with this object."""
        return self.var.variables

    def build(self) -> Union[ArrayLike, float, int]:
        """Return the variable's item(s) values."""
        return cast(collections.abc.Sequence, self.var.build())[self.key]

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(
            self, self.var, self.key, _module="operator", _name="getitem"
        )

    def _to_abstract_repr(self) -> dict[str, Any]:
        indices = list(range(self.var.size))[self.key]
        return {"expression": "index", "lhs": self.var, "rhs": indices}

    def __str__(self) -> str:
        if isinstance(self.key, slice):
            items = [
                "" if x is None else str(x)
                for x in [self.key.start, self.key.stop, self.key.step]
            ]
            key_str = ":".join(items)
        else:
            key_str = str(self.key)
        return f"{str(self.var)}[{key_str}]"
