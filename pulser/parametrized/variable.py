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

from dataclasses import dataclass
from typing import Union

import numpy as np

from pulser.parametrized import Parametrized
from pulser.parametrized.paramobj import OpSupport


@dataclass(frozen=True, eq=False)
class Variable(Parametrized, OpSupport):
    """A variable for parametrized sequence building.

    Args:
        name (str): Unique name for the variable.
        dtype (type): Type of the variable's content. Supports `float`, `int`
            and `float`.
        size (int=1): The number of values stored. Defaults to a single value.
    """

    name: str
    dtype: type
    size: int = 1

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("Variable's 'name' has to be of type 'str'.")
        if self.dtype not in [int, float, str]:
            raise TypeError(f"Invalid data type '{self.dtype}' for Variable.")
        if not isinstance(self.size, int):
            raise TypeError("Given variable 'size' is not of type 'int'.")
        elif self.size < 1:
            raise ValueError("Variables must be of size 1 or larger.")
        self.__dict__["_count"] = -1      # Counts the updates
        self._clear()

    @property
    def variables(self):
        return {self.name: self}

    def _clear(self):
        self.__dict__["value"] = None
        self.__dict__["_count"] += 1

    def _assign(self, value):
        val = np.array(value, dtype=self.dtype)
        if val.size != self.size:
            raise ValueError(f"Can't assign array of size {val.size} to "
                             + f"variable of size {self.size}.")

        self.__dict__["value"] = self.dtype(val) if self.size == 1 else val
        self.__dict__["_count"] += 1

    def build(self):
        """Returns the variable's current value."""
        if self.value is None:
            raise ValueError(f"No value assigned to variable '{self.name}'.")

        return self.value

    def __str__(self):
        return self.name

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        if not isinstance(key, (int, slice)):
            raise TypeError(f"Invalid key type {type(key)} for '{self.name}'.")
        if self.size == 1:
            raise TypeError(f"Variable '{self.name}' is not subscriptable.")
        if isinstance(key, int):
            if not -self.size <= key < self.size:
                raise IndexError(f"{key} outside of range for '{self.name}'.")

        return _VariableItem(self, key)


@dataclass(frozen=True)
class _VariableItem(Parametrized, OpSupport):
    """Stores access to items of a variable with multiple values."""

    var: Variable
    key: Union[int, slice]

    @property
    def variables(self):
        return self.var.variables

    def build(self):
        """Return the variable's item(s) values."""
        return self.var.build()[self.key]

    def __str__(self):
        if isinstance(self.key, slice):
            items = ["" if x is None else str(x)
                     for x in [self.key.start, self.key.stop, self.key.step]]
            key_str = ":".join(items)
        else:
            key_str = str(self.key)
        return f"{str(self.var)}[{key_str}]"
