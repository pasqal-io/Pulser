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

from pulser.paramobj import Parametrized


@dataclass(frozen=True)
class Variable(Parametrized):
    name: str
    dtype: type
    size: int = 1
    value: Union[int, float, str] = None

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError("Variable's 'name' has to be of type 'str'.")
        if self.dtype not in [int, float, str]:
            raise TypeError("Invalid data type.")
        if not isinstance(self.size, int):
            raise TypeError("Given variable 'size' is not of type 'int'.")
        elif self.size < 1:
            raise ValueError("Variables must be of size 1 or larger.")

    @property
    def variables(self):
        return {self.name: self}

    def clear(self):
        self.__dict__["value"] = None

    def _assign(self, value):
        val = np.array(value, dtype=self.dtype)
        if val.size != self.size:
            raise ValueError(f"Can't assign array of size {val.size} to "
                             + f"variable of size {self.size}.")

        self.__dict__["value"] = self.dtype(val) if self.size == 1 else val

    def __call__(self):
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
                raise KeyError(f"'{key}' outside of range for '{self.name}'.")

        return _VariableItem(self, key)


class _VariableItem(Parametrized):
    def __init__(self, var, key):
        if not isinstance(var, Variable):
            return TypeError("VariableItem requires a Variable instance.")
        self.var = var
        self.key = key

    @property
    def variables(self):
        return self.var.variables

    def __call__(self):
        return self.var()[self.key]

    def __str__(self):
        if isinstance(self.key, slice):
            items = ["" if x is None else str(x)
                     for x in [self.key.start, self.key.stop, self.key.step]]
            key_str = ":".join(items)
        else:
            key_str = str(self.key)
        return f"{str(self.var)}[{key_str}]"
