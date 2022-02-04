# Copyright 2022 Pulser Development Team
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

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence as abcSequence
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from pulser.base_register import QubitId, BaseRegister
from pulser.register.register import Register
from pulser.register.regiter3d import Register3D


class RegisterLayout(ABC):
    def __init__(self, trap_coordinates: ArrayLike):
        coords = np.array(trap_coordinates, dtype=float)
        shape = coords.shape
        if len(shape) != 2:
            raise ValueError(
                "'trap_coordinates' must be an array or list of coordinates."
            )
        if shape[1] not in (2, 3):
            raise ValueError(
                f"Each coordinate must be of size 2 or 3, not {shape[1]}."
            )
        self._dim = shape[1]
        # Sorting the coordinates 1st left to right, 2nd bottom to top
        rounded_coords = np.round(coords, decimals=6)
        sorting = np.lexsort((rounded_coords[:, 1], rounded_coords[:, 0]))

        self._coords = coords[sorting]

    @property
    def traps_dict(self) -> dict:
        return dict(enumerate(self._coords))

    @property
    def number_of_traps(self) -> int:
        return len(self._coords)

    @property
    def max_atom_num(self) -> int:
        return self.number_of_traps // 2

    @property
    def dimensionality(self) -> int:
        return self._dim

    def define_register(
        self, *trap_ids: int, qubit_ids: Optional[abcSequence[QubitId]] = None
    ) -> BaseRegister:

        if len(trap_ids) > self.max_atom_num:
            raise ValueError(
                "The number of required traps is greater than the maximum "
                "number of qubits allowed for this layout "
                f"({self.max_atom_num})."
            )
        ids = (
            qubit_ids if qubit_ids else [f"q{i}" for i in range(len(trap_ids))]
        )
        coords = self._coords[list(trap_ids)]
        qubits = dict(zip(ids, coords))

        reg_class = Register3D if self._dim == 3 else Register

        return reg_class(qubits)
