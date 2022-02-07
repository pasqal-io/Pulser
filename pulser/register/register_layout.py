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

from collections.abc import Sequence as abcSequence
from dataclasses import dataclass
from sys import version_info
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from pulser.register.base_register import QubitId, BaseRegister
from pulser.register.register import Register
from pulser.register.register3d import Register3D
from pulser.register._reg_drawer import RegDrawer

if version_info[:2] >= (3, 8):  # pragma: no cover
    from functools import cached_property
else:  # pragma: no cover
    try:
        from backports.cached_property import cached_property  # type: ignore
    except ImportError:
        raise ImportError(
            "Using pulser with Python version 3.7 requires the"
            " `backports.cached-property` module. Install it by running"
            " `pip install backports.cached-property`."
        )


@dataclass(frozen=True)
class RegisterLayout(RegDrawer):
    trap_coordinates: ArrayLike

    def __post_init__(self):
        shape = self.coords.shape
        if len(shape) != 2:
            raise ValueError(
                "'trap_coordinates' must be an array or list of coordinates."
            )
        if shape[1] not in (2, 3):
            raise ValueError(
                f"Each coordinate must be of size 2 or 3, not {shape[1]}."
            )

    @cached_property
    def traps_dict(self) -> dict:
        return dict(enumerate(self.coords))

    @cached_property
    def coords(self) -> np.ndarray:
        coords = np.array(self.trap_coordinates, dtype=float)
        # Sorting the coordinates 1st left to right, 2nd bottom to top
        rounded_coords = np.round(coords, decimals=6)
        sorting = np.lexsort((rounded_coords[:, 1], rounded_coords[:, 0]))
        return coords[sorting]

    @property
    def number_of_traps(self) -> int:
        return len(self.coords)

    @property
    def max_atom_num(self) -> int:
        return self.number_of_traps // 2

    @property
    def dimensionality(self) -> int:
        return self.coords.shape[1]

    def define_register(
        self, *trap_ids: int, qubit_ids: Optional[abcSequence[QubitId]] = None
    ) -> BaseRegister:

        trap_ids_set = set(trap_ids)

        if len(trap_ids_set) != len(trap_ids):
            raise ValueError("Every 'trap_id' must a unique integer.")

        if not trap_ids_set.issubset(self.traps_dict):
            raise ValueError(
                "All 'trap_ids' must correspond to the ID of a trap."
            )

        if qubit_ids:
            if len(set(qubit_ids)) != len(qubit_ids):
                raise ValueError(
                    "'qubit_ids' must be a sequence of unique IDs."
                )
            if len(qubit_ids) != len(trap_ids):
                raise ValueError(
                    "'qubit_ids' must have the same size as the number of "
                    f"provided 'trap_ids' ({len(trap_ids)})."
                )

        if len(trap_ids) > self.max_atom_num:
            raise ValueError(
                "The number of required traps is greater than the maximum "
                "number of qubits allowed for this layout "
                f"({self.max_atom_num})."
            )
        ids = (
            qubit_ids if qubit_ids else [f"q{i}" for i in range(len(trap_ids))]
        )
        coords = self.coords[list(trap_ids)]
        qubits = dict(zip(ids, coords))

        reg_class = Register3D if self.dimensionality == 3 else Register
        reg = reg_class(qubits)
        reg._set_layout(self, *trap_ids)
        return reg

    def draw(
        self,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = False,
        draw_half_radius: bool = False,
    ) -> None:
        """Draws the entire register layout.

        Keyword Args:
            blockade_radius(float, default=None): The distance (in μm) between
                atoms below which the Rydberg blockade effect occurs.
            draw_half_radius(bool, default=False): Whether or not to draw
                half the blockade radius surrounding each trap. If `True`,
                requires `blockade_radius` to be defined.
            draw_graph(bool, default=True): Whether or not to draw the
                interaction between atoms as edges in a graph. Will only draw
                if the `blockade_radius` is defined.

        Note:
            When drawing half the blockade radius, we say there is a blockade
            effect between atoms whenever their respective circles overlap.
            This representation is preferred over drawing the full Rydberg
            radius because it helps in seeing the interactions between atoms.
        """
        self._draw_checks(
            self.number_of_traps,
            blockade_radius=blockade_radius,
            draw_graph=draw_graph,
            draw_half_radius=draw_half_radius,
        )
        fig, ax = self._initialize_fig_axes(
            self.coords,
            blockade_radius=blockade_radius,
            draw_half_radius=draw_half_radius,
        )
        self._draw_2D(
            ax,
            self.coords,
            list(range(self.number_of_traps)),
            blockade_radius=blockade_radius,
            draw_graph=draw_graph,
            draw_half_radius=draw_half_radius,
            are_traps=True,
        )
        plt.show()
