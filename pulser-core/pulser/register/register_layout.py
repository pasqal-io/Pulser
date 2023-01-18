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
"""Defines a generic Register layout, from which a Register can be created."""

from __future__ import annotations

from collections.abc import Sequence as abcSequence
from dataclasses import dataclass
from hashlib import sha256
from sys import version_info
from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from pulser.json.utils import obj_to_dict
from pulser.register._reg_drawer import RegDrawer
from pulser.register.base_register import BaseRegister, QubitId
from pulser.register.mappable_reg import MappableRegister
from pulser.register.register import Register
from pulser.register.register3d import Register3D

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

COORD_PRECISION = 6


@dataclass(init=False, repr=False, eq=False, frozen=True)
class RegisterLayout(RegDrawer):
    """A layout of traps out of which registers can be defined.

    The traps are always sorted under the same convention: ascending order
    along x, then along y, then along z (if applicable). Respecting this order,
    the traps are then numbered starting from 0.

    Args:
        trap_coordinates: The trap coordinates defining the layout.
        slug: An optional identifier for the layout.
    """

    _trap_coordinates: ArrayLike
    slug: Optional[str]

    def __init__(
        self, trap_coordinates: ArrayLike, slug: Optional[str] = None
    ):
        """Initializes a RegisterLayout."""
        array_type_error_msg = ValueError(
            "'trap_coordinates' must be an array or list of coordinates."
        )

        try:
            shape = np.array(trap_coordinates).shape
        # Following lines are only being covered starting Python 3.11.1
        except ValueError as e:  # pragma: no cover
            raise array_type_error_msg from e  # pragma: no cover

        if len(shape) != 2:
            raise array_type_error_msg

        if shape[1] not in (2, 3):
            raise ValueError(
                f"Each coordinate must be of size 2 or 3, not {shape[1]}."
            )
        object.__setattr__(self, "_trap_coordinates", trap_coordinates)
        object.__setattr__(self, "slug", slug)

    @property
    def traps_dict(self) -> dict:
        """Mapping between trap IDs and coordinates."""
        return dict(enumerate(self.coords))

    @cached_property  # Acts as an attribute in a frozen dataclass
    def _coords(self) -> np.ndarray:
        coords = np.array(self._trap_coordinates, dtype=float)
        # Sorting the coordinates 1st left to right, 2nd bottom to top
        rounded_coords = np.round(coords, decimals=COORD_PRECISION)
        dims = rounded_coords.shape[1]
        sorter = [rounded_coords[:, i] for i in range(dims - 1, -1, -1)]
        sorting = np.lexsort(tuple(sorter))
        return cast(np.ndarray, rounded_coords[sorting])

    @cached_property  # Acts as an attribute in a frozen dataclass
    def _coords_to_traps(self) -> dict[tuple[float, ...], int]:
        return {tuple(coord): id for id, coord in self.traps_dict.items()}

    @property
    def coords(self) -> np.ndarray:
        """The sorted trap coordinates."""
        # Copies to prevent direct access to self._coords
        return self._coords.copy()

    @property
    def number_of_traps(self) -> int:
        """The number of traps in the layout."""
        return len(self._coords)

    @property
    def dimensionality(self) -> int:
        """The dimensionality of the layout (2 or 3)."""
        return self._coords.shape[1]

    def get_traps_from_coordinates(self, *coordinates: ArrayLike) -> list[int]:
        """Finds the trap ID for a given set of trap coordinates.

        Args:
            coordinates: The coordinates to return the trap IDs.

        Returns:
            The list of trap IDs corresponding to the coordinates.
        """
        traps = []
        rounded_coords = np.round(
            np.array(coordinates), decimals=COORD_PRECISION
        )
        for coord, rounded in zip(coordinates, rounded_coords):
            key = tuple(rounded)
            if key not in self._coords_to_traps:
                raise ValueError(
                    f"The coordinate '{coord!s}' is not a part of the "
                    "RegisterLayout."
                )
            traps.append(self._coords_to_traps[key])
        return traps

    def define_register(
        self, *trap_ids: int, qubit_ids: Optional[abcSequence[QubitId]] = None
    ) -> BaseRegister:
        """Defines a register from selected traps.

        Args:
            trap_ids: The trap IDs selected to form the Register.
            qubit_ids: A sequence of
                unique qubit IDs to associated to the selected traps. Must be
                of the same length as the selected traps.

        Returns:
            The respective register instance.
        """
        trap_ids_set = set(trap_ids)

        if len(trap_ids_set) != len(trap_ids):
            raise ValueError("Every 'trap_id' must be a unique integer.")

        if not trap_ids_set.issubset(self.traps_dict):
            # This check makes it redundant to check # qubits <= # traps
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

        ids = (
            qubit_ids if qubit_ids else [f"q{i}" for i in range(len(trap_ids))]
        )
        coords = self._coords[list(trap_ids)]
        qubits = dict(zip(ids, coords))

        reg_class = Register3D if self.dimensionality == 3 else Register
        reg = reg_class(qubits, layout=self, trap_ids=trap_ids)
        return reg

    def draw(
        self,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = False,
        draw_half_radius: bool = False,
        projection: bool = True,
        fig_name: str = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the entire register layout.

        Args:
            blockade_radius: The distance (in μm) between
                atoms below which the Rydberg blockade effect occurs.
            draw_half_radius: Whether or not to draw
                half the blockade radius surrounding each trap. If `True`,
                requires `blockade_radius` to be defined.
            draw_graph: Whether or not to draw the
                interaction between atoms as edges in a graph. Will only draw
                if the `blockade_radius` is defined.
            projection: If the layout is in 3D, draws it
                as projections on different planes.
            fig_name: The name on which to save the figure.
                If None the figure will not be saved.
            kwargs_savefig: Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.

        Note:
            When drawing half the blockade radius, we say there is a blockade
            effect between atoms whenever their respective circles overlap.
            This representation is preferred over drawing the full Rydberg
            radius because it helps in seeing the interactions between atoms.
        """
        coords = self.coords
        self._draw_checks(
            self.number_of_traps,
            blockade_radius=blockade_radius,
            draw_graph=draw_graph,
            draw_half_radius=draw_half_radius,
        )
        ids = list(range(self.number_of_traps))
        if self.dimensionality == 2:
            fig, ax = self._initialize_fig_axes(
                coords,
                blockade_radius=blockade_radius,
                draw_half_radius=draw_half_radius,
            )
            self._draw_2D(
                ax,
                coords,
                ids,
                blockade_radius=blockade_radius,
                draw_graph=draw_graph,
                draw_half_radius=draw_half_radius,
                are_traps=True,
            )
        elif self.dimensionality == 3:
            self._draw_3D(
                coords,
                ids,
                projection=projection,
                with_labels=True,
                blockade_radius=blockade_radius,
                draw_graph=draw_graph,
                draw_half_radius=draw_half_radius,
                are_traps=True,
            )
        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)
        plt.show()

    def make_mappable_register(
        self, n_qubits: int, prefix: str = "q"
    ) -> MappableRegister:
        """Creates a mappable register associated with this layout.

        A mappable register is a register whose atoms' positions have not yet
        been defined. It can be used to create a sequence whose register is
        only defined when it is built. Note that not all the qubits 'reserved'
        in a MappableRegister need to be in the final Register, as qubits not
        associated with trap IDs won't be included. If you intend on defining
        registers of different sizes from the same mappable register, reserve
        as many qubits as you need for your largest register.

        Args:
            n_qubits: The number of qubits to reserve in the mappable
                register.
            prefix: The prefix for the qubit ids. Each qubit ID starts
                with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            A substitute for a regular register that can be
            used to initialize a Sequence.
        """
        qubit_ids = [f"{prefix}{i}" for i in range(n_qubits)]
        return MappableRegister(self, *qubit_ids)

    def _safe_hash(self) -> bytes:
        # Include dimensionality because the array is flattened with tobytes()
        hash = sha256(bytes(self.dimensionality))
        hash.update(self.coords.tobytes())
        return hash.digest()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RegisterLayout):
            return False
        return self._safe_hash() == other._safe_hash()

    def __hash__(self) -> int:
        return hash(self._safe_hash())

    def __repr__(self) -> str:
        return f"RegisterLayout_{self._safe_hash().hex()}"

    def __str__(self) -> str:
        return self.slug or self.__repr__()

    def _to_dict(self) -> dict[str, Any]:
        # Allows for serialization of subclasses without a special _to_dict()
        return obj_to_dict(
            self,
            self._trap_coordinates,
            _module=__name__,
            _name="RegisterLayout",
        )

    def _to_abstract_repr(self) -> dict[str, list[list[float]]]:
        d = {"coordinates": self.coords.tolist()}
        if self.slug is not None:
            d["slug"] = self.slug
        return d
