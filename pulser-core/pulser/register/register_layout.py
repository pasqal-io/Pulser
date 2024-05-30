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

import hashlib
import json
from collections.abc import Mapping
from collections.abc import Sequence as abcSequence
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

import pulser
import pulser.json.abstract_repr as pulser_abstract_repr
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.abstract_repr.validation import validate_abstract_repr
from pulser.json.utils import obj_to_dict
from pulser.register._reg_drawer import RegDrawer
from pulser.register.base_register import BaseRegister, QubitId
from pulser.register.mappable_reg import MappableRegister
from pulser.register.traps import Traps
from pulser.register.weight_maps import DetuningMap


@dataclass(init=False, repr=False, eq=False, frozen=True)
class RegisterLayout(Traps, RegDrawer):
    """A layout of traps out of which registers can be defined.

    The traps are always sorted under the same convention: ascending order
    along x, then along y, then along z (if applicable). Respecting this order,
    the traps are then numbered starting from 0.

    Args:
        trap_coordinates: The trap coordinates defining the layout.
        slug: An optional identifier for the layout.
    """

    @property
    def coords(self) -> np.ndarray:
        """A shorthand for 'sorted_coords'."""
        return self.sorted_coords

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
        coords = self.sorted_coords[list(trap_ids)]
        qubits = dict(zip(ids, coords))

        reg_class = (
            pulser.Register3D if self.dimensionality == 3 else pulser.Register
        )
        reg = reg_class(qubits, layout=self, trap_ids=trap_ids)
        return reg

    def define_detuning_map(
        self,
        detuning_weights: Mapping[int, float],
        slug: str | None = None,
    ) -> DetuningMap:
        """Defines a DetuningMap for some trap ids of the register layout.

        Args:
            detuning_weights: A mapping between the IDs of the targeted traps
                and detuning weights (between 0 and 1).
            slug: An optional identifier for the detuning map.

        Returns:
            A DetuningMap associating detuning weights to the trap coordinates
                of the targeted traps.
        """
        if not set(detuning_weights.keys()) <= set(self.traps_dict):
            raise ValueError(
                "The trap ids of detuning weights have to be integers"
                f" in [0, {self.number_of_traps-1}]."
            )
        return DetuningMap(
            itemgetter(*detuning_weights.keys())(self.traps_dict),
            list(detuning_weights.values()),
            slug,
        )

    def draw(
        self,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = False,
        draw_half_radius: bool = False,
        projection: bool = True,
        fig_name: str | None = None,
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

    @property
    def _hash_object(self) -> hashlib._Hash:
        return super()._hash_object

    def __eq__(self, other: Any) -> bool:
        return super().__eq__(other) and isinstance(other, RegisterLayout)

    def __repr__(self) -> str:
        return f"RegisterLayout_{self._safe_hash().hex()}"

    def __hash__(self) -> int:
        return hash(self._safe_hash())

    def _to_dict(self) -> dict[str, Any]:
        # Allows for serialization of subclasses without a special _to_dict()
        return obj_to_dict(
            self,
            self._coords_arr.tolist(),
            slug=self.slug,
            _module=__name__,
            _name="RegisterLayout",
        )

    def _to_abstract_repr(self) -> dict[str, list[list[float]]]:
        d = {"coordinates": self.coords.tolist()}
        if self.slug is not None:
            d["slug"] = self.slug
        return d

    def to_abstract_repr(self) -> str:
        """Serializes the layout into an abstract JSON object."""
        abstr_layout_str = json.dumps(self, cls=AbstractReprEncoder)
        validate_abstract_repr(abstr_layout_str, "layout")
        return abstr_layout_str

    @staticmethod
    def from_abstract_repr(obj_str: str) -> RegisterLayout:
        """Deserialize a layout from an abstract JSON object.

        Args:
            obj_str (str): the JSON string representing the layout encoded
                in the abstract JSON format.
        """
        if not isinstance(obj_str, str):
            raise TypeError(
                "The serialized layout must be given as a string. "
                f"Instead, got object of type {type(obj_str)}."
            )
        # Avoids circular imports
        return pulser_abstract_repr.deserializer.deserialize_abstract_layout(
            obj_str
        )
