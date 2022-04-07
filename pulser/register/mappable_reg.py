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
"""Allows for a temporary register to exist, when associated with a layout."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pulser.json.utils import obj_to_dict

if TYPE_CHECKING:  # pragma: no cover
    from pulser.register.base_register import BaseRegister, QubitId
    from pulser.register.register_layout import RegisterLayout
from typing import Sequence as abcSequence

class MappableRegister:
    """A register with the traps of each qubit still to be defined.

    Args:
        register_layout (RegisterLayout): The register layout on which this
            register will be defined.
        qubit_ids (QubitId): The Ids for the qubits to pre-declare on this
            register.
    """

    def __init__(self, register_layout: RegisterLayout, *qubit_ids: QubitId):
        """Initializes the mappable register."""
        self._layout = register_layout
        if len(qubit_ids) > self._layout.max_atom_num:
            raise ValueError(
                "The number of required traps is greater than the maximum "
                "number of qubits allowed for this layout "
                f"({self._layout.max_atom_num})."
            )
        self._qubit_ids = qubit_ids

    @property
    def qubit_ids(self) -> tuple[QubitId, ...]:
        """The qubit IDs of this mappable register."""
        return self._qubit_ids

    def build_register(self, qubits: Mapping[QubitId, int]) -> BaseRegister:
        """Builds an actual register.

        Args:
            qubits (Mapping[QubitId, int]): A map between the qubit IDs to use
                and the layout traps where the qubits will be placed. Qubit IDs
                declared in the MappableRegister but not defined here will
                simply be left out of the final register.

        Returns:
            BaseRegister: The resulting register.
        """
        chosen_ids = tuple(qubits.keys())
        if not set(chosen_ids) <= set(self._qubit_ids):
            raise ValueError(
                f"All qubits must be labeled with pre-declared qubit IDs. {chosen_ids!r}, {self._qubit_ids!r}"
            )
        return self._layout.define_register(
            *tuple(qubits.values()), qubit_ids=chosen_ids
        )

    def find_indices(self, chosen_ids: set[QubitId], id_list: abcSequence[QubitId]) -> list[int]:
        """
        Computes indices for the given qubit IDs for a given register mapping.

        This can especially be useful when building a Pulser Sequence with a parameter denoting qubits.
        Example:
            ``
            mapp_reg = TriangularLatticeLayout(50, 5).make_mappable_register(5)
            qubit_map = {"q0": 1, "q2": 4, "q4": 2}
            indices = mapp_reg.find_indices(qubit_map, ["q4", "q2", "q1", "q2"])
            seq.build(qubits=qubit_map, qubit_indices=indices)
            ``

        Args:
            chosen_ids: IDs of the qubits that are chosen to map the MappableRegister
            id_list: IDs of the qubits to denote

        Returns:
            Indices of the qubits to denote, only valid for the given mapping.
        """
        if not chosen_ids <= set(self._qubit_ids):
            raise ValueError("Chosen IDs must be selected among pre-declared qubit IDs.")
        if not set(id_list) <= chosen_ids:
            raise ValueError("The IDs list must be selected among the chosen IDs.")
        ordered_ids = [id for id in self.qubit_ids if id in chosen_ids]
        return [ordered_ids.index(id) for id in id_list]

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._layout, *self._qubit_ids)
