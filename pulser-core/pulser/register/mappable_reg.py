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
from typing import Sequence as abcSequence

from pulser.json.utils import obj_to_dict, stringify_qubit_ids

if TYPE_CHECKING:
    from pulser.register.base_register import BaseRegister, QubitId
    from pulser.register.register_layout import RegisterLayout


class MappableRegister:
    """A register with the traps of each qubit still to be defined.

    Args:
        register_layout: The register layout on which this
            register will be defined.
        qubit_ids: The Ids for the qubits to pre-declare on this
            register.
    """

    def __init__(self, register_layout: RegisterLayout, *qubit_ids: QubitId):
        """Initializes the mappable register."""
        self._layout = register_layout
        if len(qubit_ids) > self._layout.number_of_traps:
            raise ValueError(
                "The number of required qubits is greater than the number of "
                f"traps in this layout ({self._layout.number_of_traps})."
            )
        self._qubit_ids = qubit_ids

    @property
    def qubit_ids(self) -> tuple[QubitId, ...]:
        """The qubit IDs of this mappable register."""
        return self._qubit_ids

    @property
    def layout(self) -> RegisterLayout:
        """The layout used to define the register."""
        return self._layout

    def build_register(self, qubits: Mapping[QubitId, int]) -> BaseRegister:
        """Builds an actual register.

        Args:
            qubits: A map between the qubit IDs to use
                and the layout traps where the qubits will be placed. Qubit IDs
                declared in the MappableRegister but not defined here will
                simply be left out of the final register.

        Returns:
            The resulting register.
        """
        chosen_ids = tuple(qubits.keys())
        if not set(chosen_ids) <= set(self._qubit_ids):
            raise ValueError(
                "All qubits must be labeled with pre-declared qubit IDs."
            )
        register_ordered_qubits = {
            id: qubits[id] for id in self._qubit_ids if id in chosen_ids
        }
        return self._layout.define_register(
            *tuple(register_ordered_qubits.values()),
            qubit_ids=tuple(register_ordered_qubits.keys()),
        )

    def find_indices(
        self, chosen_ids: set[QubitId], id_list: abcSequence[QubitId]
    ) -> list[int]:
        """Computes indices of qubits according to a register mapping.

        This can especially be useful when building a Pulser Sequence
        with a parameter denoting qubits.

        Example:
            Let ``reg`` be a mappable register with qubit Ids "a", "b", "c"
            and "d".

            >>> qubit_map = dict(b=1, a=2, d=0)
            >>> reg.find_indices(
            >>>   qubit_map.keys(),
            >>>   ["a", "b", "d", "a"])

            It returns ``[0, 1, 2, 0]``, following the qubits order of the
            mappable register, but keeping only the chosen ones.

            Then, it is possible to use these indices when building a
            sequence, typically to instanciate an array of variables
            that can be provided as an argument to ``target_index``
            and ``phase_shift_index``.

            ``qubit_map`` should be provided when building the sequence,
            to tell how to instantiate the register from the mappable register.

        Args:
            chosen_ids: IDs of the qubits that are chosen to
                map the MappableRegister
            id_list: IDs of the qubits to denote.

        Returns:
            Indices of the qubits to denote, only valid for the
            given mapping.
        """
        if not chosen_ids <= set(self._qubit_ids):
            raise ValueError(
                "Chosen IDs must be selected among pre-declared qubit IDs."
            )
        if not set(id_list) <= chosen_ids:
            raise ValueError(
                "The IDs list must be selected among the chosen IDs."
            )
        ordered_ids = [id for id in self.qubit_ids if id in chosen_ids]
        return [ordered_ids.index(id) for id in id_list]

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._layout, *self._qubit_ids)

    def _to_abstract_repr(self) -> list[dict[str, str]]:
        return [dict(qid=qid) for qid in stringify_qubit_ids(self.qubit_ids)]
