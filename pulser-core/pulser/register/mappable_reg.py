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
    from pulser.register.weight_maps import DetuningMap


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
        elif set(chosen_ids) != set(self.qubit_ids[: len(chosen_ids)]):
            raise ValueError(
                f"To declare {len(qubits.keys())} qubits, 'qubits' should "
                f"contain the first {len(qubits.keys())} elements of the "
                "'qubit_ids'."
            )
        register_ordered_qubits = {
            id: qubits[id] for id in self._qubit_ids if id in chosen_ids
        }
        return self._layout.define_register(
            *tuple(register_ordered_qubits.values()),
            qubit_ids=tuple(register_ordered_qubits.keys()),
        )

    def find_indices(self, id_list: abcSequence[QubitId]) -> list[int]:
        """Computes indices of qubits.

        This can especially be useful when building a Pulser Sequence
        with a parameter denoting qubits.

        Example:
            Let ``reg`` be a mappable register with qubit Ids "a", "b", "c"
            and "d".

            >>> reg.find_indices(["a", "b", "d", "a"])

            It returns ``[0, 1, 3, 0]``, following the qubits order of the
            mappable register (defined by qubit_ids).

            Then, it is possible to use these indices when building a
            sequence, typically to instanciate an array of variables
            that can be provided as an argument to ``target_index``
            and ``phase_shift_index``.

            When building a sequence and declaring N qubits, their ids should
            refer to the first N elements of qubit_id.

        Args:
            id_list: IDs of the qubits to denote.

        Returns:
            Indices of the qubits to denote, only valid for the
            given mapping.
        """
        if not set(id_list) <= set(self._qubit_ids):
            raise ValueError(
                "The IDs list must be selected among pre-declared qubit IDs."
            )
        return [self.qubit_ids.index(id) for id in id_list]

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
        return self._layout.define_detuning_map(detuning_weights, slug)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._layout, *self._qubit_ids)

    def _to_abstract_repr(self) -> list[dict[str, str]]:
        return [dict(qid=qid) for qid in stringify_qubit_ids(self.qubit_ids)]
