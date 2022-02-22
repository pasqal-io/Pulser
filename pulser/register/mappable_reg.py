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
                "All qubits must be labeled with pre-declared qubit IDs."
            )
        return self._layout.define_register(
            *tuple(qubits.values()), qubit_ids=chosen_ids
        )

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._layout, *self._qubit_ids)
