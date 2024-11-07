# Copyright 2024 Pulser Development Team
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
"""Defines the abstract base class for a quantum state."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from typing import Generic, Type, TypeVar

from pulser.channels.base_channel import States as Eigenstate

ArgScalarType = TypeVar("ArgScalarType")
ReturnScalarType = TypeVar("ReturnScalarType")
StateType = TypeVar("StateType", bound="State")


class State(ABC, Generic[ArgScalarType, ReturnScalarType]):
    """Base class enforcing an API for quantum states.

    Each backend will implement its own type of state and the
    methods below.
    """

    eigenstates: Sequence[Eigenstate]

    @abstractmethod
    def overlap(self: StateType, other: StateType, /) -> ReturnScalarType:
        """Compute the overlap between this state and another of the same type.

        Generally computes Tr[AB] for mixed states A and B, which
        corresponds to |<a|b>|^2 for pure states A=|a><a| and B=|b><b|.

        Args:
            other: The other state.

        Returns:
            The inner product between the two states.
        """
        pass

    @abstractmethod
    def sample(
        self,
        num_shots: int,
        one_state: Eigenstate | None = None,
        p_false_pos: float = 0.0,
        p_false_neg: float = 0.0,
    ) -> Counter[str]:
        """Sample bitstrings from the state, taking into account error rates.

        Args:
            num_shots: How many bitstrings to sample.
            one_state: The eigenstate that measures to 1. Can be left undefined
                if the eigenstates form a known eigenbasis with a defined
                "one state".
            p_false_pos: The rate at which a 0 is read as a 1.
            p_false_neg: The rate at which a 1 is read as a 0.

        Returns:
            The measured bitstrings, by count.
        """
        pass

    @classmethod
    @abstractmethod
    def from_state_amplitudes(
        cls: Type[StateType],
        *,
        eigenstates: Sequence[Eigenstate],
        amplitudes: dict[str, ArgScalarType],
    ) -> StateType:
        """Construct the state from its basis states' amplitudes.

        Args:
            eigenstates: The basis states (e.g., ('r', 'g')).
            amplitudes: A mapping between basis state combinations and
                complex amplitudes.

        Returns:
            The state constructed from the amplitudes.
        """
        pass
