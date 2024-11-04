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
from typing import Generic, TypeVar

ScalarType = TypeVar("ScalarType")


class State(ABC, Generic[ScalarType]):
    """Base class enforcing an API for quantum states.

    Each backend will implement its own type of state and the
    methods below.
    """

    @abstractmethod
    def inner(self, other: State) -> ScalarType:
        """Compute the inner product between this state and other.

        Note that self is the left state in the inner product,
        so this function is linear in other, and anti-linear in self.

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
        p_false_pos: float = 0.0,
        p_false_neg: float = 0.0,
    ) -> Counter[str]:
        """Sample bitstrings from the state, taking into account error rates.

        Args:
            num_shots: How many bitstrings to sample.
            p_false_pos: The rate at which a 0 is read as a 1.
            p_false_neg: The rate at which a 1 is read as a 0.

        Returns:
            The measured bitstrings, by count.
        """
        pass

    @abstractmethod
    def __add__(self, other: State) -> State:
        """Computes the sum of two states.

        Args:
            other: The other state.

        Returns:
            The summed state.
        """
        pass

    @abstractmethod
    def __rmul__(self, scalar: ScalarType) -> State:
        """Scale the state by a scale factor.

        Args:
            scalar: The scale factor.

        Returns:
            The scaled state.
        """
        pass

    @classmethod
    @abstractmethod
    def from_state_amplitudes(
        cls,
        *,
        eigenstates: Sequence[str],
        amplitudes: dict[str, ScalarType],
    ) -> State:
        """Construct the state from its basis states' amplitudes.

        Args:
            eigenstates: The basis states (e.g., ('r', 'g')).
            amplitudes: A mapping between basis state combinations and
                complex amplitudes.

        Returns:
            The state constructed from the amplitudes.
        """
        pass
