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
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any, Generic, Literal, Protocol, Type, TypeVar, Union

import numpy as np

from pulser.channels.base_channel import States

Eigenstate = Union[States, Literal["0", "1"]]

ArgScalarType = TypeVar("ArgScalarType")
ReturnScalarType = TypeVar("ReturnScalarType")
StateType = TypeVar("StateType", bound="State")


class _ProbabilityType(Protocol):
    """A protocol for probability types.

    Defines only the methods needed to correctly type hint State.
    """

    def __float__(self) -> float: ...

    def __add__(self, other: Any) -> float: ...


class State(ABC, Generic[ArgScalarType, ReturnScalarType]):
    """Base class enforcing an API for quantum states.

    Each backend will implement its own type of state and the methods below.
    """

    _eigenstates: Sequence[Eigenstate]

    @property
    @abstractmethod
    def n_qudits(self) -> int:
        """The number of qudits in the state."""
        pass

    @property
    def eigenstates(self) -> tuple[Eigenstate, ...]:
        """The eigenstates that form a qudit's eigenbasis.

        The order of the states should match the order in a numerical (ie
        state vector or density matrix) representation.
        For example, with eigenstates ("a", "b", ...),  "a" will be associated
        to eigenvector (1, 0, ...), "b" to (0, 1, ...) and so on.
        """
        return tuple(self._eigenstates)

    @property
    def basis_states(self) -> tuple[str, ...]:
        """The basis states combinations, in order."""
        return tuple(
            map(
                "".join,
                product("".join(self.eigenstates), repeat=self.n_qudits),
            )
        )

    @property
    def qudit_dim(self) -> int:
        """The dimensions (ie number of eigenstates) of a qudit."""
        return len(self.eigenstates)

    @abstractmethod
    def overlap(self: StateType, other: StateType, /) -> ReturnScalarType:
        """Compute the overlap between this state and another of the same type.

        Generally computes Tr[AB] for mixed states A and B, which
        corresponds to |<a|b>|^2 for pure states A=|a><a| and B=|b><b|.

        Args:
            other: The other state.

        Returns:
            The overlap between the two states.
        """
        pass

    @abstractmethod
    def probabilities(
        self, *, cutoff: float = 1e-10
    ) -> Mapping[str, _ProbabilityType]:
        """Extracts the probabilties of measuring each basis state combination.

        Args:
            cutoff: The value below which a probability is considered to be
                zero.

        Returns:
            A mapping between basis state combinations and their respective
            probabilities.
        """
        pass

    def bitstring_probabilities(
        self, *, one_state: Eigenstate | None = None, cutoff: float = 1e-10
    ) -> Mapping[str, _ProbabilityType]:
        """Extracts the probabilties of measuring each bitstring.

        Args:
            one_state: The eigenstate that measures to 1. Can be left undefined
                if the state's eigenstates form a known eigenbasis with a
                defined "one state".
            cutoff: The value below which a probability is considered to be
                zero.

        Returns:
            A mapping between bitstrings and their respective probabilities.
        """
        one_state = one_state or self.infer_one_state()
        zero_states = set(self.eigenstates) - {one_state}
        probs = self.probabilities(cutoff=cutoff)
        bitstring_probs: dict[str, _ProbabilityType] = {}
        for state_str in probs:
            bitstring = state_str.replace(one_state, "1")
            for s_ in zero_states:
                bitstring = bitstring.replace(s_, "0")
            # Avoid defaultdict for typing reasons
            curr_val = bitstring_probs.setdefault(bitstring, 0.0)
            bitstring_probs[bitstring] = probs[state_str] + curr_val
        return dict(bitstring_probs)

    def sample(
        self,
        *,
        num_shots: int,
        one_state: Eigenstate | None = None,
        p_false_pos: float = 0.0,
        p_false_neg: float = 0.0,
    ) -> Counter[str]:
        """Sample bitstrings from the state, taking into account error rates.

        Args:
            num_shots: How many bitstrings to sample.
            one_state: The eigenstate that measures to 1. Can be left undefined
                if the state's eigenstates form a known eigenbasis with a
                defined "one state".
            p_false_pos: The rate at which a 0 is read as a 1.
            p_false_neg: The rate at which a 1 is read as a 0.

        Returns:
            The measured bitstrings, by count.
        """
        bitstring_probs = self.bitstring_probabilities(
            one_state=one_state, cutoff=1 / (1000 * num_shots)
        )
        bitstrings = np.array(list(bitstring_probs))
        probs = np.array(list(map(float, bitstring_probs.values())))
        dist = np.random.multinomial(num_shots, probs)
        # Filter out bitstrings without counts
        non_zero_counts = dist > 0
        bitstrings = bitstrings[non_zero_counts]
        dist = dist[non_zero_counts]
        if p_false_pos == 0.0 and p_false_neg == 0.0:
            return Counter(dict(zip(bitstrings, dist)))

        # Convert bitstrings to a 2D array
        bitstr_arr = np.array([list(bs) for bs in bitstrings], dtype=int)
        # If 1 is measured, flip_prob=p_false_neg else flip_prob=p_false_pos
        flip_probs = np.where(bitstr_arr == 1, p_false_neg, p_false_pos)
        # Repeat flip_probs of a bitstring as many times as it was measured
        flip_probs_repeated = np.repeat(flip_probs, dist, axis=0)
        # Generate random matrix of same shape
        random_matrix = np.random.uniform(size=flip_probs_repeated.shape)
        # Compare random matrix with flip probabilities to get the flips
        flips = random_matrix < flip_probs_repeated
        # Apply the flips with an XOR between original array and flips
        new_bitstrings = bitstr_arr.repeat(dist, axis=0) ^ flips

        # Count all the new_bitstrings
        # Not converting to str right away because tuple indexing is faster
        new_counts: Counter = Counter(map(tuple, new_bitstrings))
        return Counter(
            {"".join(map(str, k)): v for k, v in new_counts.items()}
        )

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

    def infer_one_state(self) -> Eigenstate:
        """Infers the state measured as 1 from the eigenstates.

        Only works when the eigenstates form a known eigenbasis with
        a well-defined "one state".
        """
        eigenstates = set(self.eigenstates)
        if eigenstates == {"0", "1"}:
            return "1"
        if eigenstates == {"r", "g"}:
            return "r"
        if eigenstates == {"g", "h"}:
            return "h"
        if eigenstates == {"u", "d"}:
            return "d"
        raise RuntimeError(
            "Failed to infer the 'one state' from the "
            f"eigenstates: {self.eigenstates}"
        )
