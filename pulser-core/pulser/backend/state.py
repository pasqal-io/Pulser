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
from typing import Any, Generic, Literal, SupportsFloat, Type, TypeVar, Union

import numpy as np

from pulser.channels.base_channel import States
from pulser.exceptions.serialization import AbstractReprError

Eigenstate = Union[States, Literal["0", "1"]]

ArgScalarType = TypeVar("ArgScalarType")
ReturnScalarType = TypeVar("ReturnScalarType", bound=SupportsFloat)
StateType = TypeVar("StateType", bound="State")


class State(ABC, Generic[ArgScalarType, ReturnScalarType]):
    """Base class enforcing an API for quantum states.

    Each backend will implement its own type of state and the methods below.
    """

    _eigenstates: Sequence[Eigenstate]
    _amplitudes: Mapping[str, complex] | None

    def __init__(self, *, eigenstates: Sequence[Eigenstate]) -> None:
        """Initializes a State."""
        self._validate_eigenstates(eigenstates)
        self._eigenstates = eigenstates
        self._amplitudes = None

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
    def qudit_dim(self) -> int:
        """The dimensions (ie number of eigenstates) of a qudit."""
        return len(self.eigenstates)

    def get_basis_state_from_index(self, index: int) -> str:
        """Generates a basis state combination from its index in the state.

        Assumes a state vector representation regardless of the actual support
        of the state.

        Args:
            index: The position of the state in a state vector.

        Returns:
            The basis state combination for the desired index.
        """
        if index < 0:
            raise ValueError(
                f"'index' must be a non-negative integer;"
                f" got {index} instead."
            )
        return "".join(
            self.eigenstates[int(dig)]
            for dig in np.base_repr(index, base=self.qudit_dim).zfill(
                self.n_qudits
            )
        )

    @abstractmethod
    def overlap(self: StateType, other: StateType, /) -> ReturnScalarType:
        """Compute the overlap between this state and another of the same type.

        Generally computes ``Tr[AB]`` for mixed states ``A`` and ``B``, which
        corresponds to ``|<a|b>|^2`` for pure states ``A=|a><a|`` and
        ``B=|b><b|``.

        Args:
            other: The other state.

        Returns:
            The overlap between the two states.
        """
        pass

    @abstractmethod
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
        pass

    @classmethod
    def from_state_amplitudes(
        cls: Type[StateType],
        *,
        eigenstates: Sequence[Eigenstate],
        amplitudes: Mapping[str, ArgScalarType],
    ) -> StateType:
        """Construct the state from its basis states' amplitudes.

        Only states constructed with this method are allowed in remote backend.

        Args:
            eigenstates: The basis states (e.g., ('r', 'g')).
            amplitudes: A mapping between basis state combinations and
                complex amplitudes (e.g., {"rgr": 0.5, "grg": 0.5}).

        Returns:
            The state constructed from the amplitudes.
        """
        cls._validate_eigenstates(eigenstates)
        n_qudits = cls._validate_amplitudes(amplitudes, eigenstates)
        obj, _amplitudes = cls._from_state_amplitudes(
            eigenstates=eigenstates, n_qudits=n_qudits, amplitudes=amplitudes
        )
        obj._amplitudes = _amplitudes
        return obj

    @classmethod
    @abstractmethod
    def _from_state_amplitudes(
        cls: Type[StateType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        amplitudes: Mapping[str, ArgScalarType],
    ) -> tuple[StateType, Mapping[str, complex]]:
        """Implements the conversion used in `from_state_amplitudes()`.

        Expected to return the State instance alongside the amplitudes used
        in serialization.
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

    @staticmethod
    def _validate_eigenstates(eigenstates: Sequence[Eigenstate]) -> None:
        if not isinstance(eigenstates, Sequence):
            raise TypeError(
                "'eigenstates' must be a 'collections.Sequence' "
                f"(list or tuple), not {type(eigenstates).__name__}."
            )
        if any(not isinstance(s, str) or len(s) != 1 for s in eigenstates):
            raise ValueError(
                "All eigenstates must be represented by single characters."
            )
        if len(eigenstates) != len(set(eigenstates)):
            raise ValueError("'eigenstates' can't contain repeated entries.")

    @staticmethod
    def _validate_amplitudes(
        amplitudes: Mapping[str, Any], eigenstates: Sequence[Eigenstate]
    ) -> int:
        """Validates the state amplitudes mapping.

        Returns:
            int: The number of qudits in the state.
        """
        basis_states = list(amplitudes)
        n_qudits = len(basis_states[0])
        if not all(
            len(bs) == n_qudits and set(bs) <= set(eigenstates)
            for bs in basis_states
        ):
            raise ValueError(
                "All basis states must be combinations of eigenstates with the"
                f" same length. Expected combinations of {eigenstates}, each "
                f"with {n_qudits} elements."
            )
        return n_qudits

    def _to_abstract_repr(self) -> dict[str, Any]:
        cls_name = self.__class__.__name__
        if self._amplitudes is None:
            raise AbstractReprError(
                f"Failed to serialize state of type {cls_name!r} because it "
                f"was not created via '{cls_name}.from_state_amplitudes()'."
            )
        stashed_state = self.from_state_amplitudes(
            eigenstates=self._eigenstates,
            amplitudes=self._amplitudes,  # type: ignore[arg-type]
        )

        if abs(float(self.overlap(stashed_state)) - 1.0) > 1e-12:
            raise AbstractReprError(
                f"Failed to serialize state of type {cls_name!r} because it "
                "was modified in place after its creation."
            )
        return {
            "eigenstates": tuple(self._eigenstates),
            "amplitudes": dict(self._amplitudes),
        }


class StateRepr(State):
    """Define a backend-independent quantum state representation.

    Allows the user to define a quantum state with the usual dedicated class
    method `from_state_amplitudes`, which requires:
    - eigenstates: The basis states (e.g., ('r', 'g')).
    - amplitudes: A mapping between basis state combinations and
        complex amplitudes (e.g., {"rgr": 0.5, "grg": 0.5}).

    The created state, supports de/serialization methods for remote backend
    execution.

    Example:
    ```python
    eigenstates = ("r", "g")
    amplitudes = {"rgr"=0.5, "grg"=0.5}
    state = StateRepr.from_state_amplitudes(
        eigenstates=eigenstates, amplitudes=amplitudes
    )
    ```
    """

    _n_qudits: int

    @classmethod
    def _from_state_amplitudes(
        cls,
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        amplitudes: Mapping[str, complex],
    ) -> tuple[StateRepr, Mapping[str, complex]]:
        """Implements the conversion used in `from_state_amplitudes()`.

        Expected to return the State instance alongside the amplitudes used
        in serialization.
        """
        state = cls(eigenstates=eigenstates)
        cls._n_qudits = n_qudits
        return state, amplitudes

    def _to_abstract_repr(self) -> dict[str, Any]:
        cls_name = self.__class__.__name__
        if self._amplitudes is None:
            raise AbstractReprError(
                f"Failed to serialize state of type {cls_name!r} because it "
                f"was not created via '{cls_name}.from_state_amplitudes()'."
            )
        return {
            "eigenstates": tuple(self._eigenstates),
            "amplitudes": dict(self._amplitudes),
        }

    @property
    def n_qudits(self) -> int:
        """The number of qudits in the state."""
        return self._n_qudits

    def overlap(self, other: StateRepr, /) -> None:
        """``overlap`` not implemented in ``StateRepr``."""
        raise NotImplementedError

    def sample(
        self,
        *,
        num_shots: int,
        one_state: Eigenstate | None = None,
        p_false_pos: float = 0.0,
        p_false_neg: float = 0.0,
    ) -> Counter[str]:
        """``sample`` not implemented in ``StateRepr``."""
        raise NotImplementedError
