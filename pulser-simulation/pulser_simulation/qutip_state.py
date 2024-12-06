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
"""Definition of QutipState and QutipOperator."""
from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import SupportsComplex, Type, TypeVar

import numpy as np
import qutip
import qutip.qobj

from pulser.backend.state import Eigenstate, State

QutipStateType = TypeVar("QutipStateType", bound="QutipState")

QuditOp = Mapping[str, SupportsComplex]
TensorOp = Sequence[tuple[QuditOp, Collection[int]]]
FullOp = Sequence[tuple[SupportsComplex, TensorOp]]


class QutipState(State[SupportsComplex, complex]):
    """A quantum state stored as a qutip.Qobj.

    Args:
        state: The state as a qutip.Qobj. Can be a statevector
            or a density matrix.
        eigenstates: The eigenstates that form a qudit's eigenbasis, each
            given as an individual character. The order of the eigenstates
            matters, as for eigenstates ("a", "b", ...),  "a" will be
            associated to eigenvector (1, 0, ...), "b" to (0, 1, ...) and
            so on.
    """

    def __init__(
        self, state: qutip.Qobj, *, eigenstates: Sequence[Eigenstate]
    ):
        """Initializes a QutipState."""
        self._validate_eigenstates(eigenstates)
        self._eigenstates = tuple(eigenstates)
        valid_types = ("ket", "bra", "oper")
        if not isinstance(state, qutip.Qobj) or state.type not in valid_types:
            raise TypeError(
                "'state' must be a qutip.Qobj with one of types "
                f"{valid_types}, not {state!r}."
            )
        self._state = state.dag() if state.isbra else state
        self._validate_shape(self._state.shape, self.qudit_dim)

    @property
    def n_qudits(self) -> int:
        """The number of qudits in the state."""
        return round(math.log(self._state.shape[0], self.qudit_dim))

    def overlap(self, other: QutipState) -> float:
        """Compute the overlap between this state and another of the same type.

        Generally computes Tr[AB] for mixed states A and B, which
        corresponds to |<a|b>|^2 for pure states A=|a><a| and B=|b><b|.

        Args:
            other: The other state.

        Returns:
            The overlap between the two states.
        """
        if not isinstance(other, QutipState):
            raise TypeError(
                "'QutipState.overlap()' expects another 'QutipState', not "
                f"{type(other)}."
            )
        if (
            self.n_qudits != other.n_qudits
            or self.qudit_dim != other.qudit_dim
        ):
            raise ValueError(
                "Can't calculate the overlap between a state with "
                f"{self.n_qudits} {self.qudit_dim}-dimensional qudits and "
                f"another with {self.n_qudits} {self.qudit_dim}-dimensional"
                "qudits."
            )
        if self.eigenstates != other.eigenstates:
            msg = (
                "Can't calculate the overlap between states with eigenstates"
                f"{self.eigenstates} and {other.eigenstates}."
            )
            if set(self.eigenstates) != set(other.eigenstates):
                raise ValueError(msg)
            raise NotImplementedError(msg)
        overlap = self._state.overlap(other._state)
        if self._state.isket and other._state.isket:
            overlap = np.abs(overlap) ** 2
        return float(overlap)

    def probabilities(self, *, cutoff: float = 1e-10) -> dict[str, float]:
        """Extracts the probabilties of measuring each basis state combination.

        Args:
            cutoff: The value below which a probability is considered to be
                zero.

        Returns:
            A mapping between basis state combinations and their respective
            probabilities.
        """
        if not self._state.isket:
            probs = np.abs(self._state.diag())
        else:
            probs = (np.abs(self._state.full()) ** 2).flatten()
        non_zero = probs > cutoff
        return dict(
            zip(np.array(self.basis_states)[non_zero], probs[non_zero])
        )

    @classmethod
    def from_state_amplitudes(
        cls: Type[QutipStateType],
        *,
        eigenstates: Sequence[Eigenstate],
        amplitudes: dict[str, SupportsComplex],
    ) -> QutipStateType:
        """Construct the state from its basis states' amplitudes.

        Args:
            eigenstates: The basis states (e.g., ('r', 'g')).
            amplitudes: A mapping between basis state combinations and
                complex amplitudes.

        Returns:
            The state constructed from the amplitudes.
        """
        cls._validate_eigenstates(eigenstates)
        basis_states = list(amplitudes)
        n_qudits = len(basis_states[0])
        if not all(
            len(bs) == n_qudits and set(bs) <= set(eigenstates)
            for bs in basis_states
        ):
            raise ValueError(
                "All basis states must be combinations of eigenstates with the"
                f" same length. Expected combinations of {eigenstates}, each "
                f" with {n_qudits} elements."
            )

        qudit_dim = len(eigenstates)

        def make_qobj(basis_state: str) -> qutip.Qobj:
            return qutip.tensor(
                [
                    qutip.basis(qudit_dim, eigenstates.index(s))
                    for s in basis_state
                ]
            )

        # Start with an empty Qobj with the right dimension
        state = make_qobj(eigenstates[0] * n_qudits) * 0
        for basis_state, amp in amplitudes.items():
            state += complex(amp) * make_qobj(basis_state)

        return cls(state, eigenstates=eigenstates)

    def __repr__(self) -> str:
        return "\n".join(
            [
                "QutipState",
                "----------",
                f"Eigenstates: {self.eigenstates}",
                self._state.__repr__(),
            ]
        )

    @staticmethod
    def _validate_eigenstates(eigenstates: Sequence[Eigenstate]) -> None:
        if any(not isinstance(s, str) or len(s) != 1 for s in eigenstates):
            raise ValueError(
                "All eigenstates must be represented by single characters."
            )
        if len(eigenstates) != len(set(eigenstates)):
            raise ValueError("'eigenstates' can't contain repeated entries.")

    @staticmethod
    def _validate_shape(shape: tuple[int, int], qudit_dim: int) -> None:
        expected_n_qudits = math.log(shape[0], qudit_dim)
        if not np.isclose(expected_n_qudits, round(expected_n_qudits)):
            raise ValueError(
                f"A qutip.Qobj with shape {shape} can't represent "
                f"a collection of {qudit_dim}-level qudits."
            )
