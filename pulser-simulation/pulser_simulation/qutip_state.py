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
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any, SupportsComplex, Type, TypeVar

import numpy as np
import qutip

from pulser.backend.state import Eigenstate, State
from pulser.math.multinomial import multinomial

QutipStateType = TypeVar("QutipStateType", bound="QutipState")

QuditOp = Mapping[str, SupportsComplex]
TensorOp = Sequence[tuple[QuditOp, Collection[int]]]
FullOp = Sequence[tuple[SupportsComplex, TensorOp]]


class QutipState(State[SupportsComplex, float]):
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
        super().__init__(eigenstates=eigenstates)
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

    def to_qobj(self) -> qutip.Qobj:
        """Returns a copy of the state's Qobj representation."""
        return self._state.copy()

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
                f"another with {other.n_qudits} {other.qudit_dim}-dimensional "
                "qudits."
            )
        if self.eigenstates != other.eigenstates:
            msg = (
                "Can't calculate the overlap between states with eigenstates "
                f"{self.eigenstates} and {other.eigenstates}."
            )
            if set(self.eigenstates) != set(other.eigenstates):
                raise ValueError(msg)
            raise NotImplementedError(msg)
        overlap = self._state.overlap(other._state)
        if self._state.isket and other._state.isket:
            overlap = np.abs(overlap) ** 2
        return float(overlap.real)

    def probabilities(self, *, cutoff: float = 1e-12) -> dict[str, float]:
        """Extracts the probabilties of measuring each basis state combination.

        The probabilities are normalized to sum to 1.

        Args:
            cutoff: The value below which a probability is considered to be
                zero.

        Returns:
            A mapping between basis state combinations and their respective
            probabilities.
        """
        if not self._state.isket:
            probs = np.abs(self._state.diag()).real
        else:
            probs = (np.abs(self._state.full()) ** 2).flatten().real
        non_zero = np.argwhere(probs > cutoff).flatten()
        # Renormalize to make the non-zero probablilites sum to 1.
        probs = probs[non_zero]
        probs = probs / np.sum(probs)
        return dict(zip(map(self.get_basis_state_from_index, non_zero), probs))

    def bitstring_probabilities(
        self, *, one_state: Eigenstate | None = None, cutoff: float = 1e-12
    ) -> Mapping[str, float]:
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
        bitstring_probs: dict[str, float] = defaultdict(float)
        for state_str in probs:
            bitstring = state_str.replace(one_state, "1")
            for s_ in zero_states:
                bitstring = bitstring.replace(s_, "0")
            bitstring_probs[bitstring] += probs[state_str]
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
        indices = multinomial(num_shots, probs)
        if p_false_pos == 0.0 and p_false_neg == 0.0:
            return Counter(bitstrings[indices])

        # Convert bitstrings to a 2D array
        bitstr_arr = np.array(
            [list(bs) for bs in bitstrings[indices]], dtype=int
        )
        # If 1 is measured, flip_prob=p_false_neg else flip_prob=p_false_pos
        flip_probs = np.where(bitstr_arr == 1, p_false_neg, p_false_pos)
        # Generate random matrix of same shape
        random_matrix = np.random.uniform(size=flip_probs.shape)
        # Compare random matrix with flip probabilities to get the flips
        flips = random_matrix < flip_probs
        # Apply the flips with an XOR between original array and flips
        new_bitstrings = bitstr_arr ^ flips

        # Count all the new_bitstrings
        # Not converting to str right away because tuple indexing is faster
        new_counts: Counter = Counter(map(tuple, new_bitstrings))
        return Counter(
            {"".join(map(str, k)): v for k, v in new_counts.items()}
        )

    @classmethod
    def _from_state_amplitudes(
        cls: Type[QutipStateType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        amplitudes: Mapping[str, SupportsComplex],
    ) -> tuple[QutipStateType, Mapping[str, complex]]:
        """Construct the state from its basis states' amplitudes.

        Args:
            eigenstates: The basis states (e.g., ('r', 'g')).
            amplitudes: A mapping between basis state combinations and
                complex amplitudes.

        Returns:
            The state constructed from the amplitudes, the eigenstates and the
            amplitudes that defined the state.
        """
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
        amps = {k: complex(v) for k, v in amplitudes.items()}
        for basis_state, amp in amps.items():
            state += amp * make_qobj(basis_state)

        return cls(state, eigenstates=eigenstates), amps

    def __repr__(self) -> str:
        return "\n".join(
            [
                "QutipState",
                "----------",
                f"Eigenstates: {self.eigenstates}",
                self._state.__repr__(),
            ]
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, QutipState):
            return False
        return (
            self.eigenstates == other.eigenstates
            and self._state == other._state
        )

    @staticmethod
    def _validate_shape(shape: tuple[int, int], qudit_dim: int) -> None:
        expected_n_qudits = math.log(shape[0], qudit_dim)
        if not np.isclose(expected_n_qudits, round(expected_n_qudits)):
            raise ValueError(
                f"A qutip.Qobj with shape {shape} is incompatible with "
                f"a system of {qudit_dim}-level qudits."
            )
