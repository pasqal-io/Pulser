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
"""Defines the abstract base class for a quantum operator."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from typing import Generic, TypeVar

from pulser.backend.state import State

ScalarType = TypeVar("ScalarType")


QuditOp = Mapping[str, ScalarType]  # single qudit operator
TensorOp = Sequence[
    tuple[QuditOp, Collection[int]]
]  # QuditOp applied to set of qudits
FullOp = Sequence[tuple[ScalarType, TensorOp]]  # weighted sum of TensorOp


class Operator(ABC, Generic[ScalarType]):
    """Base class for a quantum operator."""

    @abstractmethod
    def __mul__(self, other: State) -> State:
        """Apply the operator to a state.

        Args:
            other: The state to apply this operator to.

        Returns:
            The resulting state.
        """
        pass

    @abstractmethod
    def __add__(self, other: Operator) -> Operator:
        """Computes the sum of two operators.

        Args:
            other: The other operator.

        Returns:
            The summed operator.
        """
        pass

    @abstractmethod
    def expect(self, state: State) -> ScalarType:
        """Compute the expectation value of self on the given state.

        Args:
            state: The state with which to compute.

        Returns:
            The expectation value.
        """
        pass

    @abstractmethod
    def __rmul__(self, scalar: ScalarType) -> Operator:
        """Scale the operator by a scalar factor.

        Args:
            scalar: The scalar factor.

        Returns:
            The scaled operator.
        """
        pass

    @abstractmethod
    def __matmul__(self, other: Operator) -> Operator:
        """Compose two operators where 'self' is applied after 'other'.

        Args:
            other: The operator to compose with self.

        Returns:
            The composed operator.
        """
        pass

    @classmethod
    @abstractmethod
    def from_operator_repr(
        cls,
        *,
        eigenstates: Sequence[str],
        n_qudits: int,
        operations: FullOp,
        custom_operators: dict[str, QuditOp] = {},
    ) -> Operator:
        """Create an operator from the operator representation.

        The full operator representation (FullOp0 is a weigthed sum of tensor
        operators (TensorOp), written as a sequence of coefficient and tensor
        operator pairs, ie

        `FullOp = Sequence[tuple[ScalarType, TensorOp]]`

        Each TensorOp is itself a sequence of qudit operators (QuditOp) applied
        to mutually exclusive sets of qudits (represented by their indices), ie

        `TensorOp = Sequence[tuple[QuditOp, Collection[int]]]`

        Qudits without an associated QuditOp are applied the identity operator.

        Finally, each QuditOp is represented as weighted sum of pre-defined
        single-qudit operators. It is given as a mapping between a string
        representation of the single-qudit operator and its respective
        coefficient, ie

        `QuditOp = Mapping[str, ScalarType]`

        By default it identifies strings 'ij' as single-qudit operators, where
        i and j are eigenstates that denote |i><j|.
        Nonetheless, additional symbols can be defined in custom_operators.

        Args:
            eigenstates: The eigenstates to use.
            n_qubits: How many qubits there are in the system.
            operations: The full operator representation.
            custom_operators: Additional symbols and their corresponding
                single-qudit operators.

        Returns:
            The constructed operator.
        """
        pass
