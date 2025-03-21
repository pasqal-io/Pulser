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
from typing import Any, Generic, Type, TypeVar

from pulser.backend.state import Eigenstate, State
from pulser.json.exceptions import AbstractReprError

ArgScalarType = TypeVar("ArgScalarType")
ReturnScalarType = TypeVar("ReturnScalarType")
StateType = TypeVar("StateType", bound=State)
OperatorType = TypeVar("OperatorType", bound="Operator")

# Generic type aliases
T = TypeVar("T")
QuditOp = Mapping[str, T]  # single qudit operator
TensorOp = Sequence[
    tuple[QuditOp[T], Collection[int]]
]  # QuditOp applied to set of qudits
FullOp = Sequence[tuple[T, TensorOp[T]]]  # weighted sum of TensorOp


class Operator(ABC, Generic[ArgScalarType, ReturnScalarType, StateType]):
    """Base class for a quantum operator."""

    _eigenstates: Sequence[Eigenstate] | None
    _n_qudits: int | None
    _operations: FullOp[complex] | None

    def __init__(self) -> None:
        """Initializes an Operator."""
        self._eigenstates = None
        self._n_qudits = None
        self._operations = None

    @abstractmethod
    def apply_to(self, state: StateType, /) -> StateType:
        """Apply the operator to a state.

        Args:
            state: The state to apply this operator to.

        Returns:
            The resulting state.
        """
        pass

    @abstractmethod
    def expect(self, state: StateType, /) -> ReturnScalarType:
        """Compute the expectation value of self on the given state.

        Args:
            state: The state with which to compute.

        Returns:
            The expectation value.
        """
        pass

    @abstractmethod
    def __add__(self: OperatorType, other: OperatorType, /) -> OperatorType:
        """Computes the sum of two operators.

        Args:
            other: The other operator.

        Returns:
            The summed operator.
        """
        pass

    @abstractmethod
    def __rmul__(self: OperatorType, scalar: ArgScalarType) -> OperatorType:
        """Scale the operator by a scalar factor.

        Args:
            scalar: The scalar factor.

        Returns:
            The scaled operator.
        """
        pass

    @abstractmethod
    def __matmul__(self: OperatorType, other: OperatorType) -> OperatorType:
        """Compose two operators where 'self' is applied after 'other'.

        Args:
            other: The operator to compose with self.

        Returns:
            The composed operator.
        """
        pass

    @classmethod
    def from_operator_repr(
        cls: Type[OperatorType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[ArgScalarType],
    ) -> OperatorType:
        """Create an operator from the operator representation.

        The full operator representation (``FullOp``) is a weigthed sum of
        tensor operators (``TensorOp``), written as a sequence of coefficient
        and tensor operator pairs, ie

        ``FullOp = Sequence[tuple[ScalarType, TensorOp]]``

        Each ``TensorOp`` is itself a sequence of qudit operators (``QuditOp``)
        applied to mutually exclusive sets of qudits (represented by their
        indices), ie

        ``TensorOp = Sequence[tuple[QuditOp, Collection[int]]]``

        Qudits without an associated ``QuditOp`` are applied the identity
        operator.

        Finally, each ``QuditOp`` is represented as weighted sum of pre-defined
        single-qudit operators. It is given as a mapping between a string
        representation of the single-qudit operator and its respective
        coefficient, ie

        ``QuditOp = Mapping[str, ScalarType]``

        By default it identifies strings ``"ij"`` as single-qudit operators,
        where ``i`` and ``j`` are eigenstates that denote ``|i><j|``.

        Args:
            eigenstates: The eigenstates to use.
            n_qubits: How many qubits there are in the system.
            operations: The full operator representation.

        Returns:
            The constructed operator.
        """
        obj, _operations = cls._from_operator_repr(
            eigenstates=eigenstates, n_qudits=n_qudits, operations=operations
        )
        obj._eigenstates = eigenstates
        obj._n_qudits = n_qudits
        obj._operations = _operations
        return obj

    @classmethod
    @abstractmethod
    def _from_operator_repr(
        cls: Type[OperatorType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[ArgScalarType],
    ) -> tuple[OperatorType, FullOp[complex]]:
        """Implements the conversion used in `from_operator_repr()`.

        Expected to return the Operator instance alongside the 'operations' to
        use in serialization.
        """
        pass

    def _to_abstract_repr(self) -> dict[str, Any]:
        if (
            self._eigenstates is None
            or self._n_qudits is None
            or self._operations is None
        ):
            cls_name = self.__class__.__name__
            raise AbstractReprError(
                f"Failed to serialize state of type {cls_name!r} because it "
                f"was not created via '{cls_name}.from_operator_repr()'."
            )
        return {
            "eigenstates": tuple(self._eigenstates),
            "n_qudits": self._n_qudits,
            "operations": self._operations,
        }
