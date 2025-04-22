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

from collections.abc import Sequence
from typing import Any, SupportsComplex, Type, TypeVar, cast

import qutip

from pulser.backend.operator import FullOp, Operator, QuditOp
from pulser.backend.state import Eigenstate
from pulser_simulation.qutip_state import QutipState

QutipStateType = TypeVar("QutipStateType", bound=QutipState)
QutipOperatorType = TypeVar("QutipOperatorType", bound="QutipOperator")


class QutipOperator(Operator[SupportsComplex, complex, QutipStateType]):
    """A quantum operator stored as a qutip.Qobj.

    Args:
        state: The operator as a qutip.Qobj.
        eigenstates: The eigenstates that form a qudit's eigenbasis, each
            given as an individual character. The order of the eigenstates
            matters, as for eigenstates ("a", "b", ...),  "a" will be
            associated to eigenvector (1, 0, ...), "b" to (0, 1, ...) and
            so on.

    """

    _eigenstates: Sequence[Eigenstate]

    def __init__(
        self, operator: qutip.Qobj, eigenstates: Sequence[Eigenstate]
    ):
        """Initializes a QutipOperator."""
        super().__init__()
        QutipState._validate_eigenstates(eigenstates)
        self._eigenstates = eigenstates
        if not isinstance(operator, qutip.Qobj) or not operator.isoper:
            raise TypeError(
                "'operator' must be a qutip.Qobj with type 'oper', not "
                f"{operator!r}."
            )
        QutipState._validate_shape(operator.shape, len(self._eigenstates))
        self._operator = operator

    @property
    def eigenstates(self) -> tuple[Eigenstate, ...]:
        """The eigenstates that form a qudit's eigenbasis.

        The order of the states should match the order in a numerical (ie
        state vector or density matrix) representation.
        For example, with eigenstates ("a", "b", ...),  "a" will be associated
        to eigenvector (1, 0, ...), "b" to (0, 1, ...) and so on.
        """
        return tuple(self._eigenstates)

    def to_qobj(self) -> qutip.Qobj:
        """Returns a copy of the operators's Qobj representation."""
        return self._operator.copy()

    def apply_to(self, state: QutipStateType, /) -> QutipStateType:
        """Apply the operator to a state.

        Args:
            state: The state to apply this operator to.

        Returns:
            The resulting state.
        """
        self._validate_other(state, QutipState, "QutipOperator.apply_to()")
        out = self._operator * state._state
        if state._state.isoper:
            out = out * self._operator.dag()
        return type(state)(out, eigenstates=state.eigenstates)

    def expect(self, state: QutipState, /) -> complex:
        """Compute the expectation value of self on the given state.

        Args:
            state: The state with which to compute.

        Returns:
            The expectation value.
        """
        self._validate_other(state, QutipState, "QutipOperator.expect()")
        return cast(complex, qutip.expect(self._operator, state._state))

    def __add__(
        self: QutipOperatorType, other: QutipOperatorType, /
    ) -> QutipOperatorType:
        """Computes the sum of two operators.

        Args:
            other: The other operator.

        Returns:
            The summed operator.
        """
        self._validate_other(other, QutipOperator, "__add__")
        return type(self)(
            self._operator + other._operator, eigenstates=self.eigenstates
        )

    def __rmul__(
        self: QutipOperatorType, scalar: SupportsComplex
    ) -> QutipOperatorType:
        """Scale the operator by a scalar factor.

        Args:
            scalar: The scalar factor.

        Returns:
            The scaled operator.
        """
        return type(self)(
            complex(scalar) * self._operator, eigenstates=self.eigenstates
        )

    def __matmul__(
        self: QutipOperatorType, other: QutipOperatorType
    ) -> QutipOperatorType:
        """Compose two operators where 'self' is applied after 'other'.

        Args:
            other: The operator to compose with self.

        Returns:
            The composed operator.
        """
        self._validate_other(other, QutipOperator, "__matmul__")
        return type(self)(
            self._operator * other._operator, eigenstates=self.eigenstates
        )

    @classmethod
    def _from_operator_repr(
        cls: Type[QutipOperatorType],
        *,
        eigenstates: Sequence[Eigenstate],
        n_qudits: int,
        operations: FullOp[SupportsComplex],
    ) -> tuple[QutipOperatorType, FullOp[complex]]:
        """Create an operator from the operator representation.

        The full operator representation (FullOp is a weigthed sum of tensor
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

        Args:
            eigenstates: The eigenstates to use.
            n_qubits: How many qubits there are in the system.
            operations: The full operator representation.

        Returns:
            The constructed operator.
        """
        qudit_dim = len(eigenstates)

        def build_qudit_op(qudit_op: QuditOp[SupportsComplex]) -> qutip.Qobj:
            op = qutip.identity(qudit_dim) * 0
            for proj_str, coeff in qudit_op.items():
                ket = qutip.basis(qudit_dim, eigenstates.index(proj_str[0]))
                bra = qutip.basis(
                    qudit_dim, eigenstates.index(proj_str[1])
                ).dag()
                op += complex(coeff) * ket * bra
            return op

        coeffs: list[complex] = []
        tensor_ops: list[qutip.Qobj] = []
        reconstructed_ops = []
        for tensor_op_num, (coeff, tensor_op) in enumerate(operations):
            coeffs.append(complex(coeff))
            qobj_qudit_ops = [
                qutip.identity(qudit_dim) for _ in range(n_qudits)
            ]
            re_tensor_op = []
            for qudit_op, qudit_inds in tensor_op:
                for ind in qudit_inds:
                    qobj_qudit_ops[ind] = build_qudit_op(qudit_op)
                re_qudit_op = {k: complex(v) for k, v in qudit_op.items()}
                re_tensor_op.append((re_qudit_op, set(qudit_inds)))
            tensor_ops.append(qutip.tensor(qobj_qudit_ops))
            reconstructed_ops.append((coeffs[-1], re_tensor_op))

        full_op: qutip.Qobj = sum(c * t for c, t in zip(coeffs, tensor_ops))
        return cls(full_op, eigenstates=eigenstates), reconstructed_ops

    def __repr__(self) -> str:
        return "\n".join(
            [
                "QutipOperator",
                "-------------",
                f"Eigenstates: {self.eigenstates}",
                self._operator.__repr__(),
            ]
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, QutipOperator):
            return False
        return (
            self.eigenstates == other.eigenstates
            and self._operator == other._operator
        )

    def _validate_other(
        self,
        other: QutipState | QutipOperator,
        expected_type: Type,
        op_name: str,
    ) -> None:
        if not isinstance(other, expected_type):
            raise TypeError(
                f"'{op_name}' expects a '{expected_type.__name__}' instance, "
                f"not {type(other)}."
            )
        if self.eigenstates != other.eigenstates:
            msg = (
                f"Can't apply {op_name} between a {self.__class__.__name__} "
                f"with eigenstates {self.eigenstates} and a "
                f"{other.__class__.__name__} with {other.eigenstates}."
            )
            if set(self.eigenstates) != set(other.eigenstates):
                raise ValueError(msg)
            raise NotImplementedError(msg)
