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
"""Classes to store Operators for simulation and measurements."""


from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
import itertools
from dataclasses import dataclass
from numbers import Number
from typing import Mapping

from pulser.channels import State
from pulser.register import QubitId


def _check_kron_object(
    kron_obj: tuple[tuple[QuditString | QuditOperatorString, set[QubitId]]],
    obj_type: type,
) -> None:
    """Check if an object with Kronecker products is correctly defined."""
    if not isinstance(kron_obj, tuple) or any(
        [isinstance(operation[0], obj_type) for operation in kron_obj]
    ):
        raise ValueError(
            "The kron_"
            f"{"qudits" if str(obj_type) == "QuditString" else "operations"}"
            " should be a tuple of tuples of shape (operator as a "
            f"{obj_type}, qubit id as an int or str)."
        )
    # Check the qubit ids targeted in operations
    target_qids = [operation[1] for operation in kron_obj]
    if "global" in target_qids:
        if len(kron_obj) > 1:
            raise ValueError(
                "If a 'global' operation is defined, no other operations "
                "can be defined."
            )
    else:
        target_qids_list = list(itertools.chain(*target_qids))
        target_qids_set = set(target_qids_list)
        if len(target_qids_set) < len(target_qids_list):
            # Either the qubit id has been defined twice in an operation:
            for qids in target_qids:
                if len(set(qids)) < len(qids):
                    raise ValueError("Duplicate atom ids in argument list.")
            # Or it was defined in two different operations
            raise ValueError(
                "Each qubit can be targeted by only one operation."
            )


@classmethod
class QuditString:
    """A 1-qudit state."""

    coefficients: list[Number]
    states: list[State]

    def as_tuple(self) -> tuple[tuple[Number, State]]:
        """Returns tuple of (coefficient, state label)."""
        return zip(self.coefficients, self.states)


@classmethod
class MultiQuditString:
    """A multi-qudit state."""

    kron_qudits = tuple[tuple[QuditString, set[QubitId]]]

    def __post_init__(self):
        _check_kron_object(self.kron_qudits, QuditString)


@dataclass
class QuditOperatorString:
    """A linear combination of 1-qudit operators."""

    coefficients: list[Number]
    operators: list[str]

    @property
    def operations(self) -> tuple[tuple[Number, str]]:
        """Returns the operations associated with the operator."""
        return zip(self.coefficients, self.operators)

    def __str__(self) -> str:
        return str(self.operations)


@dataclass
class TargetedOperatorString:
    """A multi-qudit operator defined as a tensor product."""

    kron_operations: tuple[tuple[QuditOperatorString, set[QubitId]]]

    def __post_init__(self):
        _check_kron_object(self.kron_qudits, QuditOperatorString)

    @property
    def targeted_qubits(self) -> set[QubitId]:
        """Returns the qubits targeted by the operator."""
        return set().union(
            *[kron_operation[1] for kron_operation in self.kron_operations]
        )

    @property
    def operations(self) -> tuple[tuple[tuple[Number, str]], QubitId]:
        """Returns the operations associated with the operator."""
        return tuple(
            (kron_op[0].operations, tuple(kron_op[1]))
            for kron_op in self.kron_operations
        )

    def __str__(self) -> str:
        return str(self.operations)

    def kron(self, operator: TargetedOperatorString):
        """Kronecker product of the current operator with a new operator."""
        if (
            "global" in self.targeted_qubits
            or "global" in operator.targeted_qubits
        ):
            raise ValueError(
                "Cannot operate kron if 'global' operation is defined."
            )
        if not self.targeted_qubits.intersection(operator.targeted_qubits):
            raise ValueError(
                "Qubit ids defined in the two operators sould be distinct."
            )
        return TargetedOperatorString(
            self.kron_operations + operator.kron_operations
        )


@dataclass
class OperatorString:
    """A linear combination of multi-qudit operators."""

    coefficients: list[Number]
    operators: list[TargetedOperatorString]

    @property
    def operations(
        self,
    ) -> tuple[
        Number, tuple[Number, tuple[tuple[tuple[Number, str]], QubitId]]
    ]:
        """Returns the operations associated with the operator."""
        return zip(
            self.coefficients, list(op.operations for op in self.operators)
        )

    @property
    def targeted_qubits(self) -> set[QubitId]:
        """Returns the qubits targeted by the operator."""
        return set().union(
            *[operator.targeted_qubits[1] for operator in self.operators]
        )

    def __str__(self) -> str:
        return str(self.operations)

    def __add__(self, operator: OperatorString) -> OperatorString:
        """Defines the sum of two OperatorString."""
        if not isinstance(operator, OperatorString):
            raise TypeError("Right operand for + must be an OperatorString")
        return OperatorString(
            self.coefficients + operator.coefficients,
            self.operators + operator.operators,
        )

    def __rmul__(self, scalar: Number) -> OperatorString:
        """Multiplies the current OperatorString by a number."""
        if not isinstance(scalar, Number):
            raise TypeError("Left operand with * must be a Number.")
        return OperatorString(
            [coeff * scalar for coeff in self.coefficients], self.operators
        )

    def kron(
        self,
        operator: OperatorString,
        qubit_ids1: Collection[QubitId],
        qubit_ids2: Collection[QubitId],
    ) -> OperatorString:
        """Kronecker product of the current operator with a new operator."""
        if not set(self.qubit_ids).intersection(operator.qubit_ids):
            raise ValueError("Operators should have different qubit ids.")
        coeffs = []
        ops = []
        for coeff1, global_op1 in zip(self.coefficients, self.operators):
            inner_op1 = [global_op1]
            if "global" in global_op1.targeted_qubits:
                assert len(global_op1.kron_operations) == 1
                inner_op1 = [
                    TargetedOperatorString(
                        ((global_op1.kron_operations[0][0], set([qubit_id])),)
                    )
                    for qubit_id in qubit_ids1
                ]
            for op1 in inner_op1:
                for coeff2, global_op2 in zip(
                    operator.coefficients, operator.operators
                ):
                    inner_op2 = [global_op2]
                    if "global" in global_op2.targeted_qubits:
                        assert len(global_op2.kron_operations) == 1
                        inner_op2 = [
                            TargetedOperatorString(
                                (
                                    (
                                        global_op2.kron_operations[0][0],
                                        set([qubit_id]),
                                    ),
                                )
                            )
                            for qubit_id in qubit_ids2
                        ]
                    for op2 in inner_op2:
                        coeffs.append(coeff1 * coeff2)
                        ops.append(op1.kron(op2))
        return OperatorString(coeffs, ops)


@dataclass
class Operator(ABC):
    """Defines a generic operator class."""

    operator_string: OperatorString
    qubit_ids: Collection[QubitId]
    operators_dict: Mapping

    def __post_init__(self):
        if not isinstance(self.operator_string, OperatorString):
            raise TypeError(
                "operator_string should be an OperatorString instance."
            )
        used_operators = set()
        for op in self.operator_string.operators:
            for kron_op in op.kron_operations:
                used_operators.union(set(kron_op[0].operators))
        if not used_operators.issubset(self.operators_dict.keys()):
            raise ValueError(
                "All operators defined in operator_string must be mapped"
                " to a value in operators_dict."
            )
        qubit_labels = set(self.qubit_ids).union(["global"])
        if not self.operator_string.targeted_qubits.issubset(qubit_labels):
            raise ValueError(
                "Allowed qubits in operator_string are qubits in"
                "qubit_ids and 'global'."
            )

    def _check_operand(self, operator: OperatorString, operation: str) -> None:
        if not isinstance(operator, OperatorString):
            raise TypeError(
                f"Right operand for {operation} must be an Operator."
            )
        for op_key, op_value in self.operators_dict.items():
            if op_key in operator and operator[op_key] != op_value:
                raise ValueError(
                    f"Operator {op_key} is defined in the two operators"
                    " with different values."
                )

    def __add__(self, operator: Operator) -> Operator:
        """Sum the current operator with a second one."""
        self._check_operand(operator, "+")
        return Operator(
            self.operator + operator,
            self.qubit_ids.union(operator.qubit_ids),
            {**self.operators_dict, **operator.operators_dict},
        )

    def __rmul__(self, scalar: Number) -> Operator:
        """Multiplies the current Operator by a number."""
        return Operator(
            scalar * self.operator_string, self.qubit_ids, self.operators_dict
        )

    @abstractmethod
    def kron(self, operator: Operator) -> Operator:
        """Kroneker product of the current operator with a second one."""
        self._check_operand(operator, "kron")
        return Operator(
            self.operator_string.kron(
                operator.operator_string, self.qubit_ids, operator.qubit_ids
            ),
            self.qubit_ids.union(operator.qubit_ids),
            {**self.operators_dict, **operator.operators_dict},
        )

    @abstractmethod
    def __mul__(self, state: MultiQuditString) -> MultiQuditString:
        """Multiplies the current Operator by a state."""
        pass

    @abstractmethod
    def __matmul__(self, operator: Operator) -> Operator:
        """Multiplies the current Operator by a second one."""
        pass
