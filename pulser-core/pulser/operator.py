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
from dataclasses import dataclass
from numbers import Number
from typing import Mapping

from pulser.channels import State
from pulser.register import QubitId


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

    kron_operations: tuple[QuditOperatorString, set[str]]

    @property
    def operations(self) -> tuple[tuple[tuple[Number, str]], str]:
        """Returns the operations associated with the operator."""
        return tuple(
            (kron_op[0].operations, tuple(kron_op[1]))
            for kron_op in self.kron_operations
        )

    def __str__(self) -> str:
        return str(self.operations)


class OperatorString:
    """A linear combination of multi-qudit operators."""

    coefficients: list[Number]
    operators: list[TargetedOperatorString]

    @property
    def operations(
        self,
    ) -> tuple[Number, tuple[Number, tuple[tuple[tuple[Number, str]], str]]]:
        """Returns the operations associated with the operator."""
        return zip(
            self.coefficients, list(op.operations for op in self.operators)
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
        used_qubits = set()
        for op in self.operator_string.operators:
            for kron_op in op.kron_operations:
                used_operators.union(set(kron_op[0].operators))
                used_qubits.union(set(kron_op[1]))
        if not used_operators.issubset(self.operators_dict.keys()):
            raise ValueError(
                "All operators defined in operator_string must be mapped"
                " to a value in operators_dict."
            )
        if not used_qubits.issubset(self.qubit_ids):
            raise ValueError(
                "qubit_ids must contain at least all the qubits defined"
                " in operator_string."
            )

    def __add__(self, operator: Operator) -> Operator:
        """Sum the current operator with a second one."""
        if not isinstance(operator, OperatorString):
            raise TypeError("Right operand for + must be an Operator")
        for op_key, op_value in self.operators_dict.items():
            if op_key in operator and operator[op_key] != op_value:
                raise ValueError(
                    f"Operator {op_key} is defined in the two operators"
                    " with different values."
                )
        return Operator(
            self.operator + operator,
            self.qubit_ids.union(operator.qubit_ids),
            {**self.operators_dict, **operator.operators_dict},
        )

    def __rmul__(self, scalar: Number) -> Operator:
        """Multiplies the current Operator by a number."""
        return Operator(
            self.operator_string * scalar, self.qubit_ids, self.operators_dict
        )

    @abstractmethod
    def kron(self, operator: Operator):
        """Kroneker product of the current operator by a second one."""
        pass

    @abstractmethod
    def __mul__(self, state: State) -> State:
        """Multiplies the current Operator by a state."""
        pass

    @abstractmethod
    def __matmul__(self, operator: Operator) -> Operator:
        """Multiplies the current Operator by a second one."""
        pass
