# Copyright 2023 Pulser Development Team
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
"""Tools to generate operators adapted to SequenceSamples."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from numbers import Number
from typing import Collection, Sequence

import qutip
from numpy.typing import ArrayLike

from pulser.channels.base_channel import STATES_RANK, States
from pulser.register.base_register import QubitId
from pulser.sampler.samples import SequenceSamples


def default_operators(
    sampled_seq: SequenceSamples,
) -> Mapping[str, qutip.Qobj]:
    r"""Default operators associated to a SequenceSamples.

    The default operators are the projectors on states of the computational
    basis and the identity "I". The computational basis is composed of
    elements in sampled_seq.eigenbasis.
    The projectors are named "sigma_{a}{b}" with a, b elements of the
    computational basis such that ``sigma_{ab} = a * b^\dagger``.

    Args:
        sampled_seq: The SequenceSamples to consider. Provides the
            computational basis in which the operators are defined.

    Returns:
        A dictionary composed of default operators as qutip.Qobj objects
        and their associated key.
    """
    eigenbasis = sampled_seq.eigenbasis

    dim = len(eigenbasis)
    basis = {b: qutip.basis(dim, i) for i, b in enumerate(eigenbasis)}
    operators = {"I": qutip.qeye(dim)}
    for proj0 in eigenbasis:
        for proj1 in eigenbasis:
            proj_name = "sigma_" + proj0 + proj1
            operators[proj_name] = basis[proj0] * basis[proj1].dag()
    return operators


def build_projector(
    sampled_seq: SequenceSamples,
    operation: str,
) -> qutip.Qobj:
    r"""Creates a projector on a state.

    Returns the projector on a state of the computational basis, that are
    the elements of sampled_seq.eigenbasis.
    The string that can be provided are either "I" (the identity) or
    "sigma_{a}{b}" with a, b elements of the computational basis such that
    ``sigma_{ab} = a * b^\dagger``.

    Examples:
        If we have a sampled sequence with only a Rydberg Channel, by default
        the available operators are "I" and "sigma_ab" with a, b in ["r", "g"].
        The projector on the Rydberg ground state `sigma_gg` can be generated
        with::

            $ python build_1qubit_operator(sampled_seq, "sigma_gg")

    Args:
        sampled_seq: The SequenceSamples to consider. Provides the
            computational basis in which the operators are defined.
        operation: A string: either "I" or "sigma_{a}{b}" with a, b in the
            computational basis.

    Returns:
        The desired operator as a qutip.Qobj object, identity if "I",
        ``sigma_{ab} = a * b^\dagger`` if "sigma_ab".
    """
    return build_operator(
        sampled_seq,
        {0: [0.0, 0.0]},
        [(operation, [0])],
    )


def build_1qubit_operator(
    sampled_seq: SequenceSamples,
    operations: list[tuple[float, list[str | qutip.Qobj]]],
    operators: Mapping[str, qutip.Qobj] | None = None,
) -> qutip.Qobj:
    r"""Creates a 1 qubit operator summing projectors and qutip objects.

    Takes as argument a list of operations to apply on one qubit. Returns
    the sum of these operations. The elements in operations can be a
    ``qutip.Qobj`` or a string key. If ``operators`` is undefined, this string
    can be "I" or "sigma_{a}{b}" with a, b elements of the computational basis
    (sampled_seq.eigenbasis with "x" if with_leakage is True) and
    ``sigma_{ab} = a * b^\dagger``.

    Examples:
        If we have a sampled sequence with only a Rydberg Channel, by default
        the available operators are "I" and "sigma_ab" with a, b in ["r", "g"].
        The operator ``sigma_gg - sigma_rr`` can be written as::

            ```python
                build_1qubit_operator(
                    sampled_seq,
                    [(1.0, "sigma_gg"), (-1.0, "sigma_rr")],
                    with_leakage=True,
                )
            ```

        We can also define our custom set of operators. For instance, by
        setting `operators = {"X":qutip.sigmax(), "Z":qutip.sigmaz()}` you can
        make combinations of these operators such as X-Z::

        ```python
            build_1qubit_operator(
                sampled_seq,
                [(1.0, "X"), (-1.0, "Z")],
                operators=operators,
            )
        ```

    Args:
        sampled_seq: The SequenceSamples to consider. Provides the
            computational basis in which the operators are defined.
        operations: List of tuples `(operator, qubits)`.
        operators: A dict of operators and their labels. If None, it is
            composed of all the projectors on the computational basis
            (with error state if with_leakage is True) and "I".


    Returns:
        The final operator as a qutip.Qobj object.
    """
    if operators is None:
        operators = default_operators(sampled_seq)
    return sum(
        [
            operation[0]
            * build_operator(
                sampled_seq,
                {0: [0.0, 0.0]},
                [(operation[1], [0])],
                operators,
            )
            for operation in operations
        ]
    )


def build_operator(
    sampled_seq: SequenceSamples,
    qubit_ids: list[QubitId],
    operations: list[tuple],
    operators: Mapping[str, qutip.Qobj] | None = None,
) -> qutip.Qobj:
    r"""Creates an operator with non-trivial actions on some qubits.

    Takes as argument a list of tuples ``[(operator_1, qubits_1),
    (operator_2, qubits_2), ...]``. Returns the operator given by the tensor
    product of {``operator_i`` applied on ``qubits_i``} and Id on the rest.
    ``(operator, 'global')`` returns the sum for all ``j`` of operator
    applied at ``qubit_j`` and identity elsewhere.

    `operator` can be a ``qutip.Qobj`` or a string key of ``operators``. If
    ``operators`` is undefined, this string can be "I" or "sigma_{a}{b}" with
    a, b elements of the computational basis (sampled_seq.eigenbasis) and
    ``sigma_{a}{b} = a * b^\dagger``.
    `qubits` is the list on which operator will be applied. The qubits are
    passed as their label in the register.

    Examples:
        If you want to generate an effective noise operator for your sampled
        Sequence, provide a dummy register::

            ```python
                build_operator(sampled_seq, {0:[0., 0.]}, [("sigma_gg", [0])])
                build_operator(
                    sampled_seq, {0:[0., 0.]}, [(qutip.sigmax(), [0])]
                )
            ```

        Here are some operations on 4 qubits labelled ["q0", "q1", "q2", "q3"]:
            - ``[(qutip.sigmax(), 'global')]`` returns `XIII + IXII + IIXI +
                IIIX`.
            - ``[(qutip.sigmax(), ["q1"])]`` returns ``IXII``.
            - ``[(sigma_gg, ["q0"]), (sigma_rr, ["q1"])]`` applies
                ``sigma_gg sigma_rr II``.

        If you define in `operators` a dictionnary containing
        {"X": qutip.sigmax()}, then the first two operations can also be
        written ``[("X", 'global')]`` and ``[("X", ["q1"])]``.

    Args:
        sampled_seq: The SequenceSamples to consider. Provides the
            computational basis in which the operators are defined.
        qubits: A dict of {label: position} whose labels can be used in
            operations.
        operations: List of tuples `(operator, qubits)`.
        operators: A dict of operators and their labels. If None, it is
            composed of all the projectors on the computational basis
            (with error state if with_leakage is True) and "I".


    Returns:
        The final operator as qutip.Qobj object.
    """
    # Check operations list
    if not isinstance(operations, list):
        raise ValueError(
            "The operations should be a list of tuples of shape "
            "(operator as a qutip.Qobj or str, qubit id as an int or str)."
        )
    # Check the qubit ids targeted in operations
    target_qids = [operation[1] for operation in operations]
    if "global" in target_qids:
        if len(operations) > 1:
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
        if not target_qids_set.issubset(qubit_ids):
            raise ValueError(
                "Invalid qubit names: " f"{target_qids_set - set(qubit_ids)}"
            )
    # Generate default operators if no operators were given
    if operators is None:
        operators = default_operators(sampled_seq)

    # Build operator
    op_list = [operators["I"] for j in range(len(qubit_ids))]
    _qid_index = {qid: i for i, qid in enumerate(qubit_ids)}

    for operator, qids in operations:
        if qids == "global":
            return sum(
                build_operator(
                    sampled_seq,
                    qubit_ids,
                    [(operator, [q_id])],
                    operators,
                )
                for q_id in qubit_ids
            )
        if isinstance(operator, str):
            try:
                operator = operators[operator]
            except KeyError:
                raise ValueError(f"{operator} is not a valid operator")
        for qubit in qids:
            k = _qid_index[qubit]
            op_list[k] = operator
    return qutip.tensor(list(map(qutip.Qobj, op_list)))


class QuditOperator:
    r"""Creates a 1 qubit operator summing projectors and qutip objects.

    Takes as argument a list of operations to apply on one qubit. Returns
    the sum of these operations. The elements in operations can be a
    ``qutip.Qobj`` or a string key. If ``operators`` is undefined, this string
    can be "I" or "sigma_{a}{b}" with a, b elements of the computational basis
    (sampled_seq.eigenbasis with "x" if with_leakage is True) and
    ``sigma_{ab} = a * b^\dagger``.

    Examples:
        If we have a sampled sequence with only a Rydberg Channel, by default
        the available operators are "I" and "sigma_ab" with a, b in ["r", "g"].
        The operator ``sigma_gg - sigma_rr`` can be written as::

            ```python
                build_1qubit_operator(
                    sampled_seq,
                    [(1.0, "sigma_gg"), (-1.0, "sigma_rr")],
                    with_leakage=True,
                )
            ```

        We can also define our custom set of operators. For instance, by
        setting `operators = {"X":qutip.sigmax(), "Z":qutip.sigmaz()}` you can
        make combinations of these operators such as X-Z::

        ```python
            build_1qubit_operator(
                sampled_seq,
                [(1.0, "X"), (-1.0, "Z")],
                operators=operators,
            )
        ```

    Args:
        sampled_seq: The SequenceSamples to consider. Provides the
            computational basis in which the operators are defined.
        operations: List of tuples `(operator, qubits)`.
        operators: A dict of operators and their labels. If None, it is
            composed of all the projectors on the computational basis
            (with error state if with_leakage is True) and "I".


    Returns:
        The final operator as a qutip.Qobj object.
    """

    sampled_seq: SequenceSamples
    operations: tuple[tuple[Number, list[str]]]
    operators: Mapping[str, qutip.Qobj]

    def __init__(
        self,
        sampled_seq: SequenceSamples,
        operations: Sequence[tuple[Number, list[str]]],
        operators: Mapping[str, qutip.Qobj] | None = None,
    ) -> None:
        self.sampled_seq = sampled_seq
        self.eigenbasis = self.sampled_seq.eigenbasis
        self.operators = (
            default_operators(self.sampled_seq)
            if operators is None
            else operators
        )
        self.operations = tuple(operations)
        self.operations_operators = [
            operation[1] for operation in self.operations
        ]
        self.operations_coeffs = [
            operation[0] for operation in self.operations
        ]

    @classmethod
    def from_coeffs_operators(
        cls,
        sampled_seq: SequenceSamples,
        operations_coeffs: list[Number],
        operations_operators: list[str],
        operators: Mapping[str, qutip.Qobj] | None = None,
    ) -> QuditOperator:
        return QuditOperator(
            sampled_seq,
            zip(operations_coeffs, operations_operators),
            operators,
        )

    def build_operator(self):
        return sum(
            [
                self.operation[0]
                * build_operator(
                    self.sampled_seq,
                    {0: [0.0, 0.0]},
                    [(operation[1], [0])],
                    self.operators,
                )
                for operation in self.operations
            ]
        )

    def __repr__(self):
        return (
            f"Operations: {self.operations} \n"
            f"Eigenbasis: {self.eigenbasis} \n"
            + f"Operators: {self.operators}"
        )

    def __str__(self):
        return str(self.operations)


class MultiQuditOperator:
    r"""Creates a 1 qubit operator summing projectors and qutip objects.

    Takes as argument a list of operations to apply on one qubit. Returns
    the sum of these operations. The elements in operations can be a
    ``qutip.Qobj`` or a string key. If ``operators`` is undefined, this string
    can be "I" or "sigma_{a}{b}" with a, b elements of the computational basis
    (sampled_seq.eigenbasis with "x" if with_leakage is True) and
    ``sigma_{ab} = a * b^\dagger``.

    Examples:
        If we have a sampled sequence with only a Rydberg Channel, by default
        the available operators are "I" and "sigma_ab" with a, b in ["r", "g"].
        The operator ``sigma_gg - sigma_rr`` can be written as::

            ```python
                build_1qubit_operator(
                    sampled_seq,
                    [(1.0, "sigma_gg"), (-1.0, "sigma_rr")],
                    with_leakage=True,
                )
            ```

        We can also define our custom set of operators. For instance, by
        setting `operators = {"X":qutip.sigmax(), "Z":qutip.sigmaz()}` you can
        make combinations of these operators such as X-Z::

        ```python
            build_1qubit_operator(
                sampled_seq,
                [(1.0, "X"), (-1.0, "Z")],
                operators=operators,
            )
        ```

    Args:
        sampled_seq: The SequenceSamples to consider. Provides the
            computational basis in which the operators are defined.
        operations: List of tuples `(operator, qubits)`.
        operators: A dict of operators and their labels. If None, it is
            composed of all the projectors on the computational basis
            (with error state if with_leakage is True) and "I".


    Returns:
        The final operator as a qutip.Qobj object.
    """

    qubit_ids: tuple[str]
    operations: tuple[tuple[Number, tuple[tuple[QuditOperator, str]]]]
    operators: Mapping[str, qutip.Qobj]

    def __init__(
        self,
        qubit_ids: tuple[str],
        operations: (
            Sequence[tuple[QuditOperator, str]]
            | Sequence[tuple[Number, Sequence[tuple[QuditOperator, str]]]]
        ),
    ) -> None:
        self.qubit_ids = qubit_ids
        self.operations = tuple(operations)
        if not isinstance(operations[0][0], Number):
            self.operations = [(1.0, self.operations)]
        elif not isinstance(self.operations[0][1][0], QuditOperator):
            raise ValueError(
                "Operations should be a Sequence of tuple[QuditOperator, str] or a Sequence"
                " of tuple[Number, Sequence[tuple[QuditOperator, str]]]."
            )
        self.operations_operators = [
            operation[1] for operation in self.operations
        ]
        self.operations_coeffs = [
            operation[0] for operation in self.operations
        ]
        self.eigenbasis = []
        self.operators = dict()
        for operations_operator in self.operations_operators:
            qudit_operators = [
                tensor_operation[0] for tensor_operation in operations_operator
            ]
            eigenbases = set(
                tuple(qudit_operator.eigenbasis)
                for qudit_operator in qudit_operators
            )
            if len(eigenbases) > 1:
                raise ValueError(
                    "All the eigenbases defined in QuditOperators should match."
                )
            eigenbasis = list(eigenbases)[0]
            if not self.eigenbasis:
                self.eigenbasis = eigenbasis
                self.sampled_seq = qudit_operators[0].sampled_seq
            elif eigenbasis != self.eigenbasis:
                raise ValueError(
                    "All the eigenbases defined in QuditOperators should match."
                )
            for qudit_operator in qudit_operators:
                for (
                    operator_label,
                    operator_value,
                ) in qudit_operator.operators.items():
                    if operator_label not in self.operators:
                        self.operators[operator_label] = operator_value
                    elif self.operators[operator_label] != operator_value:
                        raise ValueError(
                            f"Operator {operator_label} is defined differently in two QuditOperators."
                        )

    @classmethod
    def from_coeffs_operators(
        cls,
        qubit_ids: tuple[str],
        operations_coeffs: list[float],
        operations_operators: list[Sequence[tuple[QuditOperator, str]]],
    ) -> QuditOperator:
        return QuditOperator(
            qubit_ids, zip(operations_coeffs, operations_operators)
        )

    def build_operator(self):
        return sum(
            [
                self.operation[0]
                * build_operator(
                    self.sampled_seq,
                    self.qubit_ids,
                    [
                        (
                            qudit_operation[0].build_operator(),
                            qudit_operation[1],
                        )
                        for qudit_operation in operation[1]
                    ],
                    self.operators,
                )
                for operation in self.operations
            ]
        )

    def __repr__(self):
        return (
            f"Operations: {self.__str__()} \n"
            f"Eigenbasis: {self.eigenbasis} \n"
            + f"Operators: {self.operators}"
        )

    def __str__(self):
        return str(
            tuple(
                (
                    operation[0],
                    tuple(
                        (tensor_operation[0].operations, tensor_operation[1])
                        for tensor_operation in operation[1]
                    ),
                )
                for operation in self.operations
            )
        )
