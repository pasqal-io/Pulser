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
from dataclasses import dataclass
from numbers import Number
from typing import Collection, Sequence

import qutip

from pulser.channels.base_channel import States, check_eigenbasis
from pulser.operator import (
    Operator,
    OperatorString,
    QuditOperatorString,
    TargetedOperatorString,
)
from pulser.register.base_register import QubitId
from pulser.sampler.samples import SequenceSamples


def default_operators(
    eigenbasis: list[States],
) -> Mapping[str, qutip.Qobj]:
    r"""Default operators associated to an eigenbasis.

    The default operators are the projectors on states of the eigenbasis
    and the identity "I".
    The projectors are named "sigma_{a}{b}" with a, b elements of the
    eigenbasis such that ``sigma_{ab} = a * b^\dagger``.

    Args:
        eigenbasis: The computational basis in which the operators are defined.
            A list containing elements of States ranked in decreasing order
            of their associated eigenenergy (see Channel.eigenstates for a
            complete list and the ranking, can otherwise be accessed with
            SequenceSamples.eigenbasis).

    Returns:
        A dictionary composed of default operators as qutip.Qobj objects
        and their associated key.
    """
    check_eigenbasis(eigenbasis)
    dim = len(eigenbasis)
    basis = {b: qutip.basis(dim, i) for i, b in enumerate(eigenbasis)}
    operators = {"I": qutip.qeye(dim)}
    for proj0 in eigenbasis:
        for proj1 in eigenbasis:
            proj_name = "sigma_" + proj0 + proj1
            operators[proj_name] = basis[proj0] * basis[proj1].dag()
    return operators


def build_projector(
    eigenbasis: list[States],
    operation: str,
) -> qutip.Qobj:
    r"""Creates a projector on a state.

    Returns the projector on a state of the eigenbasis.
    The string that can be provided are either "I" (the identity) or
    "sigma_{a}{b}" with a, b elements of the eigenbasis such that
    ``sigma_{ab} = a * b^\dagger``.

    Examples:
        If we have the eigenbasis ["r", "g"] (associated with a Rydberg
        Channel), by default the available operators are "I" and "sigma_ab"
        with a, b in ["r", "g"]. The projector on the Rydberg ground state
        `sigma_gg` can be generated with::

            $ python build_1qubit_operator(["r", "g"], "sigma_gg")

    Args:
        eigenbasis: The computational basis in which the operators are defined.
            A list containing elements of States ranked in decreasing order
            of their associated eigenenergy (see Channel.eigenstates for a
            complete list and the ranking, can otherwise be accessed with
            SequenceSamples.eigenbasis).
        operation: A string: either "I" or "sigma_{a}{b}" with a, b in the
            computational basis.

    Returns:
        The desired operator as a qutip.Qobj object, identity if "I",
        ``sigma_{ab} = a * b^\dagger`` if "sigma_ab".
    """
    return build_operator(
        eigenbasis,
        [0],
        [(operation, [0])],
    )


def build_1qudit_operator(
    eigenbasis: list[States],
    operations: list[tuple[float, str | qutip.Qobj]],
    operators: Mapping[str, qutip.Qobj] | None = None,
) -> qutip.Qobj:
    r"""Creates a 1 qubit operator summing projectors and qutip objects.

    Takes as argument a list of operations to apply on one qubit. Returns
    the sum of these operations. The elements in operations can be a
    ``qutip.Qobj`` or a string key. If ``operators`` is undefined, this string
    can be "I" or "sigma_{a}{b}" with a, b elements of the eigenbasis (see
    Channel.eigenstates for possible states) and
    ``sigma_{ab} = a * b^\dagger``.

    Examples:
        If we have the eigenbasis ["r", "g"] (associated with a Rydberg
        Channel), by default the available operators are "I" and "sigma_ab"
        with a, b in ["r", "g"]. The operator ``sigma_gg - sigma_rr`` can be
        written as::

            ```python
                build_1qudit_operator(
                    ["r", "g"],
                    [(1.0, "sigma_gg"), (-1.0, "sigma_rr")],
                    with_leakage=True,
                )
            ```

        We can also define our custom set of operators. For instance, by
        setting `operators = {"X":qutip.sigmax(), "Z":qutip.sigmaz()}` you can
        make combinations of these operators such as X-Z::

        ```python
            build_1qudit_operator(
                ["r", "g"],
                [(1.0, "X"), (-1.0, "Z")],
                operators=operators,
            )
        ```

    Args:
        eigenbasis: The computational basis in which the operators are defined.
            A list containing elements of States ranked in decreasing order
            of their associated eigenenergy (see Channel.eigenstates for a
            complete list and the ranking, can otherwise be accessed with
            SequenceSamples.eigenbasis).
        operations: List of tuples `(operator, qubits)`.
        operators: A dict of operators and their labels. If None, it is
            composed of all the projectors on the computational basis
            (with error state if with_leakage is True) and "I".


    Returns:
        The final operator as a qutip.Qobj object.
    """
    if operators is None:
        operators = default_operators(eigenbasis)
    return sum(
        [
            operation[0]
            * build_operator(
                eigenbasis,
                [0],
                [(operation[1], [0])],
                operators,
            )
            for operation in operations
        ]
    )


def build_operator(
    eigenbasis: list[States],
    qubit_ids: list[QubitId],
    operations: list[tuple],
    operators: Mapping[str, qutip.Qobj] | None = None,
) -> qutip.Qobj:
    r"""Creates an operator with non-trivial actions on some qubits.

    Takes as argument a list of tuples ``[(operator_1, qubits_1),
    (operator_2, qubits_2), ...]``. Returns the tensor product of
    {``operator_i`` applied on ``qubits_i``} and Id on the rest.

    ``(operator, 'global')`` returns the sum for all ``j`` of operator
    applied at ``qubit_j`` and identity elsewhere.

    `operator` can be a ``qutip.Qobj`` or a string key of ``operators``. If
    ``operators`` is undefined, this string can be "I" or "sigma_{a}{b}" with
    a, b elements of the eigenbasis and ``sigma_{a}{b} = a * b^\dagger``.

    `qubits` is the list on which operator will be applied. The qubits are
    passed as their label in the register.

    Examples:
        If you want to generate an effective noise operator for your sampled
        Sequence, provide a dummy register::

            ```python
                build_operator(
                    sampled_seq.eigenbasis,
                    [0],
                    [("sigma_gg", [0])]
                )
                build_operator(
                    sampled_seq.eigenbasis,
                    [0],
                    [(qutip.sigmax(), [0])]
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
        eigenbasis: The computational basis in which the operators are defined.
            A list containing elements of States ranked in decreasing order
            of their associated eigenenergy (see Channel.eigenstates for a
            complete list and the ranking, can otherwise be accessed with
            SequenceSamples.eigenbasis).
        qubits: A list of labels that can be used in operations.
        operations: List of tuples `(operator, qubits)`.
        operators: A dict of operators and their labels. If None, it is
            composed of all the projectors on the computational basis
            and "I".


    Returns:
        The final operator as qutip.Qobj object.
    """
    check_eigenbasis(eigenbasis)
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
        operators = default_operators(eigenbasis)

    # Build operator
    op_list = [operators["I"] for j in range(len(qubit_ids))]
    _qid_index = {qid: i for i, qid in enumerate(qubit_ids)}

    for operator, qids in operations:
        if qids == "global":
            return sum(
                build_operator(
                    eigenbasis,
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


@dataclass
class QutipOperator(Operator):
    """A class to handle Operators as Qutip.QObj objects."""

    eigenbasis: list[States]

    def __post_init__(self) -> None:
        super().__post_init__()
        check_eigenbasis(self.eigenbasis)

    def to_qutip_operator(self) -> qutip.Qobj:
        """Outputs the qutip.QObj associated with the Operator."""
        return sum(
            [
                operation[0]
                * build_operator(
                    self.eigenbasis,
                    self.qubit_ids,
                    [
                        (
                            build_1qudit_operator(
                                self.eigenbasis,
                                qudit_operation[0].operations,
                                self.operators_dict,
                            ),
                            qudit_operation[1],
                        )
                        for qudit_operation in operation[1]
                    ],
                    self.operators_dict,
                )
                for operation in self.operator_string.operations
            ]
        )

    @classmethod
    def from_qutip_Qobj(
        cls,
        qutip_obj: qutip.Qobj,
        qubit_ids: Collection[QubitId],
        eigenbasis: list[States],
    ) -> QutipOperator:
        """Builds a QutipOperator from a qutip.QObj."""
        if not (
            qutip_obj.shape[0] == len(eigenbasis) ** (len(qubit_ids))
            and qutip_obj.shape[0] == qutip_obj.shape[1]
        ):
            raise ValueError(
                "For the given eigenbasis and qubit_ids, qutip_obj should"
                f"be of shape ({len(eigenbasis)}**{(len(qubit_ids))},"
                f"({len(eigenbasis)}**{(len(qubit_ids))})"
            )
        operators_dict = default_operators(eigenbasis)
        op_basis = list(operators_dict.keys())
        # 1-qudit basis is without identity
        identity_op = op_basis.pop(0)
        assert identity_op == "I"
        # Compute coefficients in front of n-qudit basis
        operators = []
        coefficients = []
        qutip_obj_dag = qutip_obj.dag()
        for op_labels in itertools.product(op_basis for _ in range(qubit_ids)):
            operators.append(
                TargetedOperatorString(
                    tuple(
                        (
                            QuditOperatorString([1.0], [op_labels[i]]),
                            [qubit_id],
                        )
                        for i, qubit_id in enumerate(qubit_ids)
                    )
                )
            )
            coefficients.append(
                (
                    qutip.tensor(
                        operators_dict[op_label]
                        for op_label in range(op_labels)
                    )
                    * qutip_obj_dag()
                ).tr()
            )
        return QutipOperator(
            OperatorString(coefficients, operators),
            qubit_ids,
            operators_dict,
            eigenbasis,
        )
