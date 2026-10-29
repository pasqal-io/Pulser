# Copyright 2025 Pulser Development Team
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
import re

import pytest

from pulser.backend import Operator, OperatorRepr


def test_validate_operations_nonexistent_qubits():
    with pytest.raises(
        ValueError, match="Got invalid indices for a system with 2 qudits"
    ):
        Operator._validate_operations(
            eigenstates=("r", "g"),
            n_qudits=2,
            operations=[(1.0, [({"gg": 1.0, "rr": -1.0}, {3, 5, 9})])],
        )


def test_validate_operations_reoccurring_qubit():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Got invalid indices for a system with 5 qudits: {3}."
            " For TensorOp #0, only indices {0, 1, 4} were still available."
        ),
    ):
        Operator._validate_operations(
            eigenstates=("r", "g"),
            n_qudits=5,
            operations=[
                (
                    1.0,
                    [
                        ({"gg": 1.0, "rr": -1.0}, {2, 3}),
                        ({"gg": 1.0, "rr": -1.0}, {3}),
                    ],
                )
            ],
        )


def test_validate_operations_valid():
    Operator._validate_operations(
        eigenstates=("r", "g"),
        n_qudits=5,
        operations=[
            (
                1.0,
                [
                    ({"gg": 1.0, "rr": -1.0}, {3}),
                    ({"gg": 1.0, "rr": -1.0}, {1, 2}),
                ],
            )
        ],
    )


def test_operator_wrong_eigenstate_count():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Every QuditOp key must be made up of two eigenstates"
            " among ('r', 'g'); instead, "
            "got 'gggg'"
        ),
    ):
        Operator._validate_operations(
            eigenstates=("r", "g"),
            n_qudits=2,
            operations=[(1.0, [({"gggg": 1.0, "rr": -1.0}, {0})])],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Every QuditOp key must be made up of two eigenstates"
            " among ('r', 'g', 'x'); instead, "
            "got 'gggg'"
        ),
    ):
        Operator._validate_operations(
            eigenstates=("r", "g", "x"),
            n_qudits=2,
            operations=[(1.0, [({"gggg": 1.0, "rr": -1.0}, {0})])],
        )


def test_operator_nonexistent_eigenstates():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Every QuditOp key must be made up of two eigenstates"
            " among ('r', 'g'); instead, "
            "got 'hh'"
        ),
    ):
        Operator._validate_operations(
            eigenstates=("r", "g"),
            n_qudits=2,
            operations=[(1.0, [({"hh": 1.0, "rr": -1.0}, {0})])],
        )


@pytest.mark.parametrize(
    "eigenstates, paulis, expected_operations",
    [
        (
            ("r", "g"),
            {1: "Z", 3: "x", 0: "Z"},
            [
                (
                    1.0,
                    [
                        ({"rr": 1.0, "gg": -1.0}, {0, 1}),
                        ({"rg": 1.0, "gr": 1.0}, {3}),
                    ],
                )
            ],
        ),
        (
            ("0", "1"),
            {2: "y"},
            [(1.0, [({"01": -1.0j, "10": 1.0j}, {2})])],
        ),
        # The leakage state is ignored when defining the Pauli matrices
        (
            ("r", "g", "x"),
            {0: "Y", 1: "Z"},
            [
                (
                    1.0,
                    [
                        ({"rg": -1.0j, "gr": 1.0j}, {0}),
                        ({"rr": 1.0, "gg": -1.0}, {1}),
                    ],
                )
            ],
        ),
    ],
)
def test_from_pauli_string(eigenstates, paulis, expected_operations):
    op = OperatorRepr.from_pauli_string(
        eigenstates=eigenstates, n_qudits=4, paulis=paulis
    )
    assert op._to_abstract_repr() == {
        "eigenstates": eigenstates,
        "n_qudits": 4,
        "operations": expected_operations,
    }


@pytest.mark.parametrize(
    "eigenstates, n_qudits, paulis, error_type, msg",
    [
        (
            ("u", "d", "r"),
            2,
            {0: "Z"},
            ValueError,
            "Pauli strings are only defined for qubits, i.e. with exactly "
            "two eigenstates (besides the leakage state 'x'); got "
            "eigenstates ('u', 'd', 'r').",
        ),
        (("r", "g"), 2, [(0, "Z")], TypeError, "'paulis' must be a mapping"),
        (
            ("r", "g"),
            2,
            {},
            ValueError,
            "'paulis' must contain at least one entry.",
        ),
        (
            ("r", "g"),
            2,
            {"0": "Z"},
            TypeError,
            "The qudit indices in 'paulis' must be integers; got '0'",
        ),
        (
            ("r", "g"),
            2,
            {0.99: "Z"},
            TypeError,
            "The qudit indices in 'paulis' must be integers; got 0.99",
        ),
        (
            ("r", "g"),
            2,
            {0: "Z", 1: "W"},
            ValueError,
            "The Pauli matrices in 'paulis' must be one of ('X', 'Y', 'Z'); "
            "got 'W' for qudit 1.",
        ),
        (
            ("r", "g"),
            2,
            {0: 1},
            ValueError,
            "The Pauli matrices in 'paulis' must be one of",
        ),
        (
            ("r", "g"),
            2,
            {0: "Z", 2: "X"},
            ValueError,
            "Got invalid indices for a system with 2 qudits: {2}.",
        ),
        (
            ("r", "g"),
            2,
            {-1: "Z"},
            ValueError,
            "Got invalid indices for a system with 2 qudits: {-1}.",
        ),
    ],
)
def test_from_pauli_string_errors(
    eigenstates, n_qudits, paulis, error_type, msg
):
    with pytest.raises(error_type, match=re.escape(msg)):
        OperatorRepr.from_pauli_string(
            eigenstates=eigenstates, n_qudits=n_qudits, paulis=paulis
        )
