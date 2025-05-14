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

from pulser.backend import Operator


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
