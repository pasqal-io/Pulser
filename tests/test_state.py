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
import math
import re

import pytest

from pulser.backend import State


@pytest.mark.parametrize(
    "amplitudes",
    [
        {"rrh": 1.0},
        {"rr": 0.5, "rgg": math.sqrt(0.75)},
    ],
)
def test_from_state_amplitudes_error(amplitudes):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All basis states must be combinations of eigenstates with "
            f"the same length. Expected combinations of ('r', 'g'), each "
            f"with {len(list(amplitudes)[0])} elements."
        ),
    ):
        State._validate_amplitudes(
            eigenstates=("r", "g"), amplitudes=amplitudes
        )


def test_from_state_amplitudes_valid():
    n_qudits = State._validate_amplitudes(
        eigenstates=("r", "g", "x"),
        amplitudes={"rrgg": 0.5, "rggr": math.sqrt(0.75)},
    )

    assert n_qudits == 4


def test_validate_eigenstates():
    with pytest.raises(
        ValueError,
        match="eigenstates must be represented by single characters",
    ):
        State._validate_eigenstates(eigenstates=["ground", "rydberg"])

    with pytest.raises(
        ValueError, match=re.escape("can't contain repeated entries")
    ):
        State._validate_eigenstates(eigenstates=["r", "g", "r"])

    with pytest.raises(
        TypeError,
        match=re.escape("must be a 'collections.Sequence'"),
    ):
        State._validate_eigenstates(eigenstates={"r", "g"})

    State._validate_eigenstates(eigenstates=("r", "g"))
