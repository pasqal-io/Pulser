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
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pytest
import qutip

from pulser.result import SampledResult
from pulser_simulation.qutip_result import QutipResult


def test_sampled_result(patch_plt_show):
    samples = Counter({"000": 50, "111": 50})
    result = SampledResult(
        atom_order=("a", "b", "c"),
        meas_basis="ground-rydberg",
        bitstring_counts=samples,
    )
    assert result.n_samples == 100
    assert result.sampling_dist == {"000": 0.5, "111": 0.5}
    sampling_err = np.sqrt(0.5**2 / 100)
    assert result.sampling_errors == {
        "000": sampling_err,
        "111": sampling_err,
    }
    n_samples = 100
    np.random.seed(3052023)
    new_samples = result.get_samples(100)
    new_samples.subtract(samples)
    assert all(
        abs(counts_diff) < sampling_err * n_samples
        for counts_diff in new_samples.values()
    )

    with pytest.raises(
        NotImplementedError,
        match=re.escape("`SampledResult.get_state()` is not implemented"),
    ):
        result.get_state()

    result.plot_histogram()


def test_qutip_result():
    qutrit_state = qutip.tensor(qutip.basis(3, 0), qutip.basis(3, 1))
    result = QutipResult(
        atom_order=("q0", "q1"),
        meas_basis="ground-rydberg",
        state=qutrit_state,
        matching_meas_basis=True,
    )
    assert result.sampling_dist == {"10": 1.0}
    assert result.sampling_errors == {"10": 0.0}
    assert result._basis_name == "all"

    assert result.get_state() == qutrit_state
    qubit_state = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 1))
    np.testing.assert_array_equal(
        result.get_state(reduce_to_basis="ground-rydberg").full(),
        qubit_state.full(),
    )

    result.meas_basis = "digital"
    assert result.sampling_dist == {"00": 1.0}

    result.meas_basis = "XY"
    with pytest.raises(RuntimeError, match="Unknown measurement basis 'XY'"):
        result.sampling_dist

    new_result = QutipResult(
        atom_order=("q0", "q1"),
        meas_basis="digital",
        state=qubit_state,
        matching_meas_basis=True,
    )
    assert new_result.sampling_dist == {"01": 1.0}

    new_result.meas_basis = "ground-rydberg"
    assert new_result.sampling_dist == {"10": 1.0}

    new_result.matching_meas_basis = False
    assert new_result.sampling_dist == {"00": 1.0}
    # The state basis should be infered to be "digital"
    with pytest.raises(
        TypeError,
        match="Can't reduce a system in digital to the ground-rydberg basis",
    ):
        new_result.get_state(reduce_to_basis="ground-rydberg")

    oversized_state = qutip.Qobj(np.eye(16) / 16)
    result.state = oversized_state
    assert result._dim == 4
    with pytest.raises(
        NotImplementedError,
        match="Cannot sample system with single-atom state vectors of"
        " dimension > 3",
    ):
        result.sampling_dist

    density_matrix = qutip.Qobj(np.eye(4) / 4)
    result = QutipResult(
        atom_order=("a", "b"),
        meas_basis="ground-rydberg",
        state=density_matrix,
        matching_meas_basis=True,
    )
    assert result.state.isoper
    assert result._dim == 2
    assert result.sampling_dist == {
        "00": 0.25,
        "01": 0.25,
        "10": 0.25,
        "11": 0.25,
    }
