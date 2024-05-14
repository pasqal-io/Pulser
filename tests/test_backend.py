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
import typing
from dataclasses import replace

import pytest

import pulser
from pulser.backend.abc import Backend
from pulser.backend.config import EmulatorConfig
from pulser.backend.qpu import QPUBackend
from pulser.backend.remote import (
    RemoteConnection,
    RemoteResults,
    RemoteResultsError,
    SubmissionStatus,
)
from pulser.devices import DigitalAnalogDevice, MockDevice
from pulser.register import SquareLatticeLayout
from pulser.result import Result, SampledResult


@pytest.fixture
def sequence() -> pulser.Sequence:
    reg = SquareLatticeLayout(5, 5, 5).square_register(2)
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    seq.add(pulser.Pulse.ConstantPulse(1000, 1, -1, 0), "rydberg_global")
    return seq


def test_abc_backend(sequence):
    with pytest.raises(
        TypeError, match="Can't instantiate abstract class Backend"
    ):
        Backend(sequence)

    class ConcreteBackend(Backend):
        def run(self):
            pass

    with pytest.raises(
        TypeError, match="'sequence' should be a `Sequence` instance"
    ):
        ConcreteBackend(sequence.to_abstract_repr())


@pytest.mark.parametrize(
    "param, value, msg",
    [
        ("sampling_rate", 0, "must be greater than 0"),
        ("evaluation_times", "full", "one of the following"),
        ("evaluation_times", 1.001, "less than or equal to 1"),
        ("evaluation_times", [-1e9, 1], "must not contain negative values"),
        ("initial_state", "all_ground", "must be 'all-ground'"),
    ],
)
def test_emulator_config_value_errors(param, value, msg):
    with pytest.raises(ValueError, match=msg):
        EmulatorConfig(**{param: value})


@pytest.mark.parametrize(
    "param, msg",
    [
        ("evaluation_times", "not a valid type for 'evaluation_times'"),
        ("initial_state", "not a valid type for 'initial_state'"),
        ("noise_model", "must be a NoiseModel instance"),
    ],
)
def test_emulator_config_type_errors(param, msg):
    with pytest.raises(TypeError, match=msg):
        EmulatorConfig(**{param: None})


class _MockConnection(RemoteConnection):
    def __init__(self):
        self._status_calls = 0

    def submit(self, sequence, wait: bool = False, **kwargs) -> RemoteResults:
        return RemoteResults("abcd", self)

    def _fetch_result(self, submission_id: str) -> typing.Sequence[Result]:
        return (
            SampledResult(
                ("q0", "q1"),
                meas_basis="ground-rydberg",
                bitstring_counts={"00": 100},
            ),
        )

    def _get_submission_status(self, submission_id: str) -> SubmissionStatus:
        self._status_calls += 1
        if self._status_calls == 1:
            return SubmissionStatus.RUNNING
        return SubmissionStatus.DONE


def test_qpu_backend(sequence):
    connection = _MockConnection()

    with pytest.raises(
        TypeError, match="must be a real device, instance of 'Device'"
    ):
        QPUBackend(sequence, connection)
    with pytest.warns(DeprecationWarning, match="From v0.18"):
        seq = sequence.switch_device(replace(DigitalAnalogDevice, max_runs=10))
    qpu_backend = QPUBackend(seq, connection)
    with pytest.raises(ValueError, match="'job_params' must be specified"):
        qpu_backend.run()
    with pytest.raises(
        ValueError,
        match="All elements of 'job_params' must specify 'runs'",
    ):
        qpu_backend.run(job_params=[{"n_runs": 10}, {"runs": 11}])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "All 'runs' must be below the maximum allowed by the device (10)"
        ),
    ):
        qpu_backend.run(job_params=[{"runs": 11}])

    remote_results = qpu_backend.run(job_params=[{"runs": 10}])

    with pytest.raises(AttributeError, match="no attribute 'result'"):
        # Cover the custom '__getattr__' default behavior
        remote_results.result

    with pytest.raises(
        RemoteResultsError,
        match="The results are not available. The submission's status is"
        " SubmissionStatus.RUNNING",
    ):
        remote_results.results

    results = remote_results.results
    assert results[0].sampling_dist == {"00": 1.0}
