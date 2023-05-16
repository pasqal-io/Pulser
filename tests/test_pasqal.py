# Copyright 2022 Pulser Development Team
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

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pasqal_cloud.device.configuration import EmuFreeConfig, EmuTNConfig

import pulser
import pulser_pasqal
from pulser.backend.config import EmulatorConfig
from pulser.backend.remote import (
    RemoteConnection,
    RemoteResults,
    SubmissionStatus,
)
from pulser.devices import Chadoq2
from pulser.register import Register
from pulser.result import SampledResult
from pulser.sequence import Sequence
from pulser_pasqal import BaseConfig, EmulatorType, Endpoints, PasqalCloud
from pulser_pasqal.backends import EmuFreeBackend, EmuTNBackend
from pulser_pasqal.job_parameters import JobParameters, JobVariables

root = Path(__file__).parent.parent


def test_version():
    assert pulser_pasqal.__version__ == pulser.__version__


@dataclasses.dataclass
class CloudFixture:
    pasqal_cloud: PasqalCloud
    mock_cloud_sdk: Any


test_device = Chadoq2
virtual_device = test_device.to_virtual()


@pytest.fixture
def seq():
    reg = Register.square(2, spacing=10, prefix="q")
    return Sequence(reg, test_device)


@pytest.fixture
def mock_job():
    @dataclasses.dataclass
    class MockJob:
        variables = {"t": 100, "qubits": {"q0": 1, "q1": 2, "q2": 4, "q3": 3}}
        result = {"00": 5, "11": 5}

    return MockJob()


@pytest.fixture
def mock_batch(mock_job, seq):
    with pytest.warns(UserWarning):
        seq_ = seq.build()
        seq_.declare_channel("rydberg_global", "rydberg_global")
        seq_.measure()

    @dataclasses.dataclass
    class MockBatch:
        id = "abcd"
        status = "DONE"
        jobs = {"job1": mock_job}
        sequence_builder = seq_.to_abstract_repr()

    return MockBatch()


@pytest.fixture
def fixt(monkeypatch, mock_batch):
    with patch("pasqal_cloud.SDK", autospec=True) as mock_cloud_sdk_class:
        pasqal_cloud_kwargs = dict(
            username="abc",
            password="def",
            group_id="ghi",
            endpoints=Endpoints(core="core_url"),
            webhook="xyz",
        )

        pasqal_cloud = PasqalCloud(**pasqal_cloud_kwargs)

        with pytest.raises(NotImplementedError):
            pasqal_cloud.fetch_available_devices()

        monkeypatch.setattr(
            PasqalCloud,
            "fetch_available_devices",
            lambda _: {test_device.name: test_device},
        )

        mock_cloud_sdk_class.assert_called_once_with(**pasqal_cloud_kwargs)

        mock_cloud_sdk = mock_cloud_sdk_class.return_value

        mock_cloud_sdk_class.reset_mock()

        mock_cloud_sdk.create_batch = MagicMock(return_value=mock_batch)
        mock_cloud_sdk.get_batch = MagicMock(return_value=mock_batch)

        yield CloudFixture(
            pasqal_cloud=pasqal_cloud, mock_cloud_sdk=mock_cloud_sdk
        )

        mock_cloud_sdk_class.assert_not_called()


@pytest.mark.parametrize(
    "emulator", [None, EmulatorType.EMU_TN, EmulatorType.EMU_FREE]
)
@pytest.mark.parametrize("parametrized", [True, False])
def test_submit(fixt, parametrized, emulator, seq, mock_job):
    with pytest.raises(
        ValueError,
        match="The measurement basis can't be implicitly determined for a "
        "sequence not addressing a single basis",
    ):
        fixt.pasqal_cloud.submit(seq)

    seq.declare_channel("rydberg_global", "rydberg_global")
    t = seq.declare_variable("t", dtype=int)
    seq.delay(t if parametrized else 100, "rydberg_global")
    assert seq.is_parametrized() == parametrized

    if not emulator:
        with pytest.raises(ValueError, match="'job_params' must be specified"):
            fixt.pasqal_cloud.submit(seq)
        with pytest.raises(
            ValueError,
            match="All elements of 'job_params' must specify 'runs'",
        ):
            fixt.pasqal_cloud.submit(
                seq, job_params=[{"n_runs": 10}, {"runs": 1}]
            )

        seq2 = seq.switch_device(virtual_device)
        with pytest.raises(
            ValueError,
            match="The device used in the sequence does not match any "
            "of the devices currently avaialble through the remote "
            "connection.",
        ):
            fixt.pasqal_cloud.submit(seq2, job_params=[dict(runs=10)])

    if parametrized:
        with pytest.raises(
            TypeError, match="Did not receive values for variables"
        ):
            fixt.pasqal_cloud.submit(seq, job_params=[{"runs": 100}])

    assert not seq.is_measured()
    config = EmulatorConfig(
        sampling_rate=0.5, backend_options=dict(with_noise=False)
    )

    if emulator is None:
        sdk_config = None
    elif emulator == EmulatorType.EMU_FREE:
        sdk_config = EmuFreeConfig(with_noise=False)
    else:
        sdk_config = EmuTNConfig(dt=2, extra_config={"with_noise": False})

    assert (
        fixt.pasqal_cloud._convert_configuration(config, emulator)
        == sdk_config
    )

    job_params = [{"runs": 10, "variables": {"t": 100}}]
    remote_results = fixt.pasqal_cloud.submit(
        seq,
        job_params=job_params,
        emulator=emulator,
        config=config,
    )
    assert not seq.is_measured()
    seq.measure(basis="ground-rydberg")

    fixt.mock_cloud_sdk.create_batch.assert_called_once_with(
        **dict(
            serialized_sequence=seq.to_abstract_repr(),
            jobs=job_params,
            emulator=emulator,
            configuration=sdk_config,
            wait=False,
            fetch_results=False,
        )
    )

    assert isinstance(remote_results, RemoteResults)
    assert remote_results.get_status() == SubmissionStatus.DONE
    fixt.mock_cloud_sdk.get_batch.assert_called_once_with(
        id=remote_results._submission_id, fetch_results=False
    )

    fixt.mock_cloud_sdk.get_batch.reset_mock()
    results = remote_results.results
    fixt.mock_cloud_sdk.get_batch.assert_called_with(
        id=remote_results._submission_id, fetch_results=True
    )
    assert results == (
        SampledResult(
            atom_order=("q0", "q1", "q2", "q3"),
            meas_basis="ground-rydberg",
            bitstring_counts=mock_job.result,
        ),
    )
    assert hasattr(remote_results, "_results")


@pytest.mark.parametrize("emu_cls", [EmuTNBackend, EmuFreeBackend])
def test_emulators_init(fixt, seq, emu_cls, monkeypatch):
    with pytest.raises(
        TypeError,
        match="'connection' must be a valid RemoteConnection instance.",
    ):
        emu_cls(seq, "connection")
    with pytest.raises(
        TypeError, match="'config' must be of type 'EmulatorConfig'"
    ):
        emu_cls(seq, fixt.pasqal_cloud, {"with_noise": True})

    with pytest.raises(
        NotImplementedError,
        match="'EmulatorConfig.with_modulation' is not configurable in this "
        "backend. It should not be changed from its default value of 'False'.",
    ):
        emu_cls(
            seq,
            fixt.pasqal_cloud,
            EmulatorConfig(
                sampling_rate=0.25,
                evaluation_times="Final",
                with_modulation=True,
            ),
        )

    monkeypatch.setattr(RemoteConnection, "__abstractmethods__", set())
    with pytest.raises(
        TypeError,
        match="connection to the remote backend must be done"
        " through a 'PasqalCloud' instance.",
    ):
        emu_cls(seq, RemoteConnection())


@pytest.mark.parametrize("parametrized", [True, False])
@pytest.mark.parametrize("emu_cls", [EmuTNBackend, EmuFreeBackend])
def test_emulators_run(fixt, seq, emu_cls, parametrized: bool):
    seq.declare_channel("rydberg_global", "rydberg_global")
    t = seq.declare_variable("t", dtype=int)
    seq.delay(t if parametrized else 100, "rydberg_global")
    assert seq.is_parametrized() == parametrized
    seq.measure(basis="ground-rydberg")

    emu = emu_cls(seq, fixt.pasqal_cloud)

    bad_kwargs = {} if parametrized else {"job_params": [{"runs": 100}]}
    err_msg = (
        "'job_params' must be provided"
        if parametrized
        else "'job_params' cannot be provided"
    )
    with pytest.raises(ValueError, match=err_msg):
        emu.run(**bad_kwargs)

    good_kwargs = (
        {"job_params": [{"variables": {"t": 100}}]} if parametrized else {}
    )
    remote_results = emu.run(**good_kwargs)
    assert isinstance(remote_results, RemoteResults)

    sdk_config: EmuTNConfig | EmuFreeConfig
    if isinstance(emu, EmuTNBackend):
        emulator_type = EmulatorType.EMU_TN
        sdk_config = EmuTNConfig()
    else:
        emulator_type = EmulatorType.EMU_FREE
        sdk_config = EmuFreeConfig()
    fixt.mock_cloud_sdk.create_batch.assert_called_once()
    fixt.mock_cloud_sdk.create_batch.assert_called_once_with(
        serialized_sequence=seq.to_abstract_repr(),
        jobs=good_kwargs.get("job_params", []),
        emulator=emulator_type,
        configuration=sdk_config,
        wait=False,
        fetch_results=False,
    )


# Deprecated


def check_pasqal_cloud(fixt, seq, emulator, expected_seq_representation):
    create_batch_kwargs = dict(
        jobs=[JobParameters(runs=10, variables=JobVariables(a=[3, 5]))],
        emulator=emulator,
        configuration=BaseConfig(
            extra_config={
                "dt": 10.0,
                "precision": "normal",
            }
        ),
        wait=True,
        fetch_results=False,
    )

    expected_create_batch_kwargs = {
        **create_batch_kwargs,
        "jobs": [{"runs": 10, "variables": {"qubits": None, "a": [3, 5]}}],
    }

    with pytest.warns(UserWarning, match="No declared variables named: a"):
        fixt.pasqal_cloud.create_batch(
            seq,
            **create_batch_kwargs,
        )
        assert pulser_pasqal.__version__ < "0.14"

    fixt.mock_cloud_sdk.create_batch.assert_called_once_with(
        serialized_sequence=expected_seq_representation,
        **expected_create_batch_kwargs,
    )

    get_batch_kwargs = dict(
        id="uuid",
        fetch_results=True,
    )
    with pytest.deprecated_call():
        fixt.pasqal_cloud.get_batch(**get_batch_kwargs)
        assert pulser_pasqal.__version__ < "0.14"

    fixt.mock_cloud_sdk.get_batch.assert_called_once_with(**get_batch_kwargs)


@pytest.mark.parametrize(
    "emulator, device",
    [
        [emulator, device]
        for emulator in (EmulatorType.EMU_FREE, EmulatorType.EMU_TN)
        for device in (test_device, virtual_device)
    ],
)
def test_pasqal_cloud_emu(fixt, emulator, device):
    reg = Register.from_coordinates(
        [(0, 0), (0, 10)], center=False, prefix="q"
    )
    seq = Sequence(reg, device)

    check_pasqal_cloud(
        fixt=fixt,
        seq=seq,
        emulator=emulator,
        expected_seq_representation=seq.to_abstract_repr(),
    )


def test_pasqal_cloud_qpu(fixt):
    device = test_device

    reg = Register.from_coordinates([(0, 0), (0, 10)], prefix="q")
    seq = Sequence(reg, device)

    check_pasqal_cloud(
        fixt=fixt,
        seq=seq,
        emulator=None,
        expected_seq_representation=seq.to_abstract_repr(),
    )


def test_virtual_device_on_qpu_error(fixt):
    reg = Register.from_coordinates([(0, 0), (0, 10)], prefix="q")
    device = Chadoq2.to_virtual()
    seq = Sequence(reg, device)

    with pytest.deprecated_call(), pytest.raises(
        TypeError, match="must be a real device"
    ):
        fixt.pasqal_cloud.create_batch(
            seq,
            jobs=[JobParameters(runs=10, variables=JobVariables(a=[3, 5]))],
            emulator=None,
            wait=True,
        )


def test_wrong_parameters(fixt):
    reg = Register.from_coordinates([(0, 0), (0, 10)], prefix="q")
    seq = Sequence(reg, test_device)
    seq.declare_variable("unset", dtype=int)

    with pytest.warns(
        UserWarning, match="No declared variables named: a"
    ), pytest.raises(TypeError, match="Did not receive values for variables"):
        fixt.pasqal_cloud.create_batch(
            seq,
            jobs=[JobParameters(runs=10, variables=JobVariables(a=[3, 5]))],
            emulator=None,
            wait=True,
        )
