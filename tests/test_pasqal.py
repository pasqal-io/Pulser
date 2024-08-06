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

import copy
import dataclasses
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
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
from pulser.devices import DigitalAnalogDevice
from pulser.register.special_layouts import SquareLatticeLayout
from pulser.result import SampledResult
from pulser.sequence import Sequence
from pulser_pasqal import EmulatorType, Endpoints, PasqalCloud
from pulser_pasqal.backends import EmuFreeBackend, EmuTNBackend

root = Path(__file__).parent.parent


def test_version():
    assert pulser_pasqal.__version__ == pulser.__version__


@dataclasses.dataclass
class CloudFixture:
    pasqal_cloud: PasqalCloud
    mock_cloud_sdk: Any


test_device = dataclasses.replace(
    DigitalAnalogDevice,
    dmm_objects=(
        dataclasses.replace(
            DigitalAnalogDevice.dmm_objects[0], total_bottom_detuning=-1000
        ),
    ),
)
virtual_device = dataclasses.replace(
    test_device.to_virtual(), name="test-virtual"
)


@pytest.fixture
def seq():
    reg = SquareLatticeLayout(5, 5, 5).make_mappable_register(10)
    return Sequence(reg, test_device)


class _MockJob:
    def __init__(
        self,
        runs=10,
        variables={"t": 100, "qubits": {"q0": 1, "q1": 2, "q2": 4, "q3": 3}},
        result={"00": 5, "11": 5},
    ) -> None:
        self.runs = runs
        self.variables = variables
        self.result = result
        self.id = str(np.random.randint(10000))


@pytest.fixture
def mock_job():
    return _MockJob()


@pytest.fixture
def mock_batch(mock_job, seq):
    seq_ = copy.deepcopy(seq)
    seq_.declare_channel("rydberg_global", "rydberg_global")
    seq_.measure()

    @dataclasses.dataclass
    class MockBatch:
        id = "abcd"
        status = "DONE"
        ordered_jobs = [
            mock_job,
            _MockJob(result={"00": 10}),
            _MockJob(result={"11": 10}),
        ]
        sequence_builder = seq_.to_abstract_repr()

    return MockBatch()


@pytest.fixture
def fixt(mock_batch):
    with patch("pasqal_cloud.SDK", autospec=True) as mock_cloud_sdk_class:
        pasqal_cloud_kwargs = dict(
            username="abc",
            password="def",
            project_id="ghi",
            endpoints=Endpoints(core="core_url"),
            webhook="xyz",
        )

        pasqal_cloud = PasqalCloud(**pasqal_cloud_kwargs)

        mock_cloud_sdk_class.assert_called_once_with(**pasqal_cloud_kwargs)

        mock_cloud_sdk = mock_cloud_sdk_class.return_value

        mock_cloud_sdk_class.reset_mock()

        mock_cloud_sdk.create_batch = MagicMock(return_value=mock_batch)
        mock_cloud_sdk.get_batch = MagicMock(return_value=mock_batch)
        mock_cloud_sdk.get_device_specs_dict = MagicMock(
            return_value={test_device.name: test_device.to_abstract_repr()}
        )

        yield CloudFixture(
            pasqal_cloud=pasqal_cloud, mock_cloud_sdk=mock_cloud_sdk
        )

        mock_cloud_sdk_class.assert_not_called()


@pytest.mark.parametrize("with_job_id", [False, True])
def test_remote_results(fixt, mock_batch, with_job_id):
    with pytest.raises(
        RuntimeError, match=re.escape("does not contain jobs ['badjobid']")
    ):
        RemoteResults(mock_batch.id, fixt.pasqal_cloud, job_ids=["badjobid"])
    fixt.mock_cloud_sdk.get_batch.reset_mock()

    select_jobs = (
        mock_batch.ordered_jobs[::-1][:2]
        if with_job_id
        else mock_batch.ordered_jobs
    )
    select_job_ids = [j.id for j in select_jobs]

    remote_results = RemoteResults(
        mock_batch.id,
        fixt.pasqal_cloud,
        job_ids=select_job_ids if with_job_id else None,
    )

    assert remote_results.submission_id == mock_batch.id
    assert remote_results.job_ids == select_job_ids
    fixt.mock_cloud_sdk.get_batch.assert_called_once_with(
        id=remote_results.submission_id
    )
    fixt.mock_cloud_sdk.get_batch.reset_mock()

    assert remote_results.get_status() == SubmissionStatus.DONE
    fixt.mock_cloud_sdk.get_batch.assert_called_once_with(
        id=remote_results.submission_id
    )

    fixt.mock_cloud_sdk.get_batch.reset_mock()
    results = remote_results.results
    fixt.mock_cloud_sdk.get_batch.assert_called_with(
        id=remote_results.submission_id
    )
    assert results == tuple(
        SampledResult(
            atom_order=("q0", "q1", "q2", "q3"),
            meas_basis="ground-rydberg",
            bitstring_counts=job.result,
        )
        for job in select_jobs
    )

    assert hasattr(remote_results, "_results")


@pytest.mark.parametrize("mimic_qpu", [False, True])
@pytest.mark.parametrize(
    "emulator", [None, EmulatorType.EMU_TN, EmulatorType.EMU_FREE]
)
@pytest.mark.parametrize("parametrized", [True, False])
def test_submit(
    fixt, parametrized, emulator, mimic_qpu, seq, mock_batch, mock_job
):
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

    if not emulator or mimic_qpu:
        seq2 = seq.switch_device(virtual_device)
        with pytest.raises(
            ValueError,
            match="The device used in the sequence does not match any "
            "of the devices currently available through the remote "
            "connection.",
        ):
            fixt.pasqal_cloud.submit(
                seq2, job_params=[dict(runs=10)], mimic_qpu=mimic_qpu
            )
        mod_test_device = dataclasses.replace(test_device, max_atom_num=1000)
        seq3 = seq.switch_device(mod_test_device).switch_register(
            pulser.Register.square(11, spacing=5)
        )
        with pytest.raises(
            ValueError,
            match="sequence is not compatible with the latest device specs",
        ):
            fixt.pasqal_cloud.submit(
                seq3, job_params=[dict(runs=10)], mimic_qpu=mimic_qpu
            )
        seq4 = seq3.switch_register(pulser.Register.square(4, spacing=5))
        # The sequence goes through QPUBackend.validate_sequence()
        with pytest.raises(
            ValueError, match="defined from a `RegisterLayout`"
        ):
            fixt.pasqal_cloud.submit(
                seq4, job_params=[dict(runs=10)], mimic_qpu=mimic_qpu
            )

        # And it goes through QPUBackend.validate_job_params()
        with pytest.raises(
            ValueError,
            match="must specify 'runs'",
        ):
            fixt.pasqal_cloud.submit(seq, job_params=[{}], mimic_qpu=mimic_qpu)

    if parametrized:
        with pytest.raises(
            TypeError, match="Did not receive values for variables"
        ):
            fixt.pasqal_cloud.submit(
                seq.build(qubits={"q0": 1, "q1": 2, "q2": 4, "q3": 3}),
                job_params=[{"runs": 10}],
                mimic_qpu=mimic_qpu,
            )

    assert not seq.is_measured()
    config = EmulatorConfig(
        sampling_rate=0.5, backend_options=dict(with_noise=False)
    )

    if emulator is None:
        sdk_config = None
    elif emulator == EmulatorType.EMU_FREE:
        sdk_config = EmuFreeConfig(
            with_noise=False, strict_validation=mimic_qpu
        )
    else:
        sdk_config = EmuTNConfig(
            dt=2,
            extra_config={"with_noise": False},
            strict_validation=mimic_qpu,
        )

    assert (
        fixt.pasqal_cloud._convert_configuration(
            config, emulator, strict_validation=mimic_qpu
        )
        == sdk_config
    )

    job_params = [
        {
            "runs": 10,
            "variables": {
                "t": np.array(100),  # Check that numpy array is converted
                "qubits": {"q0": 1, "q1": 2, "q2": 4, "q3": 3},
            },
        }
    ]

    remote_results = fixt.pasqal_cloud.submit(
        seq,
        job_params=job_params,
        emulator=emulator,
        config=config,
        mimic_qpu=mimic_qpu,
    )
    assert remote_results.submission_id == mock_batch.id

    assert not seq.is_measured()
    seq.measure(basis="ground-rydberg")

    fixt.mock_cloud_sdk.create_batch.assert_called_once_with(
        **dict(
            serialized_sequence=seq.to_abstract_repr(),
            jobs=job_params,
            emulator=emulator,
            configuration=sdk_config,
            wait=False,
        )
    )

    job_params[0]["runs"] = {10}
    with pytest.raises(
        TypeError, match="Object of type set is not JSON serializable"
    ):
        # Check that the decoder still fails on unsupported types
        fixt.pasqal_cloud.submit(
            seq,
            job_params=job_params,
            emulator=emulator,
            config=config,
        )

    assert isinstance(remote_results, RemoteResults)


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

    # With mimic_qpu=True
    with pytest.raises(TypeError, match="must be a real device"):
        emu_cls(
            seq.switch_device(virtual_device),
            fixt.pasqal_cloud,
            mimic_qpu=True,
        )


@pytest.mark.parametrize("mimic_qpu", [True, False])
@pytest.mark.parametrize("parametrized", [True, False])
@pytest.mark.parametrize("emu_cls", [EmuTNBackend, EmuFreeBackend])
def test_emulators_run(fixt, seq, emu_cls, parametrized: bool, mimic_qpu):
    seq.declare_channel("rydberg_global", "rydberg_global")
    t = seq.declare_variable("t", dtype=int)
    seq.delay(t if parametrized else 100, "rydberg_global")
    assert seq.is_parametrized() == parametrized
    seq.measure(basis="ground-rydberg")

    emu = emu_cls(seq, fixt.pasqal_cloud, mimic_qpu=mimic_qpu)

    with pytest.raises(ValueError, match="'job_params' must be specified"):
        emu.run()

    with pytest.raises(TypeError, match="'job_params' must be a list"):
        emu.run(job_params={"runs": 100})
    with pytest.raises(
        TypeError, match="All elements of 'job_params' must be dictionaries"
    ):
        emu.run(job_params=[{"runs": 100}, "foo"])

    with pytest.raises(ValueError, match="must specify 'runs'"):
        emu.run(job_params=[{}])

    good_kwargs = {
        "job_params": [
            {
                "runs": 10,
                "variables": {
                    "t": 100,
                    "qubits": {"q0": 1, "q1": 2, "q2": 4, "q3": 3},
                },
            }
        ]
    }
    remote_results = emu.run(**good_kwargs)
    assert isinstance(remote_results, RemoteResults)

    sdk_config: EmuTNConfig | EmuFreeConfig
    if isinstance(emu, EmuTNBackend):
        emulator_type = EmulatorType.EMU_TN
        sdk_config = EmuTNConfig(dt=10, strict_validation=mimic_qpu)
    else:
        emulator_type = EmulatorType.EMU_FREE
        sdk_config = EmuFreeConfig(strict_validation=mimic_qpu)
    fixt.mock_cloud_sdk.create_batch.assert_called_once()
    fixt.mock_cloud_sdk.create_batch.assert_called_once_with(
        serialized_sequence=seq.to_abstract_repr(),
        jobs=good_kwargs.get("job_params", []),
        emulator=emulator_type,
        configuration=sdk_config,
        wait=False,
    )
