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
from unittest.mock import patch

import pytest

import pulser
import pulser_pasqal
from pulser.devices import Chadoq2
from pulser.register import Register
from pulser.sequence import Sequence
from pulser_pasqal import BaseConfig, EmulatorType, Endpoints, PasqalCloud
from pulser_pasqal.job_parameters import JobParameters, JobVariables

root = Path(__file__).parent.parent


def test_version():
    assert pulser_pasqal.__version__ == pulser.__version__


@dataclasses.dataclass
class CloudFixture:
    pasqal_cloud: PasqalCloud
    mock_cloud_sdk: Any


@pytest.fixture
def fixt():
    with patch("pasqal_cloud.SDK", autospec=True) as mock_cloud_sdk_class:
        pasqal_cloud_kwargs = dict(
            username="abc",
            password="def",
            group_id="ghi",
            endpoints=Endpoints(core="core_url"),
            webhook="xyz",
        )

        pasqal_cloud = PasqalCloud(**pasqal_cloud_kwargs)

        mock_cloud_sdk_class.assert_called_once_with(**pasqal_cloud_kwargs)

        mock_cloud_sdk = mock_cloud_sdk_class.return_value

        mock_cloud_sdk_class.reset_mock()

        yield CloudFixture(
            pasqal_cloud=pasqal_cloud, mock_cloud_sdk=mock_cloud_sdk
        )

        mock_cloud_sdk_class.assert_not_called()


test_device = Chadoq2
virtual_device = test_device.to_virtual()


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

    fixt.mock_cloud_sdk.create_batch.assert_called_once_with(
        serialized_sequence=expected_seq_representation,
        **expected_create_batch_kwargs,
    )

    get_batch_kwargs = dict(
        id="uuid",
        fetch_results=True,
    )

    fixt.pasqal_cloud.get_batch(**get_batch_kwargs)

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

    with pytest.raises(TypeError, match="must be a real device"):
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
