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

from pathlib import Path
from unittest.mock import patch

import pulser
import pulser_pasqal
from pulser.devices import Chadoq2
from pulser.register import Register
from pulser.sequence import Sequence
from pulser_pasqal import Configuration, DeviceType, Endpoints, PasqalCloud

root = Path(__file__).parent.parent


def test_version():
    assert pulser_pasqal.__version__ == pulser.__version__


def test_pasqal_cloud():
    with patch("sdk.SDK", autospec=True) as mock_cloud_sdk_class:
        pasqal_cloud_kwargs = dict(
            client_id="abc",
            client_secret="def",
            endpoints=Endpoints(core="core_url", account="account_url"),
            webhook="xyz",
        )

        pasqal_cloud = PasqalCloud(**pasqal_cloud_kwargs)

        mock_cloud_sdk_class.assert_called_once_with(**pasqal_cloud_kwargs)

        mock_cloud_sdk = mock_cloud_sdk_class.return_value

        reg = Register(dict(enumerate([(0, 0), (0, 10)])))
        device = Chadoq2
        seq = Sequence(reg, device)

        create_batch_kwargs = dict(
            jobs=[{"runs": 10, "variables": {"a": [3, 5]}}],
            device_type=DeviceType.QPU,
            configuration=Configuration(
                dt=0.1,
                precision="normal",
                extra_config=None,
            ),
            wait=True,
        )

        pasqal_cloud.create_batch(
            seq,
            **create_batch_kwargs,
        )

        mock_cloud_sdk.create_batch.assert_called_once_with(
            seq.serialize(),
            **create_batch_kwargs,
        )

        get_batch_kwargs = dict(
            id=10,
            fetch_results=True,
        )

        pasqal_cloud.get_batch(**get_batch_kwargs)

        mock_cloud_sdk.get_batch.assert_called_once_with(**get_batch_kwargs)
