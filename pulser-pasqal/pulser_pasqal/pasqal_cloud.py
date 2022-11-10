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
"""Allows to connect to the cloud powered by Pasqal to run sequences."""
from typing import Dict, List, Mapping, Optional, Union

import sdk
from numpy.typing import ArrayLike

from pulser import Sequence
from pulser.devices import Device
from pulser.register import QubitId


class PasqalCloud:
    """Manager of the connection to the cloud powered by Pasqal.

    The cloud connection enables to run sequences on simulators or on real
    QPUs.

    Args:
        client_id: client_id of the API key you are holding for Pasqal
            cloud.
        client_secret: client_secret of the API key you are holding for
            Pasqal cloud.
        endpoints: Optionally, Pasqal cloud connection URLs.
        webhook: Optionally, webhook.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        endpoints: Optional[sdk.endpoints.Endpoints],
        webhook: Optional[str] = None,
    ):
        """Initializes a connection to the cloud."""
        self._sdk_connection = sdk.SDK(
            client_id=client_id,
            client_secret=client_secret,
            endpoints=endpoints,
            webhook=webhook,
        )

    def create_batch(
        self,
        seq: Sequence,
        jobs: List[Dict[str, Union[ArrayLike, Mapping[QubitId, int]]]],
        device_type: sdk.DeviceType = sdk.DeviceType.QPU,
        configuration: Optional[sdk.Configuration] = None,
        wait: bool = False,
    ) -> sdk.Batch:
        """Create a new batch and send it to the API.

        For Iroise MVP, the batch must contain at least one job and will be
        declared as complete immediately.

        Args:
            seq: Pulser sequence.
            jobs: List of jobs to be added to the batch at creation.
            device_type: The type of device to use, either an emulator or a QPU
                If set to QPU, the device_type will be set to the one
                stored in the serialized sequence.
            configuration: A dictionary with extra configuration for the
                emulators that accept it.
            wait: Whether to wait for results to be sent back.

        Returns:
            Batch: The new batch that has been created in the database.
        """
        if device_type == sdk.DeviceType.QPU and not isinstance(
            seq.device, Device
        ):
            raise TypeError(
                "To be sent to a real QPU, the device of the sequence "
                "must be a real device, instance of 'Device'."
            )

        return self._sdk_connection.create_batch(
            serialized_sequence=seq.serialize(),
            jobs=jobs,
            device_type=device_type,
            configuration=configuration,
            wait=wait,
        )

    def get_batch(self, id: int, fetch_results: bool = False) -> sdk.Batch:
        """Retrieve a batch's data and all its jobs.

        Args:
            id: Id of the batch.
            fetch_results: Whether to load job results.

        Returns:
            Batch: The batch stored in the PCS database.
        """
        return self._sdk_connection.get_batch(
            id=id, fetch_results=fetch_results
        )
