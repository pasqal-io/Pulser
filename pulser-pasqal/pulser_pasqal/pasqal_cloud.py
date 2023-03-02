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
"""Allows to connect to PASQAL's cloud platform to run sequences."""
from __future__ import annotations

from typing import Any, Optional

import sdk

from pulser import Sequence
from pulser.devices import Device
from pulser_pasqal.job_parameters import JobParameters


class PasqalCloud:
    """Manager of the connection to PASQAL's cloud platform.

    The cloud connection enables to run sequences on simulators or on real
    QPUs.

    Args:
        username: your username in the PASQAL cloud platform.
        password: the password for your PASQAL cloud platform account.
        group_id: the group_id associated to the account.
        kwargs: Additional arguments to provide to SDK
    """

    def __init__(
        self,
        username: str = "",
        password: str = "",
        group_id: str = "",
        **kwargs: Any,
    ):
        """Initializes a connection to the Pasqal cloud platform."""
        self._sdk_connection = sdk.SDK(
            username=username,
            password=password,
            group_id=group_id,
            **kwargs,
        )

    def create_batch(
        self,
        seq: Sequence,
        jobs: list[JobParameters],
        device_type: sdk.DeviceType = sdk.DeviceType.QPU,
        configuration: Optional[sdk.BaseConfig] = None,
        wait: bool = False,
        fetch_results: bool = False,
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
            configuration: Optional extra configuration for emulators.
            wait: Whether to wait for the batch to be done.
            fetch_results: Whether to download the results. Implies waiting for the batch. # noqa: 501

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

        for params in jobs:
            seq.build(**params.variables.get_dict())  # type: ignore

        return self._sdk_connection.create_batch(
            serialized_sequence=seq.to_abstract_repr(),
            jobs=[j.get_dict() for j in jobs],
            device_type=device_type,
            configuration=configuration,
            wait=wait,
            fetch_results=fetch_results,
        )

    def get_batch(self, id: str, fetch_results: bool = False) -> sdk.Batch:
        """Retrieve a batch's data and all its jobs.

        Args:
            id: Id of the batch.
            fetch_results: Whether to load job results.

        Returns:
            Batch: The batch stored in the database.
        """
        return self._sdk_connection.get_batch(
            id=id, fetch_results=fetch_results
        )
