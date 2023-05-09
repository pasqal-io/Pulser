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

import copy
from typing import Any, Dict, Optional, cast

import pasqal_cloud

from pulser import Sequence
from pulser.backend.remote import (
    RemoteConnection,
    RemoteResults,
    SubmissionStatus,
)
from pulser.devices import Device
from pulser.result import Result, SampledResult
from pulser_pasqal.job_parameters import JobParameters


class PasqalCloud(RemoteConnection):
    """Manager of the connection to PASQAL's cloud platform.

    The cloud connection enables to run sequences on simulators or on real
    QPUs.

    Args:
        username: your username in the PASQAL cloud platform.
        password: the password for your PASQAL cloud platform account.
        group_id: the group_id associated to the account.
        kwargs: Additional arguments to provide to the pasqal_cloud.SDK()
    """

    def __init__(
        self,
        username: str = "",
        password: str = "",
        group_id: str = "",
        **kwargs: Any,
    ):
        """Initializes a connection to the Pasqal cloud platform."""
        self._sdk_connection = pasqal_cloud.SDK(
            username=username,
            password=password,
            group_id=group_id,
            **kwargs,
        )

    def submit(self, sequence: Sequence, **kwargs: Any) -> RemoteResults:
        """Submits the sequence for execution on a remote Pasqal backend."""
        if not sequence.is_measured():
            bases = sequence.get_addressed_bases()
            if len(bases) != 1:
                raise ValueError(
                    "The measurement basis can't be implicitly determined "
                    "for a sequence not addressing a single basis."
                )
            # The copy prevents changing the input sequence
            sequence = copy.copy(sequence)
            sequence.measure(bases[0])

        emulator = kwargs.get("emulator", None)
        job_params: list[dict[str, int | dict]] = kwargs.get("job_params", [])
        if emulator is None:
            suffix = " when executing a sequence on a real QPU."
            if not job_params:
                raise ValueError("'job_params' must be specified" + suffix)
            if any("runs" not in j for j in job_params):
                raise ValueError(
                    "All elements of 'job_params' must specify 'runs'" + suffix
                )
        if sequence.is_parametrized() or sequence.is_register_mappable():
            for params in job_params:
                vars = cast(Dict[str, Any], params.get("variables", {}))
                sequence.build(**vars)

        # TODO: Parse configuration parameters
        configuration = None

        batch = self._sdk_connection.create_batch(
            serialized_sequence=sequence.to_abstract_repr(),
            jobs=job_params,
            emulator=emulator,
            configuration=configuration,
            wait=False,
            fetch_results=False,
        )
        return RemoteResults(batch.id, self)

    def _fetch_result(self, submission_id: str) -> tuple[Result, ...]:
        # For now, the results are always sampled results
        batch = self._sdk_connection.get_batch(
            id=submission_id, fetch_results=True
        )
        seq_builder = Sequence.from_abstract_repr(batch.sequence_builder)
        reg = seq_builder.get_register(include_mappable=True)
        all_qubit_ids = reg.qubit_ids
        meas_basis = seq_builder.get_measurement_basis()

        results = []
        for job in batch.jobs.values():
            vars = job.variables
            size: int | None = None
            if vars and "qubits" in vars:
                size = len(vars["qubits"])
            counts = job.result
            assert counts is not None, "Failed to fetch the results."
            results.append(
                SampledResult(
                    atom_order=all_qubit_ids[slice(size)],
                    meas_basis=meas_basis,
                    bitstring_counts=counts,
                )
            )
        return tuple(results)

    def _get_submission_status(self, submission_id: str) -> SubmissionStatus:
        """Gets the status of a submission from its ID."""
        batch = self._sdk_connection.get_batch(
            id=submission_id, fetch_results=False
        )
        return SubmissionStatus[batch.status]

    # TODO: Deprecate
    def create_batch(
        self,
        seq: Sequence,
        jobs: list[JobParameters],
        emulator: pasqal_cloud.EmulatorType | None = None,
        configuration: Optional[pasqal_cloud.BaseConfig] = None,
        wait: bool = False,
        fetch_results: bool = False,
    ) -> pasqal_cloud.Batch:
        """Create a new batch and send it to the API.

        For Iroise MVP, the batch must contain at least one job and will be
        declared as complete immediately.

        Args:
            seq: Pulser sequence.
            jobs: List of jobs to be added to the batch at creation.
            emulator: TThe type of emulator to use. If set to None, the device
                will be set to the one stored in the serialized sequence.
            configuration: Optional extra configuration for emulators.
            wait: Whether to wait for the batch to be done.
            fetch_results: Whether to download the results. Implies waiting for the batch. # noqa: 501

        Returns:
            Batch: The new batch that has been created in the database.
        """
        if emulator is None and not isinstance(seq.device, Device):
            raise TypeError(
                "To be sent to a real QPU, the device of the sequence "
                "must be a real device, instance of 'Device'."
            )

        for params in jobs:
            seq.build(**params.variables.get_dict())  # type: ignore

        return self._sdk_connection.create_batch(
            serialized_sequence=seq.to_abstract_repr(),
            jobs=[j.get_dict() for j in jobs],
            emulator=emulator,
            configuration=configuration,
            wait=wait,
            fetch_results=fetch_results,
        )

    # TODO: Deprecate
    def get_batch(
        self, id: str, fetch_results: bool = False
    ) -> pasqal_cloud.Batch:
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
