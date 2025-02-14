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

from dataclasses import fields
from typing import Any, Mapping, Type, cast

import backoff
import pasqal_cloud
from pasqal_cloud.device.configuration import (
    BaseConfig,
    EmuFreeConfig,
    EmuTNConfig,
)

from pulser import Sequence
from pulser.abstract_repr import deserialize_device
from pulser.backend.config import EmulatorConfig
from pulser.backend.qpu import QPUBackend
from pulser.backend.remote import (
    BatchStatus,
    JobParams,
    JobStatus,
    RemoteConnection,
    RemoteResults,
    RemoteResultsError,
)
from pulser.devices import Device
from pulser.json.utils import make_json_compatible
from pulser.result import Result, SampledResult

EMU_TYPE_TO_CONFIG: dict[pasqal_cloud.EmulatorType, Type[BaseConfig]] = {
    pasqal_cloud.EmulatorType.EMU_FREE: EmuFreeConfig,
    pasqal_cloud.EmulatorType.EMU_TN: EmuTNConfig,
}

MAX_CLOUD_ATTEMPTS = 5

backoff_decorator = backoff.on_exception(
    backoff.fibo, Exception, max_tries=MAX_CLOUD_ATTEMPTS, max_value=60
)


class PasqalCloud(RemoteConnection):
    """Manager of the connection to PASQAL's cloud platform.

    The cloud connection enables to run sequences on simulators or on real
    QPUs.

    Args:
        username: Your username in the PASQAL cloud platform.
        password: The password for your PASQAL cloud platform account.
        project_id: The project ID associated to the account.
        kwargs: Additional arguments to provide to the pasqal_cloud.SDK()
    """

    def __init__(
        self,
        username: str = "",
        password: str = "",
        project_id: str = "",
        **kwargs: Any,
    ):
        """Initializes a connection to the Pasqal cloud platform."""
        self._sdk_connection = pasqal_cloud.SDK(
            username=username,
            password=password,
            project_id=project_id,
            **kwargs,
        )

    def submit(
        self,
        sequence: Sequence,
        wait: bool = False,
        open: bool = False,
        batch_id: str | None = None,
        **kwargs: Any,
    ) -> RemoteResults:
        """Submits the sequence for execution on a remote Pasqal backend."""
        sequence = self._add_measurement_to_sequence(sequence)
        emulator = kwargs.get("emulator", None)
        job_params: list[JobParams] = make_json_compatible(
            kwargs.get("job_params", [])
        )
        mimic_qpu: bool = kwargs.get("mimic_qpu", False)
        if emulator is None or mimic_qpu:
            sequence = self.update_sequence_device(sequence)
            QPUBackend.validate_job_params(
                job_params, sequence.device.max_runs
            )

        if sequence.is_parametrized() or sequence.is_register_mappable():
            for params in job_params:
                vars = params.get("variables", {})
                sequence.build(**vars)

        configuration = self._convert_configuration(
            config=kwargs.get("config", None),
            emulator=emulator,
            strict_validation=mimic_qpu,
        )

        # If batch_id is not empty, then we can submit new jobs to a
        # batch we just created otherwise, create a new one with
        #  _sdk_connection.create_batch()
        if batch_id:
            submit_jobs_fn = backoff_decorator(self._sdk_connection.add_jobs)
            old_job_ids = self._get_job_ids(batch_id)
            batch = submit_jobs_fn(
                batch_id,
                jobs=job_params or [],  # type: ignore[arg-type]
            )
            new_job_ids = [
                job_id
                for job_id in self._get_job_ids(batch_id)
                if job_id not in old_job_ids
            ]
        else:
            create_batch_fn = backoff_decorator(
                self._sdk_connection.create_batch
            )
            batch = create_batch_fn(
                serialized_sequence=sequence.to_abstract_repr(),
                jobs=job_params or [],  # type: ignore[arg-type]
                emulator=emulator,
                configuration=configuration,
                wait=wait,
                open=open,
            )
            new_job_ids = self._get_job_ids(batch.id)
        return RemoteResults(batch.id, self, job_ids=new_job_ids)

    @backoff_decorator
    def fetch_available_devices(self) -> dict[str, Device]:
        """Fetches the devices available through this connection."""
        abstract_devices = self._sdk_connection.get_device_specs_dict()
        return {
            name: cast(Device, deserialize_device(dev_str))
            for name, dev_str in abstract_devices.items()
        }

    def _fetch_result(
        self, batch_id: str, job_ids: list[str] | None
    ) -> tuple[Result, ...]:
        # For now, the results are always sampled results
        jobs = self._query_job_progress(batch_id)

        if job_ids is None:
            job_ids = list(jobs.keys())

        results: list[Result] = []
        for id in job_ids:
            status, result = jobs[id]
            if status in {JobStatus.PENDING, JobStatus.RUNNING}:
                raise RemoteResultsError(
                    f"The results are not yet available, job {id} status is "
                    f"{status}."
                )
            if result is None:
                raise RemoteResultsError(f"No results found for job {id}.")
            results.append(result)

        return tuple(results)

    def _query_job_progress(
        self, batch_id: str
    ) -> Mapping[str, tuple[JobStatus, Result | None]]:
        get_batch_fn = backoff_decorator(self._sdk_connection.get_batch)
        batch = get_batch_fn(id=batch_id)

        assert isinstance(batch.sequence_builder, str)
        seq_builder = Sequence.from_abstract_repr(batch.sequence_builder)
        reg = seq_builder.get_register(include_mappable=True)
        all_qubit_ids = reg.qubit_ids
        meas_basis = seq_builder.get_measurement_basis()

        results: dict[str, tuple[JobStatus, Result | None]] = {}

        for job in batch.ordered_jobs:
            vars = job.variables
            size: int | None = None
            if vars and "qubits" in vars:
                size = len(vars["qubits"])
            if job.result is None:
                results[job.id] = (JobStatus[job.status], None)
            else:
                results[job.id] = (
                    JobStatus[job.status],
                    SampledResult(
                        atom_order=all_qubit_ids[slice(size)],
                        meas_basis=meas_basis,
                        bitstring_counts=job.result,
                    ),
                )
        return results

    @backoff_decorator
    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        """Gets the status of a batch from its ID."""
        batch = self._sdk_connection.get_batch(id=batch_id)
        return BatchStatus[batch.status]

    @backoff_decorator
    def _get_job_ids(self, batch_id: str) -> list[str]:
        """Gets all the job IDs within a batch."""
        batch = self._sdk_connection.get_batch(id=batch_id)
        return [job.id for job in batch.ordered_jobs]

    def _convert_configuration(
        self,
        config: EmulatorConfig | None,
        emulator: pasqal_cloud.EmulatorType | None,
        strict_validation: bool = False,
    ) -> pasqal_cloud.BaseConfig | None:
        """Converts a backend configuration into a pasqal_cloud.BaseConfig."""
        if emulator is None or config is None:
            return None
        emu_cls = EMU_TYPE_TO_CONFIG[emulator]
        backend_options = config.backend_options.copy()
        pasqal_config_kwargs = {}
        for field in fields(emu_cls):
            pasqal_config_kwargs[field.name] = backend_options.pop(
                field.name, field.default
            )
        # We pass the remaining backend options to "extra_config"
        if backend_options:
            pasqal_config_kwargs["extra_config"] = backend_options
        if emulator == pasqal_cloud.EmulatorType.EMU_TN:
            pasqal_config_kwargs["dt"] = 1.0 / config.sampling_rate

        pasqal_config_kwargs["strict_validation"] = strict_validation
        return emu_cls(**pasqal_config_kwargs)

    def supports_open_batch(self) -> bool:
        """Flag to confirm this class can support creating an open batch."""
        return True

    def _close_batch(self, batch_id: str) -> None:
        """Closes the batch on pasqal cloud associated with the batch ID."""
        self._sdk_connection.close_batch(batch_id)
