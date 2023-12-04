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
import json
from dataclasses import fields
from typing import Any, Type, cast

import backoff
import numpy as np
import pasqal_cloud
from pasqal_cloud.device.configuration import (
    BaseConfig,
    EmuFreeConfig,
    EmuTNConfig,
)

from pulser import Sequence
from pulser.backend.config import EmulatorConfig
from pulser.backend.remote import (
    JobParams,
    RemoteConnection,
    RemoteResults,
    SubmissionStatus,
)
from pulser.devices import Device
from pulser.json.abstract_repr.deserializer import deserialize_device
from pulser.result import Result, SampledResult

EMU_TYPE_TO_CONFIG: dict[pasqal_cloud.EmulatorType, Type[BaseConfig]] = {
    pasqal_cloud.EmulatorType.EMU_FREE: EmuFreeConfig,
    pasqal_cloud.EmulatorType.EMU_TN: EmuTNConfig,
}

MAX_CLOUD_ATTEMPTS = 5

backoff_decorator = backoff.on_exception(
    backoff.fibo, Exception, max_tries=MAX_CLOUD_ATTEMPTS, max_value=60
)


def _make_json_compatible(obj: Any) -> Any:
    """Makes an object compatible with JSON serialization.

    For now, simply converts Numpy arrays to lists, but more can be added
    as needed.
    """

    class NumpyEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, np.ndarray):
                return o.tolist()
            return json.JSONEncoder.default(self, o)

    # Serializes with the custom encoder and then deserializes back
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


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
        project_id_ = project_id or kwargs.pop("group_id", "")
        self._sdk_connection = pasqal_cloud.SDK(
            username=username,
            password=password,
            project_id=project_id_,
            **kwargs,
        )

    def submit(
        self, sequence: Sequence, wait: bool = False, **kwargs: Any
    ) -> RemoteResults:
        """Submits the sequence for execution on a remote Pasqal backend."""
        if not sequence.is_measured():
            bases = sequence.get_addressed_bases()
            if len(bases) != 1:
                raise ValueError(
                    "The measurement basis can't be implicitly determined "
                    "for a sequence not addressing a single basis."
                )
            # The copy prevents changing the input sequence
            sequence = copy.deepcopy(sequence)
            sequence.measure(bases[0])

        emulator = kwargs.get("emulator", None)
        job_params: list[JobParams] = _make_json_compatible(
            kwargs.get("job_params", [])
        )
        if emulator is None:
            available_devices = self.fetch_available_devices()
            # TODO: Could be better to check if the devices are
            # compatible, even if not exactly equal
            if sequence.device not in available_devices.values():
                raise ValueError(
                    "The device used in the sequence does not match any "
                    "of the devices currently available through the remote "
                    "connection."
                )
            # TODO: Validate the register layout

        if sequence.is_parametrized() or sequence.is_register_mappable():
            for params in job_params:
                vars = params.get("variables", {})
                sequence.build(**vars)

        configuration = self._convert_configuration(
            config=kwargs.get("config", None), emulator=emulator
        )
        create_batch_fn = backoff_decorator(self._sdk_connection.create_batch)
        batch = create_batch_fn(
            serialized_sequence=sequence.to_abstract_repr(),
            jobs=job_params or [],  # type: ignore[arg-type]
            emulator=emulator,
            configuration=configuration,
            wait=wait,
        )

        return RemoteResults(batch.id, self)

    @backoff_decorator
    def fetch_available_devices(self) -> dict[str, Device]:
        """Fetches the devices available through this connection."""
        abstract_devices = self._sdk_connection.get_device_specs_dict()
        return {
            name: cast(Device, deserialize_device(dev_str))
            for name, dev_str in abstract_devices.items()
        }

    def _fetch_result(self, submission_id: str) -> tuple[Result, ...]:
        # For now, the results are always sampled results
        get_batch_fn = backoff_decorator(self._sdk_connection.get_batch)
        batch = get_batch_fn(id=submission_id)
        seq_builder = Sequence.from_abstract_repr(batch.sequence_builder)
        reg = seq_builder.get_register(include_mappable=True)
        all_qubit_ids = reg.qubit_ids
        meas_basis = seq_builder.get_measurement_basis()

        results = []
        for job in batch.ordered_jobs:
            vars = job.variables
            size: int | None = None
            if vars and "qubits" in vars:
                size = len(vars["qubits"])
            assert job.result is not None, "Failed to fetch the results."
            results.append(
                SampledResult(
                    atom_order=all_qubit_ids[slice(size)],
                    meas_basis=meas_basis,
                    bitstring_counts=job.result,
                )
            )
        return tuple(results)

    @backoff_decorator
    def _get_submission_status(self, submission_id: str) -> SubmissionStatus:
        """Gets the status of a submission from its ID."""
        batch = self._sdk_connection.get_batch(id=submission_id)
        return SubmissionStatus[batch.status]

    def _convert_configuration(
        self,
        config: EmulatorConfig | None,
        emulator: pasqal_cloud.EmulatorType | None,
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

        return emu_cls(**pasqal_config_kwargs)
