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
"""Defines Pasqal specific backends."""
from __future__ import annotations

from dataclasses import fields
from typing import ClassVar

import pasqal_cloud

import pulser
from pulser.backend.config import EmulatorConfig
from pulser.backend.remote import JobParams, RemoteBackend, RemoteResults
from pulser_pasqal.pasqal_cloud import PasqalCloud

DEFAULT_CONFIG_EMU_TN = EmulatorConfig(
    evaluation_times="Final", sampling_rate=0.1
)
DEFAULT_CONFIG_EMU_FREE = EmulatorConfig(
    evaluation_times="Final", sampling_rate=0.25
)


class PasqalEmulator(RemoteBackend):
    """The base class for a Pasqal emulator backend."""

    emulator: ClassVar[pasqal_cloud.EmulatorType]
    default_config: ClassVar[EmulatorConfig] = EmulatorConfig()
    configurable_fields: ClassVar[tuple[str, ...]] = ("backend_options",)

    def __init__(
        self,
        sequence: pulser.Sequence,
        connection: PasqalCloud,
        config: EmulatorConfig | None = None,
    ) -> None:
        """Initializes a new Pasqal emulator backend."""
        super().__init__(sequence, connection)
        config_ = config or self.default_config
        self._validate_config(config_)
        self._config = config_
        if not isinstance(self._connection, PasqalCloud):
            raise TypeError(
                "The connection to the remote backend must be done"
                " through a 'PasqalCloud' instance."
            )

    def run(
        self, job_params: list[JobParams] | None = None, wait: bool = False
    ) -> RemoteResults | tuple[RemoteResults, ...]:
        """Executes on the emulator backend through the Pasqal Cloud.

        Args:
            job_params: A list of parameters for each job to execute. Each
                mapping must contain a defined 'runs' field specifying
                the number of times to run the same sequence. If the sequence
                is parametrized, the values for all the variables necessary
                to build the sequence must be given in it's own mapping, for
                each job, under the 'variables' field.
            wait: Whether to wait until the results of the jobs become
                available.  If set to False, the call is non-blocking and the
                obtained results' status can be checked using their `status`
                property.

        Returns:
            The results, which can be accessed once all sequences have been
            successfully executed.

        """
        suffix = f" when executing a sequence on {self.__class__.__name__}."
        if not job_params:
            raise ValueError("'job_params' must be specified" + suffix)
        if any("runs" not in j for j in job_params):
            raise ValueError(
                "All elements of 'job_params' must specify 'runs'" + suffix
            )

        return self._connection.submit(
            self._sequence,
            job_params=job_params,
            emulator=self.emulator,
            config=self._config,
            wait=wait,
        )

    def _validate_config(self, config: EmulatorConfig) -> None:
        if not isinstance(config, EmulatorConfig):
            raise TypeError(
                "'config' must be of type 'EmulatorConfig', "
                f"not {type(config)}."
            )
        for field in fields(config):
            if field.name in self.configurable_fields:
                continue
            default_value = getattr(self.default_config, field.name)
            if getattr(config, field.name) != default_value:
                raise NotImplementedError(
                    f"'EmulatorConfig.{field.name}' is not configurable in "
                    "this backend. It should not be changed from its default "
                    f"value of '{default_value}'."
                )


class EmuTNBackend(PasqalEmulator):
    """An emulator backend using tensor network simulation.

    Configurable fields in EmulatorConfig:
        - sampling_rate: Defaults to 0.1. This value must remain low to use
          this backend efficiently.
        - backend_options:
            - precision (str): The precision of the simulation. Can be "low",
              "normal" or "high". Defaults to "normal".
            - max_bond_dim (int): The maximum bond dimension of the Matrix
              Product State (MPS). Defaults to 500.

    All other parameters should not be changed from their default values.

    Args:
        sequence: The sequence to send to the backend.
        connection: An open PasqalCloud connection.
        config: An EmulatorConfig to configure the backend. If not provided,
            the default configuration is used.
    """

    emulator = pasqal_cloud.EmulatorType.EMU_TN
    default_config = DEFAULT_CONFIG_EMU_TN
    configurable_fields = ("backend_options", "sampling_rate")


class EmuFreeBackend(PasqalEmulator):
    """An emulator backend using free Hamiltonian time evolution.

    Configurable fields in EmulatorConfig:
        - backend_options:
            - with_noise (bool): Whether to add noise to the simulation.
              Defaults to False.

    All other parameters should not be changed from their default values.

    Args:
        sequence: The sequence to send to the backend.
        connection: An open PasqalCloud connection.
        config: An EmulatorConfig to configure the backend. If not provided,
            the default configuration is used.
    """

    emulator = pasqal_cloud.EmulatorType.EMU_FREE
    default_config = DEFAULT_CONFIG_EMU_FREE
