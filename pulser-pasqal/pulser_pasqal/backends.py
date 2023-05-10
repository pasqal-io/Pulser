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

from typing import Any, ClassVar

import pasqal_cloud

import pulser
from pulser.backend.config import EmulatorConfig
from pulser.backend.remote import RemoteBackend, RemoteResults
from pulser_pasqal.pasqal_cloud import PasqalCloud


class PasqalEmulator(RemoteBackend):
    """The base class for a Pasqal emulator backend."""

    emulator: ClassVar[pasqal_cloud.EmulatorType]

    def __init__(
        self,
        sequence: pulser.Sequence,
        connection: PasqalCloud,
        config: EmulatorConfig = EmulatorConfig(),
    ) -> None:
        """Initializes a new Pasqal emulator backend."""
        super().__init__(sequence, connection)
        if not isinstance(config, EmulatorConfig):
            raise TypeError(
                "'config' must be of type 'EmulatorConfig', "
                f"not {type(config)}."
            )
        self._config = config
        if not isinstance(self._connection, PasqalCloud):
            raise TypeError(
                "The connection to the remote backend must be done"
                " through a 'PasqalCloud' instance."
            )

    def run(
        self, job_params: list[dict[str, int | dict[str, Any]]] | None = None
    ) -> RemoteResults | tuple[RemoteResults, ...]:
        """Executes on the emulator backend through the Pasqal Cloud.

        Args:
            job_params: An optional list of parameters for each job to execute.
                Must be provided only when the sequence is parametrized as
                a list of mappings, where each mapping contains one mapping
                of variable names to values under the 'variables' field.

        Returns:
            The results, which can be accessed once all sequences have been
            successfully executed.

        """
        needs_build = (
            self._sequence.is_parametrized()
            or self._sequence.is_register_mappable()
        )
        if job_params is None and needs_build:
            raise ValueError(
                "When running a sequence that requires building, "
                "'job_params' must be provided."
            )
        elif job_params and not needs_build:
            raise ValueError(
                "'job_params' cannot be provided when running built "
                "sequences on an emulator backend."
            )

        return self._connection.submit(
            self._sequence,
            emulator=self.emulator,
            config=self._config,
            job_params=job_params,
        )


class EmuTNBackend(PasqalEmulator):
    """An emulator backend using tensor network simulation."""

    emulator = pasqal_cloud.EmulatorType.EMU_TN


class EmuFreeBackend(PasqalEmulator):
    """An emulator backend using free Hamiltonian time evolution."""

    emulator = pasqal_cloud.EmulatorType.EMU_FREE
