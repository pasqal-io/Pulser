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
"""Base classes for remote backend execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pulser.backend.abc import Backend, Results
from pulser.sequence import Sequence

JobId = str


class RemoteConnection(ABC):
    """The abstract base class for a remote connection."""

    @abstractmethod
    def submit(
        self, sequence: Sequence | list[Sequence], **kwargs: Any
    ) -> JobId:
        """Submit a job for execution."""
        pass

    @abstractmethod
    def fetch_result(self, job_id: JobId) -> Results | list[Results]:
        """Fetch the results of a completed job."""
        pass


class RemoteBackend(Backend):
    """A backend for sequence execution through a remote connection.

    Args:
        sequence: A Sequence or a list of Sequences to execute on a
            backed accessible via a remote connection.
        connection: The remote connection through which the jobs
            are executed.
    """

    def __init__(
        self,
        sequence: Sequence,
        connection: RemoteConnection,
    ) -> None:
        """Starts a new remote backend instance."""
        super().__init__(sequence)
        if not isinstance(connection, RemoteConnection):
            raise TypeError(
                "'connection' must be a valid RemoteConnection instance."
            )
        self._connection = connection
        # Extra arguments to add to pass to RemoteConnection.submit()
        self._submit_kwargs: dict[str, Any] = {}

    def run(self) -> Results | list[Results]:
        """Runs on the remote backend and returns the result."""
        job_id = self._connection.submit(self._sequence, **self._submit_kwargs)
        return self._connection.fetch_result(job_id)
