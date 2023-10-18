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

import typing
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, TypedDict

from pulser.backend.abc import Backend
from pulser.devices import Device
from pulser.result import Result, Results
from pulser.sequence import Sequence


class JobParams(TypedDict, total=False):
    """The parameters for an individual job running on a backend."""

    runs: int
    variables: dict[str, Any]


class SubmissionStatus(Enum):
    """Status of a remote submission."""

    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    CANCELED = auto()
    TIMED_OUT = auto()
    ERROR = auto()
    PAUSED = auto()


class RemoteResultsError(Exception):
    """Error raised when fetching remote results fails."""

    pass


class RemoteResults(Results):
    """A collection of results obtained through a remote connection.

    Args:
        submission_id: The ID that identifies the submission linked to
            the results.
        connection: The remote connection over which to get the submission's
            status and fetch the results.
    """

    def __init__(self, submission_id: str, connection: RemoteConnection):
        """Instantiates a new collection of remote results."""
        self._submission_id = submission_id
        self._connection = connection

    @property
    def results(self) -> tuple[Result, ...]:
        """The actual results, obtained after execution is done."""
        return self._results

    def get_status(self) -> SubmissionStatus:
        """Gets the status of the remote submission."""
        return self._connection._get_submission_status(self._submission_id)

    def __getattr__(self, name: str) -> Any:
        if name == "_results":
            status = self.get_status()
            if status == SubmissionStatus.DONE:
                self._results = tuple(
                    self._connection._fetch_result(self._submission_id)
                )
                return self._results
            raise RemoteResultsError(
                "The results are not available. The submission's status is "
                f"{str(status)}."
            )
        raise AttributeError(
            f"'RemoteResults' object has no attribute '{name}'."
        )


class RemoteConnection(ABC):
    """The abstract base class for a remote connection."""

    @abstractmethod
    def submit(
        self, sequence: Sequence, wait: bool = False, **kwargs: Any
    ) -> RemoteResults | tuple[RemoteResults, ...]:
        """Submit a job for execution."""
        pass

    @abstractmethod
    def _fetch_result(self, submission_id: str) -> typing.Sequence[Result]:
        """Fetches the results of a completed submission."""
        pass

    @abstractmethod
    def _get_submission_status(self, submission_id: str) -> SubmissionStatus:
        """Gets the status of a submission from its ID.

        Not all SubmissionStatus values must be covered, but at least
        SubmissionStatus.DONE is expected.
        """
        pass

    def fetch_available_devices(self) -> dict[str, Device]:
        """Fetches the devices available through this connection."""
        raise NotImplementedError(  # pragma: no cover
            "Unable to fetch the available devices through this "
            "remote connection."
        )


class RemoteBackend(Backend):
    """A backend for sequence execution through a remote connection.

    Args:
        sequence: A Sequence or a list of Sequences to execute on a
            backend accessible via a remote connection.
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
