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
from typing import Type
from types import TracebackType
import typing
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, TypedDict, cast, Self

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

    @property
    def submission_id(self) -> str:
        """
        Return submission id  which is associated with a associated
        with the newly submitted payloads on the remote service

        Returns:
            UUID in string format representing most recent submission
        """
        return self._submission_id

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
        self,
        sequence: Sequence,
        wait: bool = False,
<<<<<<< HEAD
        open: bool = False,
        batch_id: str | None = None,
=======
        submission_id: str | None = None,
>>>>>>> 3f3a07c (change to context manager interface for open batches)
        **kwargs: Any,
    ) -> RemoteResults | tuple[RemoteResults, ...]:
        """Submit a job for execution."""
        pass

    def close_submission(self, submission_id: str) -> SubmissionStatus | None:
        """
        Close a submission and make it unavailable to submit any further jobs
        to.
        """
        pass
        # raise NotImplementedError(
        #     "Unable to close submission through this remote connection"
        # )

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
        mimic_qpu: Whether to mimic the validations necessary for
            execution on a QPU.
    """

    def __init__(
        self,
        sequence: Sequence,
        connection: RemoteConnection,
        mimic_qpu: bool = False,
    ) -> None:
        """Starts a new remote backend instance."""

        super().__init__(sequence, mimic_qpu=mimic_qpu)
        if not isinstance(connection, RemoteConnection):
            raise TypeError(
                "'connection' must be a valid RemoteConnection instance."
            )
        self._connection = connection
<<<<<<< HEAD
        self.batch_id = ""

    @staticmethod
    def _type_check_job_params(job_params: list[JobParams] | None) -> None:
        if not isinstance(job_params, list):
            raise TypeError(
                f"'job_params' must be a list; got {type(job_params)} instead."
            )
        for d in job_params:
            if not isinstance(d, dict):
                raise TypeError(
                    "All elements of 'job_params' must be dictionaries; "
                    f"got {type(d)} instead."
                )

    def open_batch(self) -> Self:
        """
        Create an open batch that can continue to recieve new job submissions
        as long as the submissions are submitted within an open
        context manager. The batch will be closed when the scope of
        the context manager ends.

        Returns:
            A class instance with an associated batch_id property
        """
        # Create batch and receive submission id
        submission = cast(
            RemoteResults, self._connection.submit(self._sequence, open=True)
        )
        self.batch_id = submission._submission_id
        return self

    def __enter__(self):
        # enter returns an instance of self to use open_batch
        # as a context manager
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # On context exit, we make a remote call to close the open batch
        self._connection._close_batch(self.batch_id)
        self.batch_id = ""
=======
        self.submission_id: str = ""
>>>>>>> 3f3a07c (change to context manager interface for open batches)
