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
from typing import Any, Mapping, TypedDict

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


class JobStatus(Enum):
    """Status of a remote job."""

    PENDING = auto()
    RUNNING = auto()
    DONE = auto()
    CANCELED = auto()
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
        job_ids: If given, specifies which jobs within the submission should
            be included in the results and in what order. If left undefined,
            all jobs are included.
    """

    def __init__(
        self,
        submission_id: str,
        connection: RemoteConnection,
        job_ids: list[str] | None = None,
    ):
        """Instantiates a new collection of remote results."""
        self._submission_id = submission_id
        self._connection = connection
        if job_ids is not None and not set(job_ids).issubset(
            all_job_ids := self._connection._get_job_ids(self._submission_id)
        ):
            unknown_ids = [id_ for id_ in job_ids if id_ not in all_job_ids]
            raise RuntimeError(
                f"Submission {self._submission_id!r} does not contain jobs "
                f"{unknown_ids}."
            )
        self._job_ids = job_ids

    @property
    def results(self) -> tuple[Result, ...]:
        """The actual results, obtained after execution is done."""
        return self._results

    @property
    def batch_id(self) -> str:
        """The ID of the batch containing these results."""
        return self._submission_id

    @property
    def job_ids(self) -> list[str]:
        """The IDs of the jobs within this results submission."""
        if self._job_ids is None:
            return self._connection._get_job_ids(self._submission_id)
        return self._job_ids

    def get_status(self) -> SubmissionStatus:
        """Gets the status of the remote submission."""
        return self._connection._get_submission_status(self._submission_id)

    def get_available_results(self, submission_id: str) -> dict[str, Result]:
        """Returns the available results of a submission.

        Unlike the `results` property, this method does not raise an error if
        some jobs associated to the submission do not have results.

        Returns:
            dict[str, Result]: A dictionary mapping the job ID to their results.
            Jobs with no result are omitted.
        """

        results = {
            k: v[1]
            for k, v in self._connection._query_job_progress(
                submission_id
            ).items()
            if v[1] is not None
        }

        if self._job_ids:
            return {k: v for k, v in results.items() if k in self._job_ids}
        return results

    def __getattr__(self, name: str) -> Any:
        if name == "_results":
            try:
                self._results = tuple(
                    self._connection._fetch_result(
                        self._submission_id, self._job_ids
                    )
                )
                return self._results
            except RemoteResultsError as e:
                raise RemoteResultsError(
                    "Results are not available for all jobs. Use the "
                    "`get_available_results` method to retrieve partial "
                    "results."
                ) from e
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
    def _fetch_result(
        self, submission_id: str, job_ids: list[str] | None
    ) -> typing.Sequence[Result]:
        """Fetches the results of a completed submission."""
        pass

    @abstractmethod
    def _query_job_progress(
        self, submission_id: str
    ) -> Mapping[str, tuple[JobStatus, Result | None]]:
        """Fetches the status and results of all the jobs in a submission.

        Unlike `_fetch_result`, this method does not raise an error if some
        jobs associated to the submission do not have results.

        It returns a dictionnary mapping the job ID to their status and results.
        """
        pass

    @abstractmethod
    def _get_submission_status(self, submission_id: str) -> SubmissionStatus:
        """Gets the status of a submission from its ID.

        Not all SubmissionStatus values must be covered, but at least
        SubmissionStatus.DONE is expected.
        """
        pass

    def _get_job_ids(self, submission_id: str) -> list[str]:
        """Gets all the job IDs within a submission."""
        raise NotImplementedError(
            "Unable to find job IDs through this remote connection."
        )

    def fetch_available_devices(self) -> dict[str, Device]:
        """Fetches the devices available through this connection."""
        raise NotImplementedError(
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
