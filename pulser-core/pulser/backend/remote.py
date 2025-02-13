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
from types import TracebackType
from typing import Any, Mapping, Type, TypedDict

from pulser.backend.abc import Backend
from pulser.backend.results import Results, ResultsSequence
from pulser.devices import Device
from pulser.sequence import Sequence


class JobParams(TypedDict, total=False):
    """The parameters for an individual job running on a backend."""

    runs: int
    variables: dict[str, Any]


class BatchStatus(Enum):
    """Status of a batch."""

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


class RemoteResults(ResultsSequence):
    """A collection of results obtained through a remote connection.

    Args:
        batch_id: The ID that identifies the batch linked to the results.
        connection: The remote connection over which to get the batch's
            status and fetch the results.
        job_ids: If given, specifies which jobs within the batch should
            be included in the results and in what order. If left undefined,
            all jobs are included.
    """

    def __init__(
        self,
        batch_id: str,
        connection: RemoteConnection,
        job_ids: list[str] | None = None,
    ):
        """Instantiates a new collection of remote results."""
        self._batch_id = batch_id
        self._connection = connection
        if job_ids is not None and not set(job_ids).issubset(
            all_job_ids := self._connection._get_job_ids(self._batch_id)
        ):
            unknown_ids = [id_ for id_ in job_ids if id_ not in all_job_ids]
            raise RuntimeError(
                f"Batch {self._batch_id!r} does not contain jobs "
                f"{unknown_ids}."
            )
        self._job_ids = job_ids

    @property
    def results(self) -> tuple[Results, ...]:
        """The actual results, obtained after execution is done."""
        return self._results_seq

    @property
    def batch_id(self) -> str:
        """The ID of the batch containing these results."""
        return self._batch_id

    @property
    def job_ids(self) -> list[str]:
        """The IDs of the jobs within these results' batch."""
        if self._job_ids is None:
            return self._connection._get_job_ids(self._batch_id)
        return self._job_ids

    def get_batch_status(self) -> BatchStatus:
        """Gets the status of the batch linked to these results."""
        return self._connection._get_batch_status(self._batch_id)

    def get_available_results(self) -> dict[str, Results]:
        """Returns the available results.

        Unlike the `results` property, this method does not raise an error if
        some of the jobs do not have results.

        Returns:
            dict[str, Results]: A dictionary mapping the job ID to its results.
            Jobs with no result are omitted.
        """
        results = {
            k: v[1]
            for k, v in self._connection._query_job_progress(
                self.batch_id
            ).items()
            if v[1] is not None
        }

        if self._job_ids:
            return {k: v for k, v in results.items() if k in self._job_ids}
        return results

    def __getattr__(self, name: str) -> Any:
        if name == "_results_seq":
            try:
                self._results_seq = tuple(
                    self._connection._fetch_result(
                        self.batch_id, self._job_ids
                    )
                )
                return self._results_seq
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
        self,
        sequence: Sequence,
        wait: bool = False,
        open: bool = True,
        batch_id: str | None = None,
        **kwargs: Any,
    ) -> RemoteResults:
        """Submit a job for execution."""
        pass

    @abstractmethod
    def _fetch_result(
        self, batch_id: str, job_ids: list[str] | None
    ) -> typing.Sequence[Results]:
        """Fetches the results of a completed batch."""
        pass

    @abstractmethod
    def _query_job_progress(
        self, batch_id: str
    ) -> Mapping[str, tuple[JobStatus, Results | None]]:
        """Fetches the status and results of all the jobs in a batch.

        Unlike `_fetch_result`, this method does not raise an error if some
        jobs in the batch do not have results.

        It returns a dictionary mapping the job ID to its status and results.
        """
        pass

    @abstractmethod
    def _get_batch_status(self, batch_id: str) -> BatchStatus:
        """Gets the status of a batch from its ID."""
        pass

    def _get_job_ids(self, batch_id: str) -> list[str]:
        """Gets all the job IDs within a batch."""
        raise NotImplementedError(
            "Unable to find job IDs through this remote connection."
        )

    def fetch_available_devices(self) -> dict[str, Device]:
        """Fetches the devices available through this connection."""
        raise NotImplementedError(
            "Unable to fetch the available devices through this "
            "remote connection."
        )

    def _close_batch(self, batch_id: str) -> None:
        """Closes a batch using its ID."""
        raise NotImplementedError(  # pragma: no cover
            "Unable to close batch through this remote connection"
        )

    @abstractmethod
    def supports_open_batch(self) -> bool:
        """Flag to confirm this class can support creating an open batch."""
        pass

    @staticmethod
    def _add_measurement_to_sequence(sequence: Sequence) -> Sequence:
        """Adds a measurement operation to a Sequence if needed and possible.

        Adding a measurement operation to the Sequence is possible if only
        one basis is addressed by the Sequence. It also converts all tensors
        in the Sequence to arrays.

        Args:
            sequence: The Sequence to add the measurement operation to.

        Returns:
            The sequence with a measurement operation.
        """
        if sequence.is_measured():
            return sequence
        bases = sequence.get_addressed_bases()
        if len(bases) != 1:
            raise ValueError(
                "The measurement basis can't be implicitly determined "
                "for a sequence not addressing a single basis."
            )
        # This is equivalent to performing a deepcopy
        # All tensors are converted to arrays but that's ok, it would
        # have happened anyway later on
        sequence = Sequence.from_abstract_repr(
            sequence.to_abstract_repr(skip_validation=True)
        )
        sequence.measure(bases[0])
        return sequence

    def update_sequence_device(self, sequence: Sequence) -> Sequence:
        """Match the Sequence's device with an available one, update it.

        Args:
            sequence: The Sequence to check.

        Returns:
            The Sequence, with the latest version for the targeted Device.
        """
        available_devices = self.fetch_available_devices()
        available_device_names = {
            dev.name: key for key, dev in available_devices.items()
        }
        err_suffix = (
            " Please fetch the latest devices with "
            f"`{type(self).__name__}.fetch_available_devices()` and rebuild "
            "the sequence with one of the options."
        )
        if (name := sequence.device.name) not in available_device_names:
            raise ValueError(
                "The device used in the sequence does not match any "
                "of the devices currently available through the remote "
                "connection." + err_suffix
            )
        if sequence.device != (
            new_device := available_devices[available_device_names[name]]
        ):
            try:
                sequence = sequence.switch_device(new_device, strict=True)
            except Exception as e:
                raise ValueError(
                    "The sequence is not compatible with the latest "
                    "device specs." + err_suffix
                ) from e
            # Validate the sequence with the new device
            RemoteBackend.validate_sequence(sequence, mimic_qpu=True)
        return sequence


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
        self._batch_id: str | None = None

    def run(
        self, job_params: list[JobParams] | None = None, wait: bool = False
    ) -> RemoteResults:
        """Runs the sequence on the remote backend and returns the result.

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
        return self._connection.submit(
            self._sequence,
            job_params=job_params,
            wait=wait,
            **self._submit_kwargs(),
        )

    def _submit_kwargs(self) -> dict[str, Any]:
        """Keyword arguments given to any call to RemoteConnection.submit()."""
        return dict(batch_id=self._batch_id)

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

    def open_batch(self) -> _OpenBatchContextManager:
        """Creates an open batch within a context manager object."""
        if not self._connection.supports_open_batch():
            raise NotImplementedError(
                "Unable to execute open_batch using this remote connection"
            )
        return _OpenBatchContextManager(self)


class _OpenBatchContextManager:
    def __init__(self, backend: RemoteBackend) -> None:
        self.backend = backend

    def __enter__(self) -> _OpenBatchContextManager:
        batch = self.backend._connection.submit(
            self.backend._sequence,
            open=True,
            **self.backend._submit_kwargs(),
        )
        self.backend._batch_id = batch.batch_id
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.backend._batch_id:
            self.backend._connection._close_batch(self.backend._batch_id)
        self.backend._batch_id = None
