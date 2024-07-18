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
"""Defines the backend class for QPU execution."""
from __future__ import annotations

from types import TracebackType
from typing import Self, Type, cast

from pulser import Sequence
from pulser.backend.remote import (
    JobParams,
    RemoteBackend,
<<<<<<< HEAD
    RemoteConnection,
    RemoteResults,
)
=======
    RemoteResults,
    SubmissionStatus,
)
from pulser.devices import Device
>>>>>>> 2970d3d (add open submission ability for jobs on pulser-pasqal)


class QPUBackend(RemoteBackend):
    """Backend for sequence execution on a QPU.

    Args:
        sequence: A Sequence or a list of Sequences to execute on a
            backend accessible via a remote connection.
        connection: The remote connection through which the jobs
            are executed.
    """

    def __init__(
        self, sequence: Sequence, connection: RemoteConnection
    ) -> None:
        """Starts a new QPU backend instance."""
        super().__init__(sequence, connection, mimic_qpu=True)

    def run(
        self,
        job_params: list[JobParams] | None = None,
        wait: bool = False,
    ) -> RemoteResults:
        """Runs the sequence on the remote QPU and returns the result.

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
            open_submission: A flag indicating whether or not the submission
                should be for a single job or accept future jobs before
                closing.

        Returns:
            The results, which can be accessed once all sequences have been
            successfully executed.
        """
<<<<<<< HEAD
        self.validate_job_params(
            job_params or [], self._sequence.device.max_runs
        )
        results = self._connection.submit(
            self._sequence,
            job_params=job_params,
            wait=wait,
            batch_id=self.batch_id,
        )
        return cast(RemoteResults, results)

<<<<<<< HEAD
    @staticmethod
    def validate_job_params(
        job_params: list[JobParams], max_runs: int | None
    ) -> None:
        """Validates a list of job parameters prior to submission."""
=======
        # if submission is already open, we don't need an ID for a preexisting
        # one
        if open_submission and submission_id:
            raise ValueError(
                """Open submission can only be used for a new submission.
                                Don't provide a preexisting submission_id"""
            )

>>>>>>> 2970d3d (add open submission ability for jobs on pulser-pasqal)
=======
>>>>>>> 3f3a07c (change to context manager interface for open batches)
        suffix = " when executing a sequence on a real QPU."
        if not job_params:
            raise ValueError("'job_params' must be specified" + suffix)
        RemoteBackend._type_check_job_params(job_params)
        for j in job_params:
            if "runs" not in j:
                raise ValueError(
                    "All elements of 'job_params' must specify 'runs'" + suffix
                )
            if max_runs is not None and j["runs"] > max_runs:
                raise ValueError(
                    "All 'runs' must be below the maximum allowed by the "
                    f"device ({max_runs})" + suffix
                )
<<<<<<< HEAD
=======
        results = self._connection.submit(
            self._sequence,
            job_params=job_params,
            wait=wait,
            submission_id=self.submission_id,
        )
        return cast(RemoteResults, results)

    def open_submission(self, submission_id: str | None = None) -> Self:
        """
        If provided a submission_id, we can add new jobs to a batch within
        the scope of the context manager otherwise an new open submission
        is created that you can continuously add jobs to.

        Args:
            submission_id: An optional unique identifier for an already
                 open submission

        Returns:
            self reference for current object
        """
        if submission_id:
            self.submission_id = submission_id
        else:
            submission = cast(
                RemoteResults, self._connection.submit(self._sequence)
            )
            self.submission_id = submission.submission_id
        return self

    def __enter__(self) -> Self:
        # open returns an instance of self to use open_submission within a
        # context manager
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # exiting the context of an open submission will
        # make a remote call to close it.
        if self.submission_id:
            self.close_submission(self.submission_id)

    def validate_sequence(self, sequence: Sequence) -> None:
        """Validates a sequence prior to submission.

        Args:
            sequence: The sequence to validate.
        """
        super().validate_sequence(sequence)
        if not isinstance(sequence.device, Device):
            raise TypeError(
                "To be sent to a QPU, the device of the sequence "
                "must be a real device, instance of 'Device'."
            )

    def close_submission(self, submission_id: str) -> SubmissionStatus | None:
        """
        Closes a submission to prevent any further jobs being added.

        Args:
            submission_id: unique identifier as a string return when
                creating a submission.

        Returns:
            SubmissionStatus representing the new stats of the submission.
        """
        return self._connection.close_submission(submission_id)
>>>>>>> 2970d3d (add open submission ability for jobs on pulser-pasqal)
