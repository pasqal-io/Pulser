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
"""Base classes for backend execution through the cloud."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pulser.backend.abc import Backend, Results
from pulser.sequence import Sequence

JobId = str


class CloudConnection(ABC):
    """The abstract base class for a cloud connection."""

    @abstractmethod
    def validate_sequence(self, sequence: Sequence) -> None:
        """Validates a sequence prior to submission."""
        if not isinstance(sequence, Sequence):
            raise TypeError(
                "'sequence' should be a `Sequence` instance"
                f", not {type(sequence)}."
            )
        # Checks specific to each cloud provider should be added

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


class CloudBackend(Backend):
    """A backend for sequence execution through a cloud connection.

    Args:
        sequence: A Sequence or a list of Sequences to execute on a
            backed accessible via a cloud connection.
        cloud_connection: The cloud connection through which the jobs
            are executed.
    """

    def __init__(
        self,
        sequence: Sequence | list[Sequence],
        cloud_connection: CloudConnection,
    ) -> None:
        """Starts a new cloud backend instance."""
        self.cloud_connection = cloud_connection
        self.sequence = sequence
        # Extra arguments to add to pass to CloudConnection.submit()
        self._submit_kwargs: dict[str, Any] = {}
        sequence_list = (
            [sequence] if isinstance(sequence, Sequence) else sequence
        )
        for seq in sequence_list:
            self.cloud_connection.validate_sequence(seq)

    def run(self) -> Results | list[Results]:
        """Runs on the backend via the cloud and returns the result."""
        job_id = self.cloud_connection.submit(
            self.sequence, **self._submit_kwargs
        )
        return self.cloud_connection.fetch_result(job_id)
