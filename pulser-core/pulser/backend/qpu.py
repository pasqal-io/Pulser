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

from typing import cast

from pulser import Sequence
from pulser.backend.remote import JobParams, RemoteBackend, RemoteResults
from pulser.devices import Device


class QPUBackend(RemoteBackend):
    """Backend for sequence execution on a QPU."""

    def run(
        self, job_params: list[JobParams] | None = None, wait: bool = False
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

        Returns:
            The results, which can be accessed once all sequences have been
            successfully executed.
        """
        suffix = " when executing a sequence on a real QPU."
        if not job_params:
            raise ValueError("'job_params' must be specified" + suffix)

        max_runs = self._sequence.device.max_runs
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
        results = self._connection.submit(
            self._sequence, job_params=job_params, wait=wait
        )
        return cast(RemoteResults, results)

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
