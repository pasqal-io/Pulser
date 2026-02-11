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

from pulser import Sequence
from pulser.backend.config import BackendConfig
from pulser.backend.remote import (
    JobParams,
    RemoteBackend,
    RemoteConnection,
    RemoteResults,
)


class QPUBackend(RemoteBackend):
    """Backend for sequence execution on a QPU.

    Args:
        sequence: A Sequence or a list of Sequences to execute on a
            backend accessible via a remote connection.
        connection: The remote connection through which the jobs
            are executed.
        config: An optional backend configuration. For a QPU, it can be used
            to define a `default_num_shots`.
    """

    def __init__(
        self,
        sequence: Sequence,
        connection: RemoteConnection,
        *,
        config: BackendConfig | None = None,
    ) -> None:
        """Starts a new QPU backend instance."""
        super().__init__(sequence, connection, mimic_qpu=True, config=config)

    def run(
        self, job_params: list[JobParams] | None = None, wait: bool = False
    ) -> RemoteResults:
        """Runs the sequence on the remote QPU and returns the result.

        Args:
            job_params: A list of dictionaries with the parameters to execute
                each job. If not given, the backend will attempt to run one
                job with 'BackendConfig.default_num_shots'.
                If the sequence is parametrized, the values for all the
                variables necessary to build the sequence must be given in
                its own dictionary, for each job, under the 'variables' field.
                Each dictionary may contain a custom 'runs' field specifying
                the number of shots to take of the same sequence; if not given,
                'BackendConfig.default_num_shots' is used when available.
            wait: Whether to wait until the results of the jobs become
                available.  If set to False, the call is non-blocking and the
                obtained results' status can be checked using their `status`
                property.

        Returns:
            The results, which can be accessed once all sequences have been
            successfully executed.
        """
        if self._config.default_num_shots is not None:
            # Use default_num_shots when runs is not defined
            if job_params is None:
                job_params = [{"runs": self._config.default_num_shots}]
            else:
                # Make sure job_params format is correct first
                self._type_check_job_params(job_params)
                job_params = [
                    # If runs is not defined in d, uses default_num_shots
                    {"runs": self._config.default_num_shots} | d
                    for d in job_params
                ]
        # super().run() includes call to `validate_job_params` since
        # _mimic_qpu = True
        return super().run(job_params, wait)
