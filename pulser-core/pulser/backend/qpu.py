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
from pulser.backend.remote import RemoteBackend, RemoteConnection


class QPUBackend(RemoteBackend):
    """Backend for sequence execution on a QPU.

    Args:
        sequence: A Sequence or a list of Sequences to execute on a
            backend accessible via a remote connection.
        connection: The remote connection through which the jobs
            are executed.
    """

    def __init__(
        self, connection: RemoteConnection, sequence: Sequence | None = None
    ) -> None:
        """Starts a new QPU backend instance."""
        super().__init__(connection, sequence, mimic_qpu=True)
