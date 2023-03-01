# Copyright 2022 Pulser Development Team
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
"""Parameters to build a sequence sent to the PASQAL cloud plaftorm."""
from __future__ import annotations

import dataclasses
from typing import Dict, Mapping, Optional, Union

from numpy.typing import ArrayLike

from pulser.register import QubitId

JobVariablesDict = Dict[str, Union[ArrayLike, Optional[Mapping[QubitId, int]]]]


class JobVariables:
    """Variables to build the sequence."""

    def __init__(
        self,
        qubits: Optional[Mapping[QubitId, int]] = None,
        **vars: Union[ArrayLike, float, int],
    ):
        """Initializes the JobVariables class.

        Args:
            qubits: A mapping between qubit IDs and trap IDs used to define
                the register. Must only be provided when the sequence is
                initialized with a MappableRegister.
            vars: The values for all the variables declared in this Sequence
                instance, indexed by the name given upon declaration. Check
                ``Sequence.declared_variables`` to see all the variables.
        """
        self._qubits = qubits
        self._vars = vars

    def get_dict(self) -> JobVariablesDict:
        """Creates a dictionary used by the Sequence building and the cloud."""
        return {"qubits": self._qubits, **self._vars}


@dataclasses.dataclass
class JobParameters:
    """Parameters representing a job to build the sequence."""

    runs: int
    variables: JobVariables

    def get_dict(self) -> dict[str, Union[int, JobVariablesDict]]:
        """Creates a dictionary to send to the cloud."""
        return dict(
            runs=self.runs,
            variables=self.variables.get_dict(),
        )
