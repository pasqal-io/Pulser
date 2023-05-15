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
"""Base class for the backend interface."""
from __future__ import annotations

import typing
from abc import ABC, abstractmethod

from pulser.result import Result
from pulser.sequence import Sequence

Results = typing.Sequence[Result]


class Backend(ABC):
    """The backend abstract base class."""

    def __init__(self, sequence: Sequence) -> None:
        """Starts a new backend instance."""
        self.validate_sequence(sequence)
        self._sequence = sequence

    @abstractmethod
    def run(self) -> Results | typing.Sequence[Results]:
        """Executes the sequence on the backend."""
        pass

    def validate_sequence(self, sequence: Sequence) -> None:
        """Validates a sequence prior to submission."""
        if not isinstance(sequence, Sequence):
            raise TypeError(
                "'sequence' should be a `Sequence` instance"
                f", not {type(sequence)}."
            )
