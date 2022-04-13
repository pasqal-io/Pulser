# Copyright 2020 Pulser Development Team
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
"""Contains the abstract base class for parametrized objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pulser.parametrized import Variable  # pragma: no cover


class Parametrized(ABC):
    """Abstract base class for a parametrized object."""

    @property
    @abstractmethod
    def variables(self) -> dict[str, Variable]:
        """All the variables involved with this object."""
        pass

    @abstractmethod
    def build(self) -> Any:
        """Builds the object."""
        pass

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        """Serializes the object in a dictionary."""
        pass
