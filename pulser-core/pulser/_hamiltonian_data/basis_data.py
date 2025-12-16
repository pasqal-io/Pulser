# Copyright 2025 Pulser Development Team
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
"""Definition of the basis used by the Sequence."""
from dataclasses import dataclass
from typing import Literal

from pulser.channels.base_channel import States


@dataclass(frozen=True)
class BasisData:
    """Some data about the basis used by the simulation."""

    dim: int
    basis_name: str
    interaction_type: Literal["XY", "ising"]
    eigenbasis: list[States]
