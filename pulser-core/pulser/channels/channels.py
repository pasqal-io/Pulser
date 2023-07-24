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
"""The Channel subclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pulser.channels.base_channel import Channel, Literal
from pulser.channels.eom import RydbergEOM


@dataclass(init=True, repr=False, frozen=True)
class Raman(Channel):
    """Raman beam channel.

    Channel targeting the transition between the hyperfine ground states, in
    which the 'digital' basis is encoded. See base class.
    """

    @property
    def basis(self) -> Literal["digital"]:
        """The addressed basis name."""
        return "digital"


@dataclass(init=True, repr=False, frozen=True)
class Rydberg(Channel):
    """Rydberg beam channel.

    Channel targeting the transition between the ground and rydberg states,
    thus encoding the 'ground-rydberg' basis. See base class.
    """

    eom_config: Optional[RydbergEOM] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.eom_config is not None and not isinstance(
            self.eom_config, RydbergEOM
        ):
            raise TypeError(
                "When defined, 'eom_config' must be a valid 'RydbergEOM'"
                f" instance, not {type(self.eom_config)}."
            )

    @property
    def basis(self) -> Literal["ground-rydberg"]:
        """The addressed basis name."""
        return "ground-rydberg"


@dataclass(init=True, repr=False, frozen=True)
class Microwave(Channel):
    """Microwave adressing channel.

    Channel targeting the transition between two rydberg states, thus encoding
    the 'XY' basis. See base class.
    """

    @property
    def basis(self) -> Literal["XY"]:
        """The addressed basis name."""
        return "XY"

    def default_id(self) -> str:
        """Generates the default ID for indexing this channel in a Device."""
        return f"mw_{self.addressing.lower()}"
