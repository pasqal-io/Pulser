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

from dataclasses import dataclass, field
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


@dataclass(init=True, repr=False, frozen=True)
class RydbergError(Channel):
    """A channel to simulate error between ground rydberg and error state.

    Channel targeting the transition between the ground rydberg state and an
    error state. This channel does not allow adding any pulse and is not
    related to any beam but is useful for simulating noise with collapse
    operators.
    """

    addressing: Literal["Global"] = field(default="Global", init=False)
    max_abs_detuning: Optional[float] = field(default=None, init=False)
    max_amp: float = field(default=0, init=False)
    min_retarget_interval: Optional[int] = field(default=None, init=False)
    fixed_retarget_t: Optional[int] = field(default=None, init=False)
    max_targets: Optional[int] = field(default=None, init=False)
    clock_period: int = field(default=1, init=False)  # ns
    min_duration: int = field(default=1, init=False)  # ns
    max_duration: int = field(default=int(1e8), init=False)  # ns
    min_avg_amp: int = field(default=0, init=False)
    mod_bandwidth: Optional[float] = field(default=None, init=False)  # MHz

    @property
    def basis(self) -> Literal["error"]:
        """The addressed basis name."""
        return "error"
