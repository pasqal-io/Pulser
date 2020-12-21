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

from dataclasses import dataclass
from typing import ClassVar


@dataclass(init=False, repr=False, frozen=True)
class Channel:
    """Base class of a hardware channel."""

    name: ClassVar[str]
    basis: ClassVar[str]
    addressing: str
    max_abs_detuning: float
    max_amp: float
    retarget_time: int = None
    max_targets: int = 1

    @classmethod
    def Local(cls, max_abs_detuning, max_amp, retarget_time, max_targets=1):
        """Initializes the channel with local addressing.

        Args:
            max_abs_detuning (float): Maximum possible detuning (in MHz), in
            absolute value.
            max_amp(float): Maximum pulse amplitude (in MHz).
            retarget_time (int): Time to change the target (in ns).

        Keyword Args:
            max_targets (int, default=1): (For local channels only) How
                many qubits can be addressed at once by the same beam."""

        return cls('Local', max_abs_detuning, max_amp, max_targets=max_targets,
                   retarget_time=retarget_time)

    @classmethod
    def Global(cls, max_abs_detuning, max_amp):
        """Initializes the channel with global addressing.

        Args:
            max_abs_detuning (tuple): Maximum possible detuning (in MHz), in
            absolute value.
            max_amp(tuple): Maximum pulse amplitude (in MHz)."""

        return cls('Global', max_abs_detuning, max_amp)

    def __repr__(self):
        s = ".{}(Max Absolute Detuning: {} MHz, Max Amplitude: {} MHz"
        config = s.format(self.addressing, self.max_abs_detuning, self.max_amp)
        if self.addressing == 'Local':
            config += f", Target time: {self.retarget_time} ns"
            if self.max_targets > 1:
                config += f", Max targets: {self.max_targets}"
        config += f", Basis: '{self.basis}'"
        return self.name + config + ")"


@dataclass(init=True, repr=False, frozen=True)
class Raman(Channel):
    """Raman beam channel.

    Channel targeting the transition between the hyperfine ground states, in
    which the 'digital' basis is encoded. See base class.
    """
    name: ClassVar[str] = 'Raman'
    basis: ClassVar[str] = 'digital'


@dataclass(init=True, repr=False, frozen=True)
class Rydberg(Channel):
    """Rydberg beam channel.

    Channel targeting the transition between the ground and rydberg states,
    thus enconding the 'ground-rydberg' basis. See base class.
    """
    name: ClassVar[str] = 'Rydberg'
    basis: ClassVar[str] = 'ground-rydberg'
