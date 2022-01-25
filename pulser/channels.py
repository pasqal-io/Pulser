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
"""The various hardware channel types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, ClassVar, Optional
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft, ifft, fftfreq

# Warnings of adjusted waveform duration appear just once
warnings.filterwarnings("once", "A duration of")


@dataclass(init=True, repr=False, frozen=True)
class Channel:
    """Base class of a hardware channel.

    Not to be initialized itself, but rather through a child class and the
    ``Local`` or ``Global`` classmethods.

    Args:
        name: The name of channel.
        basis: The addressed basis name.
        addressing: "Local" or "Global".
        max_abs_detuning: Maximum possible detuning (in rad/µs), in absolute
            value.
        max_amp: Maximum pulse amplitude (in rad/µs).
        phase_jump_time: Time taken to change the phase between consecutive
            pulses (in ns).
        min_retarget_interval: Minimum time required between the ends of two
            target instructions (in ns).
        fixed_retarget_t: Time taken to change the target (in ns).
        max_targets: How many qubits can be addressed at once by the same beam.
        clock_period: The duration of a clock cycle (in ns). The duration of a
            pulse or delay instruction is enforced to be a multiple of the
            clock cycle.
        min_duration: The shortest duration an instruction can take.
        max_duration: The longest duration an instruction can take.
        mod_bandwith: The modulation bandwith at -3dB (50% redution), in MHz.

    Example:
        To create a channel targeting the 'ground-rydberg' transition globally,
        call ``Rydberg.Global(...)``.
    """

    name: ClassVar[str]
    basis: ClassVar[str]
    addressing: str
    max_abs_detuning: float
    max_amp: float
    phase_jump_time: int = 0
    min_retarget_interval: Optional[int] = None
    fixed_retarget_t: Optional[int] = None
    max_targets: Optional[int] = None
    clock_period: int = 4  # ns
    min_duration: int = 16  # ns
    max_duration: int = 67108864  # ns
    mod_bandwith: Optional[float] = None  # MHz

    @property
    def rise_time(self) -> int:
        """The rise time (in ns).

        Defined as the time taken to go from 10% to 90% output in response to
        a step change in the input.
        """
        if self.mod_bandwith:
            return int(0.48 / self.mod_bandwith * 1e3)
        else:
            return 0

    @classmethod
    def Local(
        cls,
        max_abs_detuning: float,
        max_amp: float,
        phase_jump_time: int = 0,
        min_retarget_interval: int = 220,
        fixed_retarget_t: int = 0,
        max_targets: int = 1,
        **kwargs: int,
    ) -> Channel:
        """Initializes the channel with local addressing.

        Args:
            max_abs_detuning (float): Maximum possible detuning (in rad/µs), in
                absolute value.
            max_amp(float): Maximum pulse amplitude (in rad/µs).
            phase_jump_time (int): Time taken to change the phase between
                consecutive pulses (in ns).
            min_retarget_interval (int): Minimum time required between two
                target instructions (in ns).
            fixed_retarget_t (int): Time taken to change the target (in ns).
            max_targets (int): Maximum number of atoms the channel can target
                simultaneously.
        """
        return cls(
            "Local",
            max_abs_detuning,
            max_amp,
            phase_jump_time,
            min_retarget_interval,
            fixed_retarget_t,
            max_targets,
            **kwargs,
        )

    @classmethod
    def Global(
        cls,
        max_abs_detuning: float,
        max_amp: float,
        phase_jump_time: int = 0,
        **kwargs: int,
    ) -> Channel:
        """Initializes the channel with global addressing.

        Args:
            max_abs_detuning (float): Maximum possible detuning (in rad/µs), in
                absolute value.
            max_amp(float): Maximum pulse amplitude (in rad/µs).
            phase_jump_time (int): Time taken to change the phase between
                consecutive pulses (in ns).
        """
        return cls(
            "Global", max_abs_detuning, max_amp, phase_jump_time, **kwargs
        )

    def validate_duration(self, duration: int) -> int:
        """Validates and adapts the duration of an instruction on this channel.

        Args:
            duration (int): The duration to validate.

        Returns:
            int: The duration, potentially adapted to the channels specs.
        """
        try:
            _duration = int(duration)
        except (TypeError, ValueError):
            raise TypeError(
                "duration needs to be castable to an int but "
                "type %s was provided" % type(duration)
            )

        if duration < self.min_duration:
            raise ValueError(
                "duration has to be at least " + f"{self.min_duration} ns."
            )

        if duration > self.max_duration:
            raise ValueError(
                "duration can be at most " + f"{self.max_duration} ns."
            )

        if duration % self.clock_period != 0:
            _duration += self.clock_period - _duration % self.clock_period
            warnings.warn(
                f"A duration of {duration} ns is not a multiple of "
                f"the channel's clock period ({self.clock_period} "
                f"ns). It was rounded up to {_duration} ns.",
                stacklevel=4,
            )
        return _duration

    def modulate(self, input_samples: ArrayLike) -> np.ndarray:
        """Modulates the input according to the channel's modulation bandwith.

        Args:
            input_samples (ArrayLike): The samples to modulate.

        Returns:
            np.ndarray: The modulated output signal.
        """
        if not self.mod_bandwith:
            warnings.warn(
                f"No modulation bandwith defined for channel '{self}',"
                " 'Channel.modulate()' returns the 'input_samples' unchanged.",
                stacklevel=2,
            )
            return input_samples
        fc = self.mod_bandwith * 1e-3 / np.sqrt(np.log(2))
        samples = np.pad(input_samples, (self.rise_time,))
        freqs = fftfreq(samples.size)
        modulation = np.exp(-(freqs ** 2) / fc ** 2)
        return ifft(fft(samples) * modulation).real

    def calc_modulation_buffer(
        self,
        input_samples: ArrayLike,
        mod_samples: ArrayLike,
        max_allowed_diff: float = 1e-2,
    ) -> tuple[int, int]:
        """Calculates the minimal buffers needed around a modulated waveform.

        Args:
            input_samples (ArrayLike): The input samples.
            mod_samples (ArrayLike): The modulated samples. Must be of size
                ``len(input_samples) + 2 * self.rise_time``.
            max_allowed_diff (float): The maximum allowed difference between
                the input and modulated samples at the end points.

        Returns:
            tuple[int, int]: The minimum buffer times at the left and right of
            the samples, in ns.
        """
        if not self.mod_bandwith:
            return 0, 0

        tr = self.rise_time
        samples = np.pad(input_samples, (tr,))
        diffs = np.abs(samples - mod_samples) <= max_allowed_diff
        try:
            start = tr - np.argwhere(diffs[:tr])[-1][0] - 1
        except IndexError:
            start = tr
        try:
            end = np.argwhere(diffs[-tr:])[0][0]
        except IndexError:
            end = tr

        return start, end

    def __repr__(self) -> str:
        config = (
            f".{self.addressing}(Max Absolute Detuning: "
            f"{self.max_abs_detuning} rad/µs, Max Amplitude: {self.max_amp}"
            f" rad/µs, Phase Jump Time: {self.phase_jump_time} ns"
        )
        if self.addressing == "Local":
            config += (
                f", Minimum retarget time: {self.min_retarget_interval} ns, "
                f"Fixed retarget time: {self.fixed_retarget_t} ns"
            )
            if cast(int, self.max_targets) > 1:
                config += f", Max targets: {self.max_targets}"
        config += f", Basis: '{self.basis}'"
        return self.name + config + ")"


@dataclass(init=True, repr=False, frozen=True)
class Raman(Channel):
    """Raman beam channel.

    Channel targeting the transition between the hyperfine ground states, in
    which the 'digital' basis is encoded. See base class.
    """

    name: ClassVar[str] = "Raman"
    basis: ClassVar[str] = "digital"


@dataclass(init=True, repr=False, frozen=True)
class Rydberg(Channel):
    """Rydberg beam channel.

    Channel targeting the transition between the ground and rydberg states,
    thus enconding the 'ground-rydberg' basis. See base class.
    """

    name: ClassVar[str] = "Rydberg"
    basis: ClassVar[str] = "ground-rydberg"


@dataclass(init=True, repr=False, frozen=True)
class Microwave(Channel):
    """Microwave adressing channel.

    Channel targeting the transition between two rydberg states, thus encoding
    the 'XY' basis. See base class.
    """

    name: ClassVar[str] = "Microwave"
    basis: ClassVar[str] = "XY"
