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

import warnings
from dataclasses import dataclass
from typing import ClassVar, Optional, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import fft, fftfreq, ifft

# Warnings of adjusted waveform duration appear just once
warnings.filterwarnings("once", "A duration of")

# Conversion factor from modulation bandwith to rise time
# For more info, see https://tinyurl.com/bdeumc8k
MODBW_TO_TR = 0.48


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
        mod_bandwidth: The modulation bandwidth at -3dB (50% redution), in MHz.

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
    mod_bandwidth: Optional[float] = None  # MHz

    @property
    def rise_time(self) -> int:
        """The rise time (in ns).

        Defined as the time taken to go from 10% to 90% output in response to
        a step change in the input.
        """
        if self.mod_bandwidth:
            return int(MODBW_TO_TR / self.mod_bandwidth * 1e3)
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
            max_abs_detuning: Maximum possible detuning (in rad/µs), in
                absolute value.
            max_amp: Maximum pulse amplitude (in rad/µs).
            phase_jump_time: Time taken to change the phase between
                consecutive pulses (in ns).
            min_retarget_interval (int): Minimum time required between two
                target instructions (in ns).
            fixed_retarget_t: Time taken to change the target (in ns).
            max_targets: Maximum number of atoms the channel can target
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
            max_abs_detuning: Maximum possible detuning (in rad/µs), in
                absolute value.
            max_amp: Maximum pulse amplitude (in rad/µs).
            phase_jump_time: Time taken to change the phase between
                consecutive pulses (in ns).
        """
        return cls(
            "Global", max_abs_detuning, max_amp, phase_jump_time, **kwargs
        )

    def validate_duration(self, duration: int) -> int:
        """Validates and adapts the duration of an instruction on this channel.

        Args:
            duration: The duration to validate.

        Returns:
            The duration, potentially adapted to the channels specs.
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

    def modulate(
        self, input_samples: np.ndarray, keep_ends: bool = False
    ) -> np.ndarray:
        """Modulates the input according to the channel's modulation bandwidth.

        Args:
            input_samples: The samples to modulate.
            keep_ends: Assume the end values of the samples were kept
                constant (i.e. there is no ramp from zero on the ends).

        Returns:
            The modulated output signal.
        """
        if not self.mod_bandwidth:
            warnings.warn(
                f"No modulation bandwidth defined for channel '{self}',"
                " 'Channel.modulate()' returns the 'input_samples' unchanged.",
                stacklevel=2,
            )
            return input_samples

        # The cutoff frequency (fc) and the modulation transfer function
        # are defined in https://tinyurl.com/bdeumc8k
        fc = self.mod_bandwidth * 1e-3 / np.sqrt(np.log(2))
        if keep_ends:
            samples = np.pad(input_samples, 2 * self.rise_time, mode="edge")
        else:
            samples = np.pad(input_samples, self.rise_time)
        freqs = fftfreq(samples.size)
        modulation = np.exp(-(freqs**2) / fc**2)
        mod_samples = ifft(fft(samples) * modulation).real
        if keep_ends:
            # Cut off the extra ends
            return cast(
                np.ndarray, mod_samples[self.rise_time : -self.rise_time]
            )
        return cast(np.ndarray, mod_samples)

    def calc_modulation_buffer(
        self,
        input_samples: ArrayLike,
        mod_samples: ArrayLike,
        max_allowed_diff: float = 1e-2,
    ) -> tuple[int, int]:
        """Calculates the minimal buffers needed around a modulated waveform.

        Args:
            input_samples: The input samples.
            mod_samples: The modulated samples. Must be of size
                ``len(input_samples) + 2 * self.rise_time``.
            max_allowed_diff: The maximum allowed difference between
                the input and modulated samples at the end points.

        Returns:
            The minimum buffer times at the start and end of
            the samples, in ns.
        """
        if not self.mod_bandwidth:
            raise TypeError(
                f"The channel {self} doesn't have a modulation bandwidth."
            )

        tr = self.rise_time
        samples = np.pad(input_samples, tr)
        diffs = np.abs(samples - mod_samples) <= max_allowed_diff
        try:
            # Finds the last index in the start buffer that's below the max
            # allowed diff. Considers that the waveform could start at the next
            # indice (hence the -1, since we are subtracting from tr)
            start = tr - np.argwhere(diffs[:tr])[-1][0] - 1
        except IndexError:
            start = tr
        try:
            # Finds the first index in the end buffer that's below the max
            # allowed diff. The index value found matches the minimum length
            # for this end buffer.
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
        if self.mod_bandwidth:
            config += f", Modulation Bandwidth: {self.mod_bandwidth} MHz"
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
