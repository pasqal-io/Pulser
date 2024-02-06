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
"""Contains the Pulse class, the building block of a pulse sequence."""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union, cast

import matplotlib.pyplot as plt
import numpy as np

import pulser
from pulser.json.abstract_repr.serializer import abstract_repr
from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized, ParamObj
from pulser.parametrized.decorators import parametrize
from pulser.waveforms import ConstantWaveform, Waveform

if TYPE_CHECKING:
    from pulser.channels.base_channel import Channel

__all__ = ["Pulse"]


@dataclass(init=False, repr=False, frozen=True)
class Pulse:
    r"""A generic pulse.

    In Pulser, a Pulse is a modulation of a frequency signal in amplitude
    and/or frequency, with a specific phase, over a given duration. Amplitude
    and frequency modulations are defined by :class:`Waveform` child classes.
    Frequency modulation is determined by a detuning waveform, which describes
    the shift in frequency from the channel's central frequency over time.
    If either quantity is constant throughout the entire pulse, use the
    ``ConstantDetuning``, ``ConstantAmplitude`` or ``ConstantPulse`` class
    method to create it.

    Note:
        We define the ``amplitude`` of a pulse to be its Rabi frequency,
        :math:`\Omega`, in rad/µs. Equivalently, the ``detuning`` is
        :math:`\delta`, also in rad/µs.

    Args:
        amplitude: The pulse amplitude waveform.
        detuning: The pulse detuning waveform.
        phase: The pulse phase (in radians).
        post_phase_shift: Optionally lets you add a phase
            shift(in rads) immediately after the end of the pulse. This allows
            for enconding of arbitrary single-qubit gates into a single pulse
            (see ``Sequence.phase_shift()`` for more information).
    """

    amplitude: Waveform = field(init=False)
    detuning: Waveform = field(init=False)
    phase: float = field(init=False)
    post_phase_shift: float = field(default=0.0, init=False)

    def __new__(cls, *args, **kwargs):  # type: ignore
        """Creates a Pulse instance or a ParamObj depending on the input."""
        for x in itertools.chain(args, kwargs.values()):
            if isinstance(x, Parametrized):
                return ParamObj(cls, *args, **kwargs)
        else:
            return object.__new__(cls)

    def __init__(
        self,
        amplitude: Union[Waveform, Parametrized],
        detuning: Union[Waveform, Parametrized],
        phase: Union[float, Parametrized],
        post_phase_shift: Union[float, Parametrized] = 0.0,
    ):
        """Initializes a new Pulse."""
        if not (
            isinstance(amplitude, Waveform) and isinstance(detuning, Waveform)
        ):
            raise TypeError("'amplitude' and 'detuning' have to be waveforms.")

        if detuning.duration != amplitude.duration:
            raise ValueError(
                "The duration of detuning and amplitude waveforms must match."
            )
        if np.any(amplitude.samples < 0):
            raise ValueError(
                "All samples of an amplitude waveform must be "
                "greater than or equal to zero."
            )
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "detuning", detuning)
        phase = cast(float, phase)
        object.__setattr__(self, "phase", float(phase) % (2 * np.pi))
        post_phase_shift = cast(float, post_phase_shift)
        object.__setattr__(
            self, "post_phase_shift", float(post_phase_shift) % (2 * np.pi)
        )

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self.amplitude.duration

    @classmethod
    @parametrize
    def ConstantDetuning(
        cls,
        amplitude: Union[Waveform, Parametrized],
        detuning: Union[float, Parametrized],
        phase: Union[float, Parametrized],
        post_phase_shift: Union[float, Parametrized] = 0.0,
    ) -> Pulse:
        """Creates a Pulse with an amplitude waveform and a constant detuning.

        Args:
            amplitude: The pulse amplitude waveform.
            detuning: The detuning value (in rad/µs).
            phase: The pulse phase (in radians).
            post_phase_shift: Optionally lets you add a
                phase shift (in rads) immediately after the end of the pulse.
        """
        detuning_wf = ConstantWaveform(
            cast(Waveform, amplitude).duration, detuning
        )
        return cls(amplitude, detuning_wf, phase, post_phase_shift)

    @classmethod
    @parametrize
    def ConstantAmplitude(
        cls,
        amplitude: Union[float, Parametrized],
        detuning: Union[Waveform, Parametrized],
        phase: Union[float, Parametrized],
        post_phase_shift: Union[float, Parametrized] = 0.0,
    ) -> Pulse:
        """Pulse with a constant amplitude and a detuning waveform.

        Args:
            amplitude: The pulse amplitude value (in rad/µs).
            detuning: The pulse detuning waveform.
            phase: The pulse phase (in radians).
            post_phase_shift: Optionally lets you add a
                phase shift (in rads) immediately after the end of the pulse.
        """
        amplitude_wf = ConstantWaveform(
            cast(Waveform, detuning).duration, amplitude
        )
        return cls(amplitude_wf, detuning, phase, post_phase_shift)

    @classmethod
    def ConstantPulse(
        cls,
        duration: Union[int, Parametrized],
        amplitude: Union[float, Parametrized],
        detuning: Union[float, Parametrized],
        phase: Union[float, Parametrized],
        post_phase_shift: Union[float, Parametrized] = 0.0,
    ) -> Pulse:
        """Pulse with a constant amplitude and a constant detuning.

        Args:
            duration: The pulse duration (in ns).
            amplitude: The pulse amplitude value (in rad/µs).
            detuning: The detuning value (in rad/µs).
            phase: The pulse phase (in radians).
            post_phase_shift: Optionally lets you add a
                phase shift (in rads) immediately after the end of the pulse.
        """
        amplitude_wf = ConstantWaveform(duration, amplitude)
        detuning_wf = ConstantWaveform(duration, detuning)
        return cls(amplitude_wf, detuning_wf, phase, post_phase_shift)

    def draw(self) -> None:
        """Draws the pulse's amplitude and frequency waveforms."""
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        self.amplitude._plot(ax1, r"$\Omega$ (rad/µs)", color="darkgreen")
        self.detuning._plot(ax2, r"$\delta$ (rad/µs)", color="indigo")

        fig.tight_layout()
        plt.show()

    def fall_time(self, channel: Channel, in_eom_mode: bool = False) -> int:
        """Calculates the extra time needed to ramp down to zero."""
        aligned_start_extra_time = (
            channel.rise_time
            if not in_eom_mode
            else cast(
                pulser.channels.eom.BaseEOM, channel.eom_config
            ).rise_time
        )
        end_extra_time = max(
            self.amplitude.modulation_buffers(channel, eom=in_eom_mode)[1],
            self.detuning.modulation_buffers(channel, eom=in_eom_mode)[1],
        )
        return aligned_start_extra_time + end_extra_time

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(
            self,
            self.amplitude,
            self.detuning,
            self.phase,
            post_phase_shift=self.post_phase_shift,
        )

    def _to_abstract_repr(self) -> dict[str, Any]:
        return abstract_repr(
            "Pulse",
            self.amplitude,
            self.detuning,
            self.phase,
            post_phase_shift=self.post_phase_shift,
        )

    def __str__(self) -> str:
        return (
            f"Pulse(Amp={self.amplitude!s}, Detuning={self.detuning!s}, "
            f"Phase={self.phase:.3g})"
        )

    def __repr__(self) -> str:
        return (
            f"Pulse(amp={self.amplitude!r}, detuning={self.detuning!r}, "
            + f"phase={self.phase:.3g}, "
            + f"post_phase_shift={self.post_phase_shift:.3g})"
        )


# Replicate __init__'s signature in __new__
functools.update_wrapper(Pulse.__new__, Pulse.__init__)
