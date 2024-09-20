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
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np

import pulser
import pulser.math as pm
from pulser.json.abstract_repr.serializer import abstract_repr
from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized, ParamObj
from pulser.parametrized.decorators import parametrize
from pulser.waveforms import (
    ConstantWaveform,
    CustomWaveform,
    RampWaveform,
    Waveform,
)

if TYPE_CHECKING:
    from pulser.channels.base_channel import Channel

__all__ = ["Pulse"]

PHASE_PRECISION = 1e-6


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
    If defining the pulse's phase modulation is preferred over its frequency
    modulation, use ``Pulse.ArbitraryPhase()``.

    Note:
        We define the ``amplitude`` of a pulse to be its Rabi frequency,
        :math:`\Omega`, in rad/µs. Equivalently, the ``detuning`` is
        :math:`\delta`, also in rad/µs.

    Args:
        amplitude: The pulse amplitude waveform (in rad/µs).
        detuning: The pulse detuning waveform (in rad/µs).
        phase: The pulse phase (in radians).
        post_phase_shift: Optionally lets you add a phase
            shift(in rad) immediately after the end of the pulse. This allows
            for enconding of arbitrary single-qubit gates into a single pulse
            (see ``Sequence.phase_shift()`` for more information).
    """

    amplitude: Waveform = field(init=False)
    detuning: Waveform = field(init=False)
    phase: pm.AbstractArray = field(init=False)
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
        amplitude: Waveform | Parametrized,
        detuning: Waveform | Parametrized,
        phase: float | pm.TensorLike | Parametrized,
        post_phase_shift: float | Parametrized = 0.0,
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
        if np.any(amplitude.samples.as_array(detach=True) < 0):
            raise ValueError(
                "All samples of an amplitude waveform must be "
                "greater than or equal to zero."
            )
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "detuning", detuning)
        assert not isinstance(phase, Parametrized)
        if (phase_ := pm.AbstractArray(phase, dtype=float)).size != 1:
            raise TypeError(f"'phase' must be a single float, not {phase!r}.")
        object.__setattr__(self, "phase", phase_ % (2 * np.pi))
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
        amplitude: Waveform | Parametrized,
        detuning: float | pm.TensorLike | Parametrized,
        phase: float | pm.TensorLike | Parametrized,
        post_phase_shift: float | Parametrized = 0.0,
    ) -> Pulse:
        """Creates a Pulse with an amplitude waveform and a constant detuning.

        Args:
            amplitude: The pulse amplitude waveform (in rad/µs).
            detuning: The detuning value (in rad/µs).
            phase: The pulse phase (in radians).
            post_phase_shift: Optionally lets you add a
                phase shift (in rad) immediately after the end of the pulse.
        """
        detuning_wf = ConstantWaveform(
            cast(Waveform, amplitude).duration, detuning
        )
        return cls(amplitude, detuning_wf, phase, post_phase_shift)

    @classmethod
    @parametrize
    def ConstantAmplitude(
        cls,
        amplitude: float | pm.TensorLike | Parametrized,
        detuning: Waveform | Parametrized,
        phase: float | pm.TensorLike | Parametrized,
        post_phase_shift: float | Parametrized = 0.0,
    ) -> Pulse:
        """Pulse with a constant amplitude and a detuning waveform.

        Args:
            amplitude: The pulse amplitude value (in rad/µs).
            detuning: The pulse detuning waveform (in rad/µs).
            phase: The pulse phase (in radians).
            post_phase_shift: Optionally lets you add a
                phase shift (in rad) immediately after the end of the pulse.
        """
        amplitude_wf = ConstantWaveform(
            cast(Waveform, detuning).duration, amplitude
        )
        return cls(amplitude_wf, detuning, phase, post_phase_shift)

    @classmethod
    def ConstantPulse(
        cls,
        duration: int | Parametrized,
        amplitude: float | pm.TensorLike | Parametrized,
        detuning: float | pm.TensorLike | Parametrized,
        phase: float | pm.TensorLike | Parametrized,
        post_phase_shift: float | Parametrized = 0.0,
    ) -> Pulse:
        """Pulse with a constant amplitude and a constant detuning.

        Args:
            duration: The pulse duration (in ns).
            amplitude: The pulse amplitude value (in rad/µs).
            detuning: The detuning value (in rad/µs).
            phase: The pulse phase (in radians).
            post_phase_shift: Optionally lets you add a
                phase shift (in rad) immediately after the end of the pulse.
        """
        amplitude_wf = ConstantWaveform(duration, amplitude)
        detuning_wf = ConstantWaveform(duration, detuning)
        return cls(amplitude_wf, detuning_wf, phase, post_phase_shift)

    @classmethod
    @parametrize
    def ArbitraryPhase(
        cls,
        amplitude: Waveform | Parametrized,
        phase: Waveform | Parametrized,
        post_phase_shift: float | Parametrized = 0.0,
    ) -> Pulse:
        r"""Pulse with an arbitrary phase waveform.

        Args:
            amplitude: The amplitude waveform (in rad/µs).
            phase: The phase waveform (in rad).
            post_phase_shift: Optionally lets you add a
                phase shift (in rad) immediately after the end of the pulse.

        Note:
            Due to how the Hamiltonian is defined in Pulser, the phase and
            detuning are related by

            .. math:: \phi(t) = \phi_c - \sum_{k=0}^{t} \delta(k)

            where :math:`\phi_c` is the pulse's constant phase offset.
            From a given phase waveform, we extract the phase offset and
            detuning waveform that respect this formula for every sample of
            :math:`\phi(t)` and use these quantities to define the Pulse.

        Warning:
            Except when the phase waveform is a ``ConstantWaveform`` or a
            ``RampWaveform``, the extracted detuning waveform will be a
            ``CustomWaveform``. This makes the Pulse uncapable of automatically
            extending its duration to fit a channel's clock period.

        Returns:
            A regular Pulse, with the phase waveform translated into a
            detuning waveform and a constant phase offset.
        """
        if not isinstance(phase, Waveform):
            raise TypeError(
                f"'phase' must be a waveform, not of type {type(phase)}."
            )
        detuning: Waveform
        if isinstance(phase, ConstantWaveform):
            detuning = ConstantWaveform(phase.duration, 0.0)
        elif isinstance(phase, RampWaveform):
            detuning = ConstantWaveform(phase.duration, -phase._slope * 1e3)
        else:
            detuning_samples = -pm.diff(phase.samples) * 1e3  # rad/ns->rad/µs
            # Use the same value in the first two detuning samples
            detuning = CustomWaveform(
                pm.pad(detuning_samples, (1, 0), mode="edge")
            )
        # Adjust phase_c to incorporate the first detuning sample
        phase_c = phase[0] + detuning[0] * 1e-3
        return cls(amplitude, detuning, phase_c, post_phase_shift)

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

    def get_full_duration(
        self, channel: Channel, in_eom_mode: bool = False
    ) -> int:
        """Calculates the pulse's full duration after output modulation.

        The full duration of a pulse is the total time between the start of
        the input signal and the end of the output signal, as shown in
        the sequence.

        Args:
            channel: The pulse executing the channel.
            in_eom_mode: Whether the pulse is executed in EOM mode.
        """
        if not isinstance(channel, pulser.channels.base_channel.Channel):
            raise TypeError(
                "'channel' must be a channel object instance, not "
                f"{type(channel)}."
            )
        if in_eom_mode and not channel.supports_eom():
            raise ValueError(
                "The given channel does not support EOM mode operation."
            )
        return self.duration + self.fall_time(channel, in_eom_mode)

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
            f"Pulse(Amp={self.amplitude!s} rad/µs, "
            f"Detuning={self.detuning!s} rad/µs, "
            f"Phase={float(self.phase):.3g})"
        )

    def __repr__(self) -> str:
        return (
            f"Pulse(amp={self.amplitude!r} rad/µs, "
            f"detuning={self.detuning!r} rad/µs, "
            f"phase={float(self.phase):.3g}, "
            f"post_phase_shift={float(self.post_phase_shift):.3g})"
        )

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return False

        def check_phase_eq(phase1: float, phase2: float) -> np.bool_:
            # Comparing with an offset ensures we don't fail just because
            # we are very close to the wraping point
            return np.isclose(phase1, phase2, atol=1e-6) or np.isclose(
                (phase1 + 1) % (2 * np.pi),
                (phase2 + 1) % (2 * np.pi),
                atol=PHASE_PRECISION,
            )

        return bool(
            self.amplitude == other.amplitude
            and self.detuning == other.detuning
            and check_phase_eq(float(self.phase), float(other.phase))
            and check_phase_eq(self.post_phase_shift, other.post_phase_shift)
        )


# Replicate __init__'s signature in __new__
functools.update_wrapper(Pulse.__new__, Pulse.__init__)
