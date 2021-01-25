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

import matplotlib.pyplot as plt
import numpy as np

from pulser.waveforms import Waveform, ConstantWaveform, RampWaveform


class Pulse:
    """A generic pulse.

    In Pulser, a Pulse is a modulation of a frequency signal in amplitude
    and/or frequency, with a specific phase, over a given duration. Amplitude
    and frequency modulation are defined by objects of type pulser.Waveform.
    Frequency modulation is determined by a detuning waveform, which describes
    shift in frequency from the channel's central frequency over time.

    Args:
        amplitude (Waveform): The pulse amplitude waveform.
        detuning (Waveform): The pulse detuning waveform.
        phase (float): The pulse phase (in radians).

    Keyword Args:
        post_phase_shift (default=0): Optionally lets you add a phase shift
            (in rads) immediately after the end of the pulse. This allows for
            enconding of arbitrary single-qubit gates into a single pulse
            (see Sequence.phase_shift() for more information).
    """

    def __init__(self, amplitude, detuning, phase, post_phase_shift=0):
        """Initializes a new Pulse."""

        if not (isinstance(amplitude, Waveform) and
                isinstance(detuning, Waveform)):
            raise TypeError("'amplitude' and 'detuning' have to be waveforms.")

        if detuning.duration != amplitude.duration:
            raise ValueError(
                "Detuning and amplitude waveforms' durations don't match.")
        self.duration = amplitude.duration
        if np.any(amplitude.samples < 0):
            raise ValueError("An amplitude waveform has always to be "
                             "non-negative.")
        self.amplitude = amplitude
        self.detuning = detuning
        self.phase = float(phase) % (2 * np.pi)
        self.post_phase_shift = float(post_phase_shift) % (2 * np.pi)

    @classmethod
    def ConstantDetuning(cls, amplitude, detuning, phase, post_phase_shift=0):
        """Pulse with an amplitude waveform and a constant detuning.

        Args:
            amplitude (Waveform): The pulse amplitude waveform.
            detuning (float): The detuning value (in MHz).
            phase (float): The pulse phase (in radians).
        """

        detuning_wf = ConstantWaveform(amplitude.duration, detuning)
        return cls(amplitude, detuning_wf, phase, post_phase_shift)

    @classmethod
    def ConstantAmplitude(cls, amplitude, detuning, phase, post_phase_shift=0):
        """Pulse with a constant amplitude and a detuning waveform.

        Args:
            amplitude (float): The pulse amplitude value (in MHz).
            detuning (Waveform): The pulse detuning waveform.
            phase (float): The pulse phase (in radians).
        """

        amplitude_wf = ConstantWaveform(detuning.duration, amplitude)
        return cls(amplitude_wf, detuning, phase, post_phase_shift)

    @classmethod
    def ConstantPulse(cls, duration, amplitude, detuning, phase,
                      post_phase_shift=0):
        """Pulse with a constant amplitude and a constant detuning.

        Args:
            duration (int): The pulse duration (in multiples of 4 ns).
            amplitude (float): The pulse amplitude value (in MHz).
            detuning (float): The detuning value (in MHz).
            phase (float): The pulse phase (in radians).
        """

        amplitude_wf = ConstantWaveform(duration, amplitude)
        detuning_wf = ConstantWaveform(duration, detuning)
        return cls(amplitude_wf, detuning_wf, phase, post_phase_shift)

    def draw(self):
        """Draws the pulse's amplitude and frequency waveforms."""

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        self.amplitude._plot(ax1, "Amplitude (MHz)", color="darkgreen")
        self.detuning._plot(ax2, "Detuning (MHz)", color="indigo")

        fig.tight_layout()
        plt.show()

    def __str__(self):
        return "Pulse(Amp={!s}, Detuning={!s}, Phase={:.3g})".format(
            self.amplitude, self.detuning, self.phase)

    def __repr__(self):
        return (f"Pulse(amp={self.amplitude!r}, detuning={self.detuning!r}, " +
                f"phase={self.phase:.3g}, " +
                f"post_phase_shift={self.post_phase_shift:.3g})")

    def _chirps(self, start_det=None):
        """Describes the detuning waveform as a series of linear chirps.

        Turns the detuning into segments of linear frequency chirps, with a
        length of 4ns. If the chirp rate is constant through the entire pulse,
        returns a single value.

        Returns:
            float, np.ndarray: The chirp rate, or an array of chirp rates,
                describing the detuning waveform (in MHz/ns).
        """
        clock_t = 4  # ns
        if isinstance(self.detuning, ConstantWaveform):
            if start_det is None or start_det == self.detuning.first_value:
                return 0
            else:
                chirps = np.zeros(self.duration // clock_t, dtype=float)
                second_value = self.detuning.first_value

        elif isinstance(self.detuning, RampWaveform):
            if start_det is None or start_det == self.detuning.first_value:
                return self.detuning.slope
            else:
                chirps = np.full(self.duration // clock_t, self.detuning.slope)
                second_value = self.detuning.slope * clock_t
        else:
            samples = self.detuning.samples
            chirps = samples[clock_t-1::clock_t] - samples[:-clock_t+1:clock_t]
            chirps = chirps / clock_t
            if start_det is None or start_det == samples[0]:
                return chirps
            second_value = samples[clock_t-1]

        chirps[0] = (second_value - start_det) / clock_t
        return chirps
