import numpy as np
import matplotlib.pyplot as plt

from waveforms import Waveform, ConstantWaveform


class Pulse:
    """A generic pulse.

    Args:
        amplitude (Waveform): The pulse amplitude waveform.
        detuning (Waveform): The pulse detuning waveform.
        phase (float): The pulse phase (in radians).
    """

    def __init__(self, amplitude, detuning, phase):
        if detuning.duration != amplitude.duration:
            raise ValueError("The detuning waveform's duration doesn't"
                             " match the pulses' duration.")
        self.duration = amplitude.duration
        if np.any(amplitude.samples < 0):
            raise ValueError("An amplitude waveform has always to be "
                             "non-negative.")
        self.amplitude = amplitude
        self.detuning = detuning
        self.phase = float(phase) % (2 * np.pi)

    @classmethod
    def ConstantDetuning(cls, amplitude, detuning, phase):
        """Pulse with a constant amplitude and a detuning waveform"""

        detuning_wf = ConstantWaveform(amplitude.duration, detuning)
        return cls(amplitude, detuning_wf, phase)

    @classmethod
    def ConstantAmplitude(cls, amplitude, detuning, phase):
        """Pulse with an amplitude waveform and a constant detuning"""

        amplitude_wf = ConstantWaveform(detuning.duration, amplitude)
        return cls(amplitude_wf, detuning, phase)

    @classmethod
    def ConstantPulse(cls, amplitude, detuning, phase, duration):
        """Pulse with a constant amplitude and a constant detuning"""

        detuning_wf = ConstantWaveform(duration, detuning)
        amplitude_wf = ConstantWaveform(duration, amplitude)
        return cls(amplitude_wf, detuning_wf, phase)

    def draw(self):
        """Draw the pulse's amplitude and frequency waveforms."""

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        self.amplitude._plot(ax1, "Amplitude (MHz)", color="darkgreen")
        self.detuning._plot(ax2, "Detuning (MHz)", color="indigo")

        plt.show()

    def __str__(self):
        return "Pulse(Amp={!s}, Detuning={!s}, Phase={})".format(
            self.amplitude, self.detuning, self.phase)
