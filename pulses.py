import numpy as np
import matplotlib.pyplot as plt

from waveforms import Waveform, ConstantWaveform
from utils import validate_duration


class Pulse:
    """A generic pulse.

    Args:
        duration (int): The pulse duration (in ns).
        amplitude (float, Waveform): The pulse amplitude. Can be a float (MHz),
            in which case it's kept constant throught the entire pulse, or a
            waveform.
        detuning (float, Waveform): The pulse detuning. Can be a float (MHz),
            in which case it's kept constant throught the entire pulse, or a
            waveform.
        phase (float): The pulse phase (in radians).
    """

    def __init__(self, duration, amplitude, detuning, phase):

        self.duration = validate_duration(duration)

        if isinstance(amplitude, Waveform):
            if amplitude.duration != self.duration:
                raise ValueError("The amplitude waveform's duration doesn't"
                                 " match the pulses' duration.")
            if np.any(amplitude.samples < 0):
                raise ValueError("An amplitude waveform has always to be "
                                 "non-negative.")
            self.amplitude = amplitude
        elif amplitude > 0:
            self.amplitude = ConstantWaveform(self.duration, amplitude)

        else:
            raise ValueError("Negative amplitudes are invalid.")

        if isinstance(detuning, Waveform):
            if detuning.duration != self.duration:
                raise ValueError("The detuning waveform's duration doesn't"
                                 " match the pulses' duration.")
            self.detuning = detuning
        else:
            self.detuning = ConstantWaveform(self.duration, detuning)

        self.phase = float(phase) % (2 * np.pi)

    def draw(self):
        """Draw the pulse's amplitude and frequency waveforms."""

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        self.amplitude._plot(ax1, "Amplitude (MHz)", color="darkgreen")
        self.detuning._plot(ax2, "Detuning (MHz)", color="indigo")

        fig.tight_layout()
        plt.show()

    def __str__(self):
        return "Pulse(Amp={!s}, Detuning={!s}, Phase={})".format(
                self.amplitude, self.detuning, self.phase)
