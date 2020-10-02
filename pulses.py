import warnings
import numpy as np

from . import Waveform, ConstantWaveform


class Pulse:
    """A generic pulse.

    Args:
        duration (int): The pulse duration (in ns).
        amplitude (float, Waveform): The pulse amplitude. Can be a float, in
            which case it's kept constant throught the entire pulse, or a
            waveform.
        frequency (float, Waveform): The pulse frequency. Can be a float, in
            which case it's kept constant throught the entire pulse, or a
            waveform.
        phase (float): The pulse phase.
    """

    def __init__(self, duration, amplitude, frequency, phase):

        try:
            _duration = int(duration)
        except (TypeError, ValueError):
            raise TypeError("duration needs to be castable to an int but "
                            "type %s was provided" % type(duration))
        if duration >= 0:
            self._duration = _duration
        else:
            raise ValueError("duration has to be castable to a non-negative "
                             "integer.")
        if duration % 1 != 0:
            warnings.warn("The given duration is below the machine's precision"
                          " of 1 ns time steps. It was rounded down to the"
                          " nearest integer.")

        if isinstance(amplitude, Waveform):
            if amplitude.duration != self._duration:
                raise ValueError("The amplitude waveform's duration doesn't"
                                 " match the pulses' duration.")
            self._amplitude = amplitude
        else:
            self._amplitude = ConstantWaveform(self._duration, amplitude)

        if isinstance(frequency, Waveform):
            if frequency.duration != self._duration:
                raise ValueError("The frequency waveform's duration doesn't"
                                 " match the pulses' duration.")
            self._frequency = frequency
        else:
            self._frequency = ConstantWaveform(self._duration, frequency)

        self._phase = float(phase) % (2 * np.pi)
