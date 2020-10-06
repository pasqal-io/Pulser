import warnings
import numpy as np

from . import Waveform, ConstantWaveform


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
            if np.any(self.samples < 0):
                raise ValueError("An amplitude waveform has always to be "
                                 "non-negative.")
            self._amplitude = amplitude
        elif amplitude > 0:
            self._amplitude = ConstantWaveform(self._duration, amplitude)

        else:
            raise ValueError("Negative amplitudes are invalid.")

        if isinstance(detuning, Waveform):
            if detuning.duration != self._duration:
                raise ValueError("The detuning waveform's duration doesn't"
                                 " match the pulses' duration.")
            self._detuning = detuning
        else:
            self._detuning = ConstantWaveform(self._duration, detuning)

        self._phase = float(phase) % (2 * np.pi)
