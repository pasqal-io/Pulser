import numpy as np
import warnings
from abc import ABC, abstractmethod


class Waveform(ABC):
    """The abstract class for a pulse's waveform."""

    def __init__(self, duration):
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

    @property
    @abstractmethod
    def duration(self):
        """The duration of the pulse (in ns)."""
        pass

    @abstractmethod
    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        pass

    def __add__(self, other):
        if not isinstance(other, Waveform):
            raise TypeError("Can't add a waveform to an object of "
                            "type {}.".format(type(other)))

        if self.duration != other.duration:
            raise ValueError("The sum of two waveforms of different durations "
                             "is ambiguous.")

        new_samples = self.samples + other.samples
        return ArbitraryWaveform(new_samples)


class ArbitraryWaveform(Waveform):
    """An arbitrary waveform.

    Args:
        samples (array_like): The modulation values at each time step.
    """

    def __init__(self, samples):
        samples_arr = np.array(samples)
        if np.any(samples_arr < 0):
            raise ValueError("All values in a sample must be non-negative.")
        self._samples = samples_arr

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return len(self._samples)

    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        return self._samples


class ConstantWaveform(Waveform):
    """A waveform of constant value.

    Args:
        duration: The waveform duration (in ns).
        value: The modulation value.
    """

    def __init__(self, duration, value):
        if value < 0:
            raise ValueError("Can't accept negative modulation values.")

        super().__init__(duration)
        self._value = float(value)

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return self._duration

    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        return np.full(self.duration, self._value)


class RampWaveform(Waveform):
    """A linear ramp waveform.

    Args:
        duration: The waveform duration (in ns).
        start: The initial value.
        stop: The final value.
    """

    def __init__(self, duration, start, stop):
        if start < 0 or stop < 0:
            raise ValueError("Can't accept negative modulation values.")

        super().__init__(duration)
        self._start = float(start)
        self._stop = float(stop)

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return self._duration

    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        return np.linspace(self._start, self._stop, num=self._duration)


class GaussianWaveform(Waveform):
    """A Gaussian-shaped waveform.

    Args:
        duration: The waveform duration (in ns).
        max_val: The maximum value.
        sigma: The standard deviation of the gaussian shape (in ns).
    """

    def __init__(self, duration, max_val, sigma):
        # NOTE: This stores max_val in self._value
        ConstantWaveform.__init__(self, duration, max_val)
        if sigma <= 0:
            raise ValueError("The standard deviation has to be positive.")
        self._sigma = sigma

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return self._duration

    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        # Ensures intervals are always symmetrical
        ts = np.arange(self.duration, dtype=float) - (self.duration - 1) * 0.5
        return self._value * np.exp(-0.5 * (ts / self._sigma)**2)
