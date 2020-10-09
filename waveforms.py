import numpy as np
from abc import ABC, abstractmethod
from utils import validate_duration


class Waveform(ABC):
    """The abstract class for a pulse's waveform."""

    def __init__(self, duration):
        self._duration = validate_duration(duration)

    @property
    @abstractmethod
    def duration(self):
        """The duration of the pulse (in ns)."""
        pass

    @property
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
        self._samples = samples_arr

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return len(self._samples)

    @property
    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        return self._samples

    def __str__(self):
        return 'Arbitrary'


class ConstantWaveform(Waveform):
    """A waveform of constant value.

    Args:
        duration: The waveform duration (in ns).
        value: The modulation value.
    """

    def __init__(self, duration, value):
        super().__init__(duration)
        self._value = float(value)

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return self._duration

    @property
    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        return np.full(self.duration, self._value)

    def __str__(self):
        return f"{self._value}MHz"


class RampWaveform(Waveform):
    """A linear ramp waveform.

    Args:
        duration: The waveform duration (in ns).
        start: The initial value.
        stop: The final value.
    """

    def __init__(self, duration, start, stop):
        super().__init__(duration)
        self._start = float(start)
        self._stop = float(stop)

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return self._duration

    @property
    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        return np.linspace(self._start, self._stop, num=self._duration)

    def __str__(self):
        return f"[{self._start}->{self._stop}]MHz"


class GaussianWaveform(Waveform):
    """A Gaussian-shaped waveform.

    Args:
        duration: The waveform duration (in ns).
        max_val: The maximum value.
        sigma: The standard deviation of the gaussian shape (in ns).

    Keyword Args:
        offset (default=0): A constant offset that defines the baseline.
    """

    def __init__(self, duration, max_val, sigma, offset=0):
        super().__init__(duration)
        if max_val < offset:
            raise ValueError("Can't accept a maximum value that is smaller"
                             " than the offset of a gaussian waveform.")
        if sigma <= 0:
            raise ValueError("The standard deviation has to be positive.")
        self._top = float(max_val - offset)
        self._sigma = sigma
        self._offset = float(offset)

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return self._duration

    @property
    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        # Ensures intervals are always symmetrical
        ts = np.arange(self.duration, dtype=float) - (self.duration - 1) * 0.5
        return self._top * np.exp(-0.5 * (ts / self._sigma)**2) + self._offset

    def __str__(self):
        return (f"Gaussian([{self._offset}->{self._top+self._offset}]MHz," +
                f"sigma={self._sigma}ns)")
