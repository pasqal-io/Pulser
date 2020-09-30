import numpy as np
import warnings
from abc import ABC, abstractmethod


class Waveform(ABC):
    """The abstract class for a pulse's waveform."""

    @property
    @abstractmethod
    def duration(self):
        """The duration of the pulse (in ns)."""
        pass

    @abstractmethod
    def samples(self):
        """The amplitude at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with an amplitude value for each
                time step.
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
        samples (array_like): The amplitude values at each time step.
    """

    def __init__(self, samples):
        samples_arr = np.array(samples)
        if np.any(samples_arr < 0):
            raise ValueError("All amplitude values in a sample must be "
                             "non-negative.")
        self._samples = samples_arr

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return len(self._samples)

    def samples(self):
        """The amplitude at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with an amplitude value at each
                time step.
        """
        return self._samples


class ConstantWaveform(Waveform):
    """A waveform of constant amplitude.

    Args:
        duration: The waveform duration (in ns).
        value: The amplitude value.
    """

    def __init__(self, duration, amp):
        if amp < 0:
            raise ValueError("Can't accept negative amplitude values.")
        if duration % 1 != 0:
            warnings.warn("The given duration is below the machine's precision"
                          " of 1 ns time steps. It was rounded down to the"
                          " nearest integer.")
        self._duration = int(duration)
        self._amp = float(amp)

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return self._duration

    def samples(self):
        """The amplitude at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with an amplitude value at each
                time step.
        """
        return np.full(self.duration, self._amp)


class GaussianWaveform(Waveform):
    """A Gaussian-shaped amplitude waveform.

    Args:
        duration: The waveform duration (in ns).
        max_amp: The maximum amplitude value.
        sigma: The standard deviation of the gaussian shape (in ns).
    """

    def __init__(self, duration, max_amp, sigma):
        # NOTE: This stores max_amp in self._amp
        ConstantWaveform.__init__(self, duration, max_amp)
        if sigma <= 0:
            raise ValueError("The standard deviation has to be positive.")
        self._sigma = sigma

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        return self._duration

    def samples(self):
        """The amplitude at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with an amplitude value at each
                time step.
        """
        # Ensures intervals are always symmetrical
        ts = np.arange(self.duration, dtype=float) - (self.duration - 1) * 0.5
        return self._amp * np.exp(-0.5 * (ts / self._sigma)**2)
