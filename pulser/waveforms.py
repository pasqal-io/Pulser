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

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from pulser.utils import validate_duration


class Waveform(ABC):
    """The abstract class for a pulse's waveform."""

    def __init__(self, duration):
        """Initializes a waveform with a given duration.

        Args:
            duration (int): The waveforms duration (in ns).
        """
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

    @property
    def integral(self):
        """Determines the integral of the waveform (time: ns, value: MHz)."""
        return np.sum(self.samples) * 1e-3  # ns * MHz = 1e-3

    def draw(self):
        """Draws the waveform."""

        fig, ax = plt.subplots()
        self._plot(ax, "MHz")

        plt.show()

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    def __eq__(self, other):
        if not isinstance(other, Waveform):
            return False
        elif self.duration != other.duration:
            return False
        else:
            return np.all(np.isclose(self.samples, other.samples))

    def _plot(self, ax, ylabel, color=None):
        ax.set_xlabel('t (ns)')
        ts = np.arange(self.duration)
        if color:
            ax.set_ylabel(ylabel, color=color)
            ax.plot(ts, self.samples, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            ax.axhline(0, color=color, linestyle=':', linewidth=0.5)
        else:
            ax.set_ylabel(ylabel)
            ax.plot(ts, self.samples)
            ax.axhline(0, color='black', linestyle=':', linewidth=0.5)


class CompositeWaveform(Waveform):
    """A waveform combining multiple smaller waveforms.

    Args:
        waveforms: Two or more waveforms to combine.
    """

    def __init__(self, *waveforms):
        """Initializes a waveform from multiple waveforms."""
        if len(waveforms) < 2:
            raise ValueError("Needs at least two waveforms to form a "
                             "CompositeWaveform.")
        for wf in waveforms:
            self._validate(wf)

        self._waveforms = list(waveforms)

    @property
    def duration(self):
        """The duration of the pulse (in ns)."""
        duration = 0
        for wf in self._waveforms:
            duration += wf.duration
        return duration

    @property
    def samples(self):
        """The value at each time step that describes the waveform.

        Returns:
            samples(np.ndarray): A numpy array with a value for each time step.
        """
        return np.concatenate([wf.samples for wf in self._waveforms])

    @property
    def waveforms(self):
        """The waveforms encapsulated in the composite waveform."""
        return list(self._waveforms)

    def insert(self, waveform, where=0):
        """Insert a new waveform into the CompositeWaveform.

        Args:
            waveform: A valid waveform.

        Keyword Args:
            where (default=0): Index before which the waveform is inserted.
        """
        self._validate(waveform)
        self._waveforms.insert(where, waveform)

    def append(self, waveform):
        """Append a new waveform to the end of a CompositeWaveform.

        Args:
            waveform: A valid waveform.
        """
        self.insert(waveform, where=len(self._waveforms))

    def _validate(self, waveform):
        if not isinstance(waveform, Waveform):
            raise TypeError("{!r} is not a valid waveform. Please provide a "
                            "valid Waveform.".format(waveform))

    def __str__(self):
        contents = ["{!r}"] * len(self._waveforms)
        contents = ", ".join(contents)
        contents = contents.format(*self._waveforms)
        return f'Composite({contents})'

    def __repr__(self):
        return f'CompositeWaveform({self.duration} ns, {self._waveforms!r})'


class CustomWaveform(Waveform):
    """A custom waveform.

    Args:
        samples (array_like): The modulation values at each time step.
    """

    def __init__(self, samples):
        """Initializes a custom waveform."""
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
        return 'Custom'

    def __repr__(self):
        return f'CustomWaveform({self.duration} ns, {self.samples!r})'


class ConstantWaveform(Waveform):
    """A waveform of constant value.

    Args:
        duration: The waveform duration (in ns).
        value: The modulation value.
    """

    def __init__(self, duration, value):
        """Initializes a constant waveform."""
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
        return f"{self._value:.3g} MHz"

    def __repr__(self):
        return f"ConstantWaveform({self._duration} ns, {self._value:.3g} MHz)"


class RampWaveform(Waveform):
    """A linear ramp waveform.

    Args:
        duration: The waveform duration (in ns).
        start: The initial value.
        stop: The final value.
    """

    def __init__(self, duration, start, stop):
        """Initializes a ramp waveform."""
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
        return f"Ramp({self._start:.3g}->{self._stop:.3g} MHz)"

    def __repr__(self):
        return (f"RampWaveform({self._duration} ns, " +
                f"{self._start:.3g}->{self._stop:.3g} MHz)")


class BlackmanWaveform(Waveform):
    """A Blackman window of a specified duration and area.

    Args:
        duration: The waveform duration (in ns).
        area: The area under the waveform.
    """
    def __init__(self, duration, area):
        """Initializes a Blackman waveform."""
        super().__init__(duration)
        if area <= 0:
            raise ValueError("Area under the waveform needs to be positive.")
        self._area = area

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
        samples = np.clip(np.blackman(self._duration), 0, np.inf)
        scaling = self._area / np.sum(samples) / 1e-3
        return samples * scaling

    def __str__(self):
        return f"Blackman(Area: {self._area:.3g})"

    def __repr__(self):
        return f"BlackmanWaveform({self._duration} ns, Area: {self._area:.3g})"
