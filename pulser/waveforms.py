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
"""Contains all supported types of waveforms and the Waveform parent class."""

from __future__ import annotations

from abc import ABC, abstractmethod
import functools
import inspect
import itertools
import sys
from types import FunctionType
from typing import Any, cast, Optional, Tuple, Union
import warnings

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from pulser.parametrized import Parametrized, ParamObj
from pulser.parametrized.decorators import parametrize
from pulser.json.utils import obj_to_dict


class Waveform(ABC):
    """The abstract class for a pulse's waveform."""

    def __new__(cls, *args, **kwargs):      # type: ignore
        """Creates a Waveform instance or a ParamObj depending on the input."""
        for x in itertools.chain(args, kwargs.values()):
            if isinstance(x, Parametrized):
                return ParamObj(cls, *args, **kwargs)
        else:
            return object.__new__(cls)

    def __init__(self, duration: Union[int, Parametrized]):
        """Initializes a waveform with a given duration.

        Args:
            duration (int): The waveforms duration (in ns).
        """
        duration = cast(int, duration)
        try:
            _duration = int(duration)
        except (TypeError, ValueError):
            raise TypeError("duration needs to be castable to an int but "
                            f"type {type(duration)} was provided.")
        if _duration <= 0:
            raise ValueError("A waveform must have a positive duration, "
                             + f"not {duration}.")
        elif duration - _duration != 0:
            warnings.warn(f"A waveform duration of {duration} ns is below the"
                          " supported precision of 1 ns. It was rounded down "
                          + f"to {_duration} ns.", stacklevel=3)

        self._duration = _duration

    @property
    @abstractmethod
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        pass

    @property
    @abstractmethod
    def samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            np.ndarray: A numpy array with a value for each time step.
        """
        pass

    @property
    def first_value(self) -> float:
        """The first value in the waveform."""
        return float(self.samples[0])

    @property
    def last_value(self) -> float:
        """The last value in the waveform."""
        return float(self.samples[-1])

    @property
    def integral(self) -> float:
        """Integral of the waveform (time in ns, value in rad/µs)."""
        return float(np.sum(self.samples)) * 1e-3  # ns * rad/µs = 1e-3

    def draw(self) -> None:
        """Draws the waveform."""
        fig, ax = plt.subplots()
        self._plot(ax, "rad/µs")

        plt.show()

    def change_duration(self, new_duration: int) -> Waveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration(int): The duration of the new waveform.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support"
                                  " modifications to its duration.")

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __mul__(self, other: float) -> Waveform:
        pass

    def __neg__(self) -> Waveform:
        return self.__mul__(-1.)

    def __truediv__(self, other: float) -> Waveform:
        if other == 0:
            raise ZeroDivisionError("Can't divide a waveform by zero.")
        else:
            return self.__mul__(1/other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Waveform):
            return False
        elif self.duration != other.duration:
            return False
        else:
            return bool(np.all(np.isclose(self.samples, other.samples)))

    def __hash__(self) -> int:
        return hash(tuple(self.samples))

    def _plot(self, ax: Axes, ylabel: str,
              color: Optional[str] = None) -> None:
        ax.set_xlabel('t (ns)')
        ts = np.arange(self.duration)
        if color:
            ax.set_ylabel(ylabel, color=color, fontsize=14)
            ax.plot(ts, self.samples, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            ax.axhline(0, color=color, linestyle=':', linewidth=0.5)
        else:
            ax.set_ylabel(ylabel, fontsize=14)
            ax.plot(ts, self.samples)
            ax.axhline(0, color='black', linestyle=':', linewidth=0.5)


class CompositeWaveform(Waveform):
    """A waveform combining multiple smaller waveforms.

    Args:
        waveforms(Waveform): Two or more waveforms to combine.
    """

    def __init__(self, *waveforms: Union[Parametrized, Waveform]):
        """Initializes a waveform from multiple waveforms."""
        if len(waveforms) < 2:
            raise ValueError("Needs at least two waveforms to form a "
                             "CompositeWaveform.")
        waveforms = cast(Tuple[Waveform], waveforms)
        for wf in waveforms:
            self._validate(wf)

        self._waveforms = list(waveforms)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        duration = 0
        for wf in self._waveforms:
            duration += wf.duration
        return duration

    @property
    def samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return cast(np.ndarray,
                    np.concatenate([wf.samples for wf in self._waveforms]))

    @property
    def first_value(self) -> float:
        """The first value in the waveform."""
        return self._waveforms[0].first_value

    @property
    def last_value(self) -> float:
        """The last value in the waveform."""
        return self._waveforms[-1].last_value

    @property
    def waveforms(self) -> list[Waveform]:
        """The waveforms encapsulated in the composite waveform."""
        return list(self._waveforms)

    def _validate(self, waveform: Waveform) -> None:
        if not isinstance(waveform, Waveform):
            raise TypeError(f"{waveform!r} is not a valid waveform. "
                            "Please provide a valid Waveform.")

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, *self._waveforms)

    def __str__(self) -> str:
        contents_list = ["{!r}"] * len(self._waveforms)
        contents = ", ".join(contents_list)
        contents = contents.format(*self._waveforms)
        return f'Composite({contents})'

    def __repr__(self) -> str:
        return f'CompositeWaveform({self.duration} ns, {self._waveforms!r})'

    def __mul__(self, other: float) -> CompositeWaveform:
        return CompositeWaveform(*(wf * other for wf in self._waveforms))


class CustomWaveform(Waveform):
    """A custom waveform.

    Args:
        samples (array_like): The modulation values at each time step
            (in rad/µs). The number of samples dictates the duration, in ns.
    """

    def __init__(self, samples: ArrayLike):
        """Initializes a custom waveform."""
        samples_arr = np.array(samples, dtype=float)
        self._samples = samples_arr
        super().__init__(len(samples_arr))

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @property
    def samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return self._samples

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._samples)

    def __str__(self) -> str:
        return 'Custom'

    def __repr__(self) -> str:
        return f'CustomWaveform({self.duration} ns, {self.samples!r})'

    def __mul__(self, other: float) -> CustomWaveform:
        return CustomWaveform(self._samples * float(other))


class ConstantWaveform(Waveform):
    """A waveform of constant value.

    Args:
        duration (int): The waveform duration (in ns).
        value (float): The modulation value (in rad/µs).
    """

    def __init__(self, duration: Union[int, Parametrized],
                 value: Union[float, Parametrized]):
        """Initializes a constant waveform."""
        super().__init__(duration)
        value = cast(float, value)
        self._value = float(value)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @property
    def samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return np.full(self.duration, self._value)

    @property
    def first_value(self) -> float:
        """The first value in the waveform."""
        return self._value

    @property
    def last_value(self) -> float:
        """The last value in the waveform."""
        return self._value

    def change_duration(self, new_duration: int) -> ConstantWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration(int): The duration of the new waveform.

        Returns:
            ConstantWaveform: The new waveform with the given duration.
        """
        return ConstantWaveform(new_duration, self._value)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._value)

    def __str__(self) -> str:
        return f"{self._value:.3g} rad/µs"

    def __repr__(self) -> str:
        return (f"ConstantWaveform({self._duration} ns, "
                + f"{self._value:.3g} rad/µs)")

    def __mul__(self, other: float) -> ConstantWaveform:
        return ConstantWaveform(self._duration, self._value * float(other))


class RampWaveform(Waveform):
    """A linear ramp waveform.

    Args:
        duration (int): The waveform duration (in ns).
        start (float): The initial value (in rad/µs).
        stop (float): The final value (in rad/µs).
    """

    def __init__(self, duration: Union[int, Parametrized],
                 start: Union[float, Parametrized],
                 stop: Union[float, Parametrized]):
        """Initializes a ramp waveform."""
        super().__init__(duration)
        start = cast(float, start)
        self._start = float(start)
        stop = cast(float, stop)
        self._stop = float(stop)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @property
    def samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return np.linspace(self._start, self._stop, num=self._duration)

    @property
    def slope(self) -> float:
        r"""Slope of the ramp, in :math:`s^{-15}`."""
        return (self._stop - self._start) / self._duration

    @property
    def first_value(self) -> float:
        """The first value in the waveform."""
        return self._start

    @property
    def last_value(self) -> float:
        """The last value in the waveform."""
        return self._stop

    def change_duration(self, new_duration: int) -> RampWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration(int): The duration of the new waveform.

        Returns:
            RampWaveform: The new waveform with the given duration.
        """
        return RampWaveform(new_duration, self._start, self._stop)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._start, self._stop)

    def __str__(self) -> str:
        return f"Ramp({self._start:.3g}->{self._stop:.3g} rad/µs)"

    def __repr__(self) -> str:
        return (f"RampWaveform({self._duration} ns, " +
                f"{self._start:.3g}->{self._stop:.3g} rad/µs)")

    def __mul__(self, other: float) -> RampWaveform:
        k = float(other)
        return RampWaveform(self._duration, self._start * k, self._stop * k)


class BlackmanWaveform(Waveform):
    """A Blackman window of a specified duration and area.

    Args:
        duration (int): The waveform duration (in ns).
        area (float): The integral of the waveform. Can be negative, in which
            case it takes the positive waveform and changes the sign of all its
            values.
    """

    def __init__(self, duration: Union[int, Parametrized],
                 area: Union[float, Parametrized]):
        """Initializes a Blackman waveform."""
        super().__init__(duration)
        try:
            self._area: float = float(cast(float, area))
        except (TypeError, ValueError):
            raise TypeError("area needs to be castable to a float but "
                            f"type {type(area)} was provided.")

        self._norm_samples: np.ndarray = np.clip(
            np.blackman(self._duration), 0, np.inf)
        self._scaling: float = self._area / float(
            np.sum(self._norm_samples)) / 1e-3

    @classmethod
    @parametrize
    def from_max_val(cls, max_val: Union[float, Parametrized],
                     area: Union[float, Parametrized]) -> BlackmanWaveform:
        """Creates a Blackman waveform with a threshold on the maximum value.

        Instead of defining a duration, the waveform is defined by its area and
        the maximum value. The duration is chosen so that the maximum value is
        not surpassed, but approached as closely as possible.

        Args:
            max_val (float): The maximum value threshold (in rad/µs). If
                negative, it is taken as the lower bound i.e. the minimum
                value that can be reached. The sign of `max_val` must match the
                sign of `area`.
            area (float): The area under the waveform.
        """
        max_val = cast(float, max_val)
        area = cast(float, area)
        if np.sign(max_val) != np.sign(area):
            raise ValueError("The maximum value and the area must have "
                             "matching signs.")
        # A normalized Blackman waveform has an area of 0.42 * duration
        duration = np.ceil(area / (0.42 * max_val) * 1e3)    # in ns
        wf = cls(duration, area)
        # Adjust for rounding errors to make sure max_val is not surpassed
        while np.abs(wf._scaling) > np.abs(max_val):
            duration += 1
            wf = cls(duration, area)
        return wf

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @property
    def samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return cast(np.ndarray, self._norm_samples * self._scaling)

    @property
    def first_value(self) -> float:
        """The first value in the waveform."""
        return 0.

    @property
    def last_value(self) -> float:
        """The last value in the waveform."""
        return 0.

    def change_duration(self, new_duration: int) -> BlackmanWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration(int): The duration of the new waveform.

        Returns:
            BlackmanWaveform: The new waveform with the same area but a new
                duration.
        """
        return BlackmanWaveform(new_duration, self._area)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._area)

    def __str__(self) -> str:
        return f"Blackman(Area: {self._area:.3g})"

    def __repr__(self) -> str:
        return f"BlackmanWaveform({self._duration} ns, Area: {self._area:.3g})"

    def __mul__(self, other: float) -> BlackmanWaveform:
        return BlackmanWaveform(self._duration, self._area * float(other))


# To replicate __init__'s signature in __new__ for every Waveform subclass
def _copy_func(f: FunctionType) -> FunctionType:
    return FunctionType(f.__code__, f.__globals__, name=f.__name__,
                        argdefs=f.__defaults__, closure=f.__closure__)


for m in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if m[1].__module__ == __name__:
        _new = _copy_func(m[1].__new__)
        m[1].__new__ = functools.update_wrapper(_new, m[1].__init__)
