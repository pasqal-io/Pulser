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
from sys import version_info
from types import FunctionType
from typing import Any, cast, Optional, Tuple, Union
import warnings

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import scipy.interpolate as interpolate

from pulser.parametrized import Parametrized, ParamObj
from pulser.parametrized.decorators import parametrize
from pulser.json.utils import obj_to_dict

if version_info[:2] >= (3, 8):  # pragma: no cover
    from functools import cached_property
else:  # pragma: no cover
    try:
        from backports.cached_property import cached_property  # type: ignore
    except ImportError:
        raise ImportError(
            "Using pulser with Python version 3.7 requires the"
            " `backports.cached-property` module. Install it by running"
            " `pip install backports.cached-property`."
        )


class Waveform(ABC):
    """The abstract class for a pulse's waveform."""

    def __new__(cls, *args, **kwargs):  # type: ignore
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
            raise TypeError(
                "duration needs to be castable to an int but "
                f"type {type(duration)} was provided."
            )
        if _duration <= 0:
            raise ValueError(
                "A waveform must have a positive duration, "
                + f"not {duration}."
            )
        elif duration - _duration != 0:
            warnings.warn(
                f"A waveform duration of {duration} ns is below the"
                " supported precision of 1 ns. It was rounded down "
                + f"to {_duration} ns.",
                stacklevel=3,
            )

        self._duration = _duration

    @property
    @abstractmethod
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        pass

    @cached_property
    @abstractmethod
    def _samples(self) -> np.ndarray:
        pass

    @property
    def samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            np.ndarray: A numpy array with a value for each time step.
        """
        return self._samples.copy()

    @property
    def first_value(self) -> float:
        """The first value in the waveform."""
        return float(self[0])

    @property
    def last_value(self) -> float:
        """The last value in the waveform."""
        return float(self[-1])

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support"
            " modifications to its duration."
        )

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __getitem__(
        self, index_or_slice: Union[int, slice]
    ) -> Union[float, np.ndarray]:
        if isinstance(index_or_slice, slice):
            s: slice = self._check_slice(index_or_slice)
            return cast(np.ndarray, self._samples[s])
        else:
            index: int = self._check_index(index_or_slice)
            return cast(float, self._samples[index])

    def _check_index(self, i: int) -> int:
        if i < -self.duration or i >= self.duration:
            raise IndexError(
                "Index ('index_or_slice' = "
                f"{i}) must be in the range "
                f"0~{self.duration-1}, or "
                f"{-self.duration}~-1 from the end."
            )
        return i if i >= 0 else self.duration + i

    def _check_slice(self, s: slice) -> slice:
        if s.step is not None and s.step != 1:
            raise IndexError("The step of the slice must be None or 1.")

        # Transform start and stop indexes into positive or null values
        # since they can be omitted (None) or negative (end-indexing)
        start = (
            0
            if s.start is None
            else (s.start if s.start >= 0 else self.duration + s.start)
        )
        stop = (
            self.duration
            if s.stop is None
            else (s.stop if s.stop >= 0 else self.duration + s.stop)
        )

        # Correct out of bounds ranges
        if start < 0:
            start = 0
        if stop < 0:
            stop = 0
        if start > self.duration:
            start = self.duration
        if stop > self.duration:
            stop = self.duration
        if stop < start:
            stop = start

        return slice(start, stop)

    @abstractmethod
    def __mul__(self, other: float) -> Waveform:
        pass

    def __neg__(self) -> Waveform:
        return self.__mul__(-1.0)

    def __truediv__(self, other: float) -> Waveform:
        if other == 0:
            raise ZeroDivisionError("Can't divide a waveform by zero.")
        else:
            return self.__mul__(1 / other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Waveform):
            return False
        elif self.duration != other.duration:
            return False
        else:
            return bool(np.all(np.isclose(self.samples, other.samples)))

    def __hash__(self) -> int:
        return hash(tuple(self.samples))

    def _plot(
        self, ax: Axes, ylabel: str, color: Optional[str] = None
    ) -> None:
        ax.set_xlabel("t (ns)")
        ts = np.arange(self.duration)
        if color:
            ax.set_ylabel(ylabel, color=color, fontsize=14)
            ax.plot(ts, self.samples, color=color)
            ax.tick_params(axis="y", labelcolor=color)
            ax.axhline(0, color=color, linestyle=":", linewidth=0.5)
        else:
            ax.set_ylabel(ylabel, fontsize=14)
            ax.plot(ts, self.samples)
            ax.axhline(0, color="black", linestyle=":", linewidth=0.5)


class CompositeWaveform(Waveform):
    """A waveform combining multiple smaller waveforms.

    Args:
        waveforms(Waveform): Two or more waveforms to combine.
    """

    def __init__(self, *waveforms: Union[Parametrized, Waveform]):
        """Initializes a waveform from multiple waveforms."""
        if len(waveforms) < 2:
            raise ValueError(
                "Needs at least two waveforms to form a " "CompositeWaveform."
            )
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

    @cached_property
    def _samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return cast(
            np.ndarray, np.concatenate([wf.samples for wf in self._waveforms])
        )

    @property
    def waveforms(self) -> list[Waveform]:
        """The waveforms encapsulated in the composite waveform."""
        return list(self._waveforms)

    def _validate(self, waveform: Waveform) -> None:
        if not isinstance(waveform, Waveform):
            raise TypeError(
                f"{waveform!r} is not a valid waveform. "
                "Please provide a valid Waveform."
            )

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, *self._waveforms)

    def __str__(self) -> str:
        contents_list = ["{!r}"] * len(self._waveforms)
        contents = ", ".join(contents_list)
        contents = contents.format(*self._waveforms)
        return f"Composite({contents})"

    def __repr__(self) -> str:
        return f"CompositeWaveform({self.duration} ns, {self._waveforms!r})"

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
        self._samples: np.ndarray = samples_arr
        super().__init__(len(samples_arr))

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        # self._samples is already cached when initialized in __init__
        pass

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._samples)

    def __str__(self) -> str:
        return "Custom"

    def __repr__(self) -> str:
        return f"CustomWaveform({self.duration} ns, {self.samples!r})"

    def __mul__(self, other: float) -> CustomWaveform:
        return CustomWaveform(self._samples * float(other))


class ConstantWaveform(Waveform):
    """A waveform of constant value.

    Args:
        duration (int): The waveform duration (in ns).
        value (float): The modulation value (in rad/µs).
    """

    def __init__(
        self,
        duration: Union[int, Parametrized],
        value: Union[float, Parametrized],
    ):
        """Initializes a constant waveform."""
        super().__init__(duration)
        value = cast(float, value)
        self._value = float(value)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return np.full(self.duration, self._value)

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
        return (
            f"ConstantWaveform({self._duration} ns, "
            + f"{self._value:.3g} rad/µs)"
        )

    def __mul__(self, other: float) -> ConstantWaveform:
        return ConstantWaveform(self._duration, self._value * float(other))


class RampWaveform(Waveform):
    """A linear ramp waveform.

    Args:
        duration (int): The waveform duration (in ns).
        start (float): The initial value (in rad/µs).
        stop (float): The final value (in rad/µs).
    """

    def __init__(
        self,
        duration: Union[int, Parametrized],
        start: Union[float, Parametrized],
        stop: Union[float, Parametrized],
    ):
        """Initializes a ramp waveform."""
        super().__init__(duration)
        start = cast(float, start)
        self._start: float = float(start)
        stop = cast(float, stop)
        self._stop: float = float(stop)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return np.linspace(self._start, self._stop, num=self._duration)

    @property
    def slope(self) -> float:
        r"""Slope of the ramp, in :math:`s^{-15}`."""
        return (self._stop - self._start) / self._duration

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
        return (
            f"RampWaveform({self._duration} ns, "
            + f"{self._start:.3g}->{self._stop:.3g} rad/µs)"
        )

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

    def __init__(
        self,
        duration: Union[int, Parametrized],
        area: Union[float, Parametrized],
    ):
        """Initializes a Blackman waveform."""
        super().__init__(duration)
        try:
            self._area: float = float(cast(float, area))
        except (TypeError, ValueError):
            raise TypeError(
                "area needs to be castable to a float but "
                f"type {type(area)} was provided."
            )

        self._norm_samples: np.ndarray = np.clip(
            np.blackman(self._duration), 0, np.inf
        )
        self._scaling: float = (
            self._area / float(np.sum(self._norm_samples)) / 1e-3
        )

    @classmethod
    @parametrize
    def from_max_val(
        cls,
        max_val: Union[float, Parametrized],
        area: Union[float, Parametrized],
    ) -> BlackmanWaveform:
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
            raise ValueError(
                "The maximum value and the area must have " "matching signs."
            )

        # A normalized Blackman waveform has an area of 0.42 * duration
        duration = np.ceil(area / (0.42 * max_val) * 1e3)  # in ns
        wf = cls(duration, area)

        # Adjust for rounding errors to make sure max_val is not surpassed
        while np.abs(wf._scaling) > np.abs(max_val):
            duration += 1
            previous_wf = wf
            wf = cls(duration, area)

        # According to the documentation for numpy.blackman(), "the value one
        # appears only if the number of samples is odd". Hence, the previous
        # even duration can reach a maximal value closer to max_val.
        if (previous_wf is not None and duration % 2 == 1 and
                np.max(wf.samples) < np.max(previous_wf.samples) <= max_val):
            wf = previous_wf

        return wf

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform.

        Returns:
            numpy.ndarray: A numpy array with a value for each time step.
        """
        return cast(np.ndarray, self._norm_samples * self._scaling)

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


class InterpolatedWaveform(Waveform):
    """Creates a waveform from interpolation of a set of data points.

    Args:
        duration (int): The waveform duration (in ns).
        values (ArrayLike): Values of the interpolation points (in rad/µs).
        times (Optional[ArrayLike]): Fractions of the total duration (between 0
            and 1), indicating where to place each value on the time axis. If
            not given, the values are spread evenly throughout the full
            duration of the waveform.
        interpolator (str = "PchipInterpolator"): The SciPy interpolation class
            to use. Supports "PchipInterpolator" and "interp1d".
        **interpolator_kwargs: Extra parameters to give to the chosen
            interpolator class.
    """

    def __init__(
        self,
        duration: Union[int, Parametrized],
        values: Union[ArrayLike, Parametrized],
        times: Optional[Union[ArrayLike, Parametrized]] = None,
        interpolator: str = "PchipInterpolator",
        **interpolator_kwargs: Any,
    ):
        """Initializes a new InterpolatedWaveform."""
        super().__init__(duration)
        self._values = np.array(values, dtype=float)
        if times is not None:
            times_ = np.array(times, dtype=float)
            if len(times_) != len(self._values):
                raise ValueError(
                    "When specified, the number of time coordinates in `times`"
                    f" ({len(times_)}) must match the number of `values` "
                    f"({len(self._values)})."
                )
            if np.any(times_ < 0):
                raise ValueError(
                    "All values in `times` must be greater than or equal to 0."
                )
            if np.any(times_ > 1):
                raise ValueError(
                    "All values in `times` must be less than or equal to 1."
                )
            unique_times = np.unique(times)  # Sorted array of unique values
            if len(times_) != len(unique_times):
                raise ValueError(
                    "`times` must be an array of non-repeating values."
                )
            self._times = times_
        else:
            self._times = np.linspace(0, 1, num=len(self._values))

        valid_interpolators = ("PchipInterpolator", "interp1d")
        if interpolator not in valid_interpolators:
            raise ValueError(
                f"Invalid interpolator '{interpolator}', only "
                "accepts: " + ", ".join(valid_interpolators)
            )
        interp_cls = getattr(interpolate, interpolator)
        self._data_pts = np.array(
            [
                (round(t), v)
                for t, v in zip(
                    self._times * (self._duration - 1), self._values
                )
            ]
        )
        self._interp_func = interp_cls(
            self._data_pts[:, 0], self._data_pts[:, 1], **interpolator_kwargs
        )
        self._kwargs: dict[str, Any] = {
            "times": times,
            "interpolator": interpolator,
            **interpolator_kwargs,
        }

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> np.ndarray:
        """The value at each time step that describes the waveform."""
        return cast(
            np.ndarray,
            np.round(
                self._interp_func(np.arange(self._duration)), decimals=9
            ),  # Rounds to the order of Hz
        )

    @property
    def interp_function(
        self,
    ) -> Union[interpolate.PchipInterpolator, interpolate.interp1d]:
        """The interpolating function."""
        return self._interp_func

    @property
    def data_points(self) -> np.ndarray:
        """Points (t[ns], value[rad/µs]) that define the interpolation."""
        return self._data_pts.copy()

    def change_duration(self, new_duration: int) -> InterpolatedWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration(int): The duration of the new waveform.

        Returns:
            InterpolatedWaveform: The new waveform with the same coordinates
                for interpolation but a new duration.
        """
        return InterpolatedWaveform(new_duration, self._values, **self._kwargs)

    def _plot(
        self, ax: Axes, ylabel: str, color: Optional[str] = None
    ) -> None:
        super()._plot(ax, ylabel, color=color)
        ax.scatter(self._data_pts[:, 0], self._data_pts[:, 1], c=color)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._values, **self._kwargs)

    def __str__(self) -> str:
        coords = [f"({int(x)}, {y:.4g})" for x, y in self.data_points]
        return f"InterpolatedWaveform(Points: {', '.join(coords)})"

    def __repr__(self) -> str:
        interp_str = f", Interpolator={self._kwargs['interpolator']})"
        return self.__str__()[:-1] + interp_str

    def __mul__(self, other: float) -> InterpolatedWaveform:
        return InterpolatedWaveform(
            self._duration, self._values * other, **self._kwargs
        )


# To replicate __init__'s signature in __new__ for every Waveform subclass
def _copy_func(f: FunctionType) -> FunctionType:
    return FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )


for m in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if m[1].__module__ == __name__:
        _new = _copy_func(m[1].__new__)
        m[1].__new__ = functools.update_wrapper(_new, m[1].__init__)
