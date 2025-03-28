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
"""All built-in types of waveforms and their Waveform parent class."""

from __future__ import annotations

import functools
import inspect
import itertools
import sys
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from types import FunctionType
from typing import TYPE_CHECKING, Any, Optional, Tuple, TypeVar, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

import pulser.math as pm
from pulser.exceptions.serialization import AbstractReprError
from pulser.json.abstract_repr.serializer import abstract_repr
from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized, ParamObj
from pulser.parametrized.decorators import parametrize

if TYPE_CHECKING:
    from pulser.channels.base_channel import Channel

__all__ = [
    "Waveform",
    "CompositeWaveform",
    "CustomWaveform",
    "ConstantWaveform",
    "RampWaveform",
    "BlackmanWaveform",
    "InterpolatedWaveform",
    "KaiserWaveform",
]

T = TypeVar("T", int, float)


def _cast_check(type_: type[T], value: Any, name: str) -> T:
    try:
        return type_(value)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"'{name}' needs to be castable to {type_.__name__!s} "
            f"but type {type(value)} was provided."
        ) from e


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
            duration: The waveforms duration (in ns).
        """
        assert not isinstance(duration, Parametrized)
        _duration = _cast_check(int, duration, "duration")

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
    def _samples(self) -> pm.AbstractArray:
        pass

    @property
    def samples(self) -> pm.AbstractArray:
        """The value at each time step that describes the waveform.

        Returns:
            A numpy array with a value for each time step.
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
        """Integral of the waveform (in [waveform units].µs)."""
        return float(pm.sum(self._samples)) * 1e-3  # ns * rad/µs = 1e-3

    def draw(
        self,
        output_channel: Optional[Channel] = None,
        ylabel: str | None = None,
    ) -> None:
        """Draws the waveform.

        Args:
            output_channel: The output channel. If given, will draw the
                modulated waveform on top of the input one.
            ylabel: An optional label for the y-axis of the plot.
        """
        fig, ax = plt.subplots()
        if not output_channel:
            self._plot(ax, ylabel=ylabel)
        else:
            self._plot(
                ax,
                ylabel=ylabel,
                label="Input",
                start_t=self.modulation_buffers(output_channel)[0],
            )
            self._plot(
                ax,
                channel=output_channel,
                label="Output",
            )
        plt.show()

    def change_duration(self, new_duration: int) -> Waveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration: The duration of the new waveform.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support"
            " modifications to its duration."
        )

    def modulated_samples(
        self, channel: Channel, eom: bool = False
    ) -> pm.AbstractArray:
        """The waveform samples as output of a given channel.

        This duration is adjusted according to the minimal buffer times.

        Args:
            channel: The channel modulating the waveform.
            eom: Whether to modulate for the EOM.

        Returns:
            The array of samples after modulation.
        """
        detach = True  # We detach unless...
        if self.samples.requires_grad:
            # ... the samples require grad. In this case, we clear the cache
            # so that the modulation is recalculated with the current samples
            self._modulated_samples.cache_clear()
            detach = False
        start, end = self.modulation_buffers(channel)
        mod_samples = self._modulated_samples(channel, eom=eom)
        tr = channel.rise_time
        trim = slice(tr - start, len(mod_samples) - tr + end)
        final_samples = mod_samples[trim]
        if detach:
            # This ensures that we don't carry the `requires_grad` of a
            # cached results
            return pm.AbstractArray(final_samples.as_array(detach=True))
        return final_samples

    @functools.lru_cache()
    def modulation_buffers(
        self, channel: Channel, eom: bool = False
    ) -> tuple[int, int]:
        """The minimal buffers needed around a modulated waveform.

        Args:
            channel: The channel modulating the waveform.
            eom: Whether to calculate the modulation buffers with
                the EOM bandwidth.

        Returns:
            The minimum buffer times at the start and end of
            the samples, in ns.
        """
        if not channel.mod_bandwidth:
            return 0, 0

        return channel.calc_modulation_buffer(
            self._samples, self._modulated_samples(channel, eom=eom), eom=eom
        )

    @functools.lru_cache()
    def _modulated_samples(
        self, channel: Channel, eom: bool = False
    ) -> pm.AbstractArray:
        """The waveform samples as output of a given channel.

        This is not adjusted to the minimal buffer times. Use
        ``Waveform.modulated_samples()`` to get the output already truncated.

        Args:
            channel: The channel modulating the waveform.
            eom: Whether to modulate for the EOM.

        Returns:
            The array of samples after modulation.
        """
        return channel.modulate(self._samples, eom=eom)

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _to_abstract_repr(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __getitem__(
        self, index_or_slice: Union[int, slice]
    ) -> pm.AbstractArray:
        if isinstance(index_or_slice, slice):
            s: slice = self._check_slice(index_or_slice)
            return self._samples[s]
        else:
            index: int = self._check_index(index_or_slice)
            return self._samples[index]

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
    def __mul__(self, other: float | ArrayLike) -> Waveform:
        pass

    def __neg__(self) -> Waveform:
        return self.__mul__(-1.0)

    def __truediv__(self, other: float | ArrayLike) -> Waveform:
        other_ = pm.AbstractArray(other)
        if np.any(other_.as_array(detach=True) == 0):
            raise ZeroDivisionError("Can't divide a waveform by zero.")
        else:
            return self.__mul__(1 / other_)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Waveform):
            return False
        elif self.duration != other.duration:
            return False
        else:
            return bool(
                np.all(
                    np.isclose(
                        self.samples.as_array(detach=True),
                        other.samples.as_array(detach=True),
                    )
                )
            )

    def __hash__(self) -> int:
        return hash(tuple(self.samples.tolist()))

    def _plot(
        self,
        ax: Axes,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        channel: Optional[Channel] = None,
        label: str = "",
        start_t: int = 0,
    ) -> None:
        ax.set_xlabel("t (ns)")
        samples = (
            self.samples
            if channel is None
            else self.modulated_samples(channel)
        ).as_array(detach=True)
        ts = np.arange(len(samples)) + start_t
        if not channel and start_t:
            # Adds zero on both ends to show rise and fall
            samples = np.pad(samples, 1)
            # Repeats the times on the edges once
            ts = np.pad(ts, 1, mode="edge")

        color_dict: dict[str, Any]
        if color:
            color_dict = {"color": color}
            hline_color = color
            ax.tick_params(axis="y", labelcolor=color)
        else:
            color_dict = {}
            hline_color = "black"

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=14, **color_dict)
        ax.plot(ts, samples, label=label, **color_dict)
        ax.axhline(0, color=hline_color, linestyle=":", linewidth=0.5)

        if label:
            plt.legend()


class CompositeWaveform(Waveform):
    """A waveform combining multiple smaller waveforms.

    Args:
        waveforms: Two or more waveforms to combine.
    """

    def __init__(self, *waveforms: Union[Parametrized, Waveform]):
        """Initializes a waveform from multiple waveforms."""
        if len(waveforms) < 2:
            raise ValueError(
                "Needs at least two waveforms to form a " "CompositeWaveform."
            )
        waveforms = cast(Tuple[Waveform, ...], waveforms)
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
    def _samples(self) -> pm.AbstractArray:
        """The value at each time step that describes the waveform.

        Returns:
            A numpy array with a value for each time step.
        """
        return pm.concatenate([wf.samples for wf in self._waveforms])

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

    def _to_abstract_repr(self) -> dict[str, Any]:
        return abstract_repr("CompositeWaveform", *self._waveforms)

    def __str__(self) -> str:
        contents_list = ["{!r}"] * len(self._waveforms)
        contents = ", ".join(contents_list)
        contents = contents.format(*self._waveforms)
        return f"Composite({contents})"

    def __repr__(self) -> str:
        return f"CompositeWaveform({self.duration} ns, {self._waveforms!r})"

    def __mul__(self, other: float | ArrayLike) -> CompositeWaveform:
        other_ = pm.AbstractArray(other, dtype=float)
        return CompositeWaveform(*(wf * other_ for wf in self._waveforms))


class CustomWaveform(Waveform):
    """A custom waveform.

    Args:
        samples: The modulation values at each time step.
            The number of samples dictates the duration, in ns.
    """

    def __init__(self, samples: ArrayLike | pm.TensorLike):
        """Initializes a custom waveform."""
        samples_arr = pm.AbstractArray(samples, dtype=float)
        self._samples_arr: pm.AbstractArray = samples_arr
        super().__init__(len(samples_arr))

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return int(self._duration)

    @cached_property
    def _samples(self) -> pm.AbstractArray:
        """The value at each time step that describes the waveform.

        Returns:
            A numpy array with a value for each time step.
        """
        return self._samples_arr

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._samples)

    def _to_abstract_repr(self) -> dict[str, Any]:
        return abstract_repr("CustomWaveform", self._samples)

    def __str__(self) -> str:
        return "Custom"

    def __repr__(self) -> str:
        return f"CustomWaveform({self.duration} ns, {self.samples!r})"

    def __mul__(self, other: float | ArrayLike) -> CustomWaveform:
        return CustomWaveform(
            self._samples * pm.AbstractArray(other, dtype=float)
        )


class ConstantWaveform(Waveform):
    """A waveform of constant value.

    Args:
        duration: The waveform duration (in ns).
        value: The value.
    """

    def __init__(
        self,
        duration: Union[int, Parametrized],
        value: Union[float, pm.TensorLike, Parametrized],
    ):
        """Initializes a constant waveform."""
        super().__init__(duration)
        assert not isinstance(value, Parametrized)
        _cast_check(float, value, "value")
        self._value = pm.AbstractArray(value, dtype=float)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> pm.AbstractArray:
        """The value at each time step that describes the waveform.

        Returns:
            A numpy array with a value for each time step.
        """
        return self._value * np.ones(self.duration)

    def change_duration(self, new_duration: int) -> ConstantWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration: The duration of the new waveform.

        Returns:
            The new waveform with the given duration.
        """
        return ConstantWaveform(new_duration, self._value)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._value)

    def _to_abstract_repr(self) -> dict[str, Any]:
        return abstract_repr("ConstantWaveform", self._duration, self._value)

    def __str__(self) -> str:
        return f"{float(self._value):.3g}"

    def __repr__(self) -> str:
        return (
            f"ConstantWaveform({self._duration} ns, {float(self._value):.3g})"
        )

    def __mul__(self, other: float | ArrayLike) -> ConstantWaveform:
        return ConstantWaveform(
            self._duration, self._value * pm.AbstractArray(other, dtype=float)
        )


class RampWaveform(Waveform):
    """A linear ramp waveform.

    Args:
        duration: The waveform duration (in ns).
        start: The value at the initial sample.
        stop: The value at the final sample.
    """

    def __init__(
        self,
        duration: Union[int, Parametrized],
        start: Union[float, pm.TensorLike, Parametrized],
        stop: Union[float, pm.TensorLike, Parametrized],
    ):
        """Initializes a ramp waveform."""
        super().__init__(duration)
        assert not isinstance(start, Parametrized)
        assert not isinstance(stop, Parametrized)
        _cast_check(float, start, "start")
        _cast_check(float, stop, "stop")
        self._start = pm.AbstractArray(start, dtype=float)
        self._stop = pm.AbstractArray(stop, dtype=float)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> pm.AbstractArray:
        """The value at each time step that describes the waveform.

        Returns:
            A numpy array with a value for each time step.
        """
        return pm.clip(
            self._slope * np.arange(self._duration, dtype=float) + self._start,
            *sorted(map(float, [self._start, self._stop])),
        )

    @property
    def _slope(self) -> pm.AbstractArray:
        return (self._stop - self._start) / (self._duration - 1)

    @property
    def slope(self) -> float:
        r"""Slope of the ramp, in [waveform units] / ns."""
        return float(self._slope)

    def change_duration(self, new_duration: int) -> RampWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration: The duration of the new waveform.

        Returns:
            The new waveform with the given duration.
        """
        return RampWaveform(new_duration, self._start, self._stop)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._start, self._stop)

    def _to_abstract_repr(self) -> dict[str, Any]:
        return abstract_repr(
            "RampWaveform", self._duration, self._start, self._stop
        )

    def __str__(self) -> str:
        return f"Ramp({float(self._start):.3g}->{float(self._stop):.3g})"

    def __repr__(self) -> str:
        return (
            f"RampWaveform({self._duration} ns, "
            f"{float(self._start):.3g}->{float(self._stop):.3g})"
        )

    def __mul__(self, other: float | ArrayLike) -> RampWaveform:
        k = pm.AbstractArray(other, dtype=float)
        return RampWaveform(self._duration, self._start * k, self._stop * k)


class BlackmanWaveform(Waveform):
    """A Blackman window of a specified duration and area.

    Warning:
        The BlackmanWaveform assumes its values are in rad/µs for the
        area calculation. If this is not the case, the 'area' value should be
        scaled accordingly.

    Args:
        duration: The waveform duration (in ns).
        area: The integral of the waveform. Can be negative, in which
            case it takes the positive waveform and changes the sign of all its
            values.
    """

    def __init__(
        self,
        duration: Union[int, Parametrized],
        area: Union[float, pm.TensorLike, Parametrized],
    ):
        """Initializes a Blackman waveform."""
        super().__init__(duration)
        assert not isinstance(area, Parametrized)
        _cast_check(float, area, "area")
        self._area = pm.AbstractArray(area, dtype=float)

        self._norm_samples = pm.AbstractArray(
            np.clip(np.blackman(self._duration), 0, np.inf)
        )
        self._scaling = self._area / pm.sum(self._norm_samples) * 1e3

    @classmethod
    @parametrize
    def from_max_val(
        cls,
        max_val: Union[float, Parametrized],
        area: Union[float, pm.TensorLike, Parametrized],
    ) -> BlackmanWaveform:
        """Creates a Blackman waveform with a threshold on the maximum value.

        Instead of defining a duration, the waveform is defined by its area and
        the maximum value. The duration is chosen so that the maximum value is
        not surpassed, but approached as closely as possible.

        Warning:
            The BlackmanWaveform assumes its values are in rad/µs for the
            area calculation. If this is not the case, the 'max_val' and 'area'
            values should be scaled accordingly.

        Args:
            max_val: The maximum value threshold (in rad/µs). If
                negative, it is taken as the lower bound i.e. the minimum
                value that can be reached. The sign of `max_val` must match the
                sign of `area`.
            area: The area under the waveform.
        """
        max_val = cast(float, max_val)
        assert not isinstance(area, Parametrized)
        area_float = _cast_check(float, area, "area")
        area_sign = np.sign(area_float)
        if np.sign(max_val) != area_sign:
            raise ValueError(
                "The maximum value and the area must have matching signs."
            )

        # Deal only with positive areas
        area = pm.AbstractArray(area, dtype=float) * float(area_sign)
        max_val *= float(area_sign)

        # A normalized Blackman waveform has an area of 0.42 * duration
        duration = np.ceil(float(area) / (0.42 * max_val) * 1e3)  # in ns
        wf = cls(duration, area)
        previous_wf = None

        # Adjust for rounding errors to make sure max_val is not surpassed
        while float(wf._scaling) > max_val:
            duration += 1
            previous_wf = wf
            wf = cls(duration, area)

        # According to the documentation for numpy.blackman(), "the value one
        # appears only if the number of samples is odd". Hence, the previous
        # even duration can reach a maximal value closer to max_val.
        if (
            previous_wf is not None
            and duration % 2 == 1
            and np.max(wf.samples.as_array(detach=True))
            < np.max(previous_wf.samples.as_array(detach=True))
            <= max_val
        ):
            wf = previous_wf

        # Restore original sign to the waveform
        return wf if area_sign != -1 else cast(BlackmanWaveform, -wf)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> pm.AbstractArray:
        """The value at each time step that describes the waveform.

        Returns:
            A numpy array with a value for each time step.
        """
        return self._norm_samples * self._scaling

    def change_duration(self, new_duration: int) -> BlackmanWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration: The duration of the new waveform.

        Returns:
            The new waveform with the same area but a new
            duration.
        """
        return BlackmanWaveform(new_duration, self._area)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._area)

    def _to_abstract_repr(self) -> dict[str, Any]:
        return abstract_repr("BlackmanWaveform", self._duration, self._area)

    def __str__(self) -> str:
        return f"Blackman(Area: {float(self._area):.3g})"

    def __repr__(self) -> str:
        return (
            f"BlackmanWaveform({self._duration} ns, "
            f"Area: {float(self._area):.3g})"
        )

    def __mul__(self, other: float | ArrayLike) -> BlackmanWaveform:
        return BlackmanWaveform(
            self._duration, self._area * pm.AbstractArray(other, dtype=float)
        )


class InterpolatedWaveform(Waveform):
    """A waveform created from interpolation of a set of data points.

    Args:
        duration: The waveform duration (in ns).
        values: Values of the interpolation points. Must be a list of castable
            to float or a parametrized object.
        times: Fractions of the total duration (between 0
            and 1), indicating where to place each value on the time axis. Must
            be a list of castable to float or a parametrized object. If
            not given, the values are spread evenly throughout the full
            duration of the waveform.
        interpolator: The SciPy interpolation class
            to use. Supports "PchipInterpolator" and "interp1d".
        **interpolator_kwargs: Extra parameters to give to the chosen
            interpolator class.
    """

    def __new__(cls, *args, **kwargs):  # type: ignore
        """Creates InterpolatedWaveform or ParamObj depending on the input."""
        cls._check_values_times(
            args[1] if len(args) >= 2 else kwargs["values"],
            args[2] if len(args) >= 3 else kwargs.get("times", None),
        )
        for x in itertools.chain(args, kwargs.values()):
            if isinstance(x, Parametrized):
                return ParamObj(cls, *args, **kwargs)
        else:
            return object.__new__(cls)

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
            times = cast(ArrayLike, times)
            times_ = np.array(times, dtype=float)
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

    @staticmethod
    def _check_values_times(
        values: Union[ArrayLike, Parametrized],
        times: Optional[Union[ArrayLike, Parametrized]] = None,
    ) -> None:
        """Check whether the types of values and times are valid."""

        def _err_message(argument_name: str) -> str:
            return (
                f"`{argument_name}` must be a parametrized object or a "
                "sequence of elements castable to float. To make a sequence"
                " of parametrized objects, declare a variable with the "
                "desired size."
            )

        if not isinstance(values, Parametrized):
            try:
                values_ = np.array(values, dtype=float)
            except TypeError as e:
                raise TypeError(_err_message("values")) from e
        if times is None or isinstance(times, Parametrized):
            return
        try:
            times = cast(ArrayLike, times)
            times_ = np.array(times, dtype=float)
        except TypeError as e:
            raise TypeError(_err_message("times")) from e
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
        if (
            not isinstance(values, Parametrized)
            and times_.size != values_.size
        ):
            raise ValueError(
                "When specified, the number of time coordinates in `times`"
                f" ({times_.size}) must match the number of `values` "
                f"({values_.size})."
            )

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> pm.AbstractArray:
        """The value at each time step that describes the waveform."""
        samples = self._interp_func(np.arange(self._duration))
        value_range = np.max(np.abs(samples))
        decimals = int(
            min(np.finfo(samples.dtype).precision - np.log10(value_range), 9)
        )  # Reduces decimal values below 9 for large ranges
        return pm.AbstractArray(np.round(samples, decimals=decimals))

    @property
    def interp_function(
        self,
    ) -> Union[interpolate.PchipInterpolator, interpolate.interp1d]:
        """The interpolating function."""
        return self._interp_func

    @property
    def data_points(self) -> np.ndarray:
        """Points (t[ns], value[arb. units]) that define the interpolation."""
        return self._data_pts.copy()

    def change_duration(self, new_duration: int) -> InterpolatedWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration: The duration of the new waveform.

        Returns:
            The new waveform with the same coordinates
            for interpolation but a new duration.
        """
        return InterpolatedWaveform(new_duration, self._values, **self._kwargs)

    def _plot(
        self,
        ax: Axes,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        channel: Optional[Channel] = None,
        label: str = "",
        start_t: int = 0,
    ) -> None:
        super()._plot(
            ax,
            ylabel,
            color=color,
            channel=channel,
            label=label,
            start_t=start_t,
        )
        if not channel:
            ax.scatter(
                self._data_pts[:, 0] + start_t, self._data_pts[:, 1], c=color
            )

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._values, **self._kwargs)

    def _to_abstract_repr(self) -> dict[str, Any]:
        if self._kwargs["interpolator"] != "PchipInterpolator" or set(
            self._kwargs
        ) - {"times", "interpolator"}:
            raise AbstractReprError(
                "Export of an InterpolatedWaveform is only supported for the "
                "'PchipInterpolator' and without any 'interpolator_kwargs'."
            )
        return abstract_repr(
            "InterpolatedWaveform",
            self._duration,
            self._values,
            times=self._times,
        )

    def __str__(self) -> str:
        coords = [f"({int(x)}, {y:.4g})" for x, y in self.data_points]
        return f"InterpolatedWaveform(Points: {', '.join(coords)})"

    def __repr__(self) -> str:
        interp_str = f", Interpolator={self._kwargs['interpolator']})"
        return self.__str__()[:-1] + interp_str

    def __mul__(self, other: float | ArrayLike) -> InterpolatedWaveform:
        return InterpolatedWaveform(
            self._duration,
            self._values * np.array(other, dtype=float),
            **self._kwargs,
        )


class KaiserWaveform(Waveform):
    """A Kaiser window of a specified duration and beta parameter.

    For more information on the Kaiser window and the beta parameter,
    check the numpy documentation for the kaiser(M, beta) function:
    https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html

    Warning:
        The KaiserWaveform assumes its values are in rad/µs for the
        area calculation. If this is not the case, the 'area'
        value should be scaled accordingly.


    Args:
        duration: The waveform duration (in ns).
        area: The integral of the waveform. Can be negative,
            in which case it takes the positive waveform and changes the sign
            of all its values.
        beta: The beta parameter of the Kaiser window.
            The default value is 14.
    """

    def __init__(
        self,
        duration: Union[int, Parametrized],
        area: Union[float, pm.TensorLike, Parametrized],
        beta: Optional[Union[float, Parametrized]] = 14.0,
    ):
        """Initializes a Kaiser waveform."""
        super().__init__(duration)

        assert not isinstance(area, Parametrized)
        _cast_check(float, area, "area")
        self._area = pm.AbstractArray(area, dtype=float)

        beta = cast(float, beta)
        # This makes sure 'beta' is not a tensor that requires grad
        pm.AbstractArray(beta).as_array()
        self._beta = _cast_check(float, beta, "beta")

        if self._beta < 0.0:
            raise ValueError(
                f"The beta parameter (`beta` = {self._beta})"
                " must be greater than 0."
            )

        self._norm_samples = pm.AbstractArray(
            np.clip(np.kaiser(self._duration, self._beta), 0, np.inf)
        )

        self._scaling = self._area / pm.sum(self._norm_samples) * 1e3

    @classmethod
    @parametrize
    def from_max_val(
        cls,
        max_val: Union[float, Parametrized],
        area: Union[float, pm.TensorLike, Parametrized],
        beta: Optional[Union[float, Parametrized]] = 14.0,
    ) -> KaiserWaveform:
        """Creates a Kaiser waveform with a threshold on the maximum value.

        Instead of defining a duration, the waveform is defined by its area and
        the maximum value. The duration is chosen so that the maximum value is
        not surpassed, but approached as closely as possible.

        Warning:
            The KaiserWaveform assumes its values are in rad/µs for the
            area calculation. If this is not the case, the 'max_val' and 'area'
            values should be scaled accordingly.

        Args:
            max_val: The maximum value threshold (in rad/µs). If
                negative, it is taken as the lower bound i.e. the minimum
                value that can be reached. The sign of `max_val` must match the
                sign of `area`.
            area: The area under the waveform.
            beta: The beta parameter of the Kaiser window.
                The default value is 14.
        """
        max_val = cast(float, max_val)
        assert not isinstance(area, Parametrized)
        area_float = _cast_check(float, area, "area")
        beta = cast(float, beta)

        if np.sign(max_val) != np.sign(area_float):
            raise ValueError(
                "The maximum value and the area must have matching signs."
            )

        # All computations will be done on a positive area
        area = pm.AbstractArray(area, dtype=float)
        is_negative: bool = area_float < 0
        if is_negative:
            area_float = -area_float
            max_val = -max_val

        # Compute the ratio area / duration for a long duration
        # and use this value for a first guess of the best duration

        ratio: float = max_val * np.sum(np.kaiser(100, beta)) / 100
        duration_guess: int = int(area_float * 1000.0 / ratio)

        duration_best: int = 0

        if duration_guess < 11:
            # Because of the seesawing effect on short durations,
            # all solutions must be tested to find the best one

            max_val_best: float = 0
            for duration in range(1, 16):
                kaiser_temp = np.kaiser(duration, beta)
                scaling_temp = 1000 * area_float / np.sum(kaiser_temp)
                max_val_temp = np.max(kaiser_temp) * scaling_temp
                if max_val_best < max_val_temp <= max_val:
                    max_val_best = max_val_temp
                    duration_best = duration

        else:
            # Start with a waveform based on the duration guess

            kaiser_guess = np.kaiser(duration_guess, beta)
            scaling_guess = 1000 * area_float / np.sum(kaiser_guess)
            max_val_temp = np.max(kaiser_guess) * scaling_guess

            # Increase or decrease duration depending on
            # the max value for the guessed duration

            step = 1 if np.max(kaiser_guess) * scaling_guess >= max_val else -1
            duration = duration_guess

            while np.sign(max_val_temp - max_val) == step:
                duration += step
                kaiser_temp = np.kaiser(duration, beta)
                scaling = 1000 * area_float / np.sum(kaiser_temp)
                max_val_temp = np.max(kaiser_temp) * scaling

            duration_best = duration if step == 1 else duration + 1

        return cls(duration_best, area, beta)

    @property
    def duration(self) -> int:
        """The duration of the pulse (in ns)."""
        return self._duration

    @cached_property
    def _samples(self) -> pm.AbstractArray:
        """The value at each time step that describes the waveform.

        Returns:
            A numpy array with a value for each time step.
        """
        return self._norm_samples * self._scaling

    def change_duration(self, new_duration: int) -> KaiserWaveform:
        """Returns a new waveform with modified duration.

        Args:
            new_duration: The duration of the new waveform.

        Returns:
            The new waveform with the same area and beta but a new
            duration.
        """
        return KaiserWaveform(new_duration, self._area, self._beta)

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._duration, self._area, self._beta)

    def _to_abstract_repr(self) -> dict[str, Any]:
        return abstract_repr(
            "KaiserWaveform", self._duration, self._area, beta=self._beta
        )

    def __str__(self) -> str:
        return (
            f"Kaiser({self._duration} ns, "
            f"Area: {float(self._area):.3g}, Beta: {self._beta:.3g})"
        )

    def __repr__(self) -> str:
        return (
            f"KaiserWaveform(duration: {self._duration}, "
            f"area: {float(self._area):.3g}, beta: {self._beta:.3g})"
        )

    def __mul__(self, other: float | ArrayLike) -> KaiserWaveform:
        return KaiserWaveform(
            self._duration,
            self._area * pm.AbstractArray(other, dtype=float),
            self._beta,
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
