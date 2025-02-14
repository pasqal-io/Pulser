# Copyright 2022 Pulser Development Team
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
"""Defines the Channel ABC."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Literal, Optional, Type, TypeVar, cast, get_args

import numpy as np
from numpy.typing import ArrayLike

import pulser.math as pm
from pulser.channels.eom import MODBW_TO_TR, BaseEOM
from pulser.json.utils import get_dataclass_defaults, obj_to_dict
from pulser.pulse import Pulse

# Warnings of adjusted waveform duration appear just once
warnings.filterwarnings("once", "A duration of")

ChannelType = TypeVar("ChannelType", bound="Channel")

OPTIONAL_ABSTR_CH_FIELDS = (
    "min_avg_amp",
    "custom_phase_jump_time",
    "propagation_dir",
)

# States ranked in decreasing order of their associated eigenenergy
States = Literal["u", "d", "r", "g", "h", "x"]

STATES_RANK = get_args(States)

EIGENSTATES: dict[str, list[States]] = {
    "ground-rydberg": ["r", "g"],
    "digital": ["g", "h"],
    "XY": ["u", "d"],
}


def get_states_from_bases(bases: Collection[str]) -> list[States]:
    """The states associated to a list of bases, ranked by their energies."""
    all_states = set().union(*(set(EIGENSTATES[basis]) for basis in bases))
    return [state for state in STATES_RANK if state in all_states]


@dataclass(init=True, frozen=True)
class Channel(ABC):
    """Base class of a hardware channel.

    Not to be initialized itself, but rather through a child class and the
    ``Local`` or ``Global`` classmethods.

    Args:
        addressing: "Local" or "Global".
        max_abs_detuning: Maximum possible detuning (in rad/µs), in absolute
            value.
        max_amp: Maximum pulse amplitude (in rad/µs).
        min_retarget_interval: Minimum time required between the ends of two
            target instructions (in ns).
        fixed_retarget_t: Time taken to change the target (in ns).
        max_targets: How many qubits can be addressed at once by the same beam.
        clock_period: The duration of a clock cycle (in ns). The duration of a
            pulse or delay instruction is enforced to be a multiple of the
            clock cycle.
        min_duration: The shortest duration an instruction can take.
        max_duration: The longest duration an instruction can take.
        min_avg_amp: The minimum average amplitude of a pulse (when not zero).
        mod_bandwidth: The modulation bandwidth at -3dB (50% reduction), in
            MHz.
        custom_phase_jump_time: An optional custom value for the phase jump
            time that overrides the default value estimated from the modulation
            bandwidth. It is not enforced in EOM mode.
        propagation_dir: The propagation direction of the beam associated with
            the channel, given as a vector in 3D space.

    Example:
        To create a channel targeting the 'ground-rydberg' transition globally,
        call ``Rydberg.Global(...)``.
    """

    addressing: Literal["Global", "Local"]
    max_abs_detuning: Optional[float]
    max_amp: Optional[float]
    min_retarget_interval: Optional[int] = None
    fixed_retarget_t: Optional[int] = None
    max_targets: Optional[int] = None
    clock_period: int = 1  # ns
    min_duration: int = 1  # ns
    max_duration: Optional[int] = int(1e8)  # ns
    min_avg_amp: float = 0
    mod_bandwidth: Optional[float] = None  # MHz
    custom_phase_jump_time: int | None = None
    eom_config: Optional[BaseEOM] = field(init=False, default=None)
    propagation_dir: tuple[float, float, float] | None = None

    @property
    def name(self) -> str:
        """The name of the channel."""
        return type(self).__name__

    @property
    @abstractmethod
    def basis(self) -> str:
        """The addressed basis name."""
        pass

    @property
    def eigenstates(self) -> list[States]:
        r"""The eigenstates associated with the basis.

        Returns a tuple of labels, ranked in decreasing order
        of their associated eigenenergy, as such:

        .. list-table::
            :align: center
            :widths: 50 35 35
            :header-rows: 1

            * - Name
              - Eigenstate (see :doc:`/conventions`)
              - Associated label
            * - Up state
              - :math:`|0\rangle`
              - ``"u"``
            * - Down state
              - :math:`|1\rangle`
              - ``"d"``
            * - Rydberg state
              - :math:`|r\rangle`
              - ``"r"``
            * - Ground state
              - :math:`|g\rangle`
              - ``"g"``
            * - Hyperfine state
              - :math:`|h\rangle`
              - ``"h"``
            * - Error state
              - :math:`|x\rangle`
              - ``"x"``
        """
        return EIGENSTATES[self.basis]

    @property
    def _internal_param_valid_options(self) -> dict[str, tuple[str, ...]]:
        """Internal parameters and their valid options."""
        return dict(
            name=("Rydberg", "Raman", "Microwave", "DMM"),
            basis=tuple(EIGENSTATES.keys()),
            addressing=("Local", "Global"),
        )

    def __post_init__(self) -> None:
        """Validates the channel's parameters."""
        for param, options in self._internal_param_valid_options.items():
            value = getattr(self, param)
            assert (
                value in options
            ), f"The channel {param} must be one of {options}, not {value}."

        parameters = [
            "max_amp",
            "max_abs_detuning",
            "clock_period",
            "min_duration",
            "max_duration",
            "mod_bandwidth",
            "min_avg_amp",
            "custom_phase_jump_time",
        ]
        non_negative = [
            "max_amp",
            "max_abs_detuning",
            "min_retarget_interval",
            "fixed_retarget_t",
            "min_avg_amp",
            "custom_phase_jump_time",
        ]
        local_only = [
            "min_retarget_interval",
            "fixed_retarget_t",
            "max_targets",
        ]
        optional = [
            "max_amp",
            "max_abs_detuning",
            "max_duration",
            "mod_bandwidth",
            "max_targets",
            "custom_phase_jump_time",
        ]

        if self.addressing == "Global":
            for p in local_only:
                assert (
                    getattr(self, p) is None
                ), f"'{p}' must be left as None in a Global channel."
        else:
            assert self.addressing == "Local"
            parameters += local_only
            if self.propagation_dir is not None:
                raise NotImplementedError(
                    "'propagation_dir' must be left as None in Local channels."
                )

        for param in parameters:
            value = getattr(self, param)
            if param in optional:
                prelude = "When defined, "
                valid = value is None
            elif value is None:
                raise TypeError(
                    f"'{param}' can't be None in a '{self.addressing}' "
                    "channel."
                )
            else:
                prelude = ""
                valid = False
            if param in non_negative:
                comp = "greater than or equal to zero"
                valid = valid or value >= 0
            else:
                comp = "greater than zero"
                valid = valid or value > 0
            msg = prelude + f"'{param}' must be {comp}, not {value}."
            if not valid:
                raise ValueError(msg)

        if (
            self.max_duration is not None
            and self.max_duration < self.min_duration
        ):
            raise ValueError(
                f"When defined, 'max_duration'({self.max_duration}) must be"
                " greater than or equal to 'min_duration'"
                f"({self.min_duration})."
            )
        if (
            self.mod_bandwidth is not None
            and self.mod_bandwidth > MODBW_TO_TR * 1e3
        ):
            raise NotImplementedError(
                f"'mod_bandwidth' must be lower than {MODBW_TO_TR*1e3} MHz"
            )

        if self.eom_config is not None and self.mod_bandwidth is None:
            raise ValueError(
                "'eom_config' can't be defined in a Channel without a "
                "modulation bandwidth."
            )

        if self.propagation_dir is not None:
            dir_vector = np.array(self.propagation_dir, dtype=float)
            if dir_vector.size != 3 or np.sum(dir_vector) == 0.0:
                raise ValueError(
                    "'propagation_dir' must be given as a non-zero 3D vector;"
                    f" got {self.propagation_dir} instead."
                )
            # Make sure it's stored as a tuple
            object.__setattr__(
                self, "propagation_dir", tuple(self.propagation_dir)
            )

    @property
    def rise_time(self) -> int:
        """The rise time (in ns).

        Defined as the time taken to go from 10% to 90% output in response to
        a step change in the input.
        """
        if self.mod_bandwidth:
            return int(MODBW_TO_TR / self.mod_bandwidth * 1e3)
        else:
            return 0

    @property
    def phase_jump_time(self) -> int:
        """Time taken to change the phase between consecutive pulses (in ns).

        Corresponds to two times the rise time when `custom_phase_jump_time`
        is not defined.
        """
        return int(
            self.rise_time * 2
            if self.custom_phase_jump_time is None
            else self.custom_phase_jump_time
        )

    def is_virtual(self) -> bool:
        """Whether the channel is virtual (i.e. partially defined)."""
        return bool(self._undefined_fields())

    def supports_eom(self) -> bool:
        """Whether the channel supports EOM mode operation."""
        return hasattr(self, "eom_config") and self.eom_config is not None

    def _undefined_fields(self) -> list[str]:
        optional = [
            "max_amp",
            "max_abs_detuning",
            "max_duration",
        ]
        if self.addressing == "Local":
            optional.append("max_targets")
        return [field for field in optional if getattr(self, field) is None]

    @classmethod
    def Local(
        cls: Type[ChannelType],
        max_abs_detuning: Optional[float],
        max_amp: Optional[float],
        min_retarget_interval: int = 0,
        fixed_retarget_t: int = 0,
        max_targets: Optional[int] = None,
        **kwargs: Any,
    ) -> ChannelType:
        """Initializes the channel with local addressing.

        Args:
            max_abs_detuning: Maximum possible detuning (in rad/µs), in
                absolute value.
            max_amp: Maximum pulse amplitude (in rad/µs).
            min_retarget_interval: Minimum time required between two
                target instructions (in ns).
            fixed_retarget_t: Time taken to change the target (in ns).
            max_targets: Maximum number of atoms the channel can target
                simultaneously.

        Keyword Args:
            clock_period(int, default=4): The duration of a clock cycle
                (in ns). The duration of a pulse or delay instruction is
                enforced to be a multiple of the clock cycle.
            min_duration(int, default=1): The shortest duration an
                instruction can take.
            max_duration(Optional[int], default=10000000): The longest
                duration an instruction can take.
            mod_bandwidth(Optional[float], default=None): The modulation
                bandwidth at -3dB (50% reduction), in MHz.
            min_avg_amp: The minimum average amplitude of a pulse (when not
                zero).
            custom_phase_jump_time: An optional custom value for the phase jump
                time that overrides the default value estimated from the
                modulation bandwidth. It is not enforced in EOM mode.
        """
        # Can't initialize a channel whose addressing is determined internally
        for cls_field in fields(cls):
            if cls_field.name == "addressing":
                break
        if not cls_field.init and cls_field.default is not MISSING:
            raise NotImplementedError(
                f"{cls} cannot be initialized from `Local` method."
            )
        return cls(
            "Local",
            max_abs_detuning,
            max_amp,
            min_retarget_interval,
            fixed_retarget_t,
            max_targets,
            **kwargs,
        )

    @classmethod
    def Global(
        cls: Type[ChannelType],
        max_abs_detuning: Optional[float],
        max_amp: Optional[float],
        # TODO: Impose a default propagation_dir in pulser-core 1.3
        **kwargs: Any,
    ) -> ChannelType:
        """Initializes the channel with global addressing.

        Args:
            max_abs_detuning: Maximum possible detuning (in rad/µs), in
                absolute value.
            max_amp: Maximum pulse amplitude (in rad/µs).

        Keyword Args:
            clock_period(int, default=4): The duration of a clock cycle
                (in ns). The duration of a pulse or delay instruction is
                enforced to be a multiple of the clock cycle.
            min_duration(int, default=1): The shortest duration an
                instruction can take.
            max_duration(Optional[int], default=10000000): The longest
                duration an instruction can take.
            mod_bandwidth(Optional[float], default=None): The modulation
                bandwidth at -3dB (50% reduction), in MHz.
            min_avg_amp: The minimum average amplitude of a pulse (when not
                zero).
            custom_phase_jump_time: An optional custom value for the phase jump
                time that overrides the default value estimated from the
                modulation bandwidth. It is not enforced in EOM mode.
            propagation_dir: The propagation direction of the beam associated
                with the channel, given as a vector in 3D space.
        """
        # Can't initialize a channel whose addressing is determined internally
        for cls_field in fields(cls):
            if cls_field.name == "addressing":
                break
        if not cls_field.init and cls_field.default is not MISSING:
            raise NotImplementedError(
                f"{cls} cannot be initialized from `Global` method."
            )
        return cls("Global", max_abs_detuning, max_amp, **kwargs)

    def validate_duration(self, duration: int) -> int:
        """Validates and adapts the duration of an instruction on this channel.

        Args:
            duration: The duration to validate.

        Returns:
            The duration, potentially adapted to the channels specs.
        """
        try:
            _duration = int(duration)
        except (TypeError, ValueError):
            raise TypeError(
                "duration needs to be castable to an int but "
                "type %s was provided" % type(duration)
            )

        if duration < self.min_duration:
            raise ValueError(
                "duration has to be at least " + f"{self.min_duration} ns."
            )

        if self.max_duration is not None and duration > self.max_duration:
            raise ValueError(
                "duration can be at most " + f"{self.max_duration} ns."
            )

        if duration % self.clock_period != 0:
            _duration += self.clock_period - _duration % self.clock_period
            warnings.warn(
                f"A duration of {duration} ns is not a multiple of "
                f"the channel's clock period ({self.clock_period} "
                f"ns). It was rounded up to {_duration} ns.",
                stacklevel=4,
            )
        return _duration

    def validate_pulse(self, pulse: Pulse) -> None:
        """Checks if a pulse can be executed this channel.

        Args:
            pulse: The pulse to validate.
        """
        if not isinstance(pulse, Pulse):
            raise TypeError(
                f"'pulse' must be of type Pulse, not of type {type(pulse)}."
            )

        amp_samples_np = pulse.amplitude.samples.as_array(detach=True)
        if self.max_amp is not None and np.any(amp_samples_np > self.max_amp):
            raise ValueError(
                "The pulse's amplitude goes over the maximum "
                "value allowed for the chosen channel."
            )
        if self.max_abs_detuning is not None and np.any(
            np.round(
                np.abs(pulse.detuning.samples.as_array(detach=True)),
                decimals=6,
            )
            > self.max_abs_detuning
        ):
            raise ValueError(
                "The pulse's detuning values go out of the range "
                "allowed for the chosen channel."
            )
        avg_amp = np.average(amp_samples_np)
        if 0 < avg_amp < self.min_avg_amp:
            raise ValueError(
                "The pulse's average amplitude is below the chosen "
                f"channel's limit ({self.min_avg_amp})."
            )

    @property
    def _modulation_padding(self) -> int:
        """The padding added to the input signals before modulation.

        Defined in number of samples to pad before and after signal
        (i.e. the signal is extended by 2*_modulation_padding).
        """
        return self.rise_time

    def modulate(
        self,
        input_samples: ArrayLike,
        keep_ends: bool = False,
        eom: bool = False,
    ) -> pm.AbstractArray:
        """Modulates the input according to the channel's modulation bandwidth.

        Args:
            input_samples: The samples to modulate.
            keep_ends: Assume the end values of the samples were kept
                constant (i.e. there is no ramp from zero on the ends).
            eom: Whether to calculate the modulation using the EOM
                bandwidth.

        Returns:
            The modulated output signal.
        """
        if eom:
            if not self.supports_eom():
                raise TypeError(f"The channel {self} does not have an EOM.")
            eom_config = cast(BaseEOM, self.eom_config)
            mod_bandwidth = eom_config.mod_bandwidth
            mod_padding = eom_config.rise_time

        elif not self.mod_bandwidth:
            warnings.warn(
                f"No modulation bandwidth defined for channel '{self}',"
                " 'Channel.modulate()' returns the 'input_samples' unchanged.",
                stacklevel=2,
            )
            return pm.AbstractArray(input_samples)
        else:
            mod_bandwidth = self.mod_bandwidth
            mod_padding = self._modulation_padding

        if keep_ends:
            samples = pm.pad(
                input_samples, mod_padding + self.rise_time, mode="edge"
            )
        else:
            samples = pm.pad(input_samples, mod_padding)
        mod_samples = self.apply_modulation(samples, mod_bandwidth)
        if keep_ends:
            # Cut off the extra ends
            return mod_samples[self.rise_time : -self.rise_time]
        return mod_samples

    @staticmethod
    def apply_modulation(
        input_samples: ArrayLike, mod_bandwidth: float
    ) -> pm.AbstractArray:
        """Applies the modulation transfer fuction to the input samples.

        Note:
            This is strictly the application of the modulation transfer
            function. The samples should be padded beforehand.

        Args:
            input_samples: The samples to modulate.
            mod_bandwidth: The modulation bandwidth at -3dB (50% reduction),
                in MHz.
        """
        # The cutoff frequency (fc) and the modulation transfer function
        # are defined in https://tinyurl.com/bdeumc8k
        input_samples = pm.AbstractArray(input_samples)
        fc = mod_bandwidth * 1e-3 / np.sqrt(np.log(2))
        freqs = pm.fftfreq(input_samples.size)
        modulation = pm.exp(-(freqs**2) / fc**2)
        return pm.ifft(pm.fft(input_samples) * modulation).real

    def calc_modulation_buffer(
        self,
        input_samples: ArrayLike,
        mod_samples: ArrayLike,
        max_allowed_diff: float = 1e-2,
        eom: bool = False,
    ) -> tuple[int, int]:
        """Calculates the minimal buffers needed around a modulated waveform.

        Args:
            input_samples: The input samples.
            mod_samples: The modulated samples. Must be of size
                ``len(input_samples) + 2 * self.rise_time``.
            max_allowed_diff: The maximum allowed difference between
                the input and modulated samples at the end points.
            eom: Whether to calculate the modulation buffers with the EOM
                bandwidth.

        Returns:
            The minimum buffer times at the start and end of
            the samples, in ns.
        """
        if eom:
            if not self.supports_eom():
                raise TypeError(f"The channel {self} does not have an EOM.")
            tr = cast(BaseEOM, self.eom_config).rise_time
        else:
            if not self.mod_bandwidth:
                raise TypeError(
                    f"The channel {self} doesn't have a modulation bandwidth."
                )
            tr = self.rise_time
        samples = pm.pad(input_samples, tr)
        diffs = (
            abs(samples - mod_samples).as_array(detach=True)
            <= max_allowed_diff
        )
        try:
            # Finds the last index in the start buffer that's below the max
            # allowed diff. Considers that the waveform could start at the next
            # indice (hence the -1, since we are subtracting from tr)
            start = tr - np.argwhere(diffs[:tr])[-1][0] - 1
        except IndexError:
            start = tr
        try:
            # Finds the first index in the end buffer that's below the max
            # allowed diff. The index value found matches the minimum length
            # for this end buffer.
            end = np.argwhere(diffs[-tr:])[0][0]
        except IndexError:
            end = tr

        return start, end

    @property
    def _eom_buffer_time(self) -> int:
        # By definition, rise_time goes from 10% to 90%
        # Roughly 2*rise_time is enough to go from 0% to 100%
        # so we use that by default
        assert self.supports_eom(), "Can't define the EOM buffer time."
        return int(
            cast(BaseEOM, self.eom_config).custom_buffer_time
            or 2 * self.rise_time
        )

    @property
    def _eom_buffer_mod_bandwidth(self) -> float:
        # Takes half of the buffer time as the rise time
        rise_time_us = self._eom_buffer_time / 2 * 1e-3
        return MODBW_TO_TR / rise_time_us

    def __str__(self) -> str:
        config = (
            f".{self.addressing}(Max Absolute Detuning: "
            f"{self.max_abs_detuning}"
            f"{' rad/µs' if self.max_abs_detuning else ''}, "
            f"Max Amplitude: {self.max_amp}"
            f"{' rad/µs' if self.max_amp else ''}"
        )
        if self.addressing == "Local":
            config += (
                f", Minimum retarget time: {self.min_retarget_interval} ns, "
                f"Fixed retarget time: {self.fixed_retarget_t} ns"
            )
            if self.max_targets is not None:
                config += f", Max targets: {self.max_targets}"
        config += (
            f", Clock period: {self.clock_period} ns"
            f", Minimum pulse duration: {self.min_duration} ns"
        )
        if self.max_duration is not None:
            config += f", Maximum pulse duration: {self.max_duration} ns"
        if self.mod_bandwidth:
            config += f", Modulation Bandwidth: {self.mod_bandwidth} MHz"
        config += f", Supports EOM: {self.supports_eom()}"
        config += f", Basis: '{self.basis}')"
        return self.name + config

    def default_id(self) -> str:
        """Generates the default ID for indexing this channel in a Device."""
        return f"{self.name.lower()}_{self.addressing.lower()}"

    def _to_dict(self, _module: str = "pulser.channels") -> dict[str, Any]:
        params = {
            f.name: getattr(self, f.name) for f in fields(self) if f.init
        }
        return obj_to_dict(self, _module=_module, **params)

    def _to_abstract_repr(self, id: str) -> dict[str, Any]:
        all_fields = fields(self)
        defaults = get_dataclass_defaults(all_fields)
        params = {f.name: getattr(self, f.name) for f in all_fields}
        for p in OPTIONAL_ABSTR_CH_FIELDS:
            if params[p] == defaults[p]:
                params.pop(p, None)
        return {"id": id, "basis": self.basis, **params}
