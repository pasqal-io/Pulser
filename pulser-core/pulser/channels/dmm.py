# Copyright 2023 Pulser Development Team
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
"""Defines the detuning map modulator."""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Literal, Optional

import numpy as np

import pulser.math as pm
from pulser.channels.base_channel import Channel, _format_violation_times
from pulser.json.utils import get_dataclass_defaults
from pulser.pulse import Pulse
from pulser.register.weight_maps import DetuningMap

OPTIONAL_ABSTR_DMM_FIELDS = ["total_bottom_detuning", "min_avg_abs_detuning"]


# Deprecated DMM arguments and the arguments replacing them
_DEPRECATED_DETUNING_ARGS = {
    "bottom_detuning": "top_abs_detuning",
    "total_bottom_detuning": "total_top_abs_detuning",
}


def _flip_detuning_sign(detuning: float | None) -> float | None:
    """Converts a 'bottom' detuning into a maximum absolute one (or back)."""
    return -detuning if detuning else detuning


@dataclass(init=True, frozen=True)
class DMM(Channel):
    """Defines a Detuning Map Modulator (DMM) Channel.

    A Detuning Map Modulator can be used to define `Global` detuning Pulses
    (of zero amplitude and phase). These detuning Pulses are locally weighted
    by the weights of a `DetuningMap`, such that qubits experience a detuning
    map spot i.e. a detuning pulse equal to
    (detuning map weight on this qubit)*(detuning pulse value). The detuning
    of the pulses added to a DMM has to be negative in the 'ground-rydberg'
    basis and positive in the 'XY' basis, such that the absolute value of each
    detuning map spot is below `top_abs_detuning`, and that the absolute value
    of the sum of all the detuning map spots is below
    `total_top_abs_detuning`. By default, this Channel targets the transition
    between the ground and rydberg states, thus encoding the 'ground-rydberg'
    basis, but it can also target the transition between two rydberg states,
    encoding the 'XY' basis.

    Note:
        The protocol to add pulses to the DMM Channel is by default
        "no-delay".

    Args:
        top_abs_detuning: Maximum absolute value of the detuning on each
            detuning map spot (in rad/µs); must be positive. *Replaces
            'bottom_detuning', deprecated since v1.10.*
        total_top_abs_detuning: Maximum absolute value of the total detuning
            summed over all detuning map spots (in rad/µs); must be positive.
            *Replaces 'total_bottom_detuning', deprecated since v1.10.*
        min_avg_abs_detuning: The minimum acceptable value for the average
            absolute detuning (in rad/µs) applied on any detuning
            map spot (when not 0). Defaults to 0.
        basis: The basis addressed by this DMM, either 'ground-rydberg' or
            'XY'. Defaults to 'ground-rydberg'.
        clock_period: The duration of a clock cycle (in ns). The duration of a
            pulse or delay instruction is enforced to be a multiple of the
            clock cycle.
        min_duration: The shortest duration an instruction can take.
        max_duration: The longest duration an instruction can take.
        mod_bandwidth: The modulation bandwidth (in MHz), following Pulser's
            non-standard definition (the frequency at 75% amplitude
            attenuation).
    """

    top_abs_detuning: float | None = None
    total_top_abs_detuning: float | None = None
    min_avg_abs_detuning: float = 0.0
    basis: Literal["ground-rydberg", "XY"] = "ground-rydberg"
    addressing: Literal["Global"] = field(
        default="Global", init=False, repr=False
    )
    max_abs_detuning: Optional[float] = field(
        default=None, init=False, repr=False
    )
    max_amp: float = field(default=0, init=False, repr=False)
    min_retarget_interval: Optional[int] = field(
        default=None, init=False, repr=False
    )
    fixed_retarget_t: Optional[int] = field(
        default=None, init=False, repr=False
    )
    max_targets: Optional[int] = field(default=None, init=False, repr=False)
    propagation_dir: tuple[float, float, float] | None = field(
        default=None, init=False, repr=False
    )
    min_avg_amp: float = field(default=0, init=False, repr=False)
    custom_phase_jump_time: int | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.top_abs_detuning and self.top_abs_detuning < 0:
            raise ValueError(
                "'top_abs_detuning' must be positive (got "
                f"{self.top_abs_detuning})."
            )
        if self.total_top_abs_detuning:
            if self.total_top_abs_detuning < 0:
                raise ValueError(
                    "'total_top_abs_detuning' must be positive "
                    f"(got {self.total_top_abs_detuning})."
                )
            if (
                self.top_abs_detuning
                and self.top_abs_detuning > self.total_top_abs_detuning
            ):
                raise ValueError(
                    f"'total_top_abs_detuning' (got "
                    f"{self.total_top_abs_detuning}) must be higher than "
                    f"'top_abs_detuning' (got {self.top_abs_detuning})."
                )
        if self.min_avg_abs_detuning < 0:
            raise ValueError(
                "'min_avg_abs_detuning' must be non-negative "
                f"(got {self.min_avg_abs_detuning})."
            )
        if (
            self.top_abs_detuning
            and self.min_avg_abs_detuning >= self.top_abs_detuning
        ):
            raise ValueError(
                f"'min_avg_abs_detuning' (got {self.min_avg_abs_detuning}) "
                "must be lower than or equal to 'top_abs_detuning' (got "
                f"{self.top_abs_detuning})."
            )

    @property
    def _detuning_sign(self) -> int:
        """The sign the detuning applied by this DMM must have."""
        return 1 if self.basis == "XY" else -1

    @property
    def _internal_param_valid_options(self) -> dict[str, tuple[str, ...]]:
        """Internal parameters and their valid options."""
        return {
            **super()._internal_param_valid_options,
            "basis": ("ground-rydberg", "XY"),
        }

    @property
    def bottom_detuning(self) -> float | None:
        """Deprecated: use :attr:`top_abs_detuning` instead."""
        warnings.warn(
            "'bottom_detuning' is deprecated since pulser v1.10, use "
            "'top_abs_detuning' instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return _flip_detuning_sign(self.top_abs_detuning)

    @property
    def total_bottom_detuning(self) -> float | None:
        """Deprecated: use :attr:`total_top_abs_detuning` instead."""
        warnings.warn(
            "'total_bottom_detuning' is deprecated since pulser v1.10, use "
            "'total_top_abs_detuning' instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return _flip_detuning_sign(self.total_top_abs_detuning)

    def _undefined_fields(self) -> list[str]:
        optional = [
            "top_abs_detuning",
            "max_duration",
            "total_top_abs_detuning",
        ]
        return [field for field in optional if getattr(self, field) is None]

    def is_virtual(self) -> bool:
        """Whether the channel is virtual (i.e. partially defined)."""
        return bool(self._undefined_fields())

    def validate_pulse(
        self,
        pulse: Pulse,
        detuning_map: DetuningMap = DetuningMap(
            trap_coordinates=[(0, 0)], weights=[1.0]
        ),
    ) -> None:
        """Checks if a pulse can be executed via this DMM on a DetuningMap.

        Args:
            pulse: The pulse to validate.
            detuning_map: The detuning map on which the pulse is applied
                (defaults to a detuning map with weight 1.0).
        """
        super().validate_pulse(pulse)
        round_detuning = pm.round(pulse.detuning.samples, 6).as_array(
            detach=True
        )
        sign = self._detuning_sign
        banned = "negative" if sign > 0 else "positive"
        extremum, keep = (
            ("maximum", "below") if sign > 0 else ("minimum", "above")
        )
        # Check that the detuning has the sign imposed by the basis
        wrong_sign = sign * round_detuning < 0
        if np.any(wrong_sign):
            raise ValueError(
                f"The detuning in a DMM must not be {banned}; it is "
                f"{banned} at "
                f"{_format_violation_times(wrong_sign)} in detuning "
                f"{pulse.detuning!r}."
            )
        # Check that detuning on each atom is within top_abs_detuning
        abs_detuning = np.max(sign * round_detuning)
        extreme_detuning = sign * abs_detuning
        max_weight = np.max(detuning_map.weights)
        if (
            self.top_abs_detuning is not None
            and max_weight * abs_detuning > self.top_abs_detuning
        ):
            raise ValueError(
                f"For a detuning map with a maximum weight of {max_weight},"
                f" a DMM pulse with {extremum} detuning {extreme_detuning} "
                "rad/µs exceeds the local maximum absolute "
                f"detuning of the DMM ({self.top_abs_detuning} rad/µs). "
                f"To respect this constraint, keep the detuning {keep} "
                f"{sign * self.top_abs_detuning/max_weight} rad/µs. Got pulse "
                f"{pulse!r}."
            )
        # Check that the total detuning is within total_top_abs_detuning
        sum_weight = np.sum(detuning_map.weights)
        if (
            self.total_top_abs_detuning is not None
            and sum_weight * abs_detuning > self.total_top_abs_detuning
        ):
            raise ValueError(
                "For a detuning map with a total summed weight of "
                f"{sum_weight}, the total applied detuning from a DMM pulse "
                f"with {extremum} detuning {extreme_detuning} rad/µs exceeds"
                " the total maximum absolute detuning "
                f"of the DMM ({self.total_top_abs_detuning} rad/µs). "
                f"To respect this constraint, keep the detuning {keep} "
                f"{sign * self.total_top_abs_detuning/sum_weight} rad/µs."
            )

        weights_arr = np.array(detuning_map.weights)
        non_zero_weight_inds = np.nonzero(weights_arr)
        assert len(non_zero_weight_inds) == 1, "Weights array is not 1D"
        if len(non_zero_weight_inds[0]) == 0:
            # All weights are zero, skip min_avg_abs_detuning validation
            return

        avg_abs_detuning = np.average(np.abs(round_detuning))
        min_non_zero_weight = np.min(weights_arr[non_zero_weight_inds])
        if (
            0
            < min_non_zero_weight * avg_abs_detuning
            < self.min_avg_abs_detuning
        ):
            raise ValueError(
                "For a detuning map with a minimum non-zero weight of "
                f"{min_non_zero_weight}, a DMM pulse with an average "
                f"absolute detuning of {avg_abs_detuning:.3g} rad/µs does not"
                " respect the minimum threshold for the average absolute "
                f"detuning of the DMM ({self.min_avg_abs_detuning} rad/µs)."
                f" Got pulse {pulse!r}."
            )

    def _to_abstract_repr(self, id: str) -> dict[str, Any]:
        all_fields = fields(self)
        defaults = get_dataclass_defaults(all_fields)
        params = super()._to_abstract_repr(id)
        # The abstract representation keeps the deprecated 'bottom' arguments
        for old, new in _DEPRECATED_DETUNING_ARGS.items():
            params[old] = _flip_detuning_sign(params.pop(new))
            defaults[old] = defaults[new]
        for p in OPTIONAL_ABSTR_DMM_FIELDS:
            if params[p] == defaults[p]:
                params.pop(p, None)
        return params


def _wrap_init_for_deprecated_args(
    original_init: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap __init__ to accept deprecated arguments.

    Supported deprecated parameters:
    - bottom_detuning
    - total_bottom_detuning

    """

    @functools.wraps(original_init)
    def wrapped_init(
        self: Any,
        *args: Any,
        bottom_detuning: float | None = None,
        total_bottom_detuning: float | None = None,
        **kwargs: Any,
    ) -> None:
        deprecated_args = {
            "bottom_detuning": bottom_detuning,
            "total_bottom_detuning": total_bottom_detuning,
        }
        for name, value in deprecated_args.items():
            if value is None:
                continue
            new_name = _DEPRECATED_DETUNING_ARGS[name]
            warnings.warn(
                f"'{name}' is deprecated since pulser v1.10, use "
                f"'{new_name}' instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            if value > 0:
                raise ValueError(f"'{name}' must be negative (got {value}).")
            # Takes precedence, since 'replace()' also gives the new argument
            kwargs[new_name] = _flip_detuning_sign(value)
        original_init(self, *args, **kwargs)

    return wrapped_init


DMM.__init__ = _wrap_init_for_deprecated_args(DMM.__init__)  # type: ignore[method-assign]  # noqa: E501


def _dmm_id_from_name(dmm_name: str) -> str:
    """Converts a dmm_name into a dmm_id.

    As a reminder the dmm_name is generated automatically from dmm_id
    as dmm_id_{number of times dmm_id has been called}.

    Args:
        dmm_name: The dmm_name to convert.

    Returns:
        The associated dmm_id.
    """
    return "_".join(dmm_name.split("_")[0:2])


def _get_dmm_name(dmm_id: str, channels: list[str]) -> str:
    """Get the dmm_name to add a dmm_id to a list of channels.

    Counts the number of channels starting by dmm_id, generates the
    dmm_name as dmm_id_{number of times dmm_id has been called}.

    Args:
        dmm_id: the id of the DMM to add to the list of channels.
        channels: a list of channel names.

    Returns:
        The associated dmm_name.
    """
    dmm_count = len(
        [key for key in channels if _dmm_id_from_name(key) == dmm_id]
    )
    if dmm_count == 0:
        return dmm_id
    return dmm_id + f"_{dmm_count}"
