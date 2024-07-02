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
"""Defines the signatures of objects for the abstract representation."""
from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from pulser.parametrized.variable import Variable, VariableItem


@dataclass
class PulserSignature:
    """The signature of a Pulser object."""

    pos: tuple[str, ...] = field(default_factory=tuple)
    var_pos: Optional[str] = None
    keyword: tuple[str, ...] = field(default_factory=tuple)
    extra: dict[str, str] = field(default_factory=dict)

    def all_pos_args(self) -> tuple[str, ...]:
        """All potential positional arguments.

        Includes the keyword args if var_pos is None.
        """
        if self.var_pos is not None:
            return self.pos
        return (*self.pos, *self.keyword)


SIGNATURES: dict[str, PulserSignature] = {
    # Waveforms
    "CompositeWaveform": PulserSignature(
        var_pos="waveforms", extra=dict(kind="composite")
    ),
    "CustomWaveform": PulserSignature(
        pos=("samples",), extra=dict(kind="custom")
    ),
    "ConstantWaveform": PulserSignature(
        pos=("duration", "value"), extra=dict(kind="constant")
    ),
    "RampWaveform": PulserSignature(
        pos=("duration", "start", "stop"), extra=dict(kind="ramp")
    ),
    "BlackmanWaveform": PulserSignature(
        pos=("duration", "area"), extra=dict(kind="blackman")
    ),
    "BlackmanWaveform.from_max_val": PulserSignature(
        pos=("max_val", "area"), extra=dict(kind="blackman_max")
    ),
    "InterpolatedWaveform": PulserSignature(
        pos=("duration", "values"),
        keyword=("times",),
        extra=dict(kind="interpolated"),
    ),
    "KaiserWaveform": PulserSignature(
        pos=("duration", "area"), keyword=("beta",), extra=dict(kind="kaiser")
    ),
    "KaiserWaveform.from_max_val": PulserSignature(
        pos=("max_val", "area"),
        keyword=("beta",),
        extra=dict(kind="kaiser_max"),
    ),
    # Pulse
    "Pulse": PulserSignature(
        pos=("amplitude", "detuning", "phase"), keyword=("post_phase_shift",)
    ),
    "Pulse.ArbitraryPhase": PulserSignature(
        pos=("amplitude", "phase"), keyword=("post_phase_shift",)
    ),
    # Special case operators
    "truediv": PulserSignature(
        pos=("lhs", "rhs"), extra=dict(expression="div")
    ),
    "round_": PulserSignature(pos=("lhs",), extra=dict(expression="round")),
}


def _index_var(lhs: Variable, rhs: int) -> VariableItem:
    return lhs[rhs]


BINARY_OPERATORS: dict[str, Callable] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "pow": operator.pow,
    "mod": operator.mod,
    "index": _index_var,
}

UNARY_OPERATORS: dict[str, Callable] = {
    "neg": operator.neg,
    "abs": operator.abs,
    "ceil": np.ceil,
    "floor": np.floor,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log2": np.log2,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
}
