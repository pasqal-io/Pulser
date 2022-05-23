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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PulserSignature:
    pos: tuple[str, ...] = field(default_factory=tuple)
    var_pos: Optional[str] = None
    keyword: tuple[str, ...] = field(default_factory=tuple)
    extra: dict[str, str] = field(default_factory=dict)


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
    # Special case operators
    "truediv": PulserSignature(
        pos=("lhs", "rhs"), extra=dict(expression="div")
    ),
    "round_": PulserSignature(pos=("lhs",), extra=dict(expression="round")),
}

BINARY_OPERATORS = ("add", "sub", "mul", "truediv", "pow", "mod")

UNARY_OPERATORS = (
    "neg",
    "abs",
    "ceil",
    "floor",
    "sqrt",
    "exp",
    "log2",
    "log",
    "sin",
    "cos",
    "tan",
)
