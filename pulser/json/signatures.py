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


SIGNATURES: dict[str, PulserSignature] = {
    # Waveforms
    "CompositeWaveform": PulserSignature(var_pos="waveforms"),
    "CustomWaveform": PulserSignature(pos=("samples",)),
    "ConstantWaveform": PulserSignature(pos=("duration", "value")),
    "RampWaveform": PulserSignature(pos=("duration", "start", "stop")),
    "BlackmanWaveform": PulserSignature(pos=("duration", "area")),
    "BlackmanWaveform.from_max_val": PulserSignature(pos=("max_val", "area")),
    "InterpolatedWaveform": PulserSignature(
        pos=("duration", "values"), keyword=("times", "interpolator")
    ),
    "KaiserWaveform": PulserSignature(
        pos=("duration", "area"), keyword=("beta",)
    ),
    "KaiserWaveform.from_max_val": PulserSignature(
        pos=("max_val", "area"), keyword=("beta",)
    ),
    # Pulse
    "Pulse": PulserSignature(
        pos=("amplitude", "detuning", "phase"), keyword=("post_phase_shift",)
    ),
    "Pulse.ConstantDetuning": PulserSignature(
        pos=("amplitude", "detuning", "phase"), keyword=("post_phase_shift",)
    ),
    "Pulse.ConstantAmplitude": PulserSignature(
        pos=("amplitude", "detuning", "phase"), keyword=("post_phase_shift",)
    ),
    "Pulse.ConstantPulse": PulserSignature(
        pos=("duration", "amplitude", "detuning", "phase"),
        keyword=("post_phase_shift",),
    ),
}

EXTRA_PROPERTIES: dict[str, dict[str, str]] = {
    "CompositeWaveform": {"type": "composite"},
    "CustomWaveform": {"kind": "custom"},
    "ConstantWaveform": {"kind": "constant"},
    "RampWaveform": {"kind": "ramp"},
    "BlackmanWaveform": {"kind": "blackman"},
    "BlackmanWaveform.from_max_val": {"kind": "blackman"},  # TODO: Confirm
    "InterpolatedWaveform": {"kind": "interpolated"},
    "KaiserWaveform": {"kind": "kaiser"},
    "KaiserWaveform.from_max_val": {"kind": "kaiser"},  # TODO: Confirm
}
