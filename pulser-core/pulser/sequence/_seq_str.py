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
"""Function for representing the sequence in a string."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pulser.pulse import Pulse

if TYPE_CHECKING:
    from pulser.sequence.sequence import Sequence


def seq_to_str(sequence: Sequence) -> str:
    """Generates the string representation of a sequence."""
    full = ""
    pulse_line = "t: {}->{} | {} | Targets: {}\n"
    target_line = "t: {}->{} | Target: {} | Phase Reference: {}\n"
    delay_line = "t: {}->{} | Delay \n"
    det_delay_line = "t: {}->{} | Detuned Delay | Detuning: {:.3g} rad/µs\n"
    for ch, seq in sequence._schedule.items():
        basis = sequence.declared_channels[ch].basis
        full += f"Channel: {ch}\n"
        first_slot = True
        for ts in seq:
            if ts.type == "delay":
                full += delay_line.format(ts.ti, ts.tf)
                continue

            try:
                tgts = sorted(ts.targets)
            except TypeError:
                raise NotImplementedError(
                    "Can't print sequence with qubit IDs of different types."
                )
            tgt_txt = ", ".join(map(str, tgts))
            if isinstance(ts.type, Pulse):
                if seq.is_detuned_delay(ts.type):
                    det = ts.type.detuning[0]
                    full += det_delay_line.format(ts.ti, ts.tf, det)
                else:
                    full += pulse_line.format(ts.ti, ts.tf, ts.type, tgt_txt)
            elif ts.type == "target":
                phase = sequence._basis_ref[basis][tgts[0]].phase[ts.tf]
                if first_slot:
                    full += (
                        f"t: 0 | Initial targets: {tgt_txt} | "
                        + f"Phase Reference: {phase} \n"
                    )
                    first_slot = False
                else:
                    full += target_line.format(ts.ti, ts.tf, tgt_txt, phase)
        full += "\n"

    if hasattr(sequence, "_measurement"):
        full += f"Measured in basis: {sequence._measurement}"

    if sequence.is_parametrized():
        prelude = "Prelude\n-------\n" + full
        lines = ["Stored calls\n------------"]
        for i, c in enumerate(sequence._to_build_calls, 1):
            args = [str(a) for a in c.args]
            kwargs = [f"{key}={str(value)}" for key, value in c.kwargs.items()]
            lines.append(f"{i}. {c.name}({', '.join(args+kwargs)})")
        full = prelude + "\n\n".join(lines)

    return full
