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
"""Base class for the backend interface."""
from __future__ import annotations

import typing
from abc import ABC, abstractmethod

from pulser.devices import Device
from pulser.result import Result
from pulser.sequence import Sequence

Results = typing.Sequence[Result]


class Backend(ABC):
    """The backend abstract base class."""

    def __init__(self, sequence: Sequence, mimic_qpu: bool = False) -> None:
        """Starts a new backend instance."""
        self.validate_sequence(sequence, mimic_qpu=mimic_qpu)
        self._sequence = sequence
        self._mimic_qpu = bool(mimic_qpu)

    @abstractmethod
    def run(self) -> Results | typing.Sequence[Results]:
        """Executes the sequence on the backend."""
        pass

    @staticmethod
    def validate_sequence(sequence: Sequence, mimic_qpu: bool = False) -> None:
        """Validates a sequence prior to submission."""
        if not isinstance(sequence, Sequence):
            raise TypeError(
                "'sequence' should be a `Sequence` instance"
                f", not {type(sequence)}."
            )
        if not mimic_qpu:
            return

        if not isinstance(device := sequence.device, Device):
            raise TypeError(
                "To be sent to a QPU, the device of the sequence "
                "must be a real device, instance of 'Device'."
            )
        reg = sequence.get_register(include_mappable=True)
        if device.requires_layout and (layout := reg.layout) is None:
            raise ValueError(
                f"'{device.name}' requires the sequence's register to be"
                " defined from a `RegisterLayout`."
            )
        if (
            not device.accepts_new_layouts
            and layout is not None
            and layout not in device.pre_calibrated_layouts
        ):
            raise ValueError(
                f"'{device.name}' does not accept new register layouts so "
                "the register's layout must be one of the layouts available "
                f"in '{device.name}.calibrated_register_layouts'."
            )
