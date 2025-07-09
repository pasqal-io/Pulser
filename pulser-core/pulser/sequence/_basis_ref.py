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
"""Class for tracking the phase and usage of a qubit over time."""
from __future__ import annotations

from typing import Generator, Union

import numpy as np


class _QubitRef:
    def __init__(self) -> None:
        self.phase = _PhaseTracker(0)
        self._last_used_history = [0]

    @property
    def last_used(self) -> int:
        return max(self._last_used_history)

    def increment_phase(self, phi: float) -> None:
        self.phase[self.last_used] = self.phase.last_phase + phi

    def update_last_used(self, new_t: int) -> None:
        self._last_used_history.append(new_t)

    def truncate(self, t: int) -> None:
        self.phase.truncate(t)
        self._last_used_history = [
            t_ for t_ in self._last_used_history if t_ <= t
        ]


class _PhaseTracker:
    """Tracks a phase reference over time."""

    def __init__(self, initial_phase: float):
        self._times: list[int] = [0]
        self._phases: list[float] = [self._format(initial_phase)]

    @property
    def last_time(self) -> int:
        return self._times[-1]

    @property
    def last_phase(self) -> float:
        return self._phases[-1]

    def changes(
        self,
        ti: Union[float, int],
        tf: Union[float, int],
        time_scale: float = 1.0,
    ) -> Generator[tuple[float, float], None, None]:
        """Changes in phases within ]ti, tf]."""
        start, end = np.searchsorted(
            self._times, (ti * time_scale, tf * time_scale), side="right"
        )
        for i in range(start, end):
            change = self._phases[i] - self._phases[i - 1]
            yield (self._times[i] / time_scale, change)

    def truncate(self, threshold: int) -> None:
        new_times = []
        new_phases = []
        for t, ph in zip(self._times, self._phases):
            if t > threshold:
                continue
            new_times.append(t)
            new_phases.append(ph)
        self._times = new_times
        self._phases = new_phases

    def _format(self, phi: float) -> float:
        return phi % (2 * np.pi)

    def __setitem__(self, t: int, phi: float) -> None:
        phase = self._format(phi)
        if t in self._times:
            ind = self._times.index(t)
            self._phases[ind] = phase
        else:
            ind = int(np.searchsorted(self._times, t, side="right"))
            self._times.insert(ind, t)
            self._phases.insert(ind, phase)

    def __getitem__(self, t: int) -> float:
        ind = int(np.searchsorted(self._times, t, side="right")) - 1
        return self._phases[ind]
