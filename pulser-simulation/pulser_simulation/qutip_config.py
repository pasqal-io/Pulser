# Copyright 2024 Pulser Development Team
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
"""Defines the QutipConfig class."""
from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from pulser.backend.config import EmulationConfig
from pulser_simulation.qutip_op import QutipOperator
from pulser_simulation.qutip_state import QutipState


class QutipConfig(EmulationConfig[QutipState]):
    """The configuration of a QutipBackend.

    Args:
        observables: A sequence of observables to compute at specific
            evaluation times. The observables without specified evaluation
            times will use this configuration's 'default_evaluation_times'.
        sampling_rate: The fraction of samples to extract from the pulse
            sequence for emulation.
        default_evaluation_times: The default times at which observables
            are computed. Can be a sequence of relative times between 0
            (the start of the sequence) and 1 (the end of the sequence).
            Can also be specified as "Full", in which case every step in the
            emulation will also be an evaluation time.
        initial_state: The initial state from which emulation starts. If
            specified, the state type needs to be compatible with the emulator
            backend. If left undefined, defaults to starting with all qudits
            in the ground state.
        with_modulation: Whether to emulate the sequence with the programmed
            input or the expected output.
        prefer_device_noise_model: If the sequence's device has a default noise
            model, this option signals the backend to prefer it over the noise
            model given with this configuration.
        noise_model: An optional noise model to emulate the sequence with.
            Ignored if the sequence's device has default noise model and
            `prefer_device_noise_model=True`.

    See Also:
        EmulationConfig: The base configuration class for an EmulatorBackend.
    """

    sampling_rate: float
    _state_type = QutipState
    _operator_type = QutipOperator

    def __init__(
        self,
        *,
        sampling_rate: float = 1.0,
        **backend_options: Any,
    ):
        """Initializes a QutipConfig."""
        if backend_options.setdefault("interaction_matrix") is not None:
            raise NotImplementedError(
                "'QutipBackendV2' does not handle custom interaction matrices."
            )
        if not (0 < sampling_rate <= 1.0):
            raise ValueError(
                f"The sampling rate (`sampling_rate` = {sampling_rate}) must"
                " be greater than 0 and less than or equal to 1."
            )
        initial_state = backend_options.setdefault("initial_state")
        if initial_state and not isinstance(initial_state, QutipState):
            raise TypeError(
                "If provided, `initial_state` must be an instance of "
                f"`QutipState`, not {type(initial_state)}."
            )
        if "noise_model" in backend_options and backend_options[
            "noise_model"
        ].samples_per_run not in [None, 1]:
            warnings.warn(
                f"The number of samples per run (`samples_per_run` "
                f"= {backend_options['noise_model'].samples_per_run}) "
                f"is ignored when using QutipBackendV2.",
                stacklevel=2,
            )

        super().__init__(
            sampling_rate=sampling_rate,
            **backend_options,
        )

    def _expected_kwargs(self) -> set[str]:
        return super()._expected_kwargs() | {"sampling_rate"}

    def _get_sampling_indices(self, total_duration_ns: int) -> np.ndarray:
        """Calculates the indices at which samples are taken."""
        return self._calculate_sampling_indices(
            self.sampling_rate, total_duration_ns
        )

    @staticmethod
    def _calculate_sampling_indices(
        sampling_rate: float, total_duration_ns: int
    ) -> np.ndarray:
        return np.linspace(
            0,
            total_duration_ns - 1,
            int(sampling_rate * total_duration_ns),
            dtype=int,
        )

    def _get_legacy_evaluation_times(
        self, total_duration_ns: int
    ) -> Literal["Full"] | np.ndarray:
        extra_eval_times: set[float] = set()
        if self.callbacks:
            return "Full"
        for obs in self.observables:
            extra_eval_times.update(obs.evaluation_times or [])

        rel_eval_times = self.default_evaluation_times
        if extra_eval_times:
            if rel_eval_times == "Full":
                rel_eval_times = (
                    self._get_sampling_indices(total_duration_ns)
                    / total_duration_ns
                )
            rel_eval_times = np.union1d(rel_eval_times, list(extra_eval_times))

        return (
            "Full"
            if isinstance(rel_eval_times, str) and rel_eval_times == "Full"
            else rel_eval_times * total_duration_ns * 1e-3
        )
