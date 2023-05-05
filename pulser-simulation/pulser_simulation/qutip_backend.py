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
from __future__ import annotations

from typing import Any

from pulser import Sequence
from pulser.backend.abc import Backend
from pulser.backend.config import EmulatorConfig

from pulser_simulation.simulation import Simulation
from pulser_simulation.simresults import SimulationResults


class QutipBackend(Backend):
    """A backend for emulating the sequences using qutip."""

    # TODO: Doc properly
    def __init__(
        self, sequence: Sequence, config: EmulatorConfig = EmulatorConfig()
    ):
        super().__init__(sequence)
        if not isinstance(config, EmulatorConfig):
            raise TypeError(
                "'config' must be of type 'EmulatorConfig', "
                f"not {type(config)}."
            )
        self._config = config
        self._sim_obj = Simulation(
            sequence,
            sampling_rate=self._config.sampling_rate,
            config=self._config.noise_model,
            evaluation_times=self._config.evaluation_times,
            with_modulation=self._config.with_modulation,
        )
        self._sim_obj.initial_state = self._config.initial_state

    def run(
        self, progress_bar: bool = False, **qutip_options: Any
    ) -> SimulationResults:
        """Emulates the sequence using QuTiP's solvers.

        Args:
            progress_bar: If True, the progress bar of QuTiP's
                solver will be shown. If None or False, no text appears.
            options: Used as arguments for qutip.Options(). If specified, will
                override SimConfig solver_options. If no `max_step` value is
                provided, an automatic one is calculated from the `Sequence`'s
                schedule (half of the shortest duration among pulses and
                delays).
                Refer to the QuTiP docs_ for an overview of the parameters.

                .. _docs: https://bit.ly/3il9A2u


        Returns:
            SimulationResults: In particular, returns NoisyResults if the
            noise model in EmulatorConfig requires it.
            Otherwise, returns CoherentResults.
        """
        return self._sim_obj.run(progress_bar=progress_bar, **qutip_options)
