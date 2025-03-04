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
"""Defines the QutipBackend class."""
from __future__ import annotations

from typing import Any

import pulser
from pulser.backend.abc import Backend, EmulatorBackend
from pulser.backend.config import EmulationConfig, EmulatorConfig
from pulser.backend.default_observables import StateResult
from pulser.backend.results import Results
from pulser.noise_model import NoiseModel
from pulser_simulation.qutip_config import QutipConfig
from pulser_simulation.qutip_op import QutipOperator
from pulser_simulation.qutip_state import QutipState
from pulser_simulation.simconfig import SimConfig
from pulser_simulation.simresults import CoherentResults, SimulationResults
from pulser_simulation.simulation import QutipEmulator


class QutipBackend(Backend):
    """A backend for emulating the sequences using qutip.

    Args:
        sequence: The sequence to emulate.
        config: The configuration for the Qutip emulator.
        mimic_qpu: Whether to mimic the validations necessary for
            execution on a QPU.
    """

    def __init__(
        self,
        sequence: pulser.Sequence,
        config: EmulatorConfig = EmulatorConfig(),
        mimic_qpu: bool = False,
    ):
        """Initializes a new QutipBackend."""
        super().__init__(sequence, mimic_qpu=mimic_qpu)
        if not isinstance(config, EmulatorConfig):
            raise TypeError(
                "'config' must be of type 'EmulatorConfig', "
                f"not {type(config)}."
            )
        self._config = config
        noise_model: None | NoiseModel = None
        if self._config.prefer_device_noise_model:
            noise_model = sequence.device.default_noise_model
        simconfig = SimConfig.from_noise_model(
            noise_model or self._config.noise_model
        )
        self._sim_obj = QutipEmulator.from_sequence(
            sequence,
            sampling_rate=self._config.sampling_rate,
            config=simconfig,
            evaluation_times=self._config.evaluation_times,
            with_modulation=self._config.with_modulation,
        )
        self._sim_obj.set_initial_state(self._config.initial_state)

    def run(
        self, progress_bar: bool = False, **qutip_options: Any
    ) -> SimulationResults:
        """Emulates the sequence using QuTiP's solvers.

        Args:
            progress_bar: If True, the progress bar of QuTiP's
                solver will be shown. If None or False, no text appears.
            options: Given directly to the Qutip solver. If specified, will
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


class QutipBackendV2(EmulatorBackend):
    """A backend for emulating the sequences using qutip.

    Warning:
        Still experimental and a work-in-progress. Not all features of
        QutipBackend are supported.

    Args:
        sequence: The sequence to emulate.
        config: The configuration for the Qutip emulator.
        mimic_qpu: Whether to mimic the validations necessary for
            execution on a QPU.
    """

    default_config = QutipConfig(observables=[StateResult()])
    _config: QutipConfig

    def __init__(
        self,
        sequence: pulser.Sequence,
        *,
        config: EmulationConfig | None = None,
        mimic_qpu: bool = False,
    ) -> None:
        """Initializes the backend."""
        super().__init__(sequence, config=config, mimic_qpu=mimic_qpu)
        noise_model: None | NoiseModel = None
        if self._config.prefer_device_noise_model:
            noise_model = sequence.device.default_noise_model
        noise_model = noise_model or self._config.noise_model
        self._config._validate_noise_model(noise_model)
        simconfig = SimConfig.from_noise_model(noise_model)
        self._sim_obj = QutipEmulator.from_sequence(
            sequence,
            sampling_rate=self._config.sampling_rate,
            config=simconfig,
            with_modulation=self._config.with_modulation,
        )
        self._sim_obj.set_evaluation_times(
            self._config._get_legacy_evaluation_times(
                self._sim_obj.total_duration_ns
            ),
        )
        if self._config.initial_state:
            self._sim_obj.set_initial_state(
                self._config.initial_state.to_qobj()
            )

    def run(self) -> Results:
        """Executes the sequence on the backend."""
        _sim_res = self._sim_obj.run()
        res = Results(
            atom_order=tuple(self._sequence.qubit_info),
            total_duration=self._sim_obj.total_duration_ns,
        )
        # For now, the NoiseModel is restricted to disallow NoisyResults
        assert isinstance(
            _sim_res, CoherentResults
        ), "Noisy QutipBackendV2 simulation is unsupported"
        eigenstates = self._sim_obj.samples_obj.eigenbasis
        for qutip_res in _sim_res:
            t_ = qutip_res.evaluation_time
            state = QutipState(qutip_res.state, eigenstates=eigenstates)
            ham: QutipOperator = QutipOperator(
                self._sim_obj.get_hamiltonian(t_ * res.total_duration),
                eigenstates=eigenstates,
            )
            for obs in self._config.observables:
                obs(
                    config=self._config,
                    t=t_,
                    state=state,
                    hamiltonian=ham,
                    result=res,
                )
        return res
