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

import warnings
from typing import Any

import pulser
from pulser.backend.abc import Backend, EmulatorBackend
from pulser.backend.config import EmulationConfig, EmulatorConfig
from pulser.backend.default_observables import BitStrings, StateResult
from pulser.backend.results import Results
from pulser.noise_model import NoiseModel
from pulser_simulation.aggregators import density_matrix_aggregator
from pulser_simulation.qutip_config import QutipConfig
from pulser_simulation.qutip_op import QutipOperator
from pulser_simulation.qutip_state import QutipState
from pulser_simulation.simresults import CoherentResults, SimulationResults
from pulser_simulation.simulation import QutipEmulator, _has_stochastic_noise


def _get_state_tag(results: Results) -> str | None:
    for tag in results.get_result_tags():
        if tag.startswith(StateResult()._base_tag):
            return tag
    return None


class QutipBackend(Backend):
    """A backend for emulating the sequences using qutip.

    Warning:
        Deprecated in v1.6, please use ``pulser_simulation.QutipBackendV2``.

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
        with warnings.catch_warnings():
            warnings.simplefilter("once")
            warnings.warn(
                "'QutipBackend' is deprecated. Please use "
                "'pulser_simulation.QutipBackendV2' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
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
        self._sim_obj = QutipEmulator.from_sequence(
            sequence,
            sampling_rate=self._config.sampling_rate,
            noise_model=noise_model or self._config.noise_model,
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

    Conforms to the generic API from pulser.backend.

    Dedicated ``EmulationConfig`` class:
    :py:class:`pulser_simulation.QutipConfig`.

    Args:
        sequence: The sequence to emulate.
        config: The configuration for the Qutip emulator.
        mimic_qpu: Whether to mimic the validations necessary for
            execution on a QPU.
    """

    default_config = QutipConfig(
        observables=[BitStrings(evaluation_times=[1.0]), StateResult()]
    )
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
        self._sim_obj = QutipEmulator.from_sequence(
            sequence,
            sampling_rate=self._config.sampling_rate,
            noise_model=noise_model,
            with_modulation=self._config.with_modulation,
            solver=self._config.solver,
            n_trajectories=self._config.n_trajectories,
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
        # setup the default qutip options
        self._qutip_options = {"progress_bar": self._config.progress_bar}
        self._sim_obj._validate_options(self._qutip_options)

    def run(self) -> Results:
        """Executes the sequence on the backend."""
        eigenstates = self._sim_obj._current_hamiltonian.basis_data.eigenbasis

        if not _has_stochastic_noise(self._sim_obj.noise_model):
            # A single run is needed, regardless of self.config.runs
            single_res = self._sim_obj.run(**self._qutip_options)
            assert isinstance(single_res, CoherentResults)
            res = Results(
                atom_order=tuple(self._sequence.qubit_info),
                total_duration=self._sim_obj.total_duration_ns,
            )

            for qutip_res in single_res:
                t = qutip_res.evaluation_time
                state = QutipState(qutip_res.state, eigenstates=eigenstates)
                ham: QutipOperator = QutipOperator(
                    self._sim_obj._get_noiseless_hamiltonian(
                        self._config.noise_model.with_leakage
                    )._hamiltonian(t * res.total_duration / 1000),
                    eigenstates=eigenstates,
                )
                for callback in self._config.callbacks:
                    callback(
                        config=self._config,
                        t=float(t),
                        state=state,
                        hamiltonian=ham,
                        result=res,
                    )
                for obs in self._config.observables:
                    obs(
                        config=self._config,
                        t=float(t),
                        state=state,
                        hamiltonian=ham,
                        result=res,
                    )
            return res
        else:
            results: list[Results] = []
            for cleanres_noisyseq, reps in self._sim_obj._noisy_runs(
                **self._qutip_options
            ):
                for _ in range(reps):
                    res = Results(
                        atom_order=tuple(self._sequence.qubit_info),
                        total_duration=self._sim_obj.total_duration_ns,
                    )
                    for qutip_res in cleanres_noisyseq:
                        t = qutip_res.evaluation_time

                        state = QutipState(
                            qutip_res.state, eigenstates=eigenstates
                        )
                        ham = QutipOperator(
                            self._sim_obj._get_noiseless_hamiltonian(
                                self._config.noise_model.with_leakage
                            )._hamiltonian(t * res.total_duration / 1000),
                            eigenstates=eigenstates,
                        )

                        for callback in self._config.callbacks:
                            callback(
                                config=self._config,
                                t=float(t),
                                state=state,
                                hamiltonian=ham,
                                result=res,
                            )
                        for obs in self._config.observables:
                            obs(
                                config=self._config,
                                t=float(t),
                                state=state,
                                hamiltonian=ham,
                                result=res,
                            )
                    results.append(res)
            custom_aggregators = {}
            if (state_tag := _get_state_tag(results[0])) is not None:
                custom_aggregators[state_tag] = density_matrix_aggregator
            return Results.aggregate(results, **custom_aggregators)
