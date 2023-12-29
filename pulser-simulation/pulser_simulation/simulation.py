# Copyright 2020 Pulser Development Team
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
"""Defines the QutipEmulator, used to simulate a Sequence or its samples."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import qutip
from numpy.typing import ArrayLike

import pulser.sampler as sampler
from pulser import Sequence
from pulser.backend.noise_model import NoiseModel
from pulser.devices._device_datacls import BaseDevice
from pulser.register.base_register import BaseRegister
from pulser.result import SampledResult
from pulser.sampler.samples import SequenceSamples
from pulser.sequence._seq_drawer import draw_samples, draw_sequence
from pulser_simulation.hamiltonian import Hamiltonian
from pulser_simulation.qutip_result import QutipResult
from pulser_simulation.simconfig import SimConfig
from pulser_simulation.simresults import (
    CoherentResults,
    NoisyResults,
    SimulationResults,
)


class QutipEmulator:
    r"""Emulator of a pulse sequence using QuTiP.

    Args:
        sampled_seq: A pulse sequence samples used in the emulation.
        register: The register associating coordinates to the qubits targeted
            by the pulses within the samples.
        device: The device specifications used in the emulation. Register and
            samples have to satisfy its constraints.
        sampling_rate: The fraction of samples that we wish to extract from
            the samples to simulate. Has to be a value between 0.05 and 1.0.
        config: Configuration to be used for this simulation.
        evaluation_times: Choose between:

            - "Full": The times are set to be the ones used to define the
              Hamiltonian to the solver.

            - "Minimal": The times are set to only include initial and final
              times.

            - An ArrayLike object of times in µs if you wish to only include
              those specific times.

            - A float to act as a sampling rate for the resulting state.
    """

    def __init__(
        self,
        sampled_seq: SequenceSamples,
        register: BaseRegister,
        device: BaseDevice,
        sampling_rate: float = 1.0,
        config: Optional[SimConfig] = None,
        evaluation_times: Union[float, str, ArrayLike] = "Full",
    ) -> None:
        """Instantiates a QutipEmulator object."""
        # Initializing the samples obj
        if not isinstance(sampled_seq, SequenceSamples):
            raise TypeError(
                "The provided sequence has to be a valid "
                "SequenceSamples instance."
            )
        if sampled_seq.max_duration == 0:
            raise ValueError("SequenceSamples is empty.")
        # Check compatibility of register and device
        device.validate_register(register)
        self._register = register
        # Check compatibility of samples and device:
        if sampled_seq._slm_mask.end > 0 and not device.supports_slm_mask:
            raise ValueError(
                "Samples use SLM mask but device does not have one."
            )
        if not sampled_seq.used_bases <= device.supported_bases:
            raise ValueError(
                "Bases used in samples should be supported by device."
            )
        # Check compatibility of masked samples and register
        if not sampled_seq._slm_mask.targets <= set(register.qubit_ids):
            raise ValueError(
                "The ids of qubits targeted in SLM mask"
                " should be defined in register."
            )
        samples_list = []
        for ch, ch_samples in sampled_seq.channel_samples.items():
            if sampled_seq._ch_objs[ch].addressing == "Local":
                # Check that targets of Local Channels are defined
                # in register
                if not set().union(
                    *(slot.targets for slot in ch_samples.slots)
                ) <= set(register.qubit_ids):
                    raise ValueError(
                        "The ids of qubits targeted in Local channels"
                        " should be defined in register."
                    )
                samples_list.append(ch_samples)
            else:
                # Replace targets of Global channels by qubits of register
                samples_list.append(
                    replace(
                        ch_samples,
                        slots=[
                            replace(slot, targets=set(register.qubit_ids))
                            for slot in ch_samples.slots
                        ],
                    )
                )
        _sampled_seq = replace(sampled_seq, samples_list=samples_list)
        self._tot_duration = _sampled_seq.max_duration
        self.samples_obj = _sampled_seq.extend_duration(self._tot_duration + 1)

        # Testing sampling
        if not (0 < sampling_rate <= 1.0):
            raise ValueError(
                "The sampling rate (`sampling_rate` = "
                f"{sampling_rate}) must be greater than 0 and "
                "less than or equal to 1."
            )
        if int(self._tot_duration * sampling_rate) < 4:
            raise ValueError(
                "`sampling_rate` is too small, less than 4 data points."
            )
        # Sets the config as well as builds the hamiltonian
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            noise_model: NoiseModel = (
                config.to_noise_model()
                if config
                else SimConfig().to_noise_model()
            )
        self._hamiltonian = Hamiltonian(
            self.samples_obj,
            self._register.qubits,
            device,
            sampling_rate,
            noise_model,
        )
        # Initializing evaluation times
        self._eval_times_array: np.ndarray
        self.set_evaluation_times(evaluation_times)

        if self.samples_obj._measurement:
            self._meas_basis = self.samples_obj._measurement
        else:
            if self._hamiltonian.basis_name in {"digital", "all"}:
                self._meas_basis = "digital"
            else:
                self._meas_basis = self._hamiltonian.basis_name
        self.set_initial_state("all-ground")

    @property
    def sampling_times(self) -> np.ndarray:
        """The times at which hamiltonian is sampled."""
        return self._hamiltonian.sampling_times

    @property
    def _sampling_rate(self) -> float:
        """The sampling rate."""
        return self._hamiltonian._sampling_rate

    @property
    def dim(self) -> int:
        """The dimension of the basis."""
        return self._hamiltonian.dim

    @property
    def basis_name(self) -> str:
        """The name of the basis."""
        return self._hamiltonian.basis_name

    @property
    def basis(self) -> dict[str, Any]:
        """The basis in which result is expressed."""
        return self._hamiltonian.basis

    @property
    def config(self) -> SimConfig:
        """The current configuration, as a SimConfig instance."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            return SimConfig.from_noise_model(self._hamiltonian.config)

    def set_config(self, cfg: SimConfig) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg: New configuration.
        """
        if not isinstance(cfg, SimConfig):
            raise ValueError(f"Object {cfg} is not a valid `SimConfig`.")
        not_supported = (
            set(cfg.noise)
            - cfg.supported_noises[self._hamiltonian._interaction]
        )
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self._hamiltonian._interaction}' does not"
                " support simulation of noise types:"
                f"{', '.join(not_supported)}."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self._hamiltonian.set_config(cfg.to_noise_model())

    def add_config(self, config: SimConfig) -> None:
        """Updates the current configuration with parameters of another one.

        Mostly useful when dealing with multiple noise types in different
        configurations and wanting to merge these configurations together.
        Adds simulation parameters to noises that weren't available in the
        former SimConfig. Noises specified in both SimConfigs will keep
        former noise parameters.

        Args:
            config: SimConfig to retrieve parameters from.
        """
        if not isinstance(config, SimConfig):
            raise ValueError(f"Object {config} is not a valid `SimConfig`")

        not_supported = (
            set(config.noise)
            - config.supported_noises[self._hamiltonian._interaction]
        )
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self._hamiltonian._interaction}' does not"
                " support simulation of noise types: "
                f"{', '.join(not_supported)}."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            noise_model = config.to_noise_model()
        old_noise_set = set(self._hamiltonian.config.noise_types)
        new_noise_set = old_noise_set.union(noise_model.noise_types)
        diff_noise_set = new_noise_set - old_noise_set
        # Create temporary param_dict to add noise parameters:
        param_dict: dict[str, Any] = asdict(self._hamiltonian.config)
        # Begin populating with added noise parameters:
        param_dict["noise_types"] = tuple(new_noise_set)
        if "SPAM" in diff_noise_set:
            param_dict["state_prep_error"] = noise_model.state_prep_error
            param_dict["p_false_pos"] = noise_model.p_false_pos
            param_dict["p_false_neg"] = noise_model.p_false_neg
        if "doppler" in diff_noise_set:
            param_dict["temperature"] = noise_model.temperature
        if "amplitude" in diff_noise_set:
            param_dict["laser_waist"] = noise_model.laser_waist
        if "dephasing" in diff_noise_set:
            param_dict["dephasing_prob"] = noise_model.dephasing_prob
            param_dict["dephasing_rate"] = noise_model.dephasing_rate
        if "depolarizing" in diff_noise_set:
            param_dict["depolarizing_prob"] = noise_model.depolarizing_prob
            param_dict["depolarizing_rate"] = noise_model.depolarizing_rate
        if "eff_noise" in diff_noise_set:
            param_dict["eff_noise_opers"] = noise_model.eff_noise_opers
            param_dict["eff_noise_rates"] = noise_model.eff_noise_rates
            param_dict["eff_noise_probs"] = noise_model.eff_noise_probs
        # update runs:
        param_dict["runs"] = noise_model.runs
        param_dict["samples_per_run"] = noise_model.samples_per_run
        # set config with the new parameters:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self._hamiltonian.set_config(NoiseModel(**param_dict))

    def show_config(self, solver_options: bool = False) -> None:
        """Shows current configuration."""
        print(self.config.__str__(solver_options))

    def reset_config(self) -> None:
        """Resets configuration to default."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self._hamiltonian.set_config(SimConfig().to_noise_model())

    @property
    def initial_state(self) -> qutip.Qobj:
        """The initial state of the simulation."""
        return self._initial_state

    def set_initial_state(
        self, state: Union[str, np.ndarray, qutip.Qobj]
    ) -> None:
        """Sets the initial state of the simulation.

        Args:
            state: The initial state.
                Choose between:

                - "all-ground" for all atoms in ground state
                - An ArrayLike with a shape compatible with the system
                - A Qobj object
        """
        self._initial_state: qutip.Qobj
        if isinstance(state, str) and state == "all-ground":
            self._initial_state = qutip.tensor(
                [
                    self._hamiltonian.basis[
                        "u" if self._hamiltonian._interaction == "XY" else "g"
                    ]
                    for _ in range(self._hamiltonian._size)
                ]
            )
        else:
            state = cast(Union[np.ndarray, qutip.Qobj], state)
            shape = state.shape[0]
            legal_shape = self._hamiltonian.dim**self._hamiltonian._size
            legal_dims = [
                [self._hamiltonian.dim] * self._hamiltonian._size,
                [1] * self._hamiltonian._size,
            ]
            if shape != legal_shape:
                raise ValueError(
                    "Incompatible shape of initial state."
                    + f"Expected {legal_shape}, got {shape}."
                )
            self._initial_state = qutip.Qobj(state, dims=legal_dims)

    @property
    def evaluation_times(self) -> np.ndarray:
        """The times at which the results of this simulation are returned."""
        return np.array(self._eval_times_array)

    def set_evaluation_times(
        self, value: Union[str, ArrayLike, float]
    ) -> None:
        """Sets times at which the results of this simulation are returned.

        Args:
            value: Choose between:

                - "Full": The times are set to be the ones used to define the
                  Hamiltonian to the solver.

                - "Minimal": The times are set to only include initial and
                  final times.

                - An ArrayLike object of times in µs if you wish to only
                  include those specific times.

                - A float to act as a sampling rate for the resulting state.
        """
        if isinstance(value, str):
            if value == "Full":
                eval_times = np.copy(self._hamiltonian.sampling_times)
            elif value == "Minimal":
                eval_times = np.array([])
            else:
                raise ValueError(
                    "Wrong evaluation time label. It should "
                    "be `Full`, `Minimal`, an array of times or"
                    + " a float between 0 and 1."
                )
        elif isinstance(value, float):
            if value > 1 or value <= 0:
                raise ValueError(
                    "evaluation_times float must be between 0 and 1."
                )
            indices = np.linspace(
                0,
                len(self._hamiltonian.sampling_times) - 1,
                int(value * len(self._hamiltonian.sampling_times)),
                dtype=int,
            )
            # Note: if `value` is very small `eval_times` is an empty list:
            eval_times = self._hamiltonian.sampling_times[indices]
        elif isinstance(value, (list, tuple, np.ndarray)):
            if np.max(value, initial=0) > self._tot_duration / 1000:
                raise ValueError(
                    "Provided evaluation-time list extends "
                    "further than sequence duration."
                )
            if np.min(value, initial=0) < 0:
                raise ValueError(
                    "Provided evaluation-time list contains "
                    "negative values."
                )
            eval_times = np.array(value)
        else:
            raise ValueError(
                "Wrong evaluation time label. It should "
                "be `Full`, `Minimal`, an array of times or a "
                + "float between 0 and 1."
            )
        # Ensure 0 and final time are included:
        self._eval_times_array = np.union1d(
            eval_times, [0.0, self._tot_duration / 1000]
        )
        self._eval_times_instruction = value

    def build_operator(self, operations: Union[list, tuple]) -> qutip.Qobj:
        """Creates an operator with non-trivial actions on some qubits.

        Takes as argument a list of tuples ``[(operator_1, qubits_1),
        (operator_2, qubits_2)...]``. Returns the operator given by the tensor
        product of {``operator_i`` applied on ``qubits_i``} and Id on the rest.
        ``(operator, 'global')`` returns the sum for all ``j`` of operator
        applied at ``qubit_j`` and identity elsewhere.

        Example for 4 qubits: ``[(Z, [1, 2]), (Y, [3])]`` returns `ZZYI`
        and ``[(X, 'global')]`` returns `XIII + IXII + IIXI + IIIX`

        Args:
            operations: List of tuples `(operator, qubits)`.
                `operator` can be a ``qutip.Quobj`` or a string key for
                ``self.op_matrix``. `qubits` is the list on which operator
                will be applied. The qubits can be passed as their
                index or their label in the register.

        Returns:
            The final operator.
        """
        return self._hamiltonian.build_operator(operations)

    def get_hamiltonian(self, time: float) -> qutip.Qobj:
        r"""Get the Hamiltonian created from the sequence at a fixed time.

        Note:
            The whole Hamiltonian is divided by :math:`\hbar`, so its
            units are rad/µs.

        Args:
            time: The specific time at which we want to extract the
                Hamiltonian (in ns).

        Returns:
            A new Qobj for the Hamiltonian with coefficients
            extracted from the effective sequence (determined by
            `self.sampling_rate`) at the specified time.
        """
        if time > self._tot_duration:
            raise ValueError(
                f"Provided time (`time` = {time}) must be "
                "less than or equal to the sequence duration "
                f"({self._tot_duration})."
            )
        if time < 0:
            raise ValueError(
                f"Provided time (`time` = {time}) must be "
                "greater than or equal to 0."
            )
        return self._hamiltonian._hamiltonian(
            time / 1000
        )  # Creates new Qutip.Qobj

    # Run Simulation Evolution using Qutip
    def run(
        self,
        progress_bar: bool = False,
        **options: Any,
    ) -> SimulationResults:
        """Simulates the sequence using QuTiP's solvers.

        Will return NoisyResults if the noise in the SimConfig requires it.
        Otherwise will return CoherentResults.

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
        """
        if "max_step" not in options:
            pulse_durations = [
                slot.tf - slot.ti
                for ch_sample in self.samples_obj.samples_list
                for slot in ch_sample.slots
                if not (
                    np.all(np.isclose(ch_sample.amp[slot.ti : slot.tf], 0))
                    and np.all(np.isclose(ch_sample.det[slot.ti : slot.tf], 0))
                )
            ]
            if pulse_durations:
                options["max_step"] = 0.5 * min(pulse_durations) / 1000

        solv_ops = qutip.Options(**options)

        meas_errors: Optional[Mapping[str, float]] = None
        if "SPAM" in self.config.noise:
            meas_errors = {
                k: self.config.spam_dict[k]
                for k in ("epsilon", "epsilon_prime")
            }
            if self.config.eta > 0 and self.initial_state != qutip.tensor(
                [
                    self._hamiltonian.basis["g"]
                    for _ in range(self._hamiltonian._size)
                ]
            ):
                raise NotImplementedError(
                    "Can't combine state preparation errors with an initial "
                    "state different from the ground."
                )

        def _run_solver() -> CoherentResults:
            """Returns CoherentResults: Object containing evolution results."""
            # Decide if progress bar will be fed to QuTiP solver
            p_bar: Optional[bool]
            if progress_bar is True:
                p_bar = True
            elif (progress_bar is False) or (progress_bar is None):
                p_bar = None
            else:
                raise ValueError("`progress_bar` must be a bool.")

            if (
                "dephasing" in self.config.noise
                or "depolarizing" in self.config.noise
                or "eff_noise" in self.config.noise
            ):
                result = qutip.mesolve(
                    self._hamiltonian._hamiltonian,
                    self.initial_state,
                    self._eval_times_array,
                    self._hamiltonian._collapse_ops,
                    progress_bar=p_bar,
                    options=solv_ops,
                )
            else:
                result = qutip.sesolve(
                    self._hamiltonian._hamiltonian,
                    self.initial_state,
                    self._eval_times_array,
                    progress_bar=p_bar,
                    options=solv_ops,
                )
            results = [
                QutipResult(
                    tuple(self._hamiltonian._qdict),
                    self._meas_basis,
                    state,
                    self._meas_basis == self._hamiltonian.basis_name,
                )
                for state in result.states
            ]
            return CoherentResults(
                results,
                self._hamiltonian._size,
                self._hamiltonian.basis_name,
                self._eval_times_array,
                self._meas_basis,
                meas_errors,
            )

        # Check if noises ask for averaging over multiple runs:
        if set(self.config.noise).issubset(
            {"dephasing", "SPAM", "depolarizing", "eff_noise"}
        ):
            # If there is "SPAM", the preparation errors must be zero
            if "SPAM" not in self.config.noise or self.config.eta == 0:
                return _run_solver()

            else:
                # Stores the different initial configurations and frequency
                initial_configs = Counter(
                    "".join(
                        (
                            np.random.uniform(
                                size=len(self._hamiltonian._qid_index)
                            )
                            < self.config.eta
                        )
                        .astype(int)
                        .astype(str)  # Turns bool->int->str
                    )
                    for _ in range(self.config.runs)
                ).most_common()
                loop_runs = len(initial_configs)
                update_ham = False
        else:
            loop_runs = self.config.runs
            update_ham = True

        # Will return NoisyResults
        time_indices = range(len(self._eval_times_array))
        total_count = np.array([Counter() for _ in time_indices])
        # We run the system multiple times
        for i in range(loop_runs):
            if not update_ham:
                initial_state, reps = initial_configs[i]
                # We load the initial state manually
                self._hamiltonian._bad_atoms = dict(
                    zip(
                        self._hamiltonian._qid_index,
                        np.array(list(initial_state)).astype(bool),
                    )
                )
            else:
                reps = 1
            # At each run, new random noise: new Hamiltonian
            self._hamiltonian._construct_hamiltonian(update=update_ham)
            # Get CoherentResults instance from sequence with added noise:
            cleanres_noisyseq = _run_solver()
            # Extract statistics at eval time:
            total_count += np.array(
                [
                    cleanres_noisyseq.sample_state(
                        t, n_samples=self.config.samples_per_run * reps
                    )
                    for t in self._eval_times_array
                ]
            )
        n_measures = self.config.runs * self.config.samples_per_run
        results = [
            SampledResult(
                tuple(self._hamiltonian._qdict),
                self._meas_basis,
                total_count[t],
            )
            for t in time_indices
        ]
        return NoisyResults(
            results,
            self._hamiltonian._size,
            self._hamiltonian.basis_name,
            self._eval_times_array,
            n_measures,
        )

    def draw(
        self,
        draw_phase_area: bool = False,
        draw_phase_shifts: bool = False,
        draw_phase_curve: bool = False,
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the samples of a sequence of operations used for simulation.

        Args:
            draw_phase_area: Whether phase and area values need
                to be shown as text on the plot, defaults to False.
            draw_phase_shifts: Whether phase shift and reference
                information should be added to the plot, defaults to False.
            draw_phase_curve: Draws the changes in phase in its own curve
                (ignored if the phase doesn't change throughout the channel).
            fig_name: The name on which to save the figure.
                If None the figure will not be saved.
            kwargs_savefig: Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.

        See Also:
            Sequence.draw(): Draws the sequence in its current state.
        """
        draw_samples(
            self.samples_obj,
            self._register,
            self._sampling_rate,
            draw_phase_area=draw_phase_area,
            draw_phase_shifts=draw_phase_shifts,
            draw_phase_curve=draw_phase_curve,
        )
        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)
        plt.show()

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence,
        sampling_rate: float = 1.0,
        config: Optional[SimConfig] = None,
        evaluation_times: Union[float, str, ArrayLike] = "Full",
        with_modulation: bool = False,
    ) -> QutipEmulator:
        r"""Simulation of a pulse sequence using QuTiP.

        Args:
            sequence: An instance of a Pulser Sequence that we
                want to simulate.
            sampling_rate: The fraction of samples that we wish to
                extract from the pulse sequence to simulate. Has to be a
                value between 0.05 and 1.0.
            config: Configuration to be used for this simulation.
            evaluation_times: Choose between:

                - "Full": The times are set to be the ones used to define the
                  Hamiltonian to the solver.

                - "Minimal": The times are set to only include initial and
                  final times.

                - An ArrayLike object of times in µs if you wish to only
                  include those specific times.

                - A float to act as a sampling rate for the resulting state.
            with_modulation: Whether to simulate the sequence with the
                programmed input or the expected output.
        """
        if not isinstance(sequence, Sequence):
            raise TypeError(
                "The provided sequence has to be a valid "
                "pulser.Sequence instance."
            )
        if sequence.is_parametrized() or sequence.is_register_mappable():
            raise ValueError(
                "The provided sequence needs to be built to be simulated. Call"
                " `Sequence.build()` with the necessary parameters."
            )
        if not sequence._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(
            sequence._schedule[x][-1].tf == 0
            for x in sequence.declared_channels
        ):
            raise ValueError(
                "No instructions given for the channels in the sequence."
            )
        if with_modulation and sequence._slm_mask_targets:
            raise NotImplementedError(
                "Simulation of sequences combining an SLM mask and output "
                "modulation is not supported."
            )
        return cls(
            sampler.sample(
                sequence,
                modulation=with_modulation,
                extended_duration=sequence.get_duration(
                    include_fall_time=with_modulation
                ),
            ),
            sequence.register,
            sequence.device,
            sampling_rate,
            config,
            evaluation_times,
        )


class Simulation:
    r"""Simulation of a pulse sequence using QuTiP.

    Warning:
        This class is deprecated in favour of ``QutipEmulator.from_sequence``.

    Args:
        sequence: An instance of a Pulser Sequence that we
            want to simulate.
        sampling_rate: The fraction of samples that we wish to
            extract from the pulse sequence to simulate. Has to be a
            value between 0.05 and 1.0.
        config: Configuration to be used for this simulation.
        evaluation_times: Choose between:

            - "Full": The times are set to be the ones used to define the
              Hamiltonian to the solver.

            - "Minimal": The times are set to only include initial and final
              times.

            - An ArrayLike object of times in µs if you wish to only include
              those specific times.

            - A float to act as a sampling rate for the resulting state.
        with_modulation: Whether to simulated the sequence with the programmed
            input or the expected output.
    """

    def __init__(
        self,
        sequence: Sequence,
        sampling_rate: float = 1.0,
        config: Optional[SimConfig] = None,
        evaluation_times: Union[float, str, ArrayLike] = "Full",
        with_modulation: bool = False,
    ) -> None:
        """Instantiates a Simulation object."""
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                DeprecationWarning(
                    "The `Simulation` class is deprecated,"
                    " use `QutipEmulator.from_sequence` instead."
                )
            )
        self._seq = sequence
        self._modulated = with_modulation
        self._emulator = QutipEmulator.from_sequence(
            self._seq, sampling_rate, config, evaluation_times, self._modulated
        )

    @property
    def evaluation_times(self) -> np.ndarray:
        """The times at which the results of this simulation are returned.

        Args:
            value: Choose between:

                - "Full": The times are set to be the ones used to define the
                  Hamiltonian to the solver.

                - "Minimal": The times are set to only include initial and
                  final times.

                - An ArrayLike object of times in µs if you wish to only
                  include those specific times.

                - A float to act as a sampling rate for the resulting state.
        """
        return self._emulator.evaluation_times

    @evaluation_times.setter
    def evaluation_times(self, value: Union[str, ArrayLike, float]) -> None:
        """Sets times at which the results of this simulation are returned."""
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                DeprecationWarning(
                    "Setting `evaluation_times` is deprecated,"
                    " use `set_evaluation_times` instead."
                )
            )
        self._emulator.set_evaluation_times(value)

    @property
    def initial_state(self) -> qutip.Qobj:
        """The initial state of the simulation.

        Args:
            state: The initial state.
                Choose between:

                - "all-ground" for all atoms in ground state
                - An ArrayLike with a shape compatible with the system
                - A Qobj object
        """
        return self._emulator.initial_state

    @initial_state.setter
    def initial_state(self, value: Union[str, np.ndarray, qutip.Qobj]) -> None:
        """Sets the initial state of the simulation."""
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                DeprecationWarning(
                    "Setting `initial_state` is deprecated,"
                    " use `set_initial_state` instead."
                )
            )
        self._emulator.set_initial_state(value)

    def draw(
        self,
        draw_phase_area: bool = False,
        draw_interp_pts: bool = False,
        draw_phase_shifts: bool = False,
        draw_phase_curve: bool = False,
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the input sequence and the one used by the solver.

        Args:
            draw_phase_area: Whether phase and area values need
                to be shown as text on the plot, defaults to False.
            draw_interp_pts: When the sequence has pulses with waveforms
                of type InterpolatedWaveform, draws the points of interpolation
                on top of the respective waveforms (defaults to False). Can't
                be used if the sequence is modulated.
            draw_phase_shifts: Whether phase shift and reference
                information should be added to the plot, defaults to False.
            draw_phase_curve: Draws the changes in phase in its own curve
                (ignored if the phase doesn't change throughout the channel).
            fig_name: The name on which to save the figure.
                If None the figure will not be saved.
            kwargs_savefig: Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.

        See Also:
            Sequence.draw(): Draws the sequence in its current state.
        """
        if draw_interp_pts and self._modulated:
            raise ValueError(
                "Can't draw the interpolation points when the sequence is "
                "modulated; `draw_interp_pts` must be `False`."
            )
        draw_sequence(
            self._seq,
            self._emulator._sampling_rate,
            draw_input=not self._modulated,
            draw_modulation=self._modulated,
            draw_phase_area=draw_phase_area,
            draw_interp_pts=draw_interp_pts,
            draw_phase_shifts=draw_phase_shifts,
            draw_phase_curve=draw_phase_curve,
        )
        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)
        plt.show()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._emulator, name)
