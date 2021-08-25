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
"""Contains the Simulation class, used for simulation of a Sequence."""

from __future__ import annotations

from typing import Optional, Union, cast, Any
from collections.abc import Mapping
import itertools
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import warnings

import qutip
import numpy as np
from numpy.typing import ArrayLike

from pulser import Pulse, Sequence
from pulser.simulation.simresults import (
    SimulationResults,
    CoherentResults,
    NoisyResults,
)
from pulser.simulation.simconfig import SimConfig
from pulser._seq_drawer import draw_sequence
from pulser.sequence import _TimeSlot


SUPPORTED_BASES = {"ground-rydberg", "digital"}


class Simulation:
    r"""Simulation of a pulse sequence using QuTiP.

    Args:
        sequence (Sequence): An instance of a Pulser Sequence that we
            want to simulate.
        sampling_rate (float): The fraction of samples that we wish to
            extract from the pulse sequence to simulate. Has to be a
            value between 0.05 and 1.0.
        config (SimConfig): Configuration to be used for this simulation.
        evaluation_times (Union[str, ArrayLike, float]): Choose between:

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
        sequence: Sequence,
        sampling_rate: float = 1.0,
        config: Optional[SimConfig] = None,
        evaluation_times: Union[float, str, ArrayLike] = "Full",
    ) -> None:
        """Instantiates a Simulation object."""
        if not isinstance(sequence, Sequence):
            raise TypeError(
                "The provided sequence has to be a valid "
                "pulser.Sequence instance."
            )
        if not sequence._schedule:
            raise ValueError("The provided sequence has no declared channels.")
        if all(sequence._schedule[x][-1].tf == 0 for x in sequence._channels):
            raise ValueError(
                "No instructions given for the channels in the " "sequence."
            )
        not_supported = (
            set(ch.basis for ch in sequence._channels.values())
            - SUPPORTED_BASES
        )
        if not_supported:
            raise NotImplementedError(
                "Sequence with unsupported bases: " + "".join(not_supported)
            )
        self._seq = sequence
        self._qdict = self._seq.qubit_info
        self._size = len(self._qdict)
        self._tot_duration = self._seq.get_duration()
        if not (0 < sampling_rate <= 1.0):
            raise ValueError(
                "The sampling rate (`sampling_rate` = "
                f"{sampling_rate}) must be greater than 0 and "
                "less than or equal to 1."
            )
        if int(self._tot_duration * sampling_rate) < 4:
            raise ValueError(
                "`sampling_rate` is too small, less than 4 data " "points."
            )
        self._sampling_rate = sampling_rate
        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}
        self._collapse_ops: list[qutip.Qobj] = []
        self._times = self._adapt_to_sampling_rate(
            np.arange(self._tot_duration, dtype=np.double) / 1000
        )
        self.evaluation_times = evaluation_times
        self._bad_atoms: dict[Union[str, int], bool] = {}
        self._doppler_detune: dict[Union[str, int], float] = {}
        # Sets the config as well as builds the hamiltonian
        self.set_config(config) if config else self.set_config(SimConfig())
        if hasattr(self._seq, "_measurement"):
            self._meas_basis = cast(str, self._seq._measurement)
        else:
            if self.basis_name in {"digital", "all"}:
                self._meas_basis = "digital"
            else:
                self._meas_basis = "ground-rydberg"
        self.initial_state = "all-ground"

    @property
    def config(self) -> SimConfig:
        """The current configuration, as a SimConfig instance."""
        return self._config

    def set_config(self, cfg: SimConfig) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg (SimConfig): New configuration.
        """
        if not isinstance(cfg, SimConfig):
            raise ValueError(f"Object {cfg} is not a valid `SimConfig`.")
        self._config = cfg
        if not ("SPAM" in self.config.noise and self.config.eta > 0):
            self._bad_atoms = {qid: False for qid in self._qid_index}
        if "doppler" not in self.config.noise:
            self._doppler_detune = {qid: 0.0 for qid in self._qid_index}
        # Noise, samples and Hamiltonian update routine
        self._construct_hamiltonian()
        if "dephasing" in self.config.noise:
            if self.basis_name == "digital" or self.basis_name == "all":
                raise NotImplementedError(
                    "Cannot include dephasing noise in"
                    + " digital- or all-basis."
                )
            # Probability of phase (Z) flip:
            # First order in prob
            prob = self.config.dephasing_prob / 2
            n = self._size
            if prob > 0.1 and n > 1:
                warnings.warn(
                    "The dephasing model is a first-order approximation in the"
                    + f" dephasing probability. p = {2*prob} is too large for "
                    "realistic results.",
                    stacklevel=2,
                )
            k = np.sqrt(prob * (1 - prob) ** (n - 1))
            self._collapse_ops = [
                np.sqrt((1 - prob) ** n)
                * qutip.tensor([self.op_matrix["I"] for _ in range(n)])
            ]
            self._collapse_ops += [
                k
                * (
                    self._build_operator("sigma_rr", qid)
                    - self._build_operator("sigma_gg", qid)
                )
                for qid in self._qid_index
            ]

    def add_config(self, config: SimConfig) -> None:
        """Updates the current configuration with parameters of another one.

        Mostly useful when dealing with multiple noise types in different
        configurations and wanting to merge these configurations together.
        Adds simulation parameters to noises that weren't available in the
        former SimConfig. Noises specified in both SimConfigs will keep
        former noise parameters.

        Args:
            config (SimConfig): SimConfig to retrieve parameters from.
        """
        if not isinstance(config, SimConfig):
            raise ValueError(f"Object {config} is not a valid `SimConfig`")

        old_noise_set = set(self.config.noise)
        new_noise_set = old_noise_set.union(config.noise)
        diff_noise_set = new_noise_set - old_noise_set
        param_dict: dict[str, Any] = asdict(self._config)
        del param_dict["spam_dict"]
        del param_dict["doppler_sigma"]
        param_dict["noise"] = tuple(new_noise_set)
        if "SPAM" in diff_noise_set:
            param_dict["eta"] = config.eta
            param_dict["epsilon"] = config.epsilon
            param_dict["epsilon_prime"] = config.epsilon_prime
        if "doppler" in diff_noise_set:
            param_dict["temperature"] = config.temperature
        if "amplitude" in diff_noise_set:
            param_dict["laser_waist"] = config.laser_waist
        if "dephasing" in diff_noise_set:
            param_dict["dephasing_prob"] = config.dephasing_prob
        param_dict["temperature"] *= 1.0e6
        self.set_config(SimConfig(**param_dict))

    def show_config(self, solver_options: bool = False) -> None:
        """Shows current configuration."""
        print(self._config.__str__(solver_options))

    def reset_config(self) -> None:
        """Resets configuration to default."""
        self.set_config(SimConfig())

    @property
    def initial_state(self) -> qutip.Qobj:
        """The initial state of the simulation.

        Args:
            state (Union[str, ArrayLike, qutip.Qobj]): The initial state.
                Choose between:

                - "all-ground" for all atoms in ground state
                - An ArrayLike with a shape compatible with the system
                - A Qobj object
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: Union[str, np.ndarray, qutip.Qobj]) -> None:
        """Sets the initial state of the simulation."""
        self._initial_state: qutip.Qobj
        if isinstance(state, str) and state == "all-ground":
            self._initial_state = qutip.tensor(
                [self.basis["g"] for _ in range(self._size)]
            )
        else:
            state = cast(Union[np.ndarray, qutip.Qobj], state)
            shape = state.shape[0]
            legal_shape = self.dim ** self._size
            legal_dims = [[self.dim] * self._size, [1] * self._size]
            if shape != legal_shape:
                raise ValueError(
                    "Incompatible shape of initial state."
                    + f"Expected {legal_shape}, got {shape}."
                )
            self._initial_state = qutip.Qobj(state, dims=legal_dims)

    @property
    def evaluation_times(self) -> Union[str, float, ArrayLike]:
        """The times at which the results of this simulation are returned.

        Args:
            value (Union[str, ArrayLike, float]): Choose between:

                - "Full": The times are set to be the ones used to define the
                  Hamiltonian to the solver.

                - "Minimal": The times are set to only include initial and
                  final times.

                - An ArrayLike object of times in µs if you wish to only
                  include those specific times.

                - A float to act as a sampling rate for the resulting state.
        """
        return self._evaluation_times

    @evaluation_times.setter
    def evaluation_times(self, value: Union[str, ArrayLike, float]) -> None:
        """Sets times at which the results of this simulation are returned."""
        if isinstance(value, str):
            if value == "Full":
                self._eval_times_array = self._times
            elif value == "Minimal":
                self._eval_times_array = np.array(
                    [self._times[0], self._times[-1]]
                )
            else:
                raise ValueError(
                    "Wrong evaluation time label. It should "
                    "be `Full`, `Minimal`, an array of times or"
                    + " a float between 0 and 1."
                )
        elif isinstance(value, float):
            if value > 1 or value <= 0:
                raise ValueError(
                    "evaluation_times float must be between 0 " "and 1."
                )
            indices = np.linspace(
                0,
                len(self._times) - 1,
                int(value * len(self._times)),
                dtype=int,
            )
            self._eval_times_array = self._times[indices]
        elif isinstance(value, (list, tuple, np.ndarray)):
            t_max = np.max(value)
            t_min = np.min(value)
            if t_max > self._times[-1]:
                raise ValueError(
                    "Provided evaluation-time list extends "
                    "further than sequence duration."
                )
            if t_min < 0:
                raise ValueError(
                    "Provided evaluation-time list contains "
                    "negative values."
                )
            # Ensure the list of times is sorted
            eval_times = np.array(np.sort(value))
            if t_min > 0:
                eval_times = np.insert(eval_times, 0, 0.0)
            if t_max < self._times[-1]:
                eval_times = np.append(eval_times, self._times[-1])
            self._eval_times_array = eval_times
            # always include initial and final times
        else:
            raise ValueError(
                "Wrong evaluation time label. It should "
                "be `Full`, `Minimal`, an array of times or a "
                + "float between 0 and 1."
            )
        self._evaluation_times: Union[str, ArrayLike, float] = value

    def draw(
        self,
        draw_phase_area: bool = False,
        draw_interp_pts: bool = False,
        draw_phase_shifts: bool = False,
    ) -> None:
        """Draws the input sequence and the one used by the solver.

        Keyword Args:
            draw_phase_area (bool): Whether phase and area values need
                to be shown as text on the plot, defaults to False.
            draw_interp_pts (bool): When the sequence has pulses with waveforms
                of type InterpolatedWaveform, draws the points of interpolation
                on top of the respective waveforms (defaults to False).
            draw_phase_shifts (bool): Whether phase shift and reference
                information should be added to the plot, defaults to False.

        See Also:
            Sequence.draw(): Draws the sequence in its current state.
        """
        draw_sequence(
            self._seq,
            self._sampling_rate,
            draw_phase_area=draw_phase_area,
            draw_interp_pts=draw_interp_pts,
            draw_phase_shifts=draw_phase_shifts,
        )

    def _extract_samples(self) -> None:
        """Populates samples dictionary with every pulse in the sequence."""
        self.samples: dict[str, dict[str, dict]] = {
            addr: {basis: {} for basis in ["ground-rydberg", "digital"]}
            for addr in ["Global", "Local"]
        }
        if not hasattr(self, "operators"):
            self.operators = deepcopy(self.samples)

        def prepare_dict() -> dict[str, np.ndarray]:
            # Duration includes retargeting, delays, etc.
            return {
                "amp": np.zeros(self._tot_duration),
                "det": np.zeros(self._tot_duration),
                "phase": np.zeros(self._tot_duration),
            }

        def write_samples(
            slot: _TimeSlot,
            samples_dict: Mapping[str, np.ndarray],
            is_global_pulse: bool,
            *qid: Union[int, str],
        ) -> None:
            """Builds hamiltonian coefficients.

            Taking into account, if necessary, noise effects, which are local
            and depend on the qubit's id qid.
            """
            _pulse = cast(Pulse, slot.type)
            noise_det = 0.0
            noise_amp = 1.0
            if "doppler" in self.config.noise:
                noise_det += self._doppler_detune[qid[0]]
            # Gaussian beam loss in amplitude for global pulses only
            # Noise is drawn at random for each pulse
            if "amplitude" in self.config.noise and is_global_pulse:
                position = self._qdict[qid[0]]
                r = np.linalg.norm(position)
                w0 = self.config.laser_waist
                noise_amp = np.random.normal(1.0, 1.0e-3) * np.exp(
                    -((r / w0) ** 2)
                )
            samples_dict["amp"][slot.ti : slot.tf] += (
                _pulse.amplitude.samples * noise_amp
            )
            samples_dict["det"][slot.ti : slot.tf] += (
                _pulse.detuning.samples + noise_det
            )
            samples_dict["phase"][slot.ti : slot.tf] += _pulse.phase

        for channel in self._seq.declared_channels:
            addr = self._seq.declared_channels[channel].addressing
            basis = self._seq.declared_channels[channel].basis

            # Case of coherent global simulations
            if addr == "Global" and (
                set(self.config.noise).issubset({"dephasing"})
            ):
                samples_dict = self.samples["Global"][basis]
                if not samples_dict:
                    samples_dict = prepare_dict()
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        write_samples(slot, samples_dict, True)
                self.samples["Global"][basis] = samples_dict

            # Any noise : global becomes local for each qubit in the reg
            # Since coefficients are modified locally by all noises
            else:
                is_global = addr == "Global"
                samples_dict = self.samples["Local"][basis]
                for slot in self._seq._schedule[channel]:
                    if isinstance(slot.type, Pulse):
                        for qubit in slot.targets:
                            if qubit not in samples_dict:
                                samples_dict[qubit] = prepare_dict()
                            # We don't write samples for badly prep qubits
                            if not self._bad_atoms[qubit]:
                                write_samples(
                                    slot, samples_dict[qubit], is_global, qubit
                                )
                self.samples["Local"][basis] = samples_dict

    def _build_operator(
        self, op_id: str, *qubit_ids: Union[str, int], global_op: bool = False
    ) -> qutip.Qobj:
        """Create qutip.Qobj with nontrivial action at *qubit_ids."""
        if global_op:
            return sum(
                self._build_operator(op_id, q_id) for q_id in self._qdict
            )
        if len(set(qubit_ids)) < len(qubit_ids):
            raise ValueError("Duplicate atom ids in argument list.")
        # List of identity operators, except for op_id where requested:
        op_list = [
            self.op_matrix[op_id]
            if j in map(self._qid_index.get, qubit_ids)
            else self.op_matrix["I"]
            for j in range(self._size)
        ]
        return qutip.tensor(op_list)

    def _adapt_to_sampling_rate(self, full_array: np.ndarray) -> np.ndarray:
        """Adapt list to correspond to sampling rate."""
        indices = np.linspace(
            0,
            self._tot_duration - 1,
            int(self._sampling_rate * self._tot_duration),
            dtype=int,
        )
        return cast(np.ndarray, full_array[indices])

    def _update_noise(self) -> None:
        """Updates noise random parameters.

        Used at the start of each run. If SPAM isn't in chosen noises, all
        atoms are set to be correctly prepared.
        """
        if "SPAM" in self.config.noise and self.config.eta > 0:
            dist = (
                np.random.uniform(size=len(self._qid_index))
                < self.config.spam_dict["eta"]
            )
            self._bad_atoms = dict(zip(self._qid_index, dist))
        if "doppler" in self.config.noise:
            detune = np.random.normal(
                0, self.config.doppler_sigma, size=len(self._qid_index)
            )
            self._doppler_detune = dict(zip(self._qid_index, detune))

    def _build_basis_and_op_matrices(self) -> None:
        """Determine dimension, basis and projector operators."""
        # No samples => Empty dict entry => False
        if (
            not self.samples["Global"]["digital"]
            and not self.samples["Local"]["digital"]
        ):
            self.basis_name = "ground-rydberg"
            self.dim = 2
            basis = ["r", "g"]
            projectors = ["gr", "rr", "gg"]
        elif (
            not self.samples["Global"]["ground-rydberg"]
            and not self.samples["Local"]["ground-rydberg"]
        ):
            self.basis_name = "digital"
            self.dim = 2
            basis = ["g", "h"]
            projectors = ["hg", "hh", "gg"]
        else:
            self.basis_name = "all"  # All three states
            self.dim = 3
            basis = ["r", "g", "h"]
            projectors = ["gr", "hg", "rr", "gg", "hh"]

        self.basis = {b: qutip.basis(self.dim, i) for i, b in enumerate(basis)}
        self.op_matrix = {"I": qutip.qeye(self.dim)}

        for proj in projectors:
            self.op_matrix["sigma_" + proj] = (
                self.basis[proj[0]] * self.basis[proj[1]].dag()
            )

    def _construct_hamiltonian(self) -> None:
        """Constructs the hamiltonian from the Sequence.

        Also builds qutip.Qobjs related to the Sequence if not built already,
        and refreshes potential noise parameters by drawing new at random.
        """
        self._update_noise()
        self._extract_samples()
        if not hasattr(self, "basis_name"):
            self._build_basis_and_op_matrices()

        def make_vdw_term() -> qutip.Qobj:
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them, then
            assign the local operator "sigma_rr" at each pair. The units are
            given so that the coefficient includes a 1/hbar factor.
            """
            vdw = 0 * self._build_operator("I")
            # Get every pair without duplicates
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                # no VdW interaction with other qubits for a badly prep. qubit
                if not (self._bad_atoms[q1] or self._bad_atoms[q2]):
                    dist = np.linalg.norm(self._qdict[q1] - self._qdict[q2])
                    U = 0.5 * self._seq._device.interaction_coeff / dist ** 6
                    vdw += U * self._build_operator("sigma_rr", q1, q2)
            return vdw

        def build_coeffs_ops(basis: str, addr: str) -> list[list]:
            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = self.samples[addr][basis]
            operators = self.operators[addr][basis]
            # Choose operator names according to addressing:
            if basis == "ground-rydberg":
                op_ids = ["sigma_gr", "sigma_rr"]
            elif basis == "digital":
                op_ids = ["sigma_hg", "sigma_gg"]

            terms = []
            if addr == "Global":
                coeffs = [
                    0.5 * samples["amp"] * np.exp(-1j * samples["phase"]),
                    -0.5 * samples["det"],
                ]
                for op_id, coeff in zip(op_ids, coeffs):
                    if np.any(coeff != 0):
                        # Build once global operators as they are needed
                        if op_id not in operators:
                            operators[op_id] = self._build_operator(
                                op_id, global_op=True
                            )
                        terms.append(
                            [
                                operators[op_id],
                                self._adapt_to_sampling_rate(coeff),
                            ]
                        )
            elif addr == "Local":
                for q_id, samples_q in samples.items():
                    if q_id not in operators:
                        operators[q_id] = {}
                    coeffs = [
                        0.5
                        * samples_q["amp"]
                        * np.exp(-1j * samples_q["phase"]),
                        -0.5 * samples_q["det"],
                    ]
                    for coeff, op_id in zip(coeffs, op_ids):
                        if np.any(coeff != 0):
                            if op_id not in operators[q_id]:
                                operators[q_id][op_id] = self._build_operator(
                                    op_id, q_id
                                )
                            terms.append(
                                [
                                    operators[q_id][op_id],
                                    self._adapt_to_sampling_rate(coeff),
                                ]
                            )
            self.operators[addr][basis] = operators
            return terms

        # Time independent term:
        if self.basis_name == "digital" or self._size == 1:
            qobj_list = [0 * self._build_operator("I")]
        else:
            # Van der Waals Interaction Terms
            qobj_list = [make_vdw_term()]

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += cast(list, build_coeffs_ops(basis, addr))

        ham = qutip.QobjEvo(qobj_list, tlist=self._times)
        ham = ham + ham.dag()
        ham.compress()
        self._hamiltonian = ham

    def get_hamiltonian(self, time: float) -> qutip.Qobj:
        """Get the Hamiltonian created from the sequence at a fixed time.

        Args:
            time (float): The specific time at which we want to extract the
                Hamiltonian (in ns).

        Returns:
            qutip.Qobj: A new Qobj for the Hamiltonian with coefficients
            extracted from the effective sequence (determined by
            `self.sampling_rate`) at the specified time.
        """
        if time > 1000 * self._times[-1]:
            raise ValueError(
                f"Provided time (`time` = {time}) must be "
                "less than or equal to the sequence duration "
                f"({1000 * self._times[-1]})."
            )
        if time < 0:
            raise ValueError(
                f"Provided time (`time` = {time}) must be "
                "greater than or equal to 0."
            )
        return self._hamiltonian(time / 1000)  # Creates new Qutip.Qobj

    # Run Simulation Evolution using Qutip
    def run(
        self,
        progress_bar: Optional[bool] = None,
        **options: qutip.solver.Options,
    ) -> SimulationResults:
        """Simulates the sequence using QuTiP's solvers.

        Will return NoisyResults if the noise in the SimConfig requires it.
        Otherwise will return CoherentResults.

        Keyword Args:
            progress_bar (bool): If True, the progress bar of QuTiP's solver
                will be shown.
            options (qutip.solver.Options): If specified, will override
                SimConfig solver_options. If no `max_step` value is provided,
                an automatic one is calculated from the `Sequence`'s schedule
                (half of the shortest duration among pulses and delays).
        """
        if "max_step" in options.keys():
            solv_ops = qutip.Options(**options)
        else:
            auto_max_step = 0.5 * (self._seq._min_pulse_duration() / 1000)
            solv_ops = qutip.Options(max_step=auto_max_step, **options)

        meas_errors: Optional[Mapping[str, float]] = None
        if "SPAM" in self.config.noise:
            meas_errors = {
                k: self.config.spam_dict[k]
                for k in ("epsilon", "epsilon_prime")
            }
            if self.config.eta > 0 and self.initial_state != qutip.tensor(
                [self.basis["g"] for _ in range(self._size)]
            ):
                raise NotImplementedError(
                    "Can't combine state preparation errors with an initial "
                    "state different from the ground."
                )

        def _run_solver() -> CoherentResults:
            """Returns CoherentResults: Object containing evolution results."""
            if "dephasing" in self.config.noise:
                # temporary workaround due to a qutip bug when using mesolve
                liouvillian = qutip.liouvillian(
                    self._hamiltonian, self._collapse_ops
                )
                result = qutip.mesolve(
                    liouvillian,
                    self.initial_state,
                    self._eval_times_array,
                    progress_bar=progress_bar,
                    options=solv_ops,
                )
            else:
                result = qutip.sesolve(
                    self._hamiltonian,
                    self.initial_state,
                    self._eval_times_array,
                    progress_bar=progress_bar,
                    options=solv_ops,
                )
            return CoherentResults(
                result.states,
                self._size,
                self.basis_name,
                self._eval_times_array,
                self._meas_basis,
                meas_errors,
            )

        # Check if noises ask for averaging over multiple runs:
        if set(self.config.noise).issubset({"dephasing", "SPAM"}):
            # If there is "SPAM", the preparation errors must be zero
            if "SPAM" not in self.config.noise or self.config.eta == 0:
                return _run_solver()

        # Did not obey the conditions above, will return NoisyResults
        time_indices = range(len(self._eval_times_array))
        total_count = np.array([Counter() for _ in time_indices])
        # We run the system multiple times
        for _ in range(self.config.runs):
            # At each run, new random noise: new Hamiltonian
            self._construct_hamiltonian()
            # Get CoherentResults instance from sequence with added noise:
            cleanres_noisyseq = _run_solver()
            # Extract statistics at eval time:
            total_count += np.array(
                [
                    cleanres_noisyseq.sample_state(
                        t, n_samples=self.config.samples_per_run
                    )
                    for t in self._eval_times_array
                ]
            )
        n_measures = self.config.runs * self.config.samples_per_run
        total_run_prob = [
            Counter({k: v / n_measures for k, v in total_count[t].items()})
            for t in time_indices
        ]
        return NoisyResults(
            total_run_prob,
            self._size,
            self.basis_name,
            self._eval_times_array,
            n_measures,
        )
