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
"""Defines the Hamiltonian class."""

from __future__ import annotations

import itertools
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Union, cast

import numpy as np
import qutip

from pulser.devices._device_datacls import BaseDevice
from pulser.register.base_register import QubitId
from pulser.sampler.samples import SequenceSamples, _PulseTargetSlot
from pulser_simulation.simconfig import SimConfig


class Hamiltonian:
    r"""Generates Hamiltonian from a sampled sequence and noise.

    Args:
        samples_obj: A sampled sequence whose ChannelSamples have same
            duration.
        qdict: A dictionary associating coordinates to qubit ids.
        device: The device specifications.
        sampling_rate: The fraction of samples that we wish to extract from
            the samples to simulate. Has to be a value between 0.05 and 1.0.
        config: Configuration to be used for this simulation.
    """

    def __init__(
        self,
        samples_obj: SequenceSamples,
        qdict: dict[QubitId, np.ndarray],
        device: BaseDevice,
        sampling_rate: float,
        config: SimConfig,
    ) -> None:
        """Instantiates a Hamiltonian object."""
        self.samples_obj = samples_obj
        self._qdict = qdict
        self._device = device
        self._sampling_rate = sampling_rate

        # Type hints for attributes defined outside of __init__
        self.basis_name: str
        self._config: SimConfig
        self.op_matrix: dict[str, qutip.Qobj]
        self.basis: dict[str, qutip.Qobj]
        self.dim: int
        self._bad_atoms: dict[Union[str, int], bool] = {}
        self._doppler_detune: dict[Union[str, int], float] = {}

        # Define interaction
        self._interaction = "XY" if self.samples_obj._in_xy else "ising"

        # Initializing qubit infos
        self._size = len(self._qdict)
        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}

        # Compute sampling times
        self._duration = self.samples_obj.max_duration
        self.sampling_times = self._adapt_to_sampling_rate(
            # Include extra time step for final instruction from samples:
            np.arange(self._duration, dtype=np.double)
            / 1000
        )

        # Stores the qutip operators used in building the Hamiltonian
        self.operators: dict[str, defaultdict[str, dict]] = {
            addr: defaultdict(dict) for addr in ["Global", "Local"]
        }
        self._collapse_ops: list[qutip.Qobj] = []

        self.set_config(config)

    def _adapt_to_sampling_rate(self, full_array: np.ndarray) -> np.ndarray:
        """Adapt list to correspond to sampling rate."""
        indices = np.linspace(
            0,
            len(full_array) - 1,
            int(self._sampling_rate * self._duration),
            dtype=int,
        )
        return cast(np.ndarray, full_array[indices])

    @property
    def config(self) -> SimConfig:
        """The current configuration, as a SimConfig instance."""
        return self._config

    def _build_collapse_operators(self, config: SimConfig) -> None:
        kraus_ops = []
        self._collapse_ops = []
        if "dephasing" in config.noise:
            if self.basis_name == "digital" or self.basis_name == "all":
                # Go back to previous config
                raise NotImplementedError(
                    "Cannot include dephasing noise in digital- or all-basis."
                )
            # Probability of phase (Z) flip:
            # First order in prob
            prob = config.dephasing_prob / 2
            n = self._size
            if prob > 0.1 and n > 1:
                warnings.warn(
                    "The dephasing model is a first-order approximation in the"
                    f" dephasing probability. p = {2*prob} is too large for "
                    "realistic results.",
                    stacklevel=2,
                )
            k = np.sqrt(prob * (1 - prob) ** (n - 1))

            self._collapse_ops += [
                np.sqrt((1 - prob) ** n)
                * qutip.tensor([self.op_matrix["I"] for _ in range(n)])
            ]
            kraus_ops.append(k * qutip.sigmaz())

        if "depolarizing" in config.noise:
            if self.basis_name == "digital" or self.basis_name == "all":
                # Go back to previous config
                raise NotImplementedError(
                    "Cannot include depolarizing "
                    + "noise in digital- or all-basis."
                )
            # Probability of error occurrence

            prob = config.depolarizing_prob / 4
            n = self._size
            if prob > 0.1 and n > 1:
                warnings.warn(
                    "The depolarizing model is a first-order approximation"
                    f" in the depolarizing probability. p = {4*prob}"
                    " is too large for realistic results.",
                    stacklevel=2,
                )

            k = np.sqrt((prob) * (1 - 3 * prob) ** (n - 1))
            self._collapse_ops += [
                np.sqrt((1 - 3 * prob) ** n)
                * qutip.tensor([self.op_matrix["I"] for _ in range(n)])
            ]
            kraus_ops.append(k * qutip.sigmax())
            kraus_ops.append(k * qutip.sigmay())
            kraus_ops.append(k * qutip.sigmaz())

        if "eff_noise" in config.noise:
            if self.basis_name == "digital" or self.basis_name == "all":
                # Go back to previous config
                raise NotImplementedError(
                    "Cannot include general "
                    + "noise in digital- or all-basis."
                )
            # Probability distribution of error occurences
            n = self._size
            m = len(config.eff_noise_opers)
            if n > 1:
                for i in range(1, m):
                    prob_i = config.eff_noise_probs[i]
                    if prob_i > 0.1:
                        warnings.warn(
                            "The effective noise model is a first-order"
                            " approximation in the noise probability."
                            f"p={prob_i} is large for realistic results.",
                            stacklevel=2,
                        )
                        break
            # Deriving Kraus operators
            prob_id = config.eff_noise_probs[0]
            self._collapse_ops += [
                np.sqrt(prob_id**n)
                * qutip.tensor([self.op_matrix["I"] for _ in range(n)])
            ]
            for i in range(1, m):
                k = np.sqrt(config.eff_noise_probs[i] * prob_id ** (n - 1))
                k_op = k * config.eff_noise_opers[i]
                kraus_ops.append(k_op)

        # Building collapse operators
        for operator in kraus_ops:
            self._collapse_ops += [
                self.build_operator([(operator, [qid])])
                for qid in self._qid_index
            ]

    def set_config(self, cfg: SimConfig) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg: New configuration.
        """
        if not isinstance(cfg, SimConfig):
            raise ValueError(f"Object {cfg} is not a valid `SimConfig`.")
        not_supported = (
            set(cfg.noise) - cfg.supported_noises[self._interaction]
        )
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self._interaction}' does not support "
                f"simulation of noise types: {', '.join(not_supported)}."
            )
        if not hasattr(self, "basis_name"):
            self._build_basis_and_op_matrices()
        self._build_collapse_operators(cfg)
        self._config = cfg
        if not ("SPAM" in self.config.noise and self.config.eta > 0):
            self._bad_atoms = {qid: False for qid in self._qid_index}
        if "doppler" not in self.config.noise:
            self._doppler_detune = {qid: 0.0 for qid in self._qid_index}
        # Noise, samples and Hamiltonian update routine
        self._construct_hamiltonian()

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
            set(config.noise) - config.supported_noises[self._interaction]
        )
        if not_supported:
            raise NotImplementedError(
                f"Interaction mode '{self._interaction}' does not support "
                f"simulation of noise types: {', '.join(not_supported)}."
            )

        old_noise_set = set(self.config.noise)
        new_noise_set = old_noise_set.union(config.noise)
        diff_noise_set = new_noise_set - old_noise_set
        # Create temporary param_dict to add noise parameters:
        param_dict: dict[str, Any] = asdict(self._config)
        # Begin populating with added noise parameters:
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
        if "depolarizing" in diff_noise_set:
            param_dict["depolarizing_prob"] = config.depolarizing_prob
        if "eff_noise" in diff_noise_set:
            param_dict["eff_noise_opers"] = config.eff_noise_opers
            param_dict["eff_noise_probs"] = config.eff_noise_probs
        param_dict["temperature"] *= 1.0e6
        # update runs:
        param_dict["runs"] = config.runs
        param_dict["samples_per_run"] = config.samples_per_run

        # set config with the new parameters:
        self.set_config(SimConfig(**param_dict))

    def show_config(self, solver_options: bool = False) -> None:
        """Shows current configuration."""
        print(self._config.__str__(solver_options))

    def reset_config(self) -> None:
        """Resets configuration to default."""
        self.set_config(SimConfig())

    def _extract_samples(self) -> None:
        """Populates samples dictionary with every pulse in the sequence."""
        local_noises = True
        if set(self.config.noise).issubset(
            {"dephasing", "SPAM", "depolarizing", "eff_noise"}
        ):
            local_noises = "SPAM" in self.config.noise and self.config.eta > 0
        samples = self.samples_obj.to_nested_dict(all_local=local_noises)

        def add_noise(
            slot: _PulseTargetSlot,
            samples_dict: Mapping[QubitId, dict[str, np.ndarray]],
            is_global_pulse: bool,
        ) -> None:
            """Builds hamiltonian coefficients.

            Taking into account, if necessary, noise effects, which are local
            and depend on the qubit's id qid.
            """
            noise_amp_base = max(
                0, np.random.normal(1.0, self.config.amp_sigma)
            )
            for qid in slot.targets:
                if "doppler" in self.config.noise:
                    noise_det = self._doppler_detune[qid]
                    samples_dict[qid]["det"][slot.ti : slot.tf] += noise_det
                # Gaussian beam loss in amplitude for global pulses only
                # Noise is drawn at random for each pulse
                if "amplitude" in self.config.noise and is_global_pulse:
                    position = self._qdict[qid]
                    r = np.linalg.norm(position)
                    w0 = self.config.laser_waist
                    noise_amp = noise_amp_base * np.exp(-((r / w0) ** 2))
                    samples_dict[qid]["amp"][slot.ti : slot.tf] *= noise_amp

        if local_noises:
            for ch, ch_samples in self.samples_obj.channel_samples.items():
                addr = self.samples_obj._ch_objs[ch].addressing
                basis = self.samples_obj._ch_objs[ch].basis
                samples_dict = samples["Local"][basis]
                for slot in ch_samples.slots:
                    add_noise(slot, samples_dict, addr == "Global")
            # Delete samples for badly prepared atoms
            for basis in samples["Local"]:
                for qid in samples["Local"][basis]:
                    if self._bad_atoms[qid]:
                        for qty in ("amp", "det", "phase"):
                            samples["Local"][basis][qid][qty] = 0.0
        self.samples = samples

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
        op_list = [self.op_matrix["I"] for j in range(self._size)]

        if not isinstance(operations, list):
            operations = [operations]

        for operator, qubits in operations:
            if qubits == "global":
                return sum(
                    self.build_operator([(operator, [q_id])])
                    for q_id in self._qdict
                )
            else:
                qubits_set = set(qubits)
                if len(qubits_set) < len(qubits):
                    raise ValueError("Duplicate atom ids in argument list.")
                if not qubits_set.issubset(self._qdict.keys()):
                    raise ValueError(
                        "Invalid qubit names: "
                        f"{qubits_set - self._qdict.keys()}"
                    )
                if isinstance(operator, str):
                    try:
                        operator = self.op_matrix[operator]
                    except KeyError:
                        raise ValueError(f"{operator} is not a valid operator")
                for qubit in qubits:
                    k = self._qid_index[qubit]
                    op_list[k] = operator
        return qutip.tensor(op_list)

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
        if self._interaction == "XY":
            self.basis_name = "XY"
            self.dim = 2
            basis = ["u", "d"]
            projectors = ["uu", "du", "ud", "dd"]
        else:
            if "digital" not in self.samples_obj.used_bases:
                self.basis_name = "ground-rydberg"
                self.dim = 2
                basis = ["r", "g"]
                projectors = ["gr", "rr", "gg"]
            elif "ground-rydberg" not in self.samples_obj.used_bases:
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

    def _construct_hamiltonian(self, update: bool = True) -> None:
        """Constructs the hamiltonian from the sampled Sequence and noise.

        Also builds qutip.Qobjs related to the Sequence if not built already,
        and refreshes potential noise parameters by drawing new at random.

        Args:
            update: Whether to update the noise parameters.
        """
        if update:
            self._update_noise()
        self._extract_samples()
        if not hasattr(self, "basis_name"):
            self._build_basis_and_op_matrices()

        def make_vdw_term(q1: QubitId, q2: QubitId) -> qutip.Qobj:
            """Construct the Van der Waals interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_rr" at each pair.
            The units are given so that the coefficient includes a
            1/hbar factor.
            """
            dist = np.linalg.norm(self._qdict[q1] - self._qdict[q2])
            U = 0.5 * self._device.interaction_coeff / dist**6
            return U * self.build_operator([("sigma_rr", [q1, q2])])

        def make_xy_term(q1: QubitId, q2: QubitId) -> qutip.Qobj:
            """Construct the XY interaction Term.

            For each pair of qubits, calculate the distance between them,
            then assign the local operator "sigma_ud * sigma_du" at each pair.
            The units are given so that the coefficient
            includes a 1/hbar factor.
            """
            dist = np.linalg.norm(self._qdict[q1] - self._qdict[q2])
            coords_dim = len(self._qdict[q1])
            mag_field = cast(np.ndarray, self.samples_obj._magnetic_field)[
                :coords_dim
            ]
            mag_norm = np.linalg.norm(mag_field)
            if mag_norm < 1e-8:
                cosine = 0.0
            else:
                cosine = np.dot(
                    (self._qdict[q1] - self._qdict[q2]),
                    mag_field,
                ) / (dist * mag_norm)
            U = (
                0.5
                * cast(float, self._device.interaction_coeff_xy)
                * (1 - 3 * cosine**2)
                / dist**3
            )
            return U * self.build_operator(
                [("sigma_ud", [q1]), ("sigma_du", [q2])]
            )

        def make_interaction_term(masked: bool = False) -> qutip.Qobj:
            if masked:
                # Calculate the total number of good, unmasked qubits
                effective_size = self._size - sum(self._bad_atoms.values())
                for q in self.samples_obj._slm_mask.targets:
                    if not self._bad_atoms[q]:
                        effective_size -= 1
                if effective_size < 2:
                    return 0 * self.build_operator([("I", "global")])

            # make interaction term
            dipole_interaction = cast(qutip.Qobj, 0)
            for q1, q2 in itertools.combinations(self._qdict.keys(), r=2):
                if (
                    self._bad_atoms[q1]
                    or self._bad_atoms[q2]
                    or (
                        masked
                        and self._interaction == "XY"
                        and (
                            q1 in self.samples_obj._slm_mask.targets
                            or q2 in self.samples_obj._slm_mask.targets
                        )
                    )
                ):
                    continue

                if self._interaction == "XY":
                    dipole_interaction += make_xy_term(q1, q2)
                else:
                    dipole_interaction += make_vdw_term(q1, q2)
            return dipole_interaction

        def build_coeffs_ops(basis: str, addr: str) -> list[list]:
            """Build coefficients and operators for the hamiltonian QobjEvo."""
            samples = self.samples[addr][basis]
            operators = self.operators[addr][basis]
            # Choose operator names according to addressing:
            if basis == "ground-rydberg":
                op_ids = ["sigma_gr", "sigma_rr"]
            elif basis == "digital":
                op_ids = ["sigma_hg", "sigma_gg"]
            elif basis == "XY":
                op_ids = ["sigma_du", "sigma_uu"]

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
                            operators[op_id] = self.build_operator(
                                [(op_id, "global")]
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
                                operators[q_id][op_id] = self.build_operator(
                                    [(op_id, [q_id])]
                                )
                            terms.append(
                                [
                                    operators[q_id][op_id],
                                    self._adapt_to_sampling_rate(coeff),
                                ]
                            )
            self.operators[addr][basis] = operators
            return terms

        qobj_list = []
        # Time independent term:
        effective_size = self._size - sum(self._bad_atoms.values())
        if self.basis_name != "digital" and effective_size > 1:
            # Build time-dependent or time-independent interaction term based
            # on whether an SLM mask was defined or not
            if (
                self.samples_obj._slm_mask.end > 0
                and self._interaction == "XY"
            ):
                # Build an array of binary coefficients for the interaction
                # term of unmasked qubits
                coeff = np.ones(self._duration - 1)
                coeff[0 : self.samples_obj._slm_mask.end] = 0
                # Build the interaction term for unmasked qubits
                qobj_list = [
                    [
                        make_interaction_term(),
                        self._adapt_to_sampling_rate(coeff),
                    ]
                ]
                # Build the interaction term for masked qubits
                qobj_list += [
                    [
                        make_interaction_term(masked=True),
                        self._adapt_to_sampling_rate(
                            np.logical_not(coeff).astype(int),
                        ),
                    ]
                ]
            else:
                qobj_list = [make_interaction_term()]

        # Time dependent terms:
        for addr in self.samples:
            for basis in self.samples[addr]:
                if self.samples[addr][basis]:
                    qobj_list += cast(list, build_coeffs_ops(basis, addr))

        if not qobj_list:  # If qobj_list ends up empty
            qobj_list = [0 * self.build_operator([("I", "global")])]

        ham = qutip.QobjEvo(qobj_list, tlist=self.sampling_times)
        ham = ham + ham.dag()
        ham.compress()
        self._hamiltonian = ham
