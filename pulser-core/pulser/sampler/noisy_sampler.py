# Copyright 2025 Pulser Development Team
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

import functools
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from typing import Union

import numpy as np

from pulser.channels.base_channel import STATES_RANK, States
from pulser.noise_model import NoiseModel, doppler_sigma
from pulser.register.base_register import BaseRegister, QubitId
from pulser.sampler import sampler
from pulser.sampler.samples import SequenceSamples, _PulseTargetSlot
from pulser.sequence import Sequence


class AbstractHamiltonian:
    r"""Information that can be used to generate an Hamiltonian.

    Args:
        samples_obj: A sampled sequence whose ChannelSamples have same
            duration.
        qdict: A dictionary associating coordinates to qubit ids.
        sampling_rate: The fraction of samples that we wish to extract from
            the samples to simulate. Has to be a value between 0.05 and 1.0.
        config: Configuration to be used for this simulation.
    """

    def __init__(
        self,
        samples_obj: SequenceSamples,
        register: BaseRegister,
        config: NoiseModel,
    ) -> None:
        """Instantiates a Hamiltonian object."""
        # Initializing the samples obj
        if not isinstance(samples_obj, SequenceSamples):
            raise TypeError(
                "The provided sequence has to be a valid "
                "SequenceSamples instance."
            )
        if samples_obj.max_duration == 0:
            raise ValueError("SequenceSamples is empty.")
        # Check correct definition of qdict
        self._register = register
        self._qdict = register.qubits
        # Check compatibility of masked samples and register
        if not samples_obj._slm_mask.targets <= set(self._qdict.keys()):
            raise ValueError(
                "The ids of qubits targeted in SLM mask"
                " should be defined in register."
            )
        samples_list = []
        for ch, ch_samples in samples_obj.channel_samples.items():
            if samples_obj._ch_objs[ch].addressing == "Local":
                # Check that targets of Local Channels are defined
                # in register
                if not set().union(
                    *(slot.targets for slot in ch_samples.slots)
                ) <= set(self._qdict.keys()):
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
                            replace(slot, targets=set(self._qdict.keys()))
                            for slot in ch_samples.slots
                        ],
                    )
                )
        self._samples_obj = replace(samples_obj, samples_list=samples_list)

        self._qdict = {
            k: v.as_array(detach=True) for k, v in self._qdict.items()
        }

        # Type hints for attributes defined outside of __init__
        self.basis_name: str
        self._config: NoiseModel
        self.op_matrix_names: list[str]
        self.dim: int
        self._bad_atoms: dict[Union[str, int], bool] = {}
        self._doppler_detune: dict[Union[str, int], float] = {}

        # Initializing qubit infos
        self._size = len(self._qdict)
        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}

        # Stores the qutip operators used in building the Hamiltonian
        self._local_collapse_ops: list[str | np.ndarray] = []

        self.set_config(config)

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence,
        with_modulation: bool = False,
        noise_model: NoiseModel | None = None,
    ) -> AbstractHamiltonian:
        r"""Simulation of a pulse sequence using QuTiP.

        Args:
            sequence: An instance of a Pulser Sequence that we
                want to simulate.
            with_modulation: Whether to simulate the sequence with the
                programmed input or the expected output.
            noise_model: The noise model for the simulation. Replaces and
                should be preferred over 'config'.
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
            sequence.register.qubits,
            noise_model=noise_model or NoiseModel(),
        )

    @property
    def samples_obj(self) -> SequenceSamples:
        """The samples without noise, as a SequenceSamples."""
        return self._samples_obj

    @property
    def noisy_samples(self) -> dict:
        """The latest samples generated with noise, as a nested dict."""
        return self._samples_obj

    @property
    def register(self) -> BaseRegister:
        return self._register

    @property
    def noise_model(self) -> NoiseModel:
        """The current configuration, as a NoiseModel instance."""
        return self._config

    @property
    def local_collapse_operators(self) -> list[str | np.ndarray]:
        return self._local_collapse_ops

    @property
    def bad_atoms(self) -> dict[Union[str, int], bool]:
        return self._bad_atoms

    def _build_collapse_operators(
        self,
        config: NoiseModel,
        basis_name: str,
        eigenbasis: list[States],
        op_matrix: list[str],
    ) -> None:

        local_collapse_ops = []
        if "dephasing" in config.noise_types:
            dephasing_rates = {
                "d": config.dephasing_rate,
                "r": config.dephasing_rate,
                "h": config.hyperfine_dephasing_rate,
            }
            for state in eigenbasis:
                if state in dephasing_rates:
                    coeff = np.sqrt(2 * dephasing_rates[state])
                    op = f"sigma_{state}{state}"
                    assert op in op_matrix
                    local_collapse_ops.append((coeff, op))

        if "relaxation" in config.noise_types:
            coeff = np.sqrt(config.relaxation_rate)
            op = "sigma_gr"

            if op not in op_matrix:
                raise ValueError(
                    "'relaxation' noise requires addressing of the"
                    " 'ground-rydberg' basis."
                )
            local_collapse_ops.append((coeff, op))

        if "depolarizing" in config.noise_types:
            if "all" in basis_name:
                # Go back to previous config
                raise NotImplementedError(
                    "Cannot include depolarizing noise in all-basis."
                )
            # NOTE: These operators only make sense when basis != "all"
            b, a = eigenbasis[:2]
            pauli_2d = [
                (1, f"sigma_{a}{b}"),
                (1, f"sigma_{b}{a}"),
                (1j, f"sigma_{a}{b}"),
                (-1j, f"sigma_{b}{a}"),
                (1, f"sigma_{b}{b}")(-1, f"sigma_{a}{a}"),
            ]
            coeff = np.sqrt(config.depolarizing_rate / 4)
            for pauli_coeff, pauli_op in pauli_2d:
                local_collapse_ops.append((pauli_coeff * coeff, pauli_op))

        if "eff_noise" in config.noise_types:
            for id_, rate in enumerate(config.eff_noise_rates):
                op = config.eff_noise_opers[id_]
                # This supports the case where the operators are given as Qobj
                # (even though they shouldn't per the NoiseModel signature)
                try:
                    op = op.full()
                except AttributeError:
                    pass
                op = np.array(op)

                basis_dim = len(eigenbasis)
                op_shape = (basis_dim, basis_dim)
                if op.shape != op_shape:
                    raise ValueError(
                        "Incompatible shape for effective noise operator n°"
                        f"{id_}. Operator {op} should be of shape {op_shape}."
                    )
                local_collapse_ops.append((np.sqrt(rate), op))
        # Building collapse operators
        self._local_collapse_ops = local_collapse_ops

    def set_config(self, cfg: NoiseModel) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg: New configuration.
        """
        if not isinstance(cfg, NoiseModel):
            raise ValueError(f"Object {cfg} is not a valid `NoiseModel`.")
        if not hasattr(self, "_config") or (
            hasattr(self, "_config")
            and self.config.with_leakage != cfg.with_leakage
        ):
            basis_name = self._get_basis_name(cfg.with_leakage)
            eigenbasis = self._get_eigenbasis(cfg.with_leakage)
            op_matrix = self._get_projectors(eigenbasis)
            self._build_collapse_operators(
                cfg, basis_name, eigenbasis, op_matrix
            )
            self.basis_name = basis_name
            self.eigenbasis = eigenbasis
            self.op_matrix_names = op_matrix
            self.dim = len(eigenbasis)
            self.operators: dict[str, defaultdict[str, dict]] = {
                addr: defaultdict(dict) for addr in ["Global", "Local"]
            }
        else:
            self._build_collapse_operators(
                cfg, self.basis_name, self.eigenbasis, self.op_matrix_names
            )
        self._config = cfg
        if not (
            "SPAM" in self.config.noise_types
            and self.config.state_prep_error > 0
        ):
            self._bad_atoms = {qid: False for qid in self._qid_index}
        if "doppler" not in self.config.noise_types:
            self._doppler_detune = {qid: 0.0 for qid in self._qid_index}
        # Noise, samples and Hamiltonian update routine
        self._construct_hamiltonian()

    @staticmethod
    @functools.cache
    def _finite_waist_amp_fraction(
        coords: tuple[float, ...],
        propagation_dir: tuple[float, float, float],
        laser_waist: float,
    ) -> float:
        pos_vec = np.zeros(3, dtype=float)
        pos_vec[: len(coords)] = np.array(coords, dtype=float)
        u_vec = np.array(propagation_dir, dtype=float)
        u_vec = u_vec / np.linalg.norm(u_vec)
        # Given a line crossing the origin with normalized direction vector
        # u_vec, the closest point along said line and an arbitrary point
        # pos_vec is at k*u_vec, where
        k = np.dot(pos_vec, u_vec)
        # The distance between pos_vec and the line is then given by
        dist = np.linalg.norm(pos_vec - k * u_vec)
        # We assume the Rayleigh length of the gaussian beam is very large,
        # resulting in a negligble drop in amplitude along the propagation
        # direction. Therefore, the drop in amplitude at a given position
        # is dictated solely by its distance to the optical axis (as dictated
        # by the propagation direction), ie
        return float(np.exp(-((dist / laser_waist) ** 2)))

    def _extract_samples(self) -> None:
        """Populates samples dictionary with every pulse in the sequence."""
        local_noises = True
        if set(self.config.noise_types).issubset(
            {
                "dephasing",
                "relaxation",
                "SPAM",
                "depolarizing",
                "eff_noise",
                "leakage",
            }
        ):
            local_noises = (
                "SPAM" in self.config.noise_types
                and self.config.state_prep_error > 0
            )
        samples = self._samples_obj.to_nested_dict(all_local=local_noises)

        def add_noise(
            slot: _PulseTargetSlot,
            samples_dict: Mapping[QubitId, dict[str, np.ndarray]],
            is_global_pulse: bool,
            amp_fluctuation: float,
            det_fluctuation: float,
            propagation_dir: tuple | None,
        ) -> None:
            """Builds hamiltonian coefficients.

            Taking into account, if necessary, noise effects, which are local
            and depend on the qubit's id qid.
            """
            for qid in slot.targets:
                if "doppler" in self.config.noise_types:
                    noise_det = self._doppler_detune[qid]
                    samples_dict[qid]["det"][slot.ti : slot.tf] += noise_det
                # Gaussian beam loss in amplitude for global pulses only
                # Noise is drawn at random for each pulse
                if "amplitude" in self.config.noise_types:
                    amp_fraction = amp_fluctuation
                    if self.config.laser_waist is not None and is_global_pulse:
                        # Default to an optical axis along y
                        prop_dir = propagation_dir or (0.0, 1.0, 0.0)
                        amp_fraction *= self._finite_waist_amp_fraction(
                            tuple(self._qdict[qid]),
                            tuple(prop_dir),
                            self.config.laser_waist,
                        )
                    samples_dict[qid]["amp"][slot.ti : slot.tf] *= amp_fraction
                if "detuning" in self.config.noise_types:
                    samples_dict[qid]["det"][
                        slot.ti : slot.tf
                    ] += det_fluctuation

        if local_noises:
            for ch, ch_samples in self._samples_obj.channel_samples.items():
                _ch_obj = self._samples_obj._ch_objs[ch]
                samples_dict = samples["Local"][_ch_obj.basis]
                ch_amp_fluctuation = max(
                    0, np.random.normal(1.0, self.config.amp_sigma)
                )
                ch_det_fluctuation = (
                    np.random.normal(0.0, self.config.detuning_sigma)
                    if self.config.detuning_sigma
                    else 0.0
                )
                for slot in ch_samples.slots:
                    add_noise(
                        slot,
                        samples_dict,
                        _ch_obj.addressing == "Global",
                        amp_fluctuation=ch_amp_fluctuation,
                        det_fluctuation=ch_det_fluctuation,
                        propagation_dir=_ch_obj.propagation_dir,
                    )
            # Delete samples for badly prepared atoms
            for basis in samples["Local"]:
                for qid in samples["Local"][basis]:
                    if self._bad_atoms[qid]:
                        for qty in ("amp", "det", "phase"):
                            samples["Local"][basis][qid][qty] = 0.0
        self.samples = samples

    def _update_noise(self) -> None:
        """Updates noise random parameters.

        Used at the start of each run. If SPAM isn't in chosen noises, all
        atoms are set to be correctly prepared.
        """
        if (
            "SPAM" in self.config.noise_types
            and self.config.state_prep_error > 0
        ):
            dist = (
                np.random.uniform(size=len(self._qid_index))
                < self.config.state_prep_error
            )
            self._bad_atoms = dict(zip(self._qid_index, dist))
        if "doppler" in self.config.noise_types:
            temp = self.config.temperature * 1e-6
            detune = np.random.normal(
                0, doppler_sigma(temp), size=len(self._qid_index)
            )
            self._doppler_detune = dict(zip(self._qid_index, detune))

    def _get_basis_name(self, with_leakage: bool) -> str:
        if len(self._samples_obj.used_bases) == 0:
            if self._samples_obj._in_xy:
                basis_name = "XY"
            else:
                basis_name = "ground-rydberg"
        elif len(self._samples_obj.used_bases) == 1:
            basis_name = list(self._samples_obj.used_bases)[0]
        else:
            basis_name = "all"  # All three rydberg states
        if with_leakage:
            basis_name += "_with_error"
        return basis_name

    def _get_eigenbasis(self, with_leakage: bool) -> list[States]:
        eigenbasis = self._samples_obj.eigenbasis
        if with_leakage:
            eigenbasis.append("x")
        return [state for state in STATES_RANK if state in eigenbasis]

    @staticmethod
    def _get_projectors(
        eigenbasis: list[States],
    ) -> list[str]:
        """Determine basis and projector operators."""
        op_matrix_names = ["I"]
        for proj0 in eigenbasis:
            for proj1 in eigenbasis:
                proj_name = "sigma_" + proj0 + proj1
                op_matrix_names.append(proj_name)
        return op_matrix_names

    def _construct_hamiltonian(self, update: bool = True) -> None:
        """Constructs the hamiltonian from the sampled Sequence and noise.

        Also builds qutip.Qobjs related to the Sequence if not built already,
        and refreshes potential noise parameters by drawing new at random.

        Warning:
            The refreshed noise parameters (when update=True) are only those
            that change from shot to shot (ie doppler and state preparation).
            Amplitude fluctuations change from pulse to pulse and are always
            applied in `_extract_samples()`.

        Args:
            update: Whether to update the noise parameters.
        """
        if update:
            self._update_noise()
        self._extract_samples()
