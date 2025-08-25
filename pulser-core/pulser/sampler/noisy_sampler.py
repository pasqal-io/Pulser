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
from typing import Literal, cast

import numpy as np
from scipy.spatial.distance import cdist

import pulser.math as pm
from pulser.channels.base_channel import STATES_RANK, States
from pulser.devices._device_datacls import COORD_PRECISION, BaseDevice
from pulser.noise_model import NoiseModel, doppler_sigma
from pulser.register.base_register import BaseRegister, QubitId
from pulser.sampler import sampler
from pulser.sampler.samples import SequenceSamples, _PulseTargetSlot
from pulser.sequence import Sequence


class HamiltonianData:
    r"""Information that can be used to generate an Hamiltonian.

    Args:
        samples_obj: A sampled sequence whose ChannelSamples have same
            duration.
        device: The device specifications.
        register: A Register associating coordinates to qubit ids.
        config: NoiseModel to be used to generate noise.

    Keyword Args:
        assign_config: Whether to assign a configuration at initialization
            (defaults to True).
    """

    def __init__(
        self,
        samples_obj: SequenceSamples,
        register: BaseRegister,
        device: BaseDevice,
        config: NoiseModel,
        **kwargs: bool,
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
        # Check compatibility of register and device
        if not isinstance(device, BaseDevice):
            raise TypeError("The device must be a Device or BaseDevice.")
        self._device = device
        self.device.validate_register(register)
        self._register = register
        self._qdict = register.qubits
        # Check compatibility of samples and device:
        if samples_obj._slm_mask.end > 0 and not self.device.supports_slm_mask:
            raise ValueError(
                "Samples use SLM mask but device does not have one."
            )
        if not samples_obj.used_bases <= self.device.supported_bases:
            raise ValueError(
                "Bases used in samples should be supported by device."
            )
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

        # Type hints for attributes defined outside of __init__
        self.basis_name: str
        self._config: NoiseModel
        self.op_matrix_names: list[str]
        self.dim: int
        self._bad_atoms: dict[str, bool] = {}
        self._doppler_detune: dict[str, float] = {}
        self._amp_fluctuations: dict[str, float] = {}
        self._det_fluctuations: dict[str, float] = {}

        # Define interaction
        self._interaction: Literal["XY", "ising"] = (
            "XY" if self.samples_obj._in_xy else "ising"
        )

        # Initializing qubit infos
        self._size = len(self._qdict)
        self._qid_index = {qid: i for i, qid in enumerate(self._qdict)}

        # Stores the qutip operators used in building the Hamiltonian
        self._local_collapse_ops: list[
            tuple[int | float | complex, str | np.ndarray]
        ] = []
        self._depolarizing_pauli_2ds: dict[
            str, list[tuple[int | complex, str]]
        ] = {}

        if kwargs.get("assign_config", True):
            self.set_config(config)

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence,
        *,
        with_modulation: bool = False,
        noise_model: NoiseModel | None = None,
    ) -> HamiltonianData:
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
            sequence.register,
            sequence.device,
            noise_model or NoiseModel(),
        )

    @property
    def samples_obj(self) -> SequenceSamples:
        """The samples without noise, as a SequenceSamples."""
        return self._samples_obj

    @property
    def noisy_samples(self) -> dict:
        """The latest samples generated with noise, as a nested dict."""
        return self.samples

    @property
    def noisy_samples_obj(self) -> SequenceSamples:
        """The latest samples generated with noise, as a SequenceSamples."""
        raise NotImplementedError  # TODO: nested_dict -> SequenceSamples

    @property
    def register(self) -> BaseRegister:
        """The register used."""
        return self._register

    @property
    def device(self) -> BaseDevice:
        """The device used."""
        return self._device

    @property
    def noise_model(self) -> NoiseModel:
        """The current configuration, as a NoiseModel instance."""
        return self._config

    @property
    def local_collapse_operators(
        self,
    ) -> list[tuple[int | float | complex, str | np.ndarray]]:
        """The 1-qudit collapse operators, as string or array."""
        return self._local_collapse_ops

    @property
    def interaction_type(self) -> Literal["XY", "ising"]:
        """The interaction associated with the used samples."""
        return self._interaction

    @property
    def bad_atoms(self) -> dict[str, bool]:
        """The badly prepared atoms at the beginning of the run."""
        return self._bad_atoms

    @functools.cached_property
    def distances(self) -> "pm.torch.Tensor" | np.ndarray:
        r"""Distances between each qubits (in :math:`\mu m`)."""
        # TODO: Handle torch arrays
        positions = list(self._qdict.values())
        if not positions[0].is_tensor:
            return cast(
                np.ndarray,
                np.round(
                    cast(
                        np.ndarray,
                        cdist(positions, positions, metric="euclidean"),
                    ),
                    COORD_PRECISION,
                ),
            )
        else:
            size = len(positions)
            distances = pm.torch.zeros(
                size, size, dtype=positions[0]._array.dtype
            )
            for i in range(size):
                for j in range(i + 1, size):
                    distances[[i, j], [j, i]] = pm.torch.linalg.vector_norm(
                        positions[i]._array - positions[j]._array
                    )
            return distances

    @functools.cached_property
    def nbqudits(self) -> int:
        """Number of qudits in the Register."""
        return self._size

    @functools.cached_property
    def interaction_matrix(self) -> "pm.torch.Tensor" | np.ndarray:
        r"""C6/C3 Interactions between the qubits (in :math:`rad/\mu s`)."""
        # TODO: Include masked qubits in XY + bad atoms in the interaction
        d = pm.AbstractArray(self.distances)
        interactions = pm.AbstractArray.zeros_like(d)._array
        if self.interaction_type == "XY":
            positions = list(self._qdict.values())
            assert self.samples_obj._magnetic_field is not None
            assert self._device.interaction_coeff_xy is not None
            mag_arr = pm.AbstractArray(self.samples_obj._magnetic_field)
            mag_norm = mag_arr.norm()
            assert mag_norm > 0, "There must be a magnetic field in XY mode."
            for i in range(self.nbqudits):
                for j in range(i + 1, self.nbqudits):
                    diff = positions[i] - positions[j]
                    if len(diff) == 2:
                        diff = pm.hstack(
                            [diff, pm.AbstractArray(np.array(0.0))]
                        )
                    cosine = pm.dot(diff, mag_arr) / (
                        diff.norm() * mag_arr.norm()
                    )
                    interactions[[i, j], [j, i]] = (
                        self._device.interaction_coeff_xy  # type: ignore
                        * (1 - 3 * cosine._array**2)
                        / self.distances[i, j] ** 3
                    )
        else:
            for i in range(self.nbqudits):
                for j in range(i + 1, self.nbqudits):
                    interactions[[i, j], [j, i]] = (
                        self._device.interaction_coeff
                        / self.distances[i, j] ** 6
                    )
        return interactions

    @property
    def noisy_interaction_matrix(self) -> "pm.torch.Tensor" | np.ndarray:
        """Return the noisy interaction matrix."""
        mask = [False for _ in range(self.nbqudits)]
        for ind, value in enumerate(self.bad_atoms.values()):
            mask[ind] = True if value else False  # convert to python bool
        if isinstance(self.interaction_matrix, np.ndarray):
            mask2 = np.outer(mask, mask)
            mat = self.interaction_matrix.copy()
            mat[mask2] = 0.0
            return mat
        else:
            ten = pm.torch.tensor(mask, dtype=pm.torch.bool)
            mask3 = pm.torch.outer(ten, ten)
            mat2 = self.interaction_matrix.clone()
            mat2[mask3] = 0.0
            return mat2

    def _build_local_collapse_operators(
        self,
        config: NoiseModel,
        basis_name: str,
        eigenbasis: list[States],
        op_matrix: list[str],
    ) -> None:

        local_collapse_ops: list[
            tuple[int | float | complex, str | np.ndarray]
        ] = []
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
            self._depolarizing_pauli_2ds["x"] = [
                (1, f"sigma_{a}{b}"),
                (1, f"sigma_{b}{a}"),
            ]
            self._depolarizing_pauli_2ds["y"] = [
                (1j, f"sigma_{a}{b}"),
                (-1j, f"sigma_{b}{a}"),
            ]
            self._depolarizing_pauli_2ds["z"] = [
                (1, f"sigma_{b}{b}"),
                (-1, f"sigma_{a}{a}"),
            ]
            coeff = np.sqrt(config.depolarizing_rate / 4)
            for pauli_label in self._depolarizing_pauli_2ds.keys():
                local_collapse_ops.append((coeff, pauli_label))

        if "eff_noise" in config.noise_types:
            for id_, rate in enumerate(config.eff_noise_rates):
                operator = config.eff_noise_opers[id_]
                # This supports the case where the operators are given as Qobj
                # (even though they shouldn't per the NoiseModel signature)
                try:
                    operator = operator.full()  # type: ignore
                except AttributeError:
                    pass
                operator = np.array(operator)

                basis_dim = len(eigenbasis)
                op_shape = (basis_dim, basis_dim)
                if operator.shape != op_shape:
                    raise ValueError(
                        "Incompatible shape for effective noise operator n°"
                        f"{id_}. Operator {operator} should be of shape "
                        f"{op_shape}."
                    )
                local_collapse_ops.append((np.sqrt(rate), operator))
        # Building collapse operators
        self._local_collapse_ops = local_collapse_ops

    @staticmethod
    def _check_config(cfg: NoiseModel) -> None:
        """Checks that the provided config is a NoiseModel."""
        if not isinstance(cfg, NoiseModel):
            raise ValueError(f"Object {cfg} is not a valid `NoiseModel`.")

    def set_config(self, cfg: NoiseModel, **kwargs: bool) -> None:
        """Sets current config to cfg and updates simulation parameters.

        Args:
            cfg: New configuration.

        Keyword Args:
            construct_hamiltonian: Whether or not to update noisy values.
        """
        self._check_config(cfg)
        basis_name = self._get_basis_name(cfg.with_leakage)
        eigenbasis = self._get_eigenbasis(cfg.with_leakage)
        op_matrix_names = self._get_projectors(eigenbasis)
        self.basis_name = basis_name
        self.eigenbasis = eigenbasis
        self.op_matrix_names = op_matrix_names
        self.dim = len(eigenbasis)
        self.operators: dict[str, defaultdict[str, dict]] = {
            addr: defaultdict(dict) for addr in ["Global", "Local"]
        }
        self._build_local_collapse_operators(
            cfg, self.basis_name, self.eigenbasis, self.op_matrix_names
        )
        self._config = cfg
        if not (
            "SPAM" in self.noise_model.noise_types
            and self.noise_model.state_prep_error > 0
        ):
            self._bad_atoms = {qid: False for qid in self._qid_index}
        if "doppler" not in self.noise_model.noise_types:
            self._doppler_detune = {qid: 0.0 for qid in self._qid_index}
        # Noise, samples and Hamiltonian update routine
        if kwargs.get("construct_hamiltonian", True):
            self._create_noise_representation()
            self.construct_hamiltonian()

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
        if set(self.noise_model.noise_types).issubset(
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
                "SPAM" in self.noise_model.noise_types
                and self.noise_model.state_prep_error > 0
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
                if "doppler" in self.noise_model.noise_types:
                    noise_det = self._doppler_detune[qid]
                    samples_dict[qid]["det"][slot.ti : slot.tf] += noise_det
                # Gaussian beam loss in amplitude for global pulses only
                # Noise is drawn at random for each pulse
                if "amplitude" in self.noise_model.noise_types:
                    amp_fraction = amp_fluctuation
                    if (
                        self.noise_model.laser_waist is not None
                        and is_global_pulse
                    ):
                        # Default to an optical axis along y
                        prop_dir = propagation_dir or (0.0, 1.0, 0.0)
                        amp_fraction *= self._finite_waist_amp_fraction(
                            tuple(self._qdict[qid].as_array()),
                            tuple(prop_dir),
                            self.noise_model.laser_waist,
                        )
                    samples_dict[qid]["amp"][slot.ti : slot.tf] *= amp_fraction
                if "detuning" in self.noise_model.noise_types:
                    samples_dict[qid]["det"][
                        slot.ti : slot.tf
                    ] += det_fluctuation

        if local_noises:
            for ch, ch_samples in self._samples_obj.channel_samples.items():
                _ch_obj = self._samples_obj._ch_objs[ch]
                samples_dict = samples["Local"][_ch_obj.basis]
                for slot in ch_samples.slots:
                    add_noise(
                        slot,
                        samples_dict,
                        _ch_obj.addressing == "Global",
                        amp_fluctuation=self._amp_fluctuations[ch],
                        det_fluctuation=self._det_fluctuations[ch],
                        propagation_dir=_ch_obj.propagation_dir,
                    )
            # Delete samples for badly prepared atoms
            for basis in samples["Local"]:
                for qid in samples["Local"][basis]:
                    if self._bad_atoms[qid]:
                        for qty in ("amp", "det", "phase"):
                            samples["Local"][basis][qid][qty] = 0.0
        self.samples = samples

    def _create_noise_representation(self) -> None:
        """Updates noise random parameters.

        Used at the start of each run. If SPAM isn't in chosen noises, all
        atoms are set to be correctly prepared.
        """
        if (
            "SPAM" in self.noise_model.noise_types
            and self.noise_model.state_prep_error > 0
        ):
            dist = (
                np.random.uniform(size=len(self._qid_index))
                < self.noise_model.state_prep_error
            )
            self._bad_atoms = dict(zip(self._qid_index, dist))
        if "doppler" in self.noise_model.noise_types:
            temp = self.noise_model.temperature * 1e-6
            detune = np.random.normal(
                0, doppler_sigma(temp), size=len(self._qid_index)
            )
            self._doppler_detune = dict(zip(self._qid_index, detune))
        pass
        for ch in self._samples_obj.channel_samples:
            self._amp_fluctuations[ch] = max(
                0, np.random.normal(1.0, self.noise_model.amp_sigma)
            )
            self._det_fluctuations[ch] = (
                np.random.normal(0.0, self.noise_model.detuning_sigma)
                if self.noise_model.detuning_sigma
                else 0.0
            )

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
        """Determine projector operators."""
        op_matrix_names = ["I"]
        for proj0 in eigenbasis:
            for proj1 in eigenbasis:
                proj_name = "sigma_" + proj0 + proj1
                op_matrix_names.append(proj_name)
        return op_matrix_names

    def construct_hamiltonian(self) -> None:
        """Constructs the noisy samples.

        Refreshes potential noise parameters by drawing new at random.

        Warning:
            The refreshed noise parameters (when update=True) are only those
            that change from shot to shot (eg doppler and state preparation).
            Amplitude fluctuations change from pulse to pulse and are always
            applied in `_extract_samples()`.

        Args:
            update: Whether to update the noise parameters.
        """
        self._extract_samples()
