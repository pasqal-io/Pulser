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
"""Defines a noise model class for emulator backends."""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Literal, Union, cast, get_args

import numpy as np
from numpy.typing import ArrayLike

import pulser.json.abstract_repr as pulser_abstract_repr
from pulser.constants import KB, KEFF, MASS, TRAP_WAVELENGTH
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.abstract_repr.validation import validate_abstract_repr
from pulser.json.utils import get_dataclass_defaults

__all__ = ["NoiseModel"]

NoiseTypes = Literal[
    "leakage",
    "doppler",
    "amplitude",
    "detuning",
    "register",
    "SPAM",
    "dephasing",
    "relaxation",
    "depolarizing",
    "eff_noise",
]

_NOISE_TYPE_PARAMS: dict[NoiseTypes, tuple[str, ...]] = {
    "leakage": ("with_leakage",),
    "doppler": ("temperature",),
    "register": ("trap_waist", "trap_depth"),
    "amplitude": ("laser_waist", "amp_sigma"),
    "detuning": ("detuning_sigma", "detuning_hf_psd", "detuning_hf_omegas"),
    "SPAM": ("p_false_pos", "p_false_neg", "state_prep_error"),
    "dephasing": ("dephasing_rate", "hyperfine_dephasing_rate"),
    "relaxation": ("relaxation_rate",),
    "depolarizing": ("depolarizing_rate",),
    "eff_noise": ("eff_noise_rates", "eff_noise_opers"),
}

_PARAM_TO_NOISE_TYPE: dict[str, NoiseTypes] = {
    param: noise_type
    for noise_type, params in _NOISE_TYPE_PARAMS.items()
    for param in params
}


_POSITIVE = {
    "dephasing_rate",
    "hyperfine_dephasing_rate",
    "relaxation_rate",
    "depolarizing_rate",
    "temperature",
    "detuning_sigma",
    "trap_waist",
}
_STRICT_POSITIVE = {"runs", "samples_per_run", "laser_waist", "trap_depth"}
_PROBABILITY_LIKE = {
    "state_prep_error",
    "p_false_pos",
    "p_false_neg",
    "amp_sigma",
}

_BOOLEAN = {"with_leakage", "disable_doppler"}

_LEGACY_DEFAULTS = {
    "runs": 15,
    "samples_per_run": 5,
    "state_prep_error": 0.005,
    "p_false_pos": 0.01,
    "p_false_neg": 0.05,
    "temperature": 50.0,
    "laser_waist": 175.0,
    "amp_sigma": 5e-2,
    "relaxation_rate": 0.01,
    "dephasing_rate": 0.05,
    "hyperfine_dephasing_rate": 1e-3,
    "depolarizing_rate": 0.05,
}

OPTIONAL_IN_ABSTR_REPR = (
    "detuning_sigma",
    "trap_waist",
    "trap_depth",
    "detuning_hf_psd",
    "detuning_hf_omegas",
)


def _doppler_sigma(temperature: float) -> float:
    """Standard deviation for Doppler shifting due to thermal motion.

    Arg:
        temperature: The temperature in K.
    """
    return KEFF * math.sqrt(KB * temperature / MASS)


def _register_sigma_xy_z(
    temperature: float, trap_waist: float, trap_depth: float
) -> tuple[float, float]:
    """Standard deviation for fluctuations in atom position in the trap.

    - Plane fluctuation: 𝜎ˣʸ = √(T w²/(4 Uₜᵣₐₚ)), where T is temperature,
      w is the trap waist and Uₜᵣₐₚ is the trap depth.
    - Off plane fluctuation: 𝜎ᶻ = 𝜋 / 𝜆 √2 w 𝜎ˣʸ, where 𝜆 is the trap
    wavelength with a constant value of 0.85 µm

    Note: a k_B factor is absorbed in the trap depth (Uₜᵣₐₚ), so the units
    of temperature and trap depth are the same.

    Args:
        temperature (float): Temperature (T) of the atoms in the trap
            (in Kelvin).
        trap_depth (float): Depth of the trap (Uₜᵣₐₚ)
            (same units as temperature).
        trap_waist (float): Waist of the trap (w) (in µmeters).

    Returns:
        tuple: The standard deviations of the spatial position fluctuations
        in the xy-plane (register_sigma_xy) and along the z-axis
        (register_sigma_z).
    """
    register_sigma_xy = math.sqrt(
        temperature * trap_waist**2 / (4 * trap_depth)
    )
    register_sigma_z = (
        math.pi
        / TRAP_WAVELENGTH
        * math.sqrt(2)
        * trap_waist
        * register_sigma_xy
    )
    return register_sigma_xy, register_sigma_z


@dataclass(init=True, repr=False, frozen=True)
class NoiseModel:
    r"""Specifies the noise model parameters for emulation.

    **Supported noise types:**

    - **leakage**: Adds an error state 'x' to the computational
      basis, that can interact with the other states via an
      effective noise channel. Must be defined with an effective
      noise channel, but is incompatible with dephasing and
      depolarizing noise channels.
    - **relaxation**: Noise due to a decay from the Rydberg to
      the ground state (parametrized by ``relaxation_rate``),
      commonly characterized experimentally by the T1 time.
    - **dephasing**: Random phase (Z) flip (parametrized
      by ``dephasing_rate``), commonly characterized
      experimentally by the T2* time.
    - **depolarizing**: Quantum noise where the state is
      turned into the maximally mixed state with rate
      ``depolarizing_rate``. While it does not describe a
      physical phenomenon, it is a commonly used tool to test
      the system under a uniform combination of phase flip (Z) and
      bit flip (X) errors.
    - **eff_noise**: General effective noise channel defined by the
      set of collapse operators ``eff_noise_opers`` and their
      corresponding rates ``eff_noise_rates``.
    - **doppler**: Local atom detuning due to termal motion of the
      atoms and Doppler effect with respect to laser frequency.
      Parametrized by the ``temperature`` field. Can be disabled with
      the ``disable_doppler`` field.
    - **register**: Thermal fluctuations in the
      register positions, parametrized by ``temperature``, ``trap_waist``
      and, ``trap_depth``, which must all be defined.

      (1) Plane standard deviation fluctuation given by:
      :math:`\sigma^{xy} = \sqrt{\frac{T w²}{4 U_{trap}}}`, where T is
      temperature, w is the trap waist and :math:`U_{trap}` is
      the trap depth.

      (2) Off plane standard deviation fluctuation given by:
      :math:`\sigma^z = \frac{\pi}{\lambda}\sqrt{2} w \sigma^{xy}`, where
      :math:`\lambda` is the trap wavelength with a constant value of 0.85 µm.
    - **amplitude**: Gaussian damping due to finite laser waist and
      laser amplitude fluctuations. Parametrized by ``laser_waist``
      and ``amp_sigma``.
    - **detuning**: Detuning fluctuations consisting of two
      components:

      (1) constant offset (zero-frequency), parameterized by
      ``detuning_sigma``

      (2) time-dependent high-frequency fluctuations, defined by the
      power spectral density (PSD) ``detuning_hf_psd`` over the relevant
      ``detuning_hf_omegas`` angular frequency support.
      :math:`\delta_{hf}(t) = \sum_k \sqrt{2*\Delta \omega_k*\mathrm{PSD}_k}
      * \cos(\omega_k * t + \phi_k)`
      where :math:`\phi_k \backsim U[0, 2\pi)` (uniform random phase),
      and :math:`\Delta \omega_k = \omega_{k+1} - \omega_k`.
    - **SPAM**: SPAM errors. Parametrized by ``state_prep_error``,
      ``p_false_pos`` and ``p_false_neg``.

    Args:
        runs: When reconstructing the Hamiltonian from random noise is
            necessary, this determines how many times that happens. Not
            to be confused with the number of times the resulting
            bitstring distribution is sampled when calculating bitstring
            counts. *Deprecated since v1.7; this parameter is only used by
            deprecated backends. Please use 'EmulationConfig.n_trajectories'
            instead.*
        samples_per_run: Number of samples per noisy Hamiltonian. Useful
            for cutting down on computing time, but unrealistic. *Deprecated
            since v1.6; this parameter is only used by deprecated backends.*
        state_prep_error: The state preparation error probability. Defaults
            to 0.
        p_false_pos: Probability of measuring a false positive. Defaults to 0.
        p_false_neg: Probability of measuring a false negative. Defaults to 0.
        temperature: Temperature, set in µK, of the atoms in the array.
            Also sets the standard deviation of the speed of the atoms.
            Defaults to 0.
        laser_waist: Waist of the gaussian lasers, set in µm, for global
            pulses. Assumed to be the same for all global channels.
        amp_sigma: Dictates the fluctuation in amplitude of a channel from
            run to run as a standard deviation of a normal distribution
            centered in 1. Assumed to be the same for all channels (though
            each channel has its own randomly sampled value in each run).
        detuning_sigma: Dictates the fluctuation in detuning (in rad/µs)
            of a channel from run to run as a standard deviation of a normal
            distribution centered in 0. Assumed to be the same for all
            channels (though each channel has its own randomly sampled
            value in each run). This noise is additive. Defaults to 0.
        trap_waist: The waist of each optical trap at the focal point (in µm).
            Defaults to 0.
        trap_depth: The potential energy well depth that confines the atoms
            (in µK). Defaults to None.
        detuning_hf_psd: Power Spectral Density (PSD) is 1D tuple (in rad/µs)
            provided together with `detuning_hf_omegas` define high frequency
            noise contribution of time dependent detuning (in rad/µs).
            Must either be empty or a tuple with at least two values,
            matching the length of `detuning_hf_omegas`. Default is ().
        detuning_hf_omegas: 1D tuple (in rad/µs) of relevant angular frequency
            support for the PSD. Along with the PSD, it is required to define
            high frequency noise contribution of time dependent detuning
            (in rad/µs). Must either be empty or a tuple with at least two
            values, matching the length of `detuning_hf_psd`. Default is ().
        relaxation_rate: The rate of relaxation from the Rydberg to the
            ground state (in 1/µs). Corresponds to 1/T1. Defaults to 0.
        dephasing_rate: The rate of a dephasing occuring (in 1/µs) in a
            Rydberg state superpostion. Only used if a Rydberg state is
            involved. Corresponds to 1/T2*. Defaults to 0.
        hyperfine_dephasing_rate: The rate of dephasing occuring (in 1/µs)
            between hyperfine ground states. Only used if the hyperfine
            state is involved. Defaults to 0.
        depolarizing_rate: The rate (in 1/µs) at which a depolarizing
            error occurs. Defaults to 0.
        eff_noise_rates: The rate associated to each effective noise operator
            (in 1/µs). Defaults to 0.
        eff_noise_opers: The operators for the effective noise model.
            Defaults to 0.
        with_leakage: Whether or not to include an error state in the
            computations (default to False).
        disable_doppler: Whether or not to disable the doppler noise,
            even if the temperature is defined. In this way, 'register' noise
            (which requires 'temperature') can be activated on its own (i.e
            without doppler).
    """

    noise_types: tuple[NoiseTypes, ...] = field(init=False)
    runs: int | None = None
    samples_per_run: int = 1
    state_prep_error: float = 0.0
    p_false_pos: float = 0.0
    p_false_neg: float = 0.0
    temperature: float = 0.0
    laser_waist: float | None = None
    amp_sigma: float = 0.0
    detuning_sigma: float = 0.0
    detuning_hf_psd: tuple[float, ...] = ()
    detuning_hf_omegas: tuple[float, ...] = ()
    relaxation_rate: float = 0.0
    dephasing_rate: float = 0.0
    # if the trap depth is not None the trap waist should be 0.0
    trap_waist: float = 0.0
    # Must be defined when trap_waist > 0
    trap_depth: float | None = None
    hyperfine_dephasing_rate: float = 0.0
    depolarizing_rate: float = 0.0
    eff_noise_rates: tuple[float, ...] = ()
    eff_noise_opers: tuple[ArrayLike, ...] = ()
    with_leakage: bool = False
    disable_doppler: bool = False

    def __post_init__(self) -> None:
        """Initializes a noise model."""

        def to_tuple(obj: tuple) -> tuple:
            try:
                # Transform qutip.Qobj objects into np.ndarray
                obj = np.array(
                    obj.to("Dense").data_as("ndarray"),  # type: ignore
                    dtype=complex,
                )
            except AttributeError:
                pass
            if isinstance(obj, (tuple, list, np.ndarray)):
                obj = tuple(to_tuple(el) for el in obj)
            return obj

        param_vals = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.init
        }

        param_vals["eff_noise_rates"] = to_tuple(self.eff_noise_rates)
        param_vals["eff_noise_opers"] = to_tuple(self.eff_noise_opers)

        param_vals["detuning_hf_psd"] = to_tuple(self.detuning_hf_psd)
        param_vals["detuning_hf_omegas"] = to_tuple(self.detuning_hf_omegas)

        # Checking the type of provided positive and probability parameters
        for p_, val in param_vals.items():
            if p_ in _PROBABILITY_LIKE | _POSITIVE:
                try:
                    param_vals[p_] = float(val)
                except (TypeError, ValueError):
                    raise TypeError(
                        f"{p_} should be castable to float, not of type"
                        f" {type(val)}."
                    )

        true_noise_types: set[NoiseTypes] = {
            _PARAM_TO_NOISE_TYPE[p_]
            for p_ in param_vals
            if param_vals[p_] and p_ in _PARAM_TO_NOISE_TYPE
        }

        self._check_leakage_noise(true_noise_types)
        self._check_detuning_hf_noise(
            param_vals["detuning_hf_psd"],
            param_vals["detuning_hf_omegas"],
        )
        self._check_eff_noise(
            cast(tuple, param_vals["eff_noise_rates"]),
            cast(tuple, param_vals["eff_noise_opers"]),
            "eff_noise" in true_noise_types,
            with_leakage=cast(bool, param_vals["with_leakage"]),
        )

        relevant_params = self._find_relevant_params(
            true_noise_types,
            cast(float, param_vals["state_prep_error"]),
            cast(float, param_vals["amp_sigma"]),
            cast(Union[float, None], param_vals["laser_waist"]),
        )

        relevant_param_vals = {
            p: param_vals[p]
            for p in param_vals
            if param_vals[p] is not None or p in relevant_params
        }

        if param_vals.get("runs") is not None:
            warnings.warn(
                "Defining the number of emulation trajectories via "
                "'NoiseModel.runs' is deprecated since pulser v1.7. "
                "Please favour using 'EmulationConfig.n_trajectories' "
                "instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        else:
            # Exclude 'runs' from validation when it's None
            relevant_param_vals.pop("runs", None)

        self._validate_parameters(relevant_param_vals)

        self._check_register_noise_params(
            true_noise_types,
            cast(float, param_vals["trap_waist"]),
            cast(Union[float, None], param_vals["trap_depth"]),
            cast(float, param_vals["temperature"]),
        )
        if self.disable_doppler:
            true_noise_types.discard("doppler")

        object.__setattr__(
            self, "noise_types", tuple(sorted(true_noise_types))
        )
        non_zero_relevant_params = [
            p for p in relevant_params if param_vals[p]
        ]
        for param_, val_ in param_vals.items():
            object.__setattr__(self, param_, val_)
            if (
                # disable_doppler is not a noise parameter
                param_ != "disable_doppler"
                and param_ not in relevant_params
                and (val_ if param_ != "samples_per_run" else val_ != 1)
            ):
                warnings.warn(
                    f"{param_!r} is not used by any active noise type "
                    f"in {self.noise_types} when the only defined parameters "
                    f"are {non_zero_relevant_params}.",
                    stacklevel=2,
                )

    @staticmethod
    def _check_register_noise_params(
        true_noise_types: Collection[NoiseTypes],
        trap_waist: float,
        trap_depth: float | None,
        temperature: float,
    ) -> None:
        if "register" not in true_noise_types:
            # trap_waist and trap_depth have default values
            return
        if trap_waist == 0.0 or trap_depth is None or temperature == 0.0:
            raise ValueError(
                "trap_waist, trap_depth, and temperature must be defined in "
                "order to simulate register noise."
            )

    @staticmethod
    def _find_relevant_params(
        noise_types: Collection[NoiseTypes],
        state_prep_error: float,
        amp_sigma: float,
        laser_waist: float | None,
    ) -> set[str]:
        relevant_params: set[str] = set()
        for nt_ in noise_types:
            relevant_params.update(_NOISE_TYPE_PARAMS[nt_])
            if nt_ == "register":
                relevant_params.add("temperature")
            if (
                nt_ == "doppler"
                or nt_ == "detuning"
                or (nt_ == "amplitude" and amp_sigma != 0.0)
                or (nt_ == "SPAM" and state_prep_error != 0.0)
                or nt_ == "register"
            ):
                relevant_params.update(("runs", "samples_per_run"))
        # Disregard laser_waist when not defined
        if laser_waist is None:
            relevant_params.discard("laser_waist")
        return relevant_params

    @staticmethod
    def _check_leakage_noise(noise_types: Collection[NoiseTypes]) -> None:
        # Can't define "dephasing", "depolarizing" with "leakage"
        if "leakage" not in noise_types:
            return
        if "eff_noise" not in noise_types:
            raise ValueError(
                "At least one effective noise operator must be defined to"
                " simulate leakage."
            )

    @staticmethod
    def _check_noise_types(noise_types: Sequence[NoiseTypes]) -> None:
        for noise_type in noise_types:
            if noise_type not in get_args(NoiseTypes):
                raise ValueError(
                    f"'{noise_type}' is not a valid noise type. "
                    + "Valid noise types: "
                    + ", ".join(get_args(NoiseTypes))
                )

    @staticmethod
    def _check_detuning_hf_noise(
        psd: tuple[float, ...],
        freqs: tuple[float, ...],
    ) -> None:
        if (psd == ()) ^ (freqs == ()):
            raise ValueError(
                "`detuning_hf_psd` and `detuning_hf_omegas` must either"
                " both be empty tuples or both be provided."
            )

        if psd == ():
            return

        psd_a = np.asarray(psd)
        freqs_a = np.asarray(freqs)

        if psd_a.ndim != 1 or freqs_a.ndim != 1:
            raise ValueError(
                "`detuning_hf_psd` and `detuning_hf_omegas`"
                " are expected to be 1D tuples."
            )

        if psd_a.size != freqs_a.size:
            raise ValueError(
                "`detuning_hf_psd` and `detuning_hf_omegas`"
                " are expected to have the same length."
            )

        if psd_a.size <= 1:
            raise ValueError(
                "`detuning_hf_psd` and `detuning_hf_omegas`"
                " are expected to have length > 1."
            )

        if not (np.all(psd_a > 0) and np.all(freqs_a > 0)):
            raise ValueError(
                "`detuning_hf_psd` and `detuning_hf_omegas`"
                " are expected to have positive values."
            )

        if np.any(np.diff(freqs_a) < 0):
            raise ValueError(
                "`detuning_hf_omegas` are expected to be monotonously growing."
            )

    @staticmethod
    def _check_eff_noise(
        eff_noise_rates: Sequence[float],
        eff_noise_opers: Sequence[ArrayLike],
        check_contents: bool,
        with_leakage: bool,
    ) -> None:
        if len(eff_noise_opers) != len(eff_noise_rates):
            raise ValueError(
                f"The operators list length({len(eff_noise_opers)}) "
                "and rates list length"
                f"({len(eff_noise_rates)}) must be equal."
            )
        for rate in eff_noise_rates:
            if not (isinstance(rate, float) or isinstance(rate, int)):
                raise TypeError(
                    "eff_noise_rates is a list of floats,"
                    f" it must not contain a {type(rate)}."
                )

        if not check_contents:
            return

        if not eff_noise_opers or not eff_noise_rates:
            raise ValueError(
                "The effective noise parameters have not been filled."
            )

        if np.any(np.array(eff_noise_rates) < 0):
            raise ValueError("The provided rates must be greater than 0.")

        # Check the validity of operators
        min_shape = 2 if not with_leakage else 3
        possible_shapes = [
            (min_shape, min_shape),
            (min_shape + 1, min_shape + 1),
        ]
        for op in eff_noise_opers:
            # type checking
            try:
                operator = np.array(op, dtype=complex)
            except TypeError as e1:
                try:
                    operator = np.array(
                        op.to("Dense").data_as("ndarray"),  # type: ignore
                        dtype=complex,
                    )
                except AttributeError:
                    raise TypeError(
                        f"Operator {op!r} is not castable to a Numpy array."
                    ) from e1
            if operator.ndim != 2:
                raise ValueError(f"Operator '{op!r}' is not a 2D array.")

            if operator.shape not in possible_shapes:
                raise ValueError(
                    f"With{'' if with_leakage else 'out'} leakage, operator's "
                    f"shape must be {possible_shapes[0]}, "
                    f"not {operator.shape}."
                )

    @staticmethod
    def _validate_parameters(param_vals: dict[str, Any]) -> None:
        for param in param_vals:
            is_valid = True
            value = param_vals[param]
            if param in _POSITIVE:
                is_valid = value >= 0
                comp = "greater than or equal to zero"
            elif param in _STRICT_POSITIVE:
                is_valid = value is not None and value > 0
                comp = "greater than zero"
            elif param in _PROBABILITY_LIKE:
                is_valid = 0 <= value <= 1
                comp = (
                    "greater than or equal to zero and smaller than "
                    "or equal to one"
                )
            elif param in _BOOLEAN:
                is_valid = isinstance(value, bool)
                comp = "a boolean"
            if not is_valid:
                raise ValueError(f"'{param}' must be {comp}, not {value}.")
            if param == "samples_per_run" and value != 1:
                warnings.warn(
                    "Setting samples_per_run different to 1 is "
                    "deprecated since pulser v1.6.",
                    DeprecationWarning,
                    stacklevel=2,
                )

    def _to_abstract_repr(self) -> dict[str, Any]:
        all_fields = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if (
                f.name in OPTIONAL_IN_ABSTR_REPR
                and get_dataclass_defaults((f,))[f.name] == value
            ):
                continue
            all_fields[f.name] = value
        # Removed fields that can be deduced from the noise_types
        all_fields.pop("disable_doppler")
        all_fields.pop("with_leakage")
        # Effective noise as a list of (rate, operator)
        eff_noise_rates = all_fields.pop("eff_noise_rates")
        eff_noise_opers = all_fields.pop("eff_noise_opers")
        all_fields["eff_noise"] = list(zip(eff_noise_rates, eff_noise_opers))

        if "detuning_hf_psd" in all_fields:
            det_hf_psd = all_fields.pop("detuning_hf_psd")
            det_hf_freqs = all_fields.pop("detuning_hf_omegas")
            all_fields["detuning_hf"] = list(zip(det_hf_psd, det_hf_freqs))

        return all_fields

    def __repr__(self) -> str:
        relevant_params = self._find_relevant_params(
            self.noise_types,
            self.state_prep_error,
            self.amp_sigma,
            self.laser_waist,
        )
        relevant_params.add("noise_types")
        params_list = []
        for f in fields(self):
            if (
                f.name in relevant_params
                and f.name != "runs"
                and f.name != "samples_per_run"
            ):
                params_list.append(f"{f.name}={getattr(self, f.name)!r}")
        return f"{self.__class__.__name__}({', '.join(params_list)})"

    def to_abstract_repr(self) -> str:
        """Serializes the noise model into an abstract JSON object."""
        abstr_str = json.dumps(self, cls=AbstractReprEncoder)
        validate_abstract_repr(abstr_str, "noise")
        return abstr_str

    @staticmethod
    def from_abstract_repr(obj_str: str) -> NoiseModel:
        """Deserialize a noise model from an abstract JSON object.

        Args:
            obj_str (str): the JSON string representing the noise model
                encoded in the abstract JSON format.
        """
        if not isinstance(obj_str, str):
            raise TypeError(
                "The serialized noise model must be given as a string. "
                f"Instead, got object of type {type(obj_str)}."
            )

        # Avoids circular imports
        return (
            pulser_abstract_repr.deserializer.deserialize_abstract_noise_model(
                obj_str
            )
        )

    def get_noise_table(self) -> dict[str, tuple[Any, str]]:
        """Maps non-zero noise quantities with their value and units.

        Returns:
            A dictionnary, associating to each quantity (a string) a tuple
            composed of their value and their units (a string, that is empty
            if the quantity doesn't have units like for probabilities).
        """
        table: dict[str, tuple[Any, str]] = {}
        if "register" in self.noise_types:
            register_sigma_xy, register_sigma_z = _register_sigma_xy_z(
                self.temperature * 1e-6,
                self.trap_waist,
                cast(float, self.trap_depth),
            )
            table["register_sigma_xy"] = (register_sigma_xy, "µm")
            table["register_sigma_z"] = (register_sigma_z, "µm")
        if self.state_prep_error > 0:
            table["state_prep_error"] = (self.state_prep_error, "")
        if self.laser_waist is not None and self.laser_waist > 0:
            table["laser_waist"] = (self.laser_waist, "µm")
        if self.amp_sigma > 0:
            table["amp_sigma"] = (self.amp_sigma * 100, "%")
        if self.detuning_sigma > 0:
            table["detuning_sigma"] = (self.detuning_sigma, "rad/µs")
        if "doppler" in self.noise_types:
            table["doppler_sigma"] = (
                _doppler_sigma(self.temperature * 1e-6),
                "rad/µs",
            )
        if len(self.detuning_hf_psd) > 0:
            psd = list(zip(self.detuning_hf_omegas, self.detuning_hf_psd))
            table["detuning_psd"] = (psd, "(rad/µs, rad/µs)")
        if "relaxation" in self.noise_types:
            table["T1"] = (1 / self.relaxation_rate, "µs")
        if self.dephasing_rate > 0:
            table["T2* (r-g)"] = (1 / self.dephasing_rate, "µs")
        if self.hyperfine_dephasing_rate > 0:
            table["T2* (g-h)"] = (1 / self.hyperfine_dephasing_rate, "µs")
        if "depolarizing" in self.noise_types:
            table["depolarizing_rate"] = (self.depolarizing_rate, "1/µs")
        if "eff_noise" in self.noise_types:
            table["eff_noise"] = (
                list(zip(self.eff_noise_rates, self.eff_noise_opers)),
                "(1/µs, '')",
            )
            table["with_leakage"] = (self.with_leakage, "")
        if self.p_false_pos > 0:
            table["p_false_pos"] = (self.p_false_pos, "")
        if self.p_false_neg > 0:
            table["p_false_neg"] = (self.p_false_neg, "")
        return table

    def summary(self) -> str:
        """A readable summary of the impact of the noise on the simulation."""

        def _repr_value_unit(value: Any, unit: str) -> str:
            if unit == "":
                return f"{value:g}"
            return f"{value:g} {unit}"

        noise_table = self.get_noise_table()
        summary_list = ["Noise summary:"]
        add_to_traj_summary = []
        # Follow the impact as in step-by-step tutorial
        if "register_sigma_xy" in noise_table:
            assert "register_sigma_z" in noise_table
            # 1. Register
            summary_list += [
                "- Register Position Fluctuations**:",
                "  - XY-Plane Position Fluctuations: "
                f"{_repr_value_unit(*noise_table['register_sigma_xy'])}",
                "  - Z-Axis Position Fluctuations: "
                f"{_repr_value_unit(*noise_table['register_sigma_z'])}",
            ]
            add_to_traj_summary.append("register")
        if "state_prep_error" in noise_table:
            # 2. State Preparation
            summary_list.append(
                "- State Preparation Error Probability**: "
                f"{_repr_value_unit(*noise_table['state_prep_error'])}"
            )
            add_to_traj_summary.append("initial state")

        # 3. Pulse Shaping
        if "laser_waist" in noise_table or "amp_sigma" in noise_table:
            summary_list.append("- Amplitude inhomogeneities:")
            if "laser_waist" in noise_table:
                summary_list += [
                    "  - Finite-waist Gaussian damping σ="
                    f"{_repr_value_unit(*noise_table['laser_waist'])}"
                ]
            if "amp_sigma" in noise_table:
                summary_list += [
                    "  - Shot-to-shot Amplitude Fluctuations**:"
                    f" {_repr_value_unit(*noise_table['amp_sigma'])}"
                ]
                add_to_traj_summary.append("amplitude")
        if (
            "detuning_sigma" in noise_table
            or "doppler_sigma" in noise_table
            or "detuning_psd" in noise_table
        ):
            summary_list.append("- Detuning fluctuations**:")
            if (
                "detuning_sigma" in noise_table
                or "doppler_sigma" in noise_table
            ):
                summary_list += ["  - Shot-to-Shot Detuning fluctuations:"]
                if "detuning_sigma" in noise_table:
                    summary_list += [
                        "       - Laser's Detuning fluctuations: "
                        f"{_repr_value_unit(*noise_table['detuning_sigma'])}"
                    ]
                if "doppler_sigma" in noise_table:
                    summary_list += [
                        "       - Doppler fluctuations: "
                        f"{_repr_value_unit(*noise_table['doppler_sigma'])}"
                    ]
            if "detuning_psd" in noise_table:
                summary_list += [
                    "  - High-Frequency Detuning fluctuations. See PSD in "
                    "get_noise_table()['detuning_psd']."
                ]
            add_to_traj_summary.append("detuning")

        # 4. Noise channels
        if (
            "T1" in noise_table
            or "T2* (r-g)" in noise_table
            or "T2* (g-h)" in noise_table
        ):
            summary_list += ["- Dissipation parameters:"]
            if "T1" in noise_table:
                summary_list += [
                    f"   - T1: {_repr_value_unit(*noise_table['T1'])}"
                ]
            if "T2* (r-g)" in noise_table:
                summary_list += [
                    "   - T2* (r-g): "
                    f"{_repr_value_unit(*noise_table['T2* (r-g)'])}"
                ]
            if "T2* (g-h)" in noise_table:
                summary_list += [
                    "   - T2* (g-h): "
                    f"{_repr_value_unit(*noise_table['T2* (g-h)'])}"
                ]

        if "eff_noise" in noise_table or "depolarizing_rate" in noise_table:
            summary_list += ["- Other Decoherence Processes:"]
            if "depolarizing_rate" in noise_table:
                summary_list += [
                    "   - Depolarization at rate "
                    f"{_repr_value_unit(*noise_table['depolarizing_rate'])}"
                ]
            if "eff_noise" in noise_table:
                summary_list += [
                    "   - Custom Lindblad operators (in 1/µs)"
                    + (
                        " including a leakage state:"
                        if noise_table["with_leakage"][0]
                        else ":"
                    )
                ]
                for rate, oper in noise_table["eff_noise"][0]:
                    oper_str = tuple(
                        tuple(float(f"{val:g}") for val in lines)
                        for lines in oper
                    )
                    summary_list += [f"       - {rate:g} * {oper_str}"]

        # 5. Measurement noises
        if "p_false_pos" in noise_table or "p_false_neg" in noise_table:
            summary_list += ["- Measurement noises:"]
            if "p_false_pos" in noise_table:
                summary_list += [
                    "   - False Positive Meas. Probability: "
                    f"{_repr_value_unit(*noise_table['p_false_pos'])}"
                ]
            if "p_false_neg" in noise_table:
                summary_list += [
                    "   - False Negative Meas. Probability: "
                    f"{_repr_value_unit(*noise_table['p_false_neg'])}"
                ]
        traj_summary = (
            "**: Emulation will generate EmulationConfig.n_trajectories"
            " trajectories with different "
        )
        if add_to_traj_summary != []:
            summary_list += [traj_summary + ", ".join(add_to_traj_summary)]
        return "\n".join(summary_list)
