"""(De)serialization logic specific to the pulser.backend module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type, TypeVar

from pulser.backend.default_observables import (
    BitStrings,
    CorrelationMatrix,
    Energy,
    EnergySecondMoment,
    EnergyVariance,
    Expectation,
    Fidelity,
    Occupation,
)
from pulser.json.abstract_repr.deserializer import (
    _convert_complex,
    _deserialize_noise_model,
)

if TYPE_CHECKING:
    from pulser.backend import EmulationConfig, Observable, Operator, State

EmulationConfigType = TypeVar("EmulationConfigType", bound="EmulationConfig")
StateType = TypeVar("StateType", bound="State")
OperatorType = TypeVar("OperatorType", bound="Operator")


def _deserialize_state(
    ser_state: dict, state_type: Type[StateType]
) -> StateType:
    return state_type.from_state_amplitudes(
        eigenstates=ser_state["eigenstates"],
        amplitudes=_convert_complex(ser_state["amplitudes"]),
    )


def _deserialize_operator(
    ser_op: dict, op_type: Type[OperatorType]
) -> OperatorType:
    return op_type.from_operator_repr(
        eigenstates=ser_op["eigenstates"],
        n_qudits=ser_op["n_qudits"],
        operations=_convert_complex(ser_op["operations"]),
    )


def _deserialize_observable(
    ser_obs: dict, state_type: Type[State], op_type: Type[Operator]
) -> Observable:
    obs = ser_obs.copy()
    obs_name = obs.pop("observable")
    if obs_name == "bitstrings":
        return BitStrings(**obs)
    if obs_name == "expectation":
        return Expectation(
            _deserialize_operator(obs.pop("operator"), op_type), **obs
        )
    if obs_name == "fidelity":
        return Fidelity(
            _deserialize_state(obs.pop("state"), state_type), **obs
        )
    if obs_name == "occupation":
        return Occupation(**obs)
    if obs_name == "correlation_matrix":
        return CorrelationMatrix(**obs)
    if obs_name == "energy":
        return Energy(**obs)
    if obs_name == "energy_second_moment":
        return EnergySecondMoment(**obs)
    if obs_name == "energy_variance":
        return EnergyVariance(**obs)
    raise RuntimeError  # TODO: Change to a better error


def _deserialize_emulation_config(
    ser_config: dict,
    config_type: Type[EmulationConfigType],
    state_type: Type[StateType],
    op_type: Type[Operator],
) -> EmulationConfigType:
    config = ser_config.copy()
    observables = [
        _deserialize_observable(obs, state_type, op_type)
        for obs in config.pop("observables")
    ]
    noise_model = _deserialize_noise_model(config.pop("noise_model"))
    initial_state = config.pop("initial_state", None)
    if initial_state is not None:
        initial_state = _deserialize_state(initial_state, state_type)
    return config_type(
        observables=observables,
        noise_model=noise_model,
        initial_state=initial_state,
        **config,
    )
