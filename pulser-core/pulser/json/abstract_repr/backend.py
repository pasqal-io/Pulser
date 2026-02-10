"""(De)serialization logic specific to the pulser.backend module."""

from __future__ import annotations

import uuid
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
from pulser.exceptions.serialization import AbstractReprError
from pulser.json.abstract_repr.deserializer import (
    _deserialize_noise_model,
    deserialize_complex,
)

if TYPE_CHECKING:
    from pulser.backend import EmulationConfig, Observable, Operator, State

EmulationConfigType = TypeVar("EmulationConfigType", bound="EmulationConfig")
StateType = TypeVar("StateType", bound="State")
OperatorType = TypeVar("OperatorType", bound="Operator")


def _deserialize_state(
    ser_state: dict, state_type: Type[StateType]
) -> StateType:
    """Deserialize a state from its abstract representation.

    Args:
        ser_state: state representation encoded in the abstract JSON format.
        state_type: type of the state to create.

    Returns:
        StateType: the deserialized state.
    """
    return state_type.from_state_amplitudes(
        eigenstates=ser_state["eigenstates"],
        amplitudes=deserialize_complex(ser_state["amplitudes"]),
    )


def _deserialize_operator(
    ser_op: dict, op_type: Type[OperatorType]
) -> OperatorType:
    """Deserialize an operator state from its abstract representation.

    Args:
        ser_op: operator representation encoded in the abstract JSON format.
        op_type: type of the operator to create.

    Returns:
        OperatorType: the deserialized operator.
    """
    # format operations to FullOp type
    operations = ser_op["operations"]
    for i, tensor_op in enumerate(operations):
        qudit_ops = tensor_op[1]
        for j, qudit_op in enumerate(qudit_ops):
            qudit_ops[j] = tuple(qudit_op)
        operations[i] = tuple(tensor_op)

    return op_type.from_operator_repr(
        eigenstates=ser_op["eigenstates"],
        n_qudits=ser_op["n_qudits"],
        operations=deserialize_complex(operations),
    )


def _deserialize_observable(
    ser_obs: dict, state_type: Type[State], op_type: Type[Operator]
) -> Observable:
    obs_params = ser_obs.copy()
    obs_name = obs_params.pop("observable")
    obs_uuid = obs_params.pop("uuid", None)
    obs: Observable
    match obs_name:
        case "bitstrings":
            obs = BitStrings(**obs_params)
        case "expectation":
            obs = Expectation(
                _deserialize_operator(obs_params.pop("operator"), op_type),
                **obs_params,
            )
        case "fidelity":
            obs = Fidelity(
                _deserialize_state(obs_params.pop("state"), state_type),
                **obs_params,
            )
        case "occupation":
            obs = Occupation(**obs_params)
        case "correlation_matrix":
            obs = CorrelationMatrix(**obs_params)
        case "energy":
            obs = Energy(**obs_params)
        case "energy_second_moment":
            obs = EnergySecondMoment(**obs_params)
        case "energy_variance":
            obs = EnergyVariance(**obs_params)
        case _:
            raise AbstractReprError(
                f"Failed to deserialize the observable tagged `{obs_name}` "
                "as unknown or not supported. This likely implies that the "
                "JSON abstract representation of the emulation configuration "
                "has not been validated or has been corrupted."
            )
    if obs_uuid is not None:
        # Replace the observable's UUID with the one in the schema
        obs._uuid = uuid.UUID(obs_uuid)
    return obs


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
