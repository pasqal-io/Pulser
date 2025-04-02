# Copyright 2022 Pulser Development Team
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
"""Deserializer from JSON in the abstract representation."""
from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Literal, Type, Union, cast, overload

import jsonschema
import jsonschema.exceptions

import pulser
import pulser.devices as devices
from pulser.channels import DMM, Microwave, Raman, Rydberg
from pulser.channels.base_channel import Channel
from pulser.channels.eom import (
    OPTIONAL_ABSTR_EOM_FIELDS,
    RydbergBeam,
    RydbergEOM,
)
from pulser.devices import Device, VirtualDevice
from pulser.devices._device_datacls import PARAMS_WITH_ABSTR_REPR
from pulser.exceptions.serialization import (
    AbstractReprError,
    DeserializeDeviceError,
)
from pulser.json.abstract_repr.signatures import (
    BINARY_OPERATORS,
    UNARY_OPERATORS,
)
from pulser.json.abstract_repr.validation import validate_abstract_repr
from pulser.json.utils import get_dataclass_defaults
from pulser.parametrized import ParamObj, Variable
from pulser.pulse import Pulse
from pulser.register.mappable_reg import MappableRegister
from pulser.register.register_layout import RegisterLayout
from pulser.register.weight_maps import DetuningMap
from pulser.waveforms import (
    BlackmanWaveform,
    CompositeWaveform,
    ConstantWaveform,
    CustomWaveform,
    InterpolatedWaveform,
    KaiserWaveform,
    RampWaveform,
    Waveform,
)

if TYPE_CHECKING:
    from pulser.noise_model import NoiseModel
    from pulser.register import Register, Register3D
    from pulser.sequence import Sequence


VARIABLE_TYPE_MAP = {"int": int, "float": float}

ExpReturnType = Union[int, float, list, ParamObj]


@overload
def _deserialize_parameter(param: int, vars: dict[str, Variable]) -> int:
    pass


@overload
def _deserialize_parameter(param: float, vars: dict[str, Variable]) -> float:
    pass


@overload
def _deserialize_parameter(
    param: list[int], vars: dict[str, Variable]
) -> list[int]:
    pass


@overload
def _deserialize_parameter(
    param: dict[str, str], vars: dict[str, Variable]
) -> Variable:
    pass


def _deserialize_parameter(
    param: Union[int, float, list[int], dict[str, Any]],
    vars: dict[str, Variable],
) -> Union[ExpReturnType, Variable]:
    """Deserialize a parameterized object.

    A parameter can be either a literal, a variable or an expression.
    In the first case, return the literal. Otherwise, return a reference
    to the variable, or build an expression referencing variables.

    Args:
        param: The JSON parametrized object to deserialize
        vars: The references to the sequence variables

    Returns:
        A literal (int | float), a ``Variable``, or a ``ParamObj``.
    """
    if not isinstance(param, dict):
        # This is a literal
        return param

    if "variable" in param:
        # This is a reference to a variable.
        if param["variable"] not in vars:
            raise AbstractReprError(
                f"Variable '{param['variable']}' used in operations "
                "but not found in declared variables."
            )
        return vars[param["variable"]]

    if "expression" not in param:
        # Can't deserialize param if it is a dict without a
        # `variable` or an `expression` key
        raise AbstractReprError(
            f"Parameter '{param}' is neither a literal nor "
            "a variable or an expression."
        )

    # This is a unary or a binary expression
    expression = (
        param["expression"] if param["expression"] != "div" else "truediv"
    )

    if expression in UNARY_OPERATORS:
        return cast(
            ExpReturnType,
            UNARY_OPERATORS[expression](
                _deserialize_parameter(param["lhs"], vars)
            ),
        )
    elif expression in BINARY_OPERATORS:
        return cast(
            ExpReturnType,
            BINARY_OPERATORS[expression](
                _deserialize_parameter(param["lhs"], vars),
                _deserialize_parameter(param["rhs"], vars),
            ),
        )
    else:
        raise AbstractReprError(f"Expression '{param['expression']}' invalid.")


def _deserialize_waveform(obj: dict, vars: dict) -> Waveform:
    if obj["kind"] == "constant":
        return ConstantWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            value=_deserialize_parameter(obj["value"], vars),
        )
    if obj["kind"] == "ramp":
        return RampWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            start=_deserialize_parameter(obj["start"], vars),
            stop=_deserialize_parameter(obj["stop"], vars),
        )
    if obj["kind"] == "blackman":
        return BlackmanWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            area=_deserialize_parameter(obj["area"], vars),
        )
    if obj["kind"] == "blackman_max":
        return BlackmanWaveform.from_max_val(
            max_val=_deserialize_parameter(obj["max_val"], vars),
            area=_deserialize_parameter(obj["area"], vars),
        )
    if obj["kind"] == "interpolated":
        return InterpolatedWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            values=_deserialize_parameter(obj["values"], vars),
            times=_deserialize_parameter(obj["times"], vars),
        )
    if obj["kind"] == "kaiser":
        return KaiserWaveform(
            duration=_deserialize_parameter(obj["duration"], vars),
            area=_deserialize_parameter(obj["area"], vars),
            beta=_deserialize_parameter(obj["beta"], vars),
        )
    if obj["kind"] == "kaiser_max":
        return KaiserWaveform.from_max_val(
            max_val=_deserialize_parameter(obj["max_val"], vars),
            area=_deserialize_parameter(obj["area"], vars),
            beta=_deserialize_parameter(obj["beta"], vars),
        )
    if obj["kind"] == "composite":
        wfs = [_deserialize_waveform(wf, vars) for wf in obj["waveforms"]]
        return CompositeWaveform(*wfs)
    if obj["kind"] == "custom":
        return CustomWaveform(
            samples=_deserialize_parameter(obj["samples"], vars)
        )

    raise AbstractReprError("The object does not encode a known waveform.")


def _deserialize_operation(seq: Sequence, op: dict, vars: dict) -> None:
    if op["op"] == "target":
        seq.target_index(
            qubits=_deserialize_parameter(op["target"], vars),
            channel=op["channel"],
        )
    elif op["op"] == "align":
        seq.align(
            *op["channels"],
            at_rest=op.get("at_rest", True),
        )
    elif op["op"] == "delay":
        seq.delay(
            duration=_deserialize_parameter(op["time"], vars),
            channel=op["channel"],
            at_rest=op.get("at_rest", False),
        )
    elif op["op"] == "phase_shift":
        seq.phase_shift_index(
            _deserialize_parameter(op["phi"], vars),
            *[_deserialize_parameter(t, vars) for t in op["targets"]],
            basis=op["basis"],
        )
    elif op["op"] == "pulse":
        phase = _deserialize_parameter(op["phase"], vars)
        post_phase_shift = _deserialize_parameter(op["post_phase_shift"], vars)

        # A waveform with a duration of 0 means the pulse was created with one
        # of Pulse's class methods (ConstantAmplitude or ConstantDetuning) and
        # the Pulse is parametrized
        if (
            op["amplitude"].get("duration") == 0
            and op["amplitude"].get("kind") == "constant"
        ):
            pulse = Pulse.ConstantAmplitude(
                amplitude=_deserialize_parameter(
                    op["amplitude"]["value"], vars
                ),
                detuning=_deserialize_waveform(op["detuning"], vars),
                phase=phase,
                post_phase_shift=post_phase_shift,
            )
        elif (
            op["detuning"].get("duration") == 0
            and op["detuning"].get("kind") == "constant"
        ):
            pulse = Pulse.ConstantDetuning(
                amplitude=_deserialize_waveform(op["amplitude"], vars),
                detuning=_deserialize_parameter(op["detuning"]["value"], vars),
                phase=phase,
                post_phase_shift=post_phase_shift,
            )
        else:
            pulse = Pulse(
                amplitude=_deserialize_waveform(op["amplitude"], vars),
                detuning=_deserialize_waveform(op["detuning"], vars),
                phase=phase,
                post_phase_shift=post_phase_shift,
            )

        seq.add(
            pulse=pulse,
            channel=op["channel"],
            protocol=op["protocol"],
        )
    elif op["op"] == "pulse_arbitrary_phase":
        pulse = Pulse.ArbitraryPhase(
            amplitude=_deserialize_waveform(op["amplitude"], vars),
            phase=_deserialize_waveform(op["phase"], vars),
            post_phase_shift=_deserialize_parameter(
                op["post_phase_shift"], vars
            ),
        )
        seq.add(
            pulse=pulse,
            channel=op["channel"],
            protocol=op["protocol"],
        )
    elif op["op"] == "enable_eom_mode":
        seq.enable_eom_mode(
            channel=op["channel"],
            amp_on=_deserialize_parameter(op["amp_on"], vars),
            detuning_on=_deserialize_parameter(op["detuning_on"], vars),
            optimal_detuning_off=_deserialize_parameter(
                op["optimal_detuning_off"], vars
            ),
            correct_phase_drift=op.get("correct_phase_drift", False),
        )
    elif op["op"] == "modify_eom_setpoint":
        seq.modify_eom_setpoint(
            channel=op["channel"],
            amp_on=_deserialize_parameter(op["amp_on"], vars),
            detuning_on=_deserialize_parameter(op["detuning_on"], vars),
            optimal_detuning_off=_deserialize_parameter(
                op["optimal_detuning_off"], vars
            ),
            correct_phase_drift=op["correct_phase_drift"],
        )
    elif op["op"] == "add_eom_pulse":
        seq.add_eom_pulse(
            channel=op["channel"],
            duration=_deserialize_parameter(op["duration"], vars),
            phase=_deserialize_parameter(op["phase"], vars),
            post_phase_shift=_deserialize_parameter(
                op["post_phase_shift"], vars
            ),
            protocol=op["protocol"],
            correct_phase_drift=op.get("correct_phase_drift", False),
        )
    elif op["op"] == "disable_eom_mode":
        seq.disable_eom_mode(
            channel=op["channel"],
            correct_phase_drift=op.get("correct_phase_drift", False),
        )
    elif op["op"] == "add_dmm_detuning":
        seq.add_dmm_detuning(
            waveform=_deserialize_waveform(op["waveform"], vars),
            dmm_name=op["dmm_name"],
            protocol=op["protocol"],
        )
    elif op["op"] == "config_slm_mask":
        seq.config_slm_mask(qubits=op["qubits"], dmm_id=op["dmm_id"])
    elif op["op"] == "config_detuning_map":
        seq.config_detuning_map(
            detuning_map=_deserialize_det_map(op["detuning_map"]),
            dmm_id=op["dmm_id"],
        )


def _deserialize_channel(obj: dict[str, Any]) -> Channel:
    params: dict[str, Any] = {}
    channel_cls: Type[Channel]
    if obj["basis"] == "ground-rydberg":
        if "bottom_detuning" in obj:
            channel_cls = DMM
        else:
            channel_cls = Rydberg
            params["eom_config"] = None
        if obj["eom_config"] is not None:
            data = obj["eom_config"]
            try:
                optional = {
                    key: data[key]
                    for key in OPTIONAL_ABSTR_EOM_FIELDS
                    if key in data
                }
                params["eom_config"] = RydbergEOM(
                    mod_bandwidth=data["mod_bandwidth"],
                    limiting_beam=RydbergBeam[data["limiting_beam"]],
                    max_limiting_amp=data["max_limiting_amp"],
                    intermediate_detuning=data["intermediate_detuning"],
                    controlled_beams=tuple(
                        RydbergBeam[beam] for beam in data["controlled_beams"]
                    ),
                    **optional,
                )
            except ValueError as e:
                raise AbstractReprError(
                    "RydbergEOM deserialization failed."
                ) from e
    elif obj["basis"] == "digital":
        channel_cls = Raman
    elif obj["basis"] == "XY":
        channel_cls = Microwave
    # No other basis allowed by the schema

    channel_fields = dataclasses.fields(channel_cls)
    channel_defaults = get_dataclass_defaults(channel_fields)
    for param in channel_fields:
        use_default = param.name not in obj and param.name in channel_defaults
        if param.init and param.name != "eom_config" and not use_default:
            params[param.name] = obj[param.name]
    try:
        return channel_cls(**params)
    except (ValueError, NotImplementedError) as e:
        raise AbstractReprError("Channel deserialization failed.") from e


def _deserialize_layout(layout_obj: dict[str, Any]) -> RegisterLayout:
    try:
        return RegisterLayout(
            layout_obj["coordinates"], slug=layout_obj.get("slug")
        )
    except ValueError as e:
        raise AbstractReprError(
            "Register layout deserialization failed."
        ) from e


def _deserialize_register(
    qubits: list[dict[str, Any]], layout: RegisterLayout | None
) -> Register:
    coords = [(q["x"], q["y"]) for q in qubits]
    qubit_ids = [q["name"] for q in qubits]
    if layout:
        trap_ids = layout.get_traps_from_coordinates(*coords)
        reg = layout.define_register(*trap_ids, qubit_ids=qubit_ids)
    else:
        reg = pulser.Register(dict(zip(qubit_ids, coords)))
    return cast(pulser.Register, reg)


def _deserialize_register3d(
    qubits: list[dict[str, Any]], layout: RegisterLayout | None
) -> Register3D:
    coords = [(q["x"], q["y"], q["z"]) for q in qubits]
    qubit_ids = [q["name"] for q in qubits]
    if layout:
        trap_ids = layout.get_traps_from_coordinates(*coords)
        reg = layout.define_register(*trap_ids, qubit_ids=qubit_ids)
    else:
        reg = pulser.Register3D(dict(zip(qubit_ids, coords)))
    return cast(pulser.Register3D, reg)


def _convert_complex(obj: Any) -> Any:
    """Searches for serialized complex numbers and converts them."""
    if isinstance(obj, list):
        return [_convert_complex(e) for e in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_complex(e) for e in obj)
    if isinstance(obj, dict):
        if obj.keys() == {"real", "imag"}:
            return obj["real"] + 1j * obj["imag"]
        return {k: _convert_complex(v) for k, v in obj.items()}
    return obj


def _deserialize_noise_model(noise_model_obj: dict[str, Any]) -> NoiseModel:
    eff_noise_rates = []
    eff_noise_opers = []
    for rate, oper in noise_model_obj.pop("eff_noise"):
        eff_noise_rates.append(rate)
        eff_noise_opers.append(_convert_complex(oper))

    noise_types = noise_model_obj.pop("noise_types")
    with_leakage = "leakage" in noise_types
    relevant_params = pulser.NoiseModel._find_relevant_params(
        noise_types,
        noise_model_obj["state_prep_error"],
        noise_model_obj["amp_sigma"],
        noise_model_obj["laser_waist"],
    ) - {  # Handled separately
        "eff_noise_rates",
        "eff_noise_opers",
        "with_leakage",
    }
    noise_model = pulser.NoiseModel(
        **{param: noise_model_obj[param] for param in relevant_params},
        eff_noise_rates=tuple(eff_noise_rates),
        eff_noise_opers=tuple(eff_noise_opers),
        with_leakage=with_leakage,
    )
    assert set(noise_model.noise_types) == set(noise_types)
    return noise_model


def _deserialize_device_object(obj: dict[str, Any]) -> Device | VirtualDevice:
    device_cls: Type[Device] | Type[VirtualDevice] = (
        VirtualDevice if obj["is_virtual"] else Device
    )
    ch_ids = []
    ch_objs = []
    for ch in obj["channels"]:
        ch_ids.append(ch["id"])
        ch_objs.append(_deserialize_channel(ch))
    params: dict[str, Any] = dict(
        channel_ids=tuple(ch_ids), channel_objects=tuple(ch_objs)
    )
    if "dmm_objects" in obj:
        params["dmm_objects"] = tuple(
            _deserialize_channel(dmm_ch) for dmm_ch in obj["dmm_objects"]
        )
    device_fields = dataclasses.fields(device_cls)
    device_defaults = get_dataclass_defaults(device_fields)
    for param in device_fields:
        use_default = param.name not in obj and param.name in device_defaults
        if (
            not param.init
            or param.name in PARAMS_WITH_ABSTR_REPR
            or use_default
        ):
            continue
        if param.name == "pre_calibrated_layouts":
            key = "pre_calibrated_layouts"
            params[key] = tuple(
                _deserialize_layout(layout) for layout in obj[key]
            )
        elif param.name == "default_noise_model":
            params[param.name] = _deserialize_noise_model(obj[param.name])
        else:
            params[param.name] = obj[param.name]
    try:
        return device_cls(**params)
    except (ValueError, TypeError) as e:
        raise AbstractReprError("Device deserialization failed.") from e


def _deserialize_det_map(ser_det_map: dict) -> DetuningMap:
    trap_coords = []
    weights = []
    for trap in ser_det_map["traps"]:
        trap_coords.append((trap["x"], trap["y"]))
        weights.append(trap["weight"])
    return DetuningMap(
        trap_coordinates=trap_coords,
        weights=weights,
        slug=ser_det_map.get("slug"),
    )


def deserialize_abstract_sequence(obj_str: str) -> Sequence:
    """Deserialize a sequence from an abstract JSON object.

    Args:
        obj_str: the JSON string representing the sequence encoded
            in the abstract JSON format.

    Returns:
        Sequence: The Pulser sequence.
    """
    # Validate the format of the data against the JSON schema.
    validate_abstract_repr(obj_str, "sequence")
    obj = json.loads(obj_str)
    # Device
    if isinstance(obj["device"], str):
        device_name = obj["device"]
        device = getattr(devices, device_name)
    else:
        device = _deserialize_device_object(obj["device"])

    # Register Layout
    layout = _deserialize_layout(obj["layout"]) if "layout" in obj else None

    # Register
    reg: Register | Register3D | MappableRegister
    qubits = obj["register"]
    if {"name", "x", "y"} == qubits[0].keys():
        # Regular 2D register
        reg = _deserialize_register(qubits, layout)
    elif {"name", "x", "y", "z"} == qubits[0].keys():
        # Regular 3D register
        reg = _deserialize_register3d(qubits, layout)
    else:
        # Mappable register
        assert (
            layout is not None
        ), "Layout must be defined in a MappableRegister."
        reg = MappableRegister(layout, *(d["qid"] for d in qubits))

    seq = pulser.Sequence(reg, device)

    # Channels
    for name, channel_id in obj["channels"].items():
        seq.declare_channel(name, channel_id)

    # Magnetic field
    if "magnetic_field" in obj:
        seq.set_magnetic_field(*obj["magnetic_field"])

    # SLM Mask
    if "slm_mask_targets" in obj:
        # This is kept for backwards compatibility
        seq.config_slm_mask(obj["slm_mask_targets"])

    # Variables
    vars = {}
    for name, desc in obj["variables"].items():
        v = seq.declare_variable(
            cast(str, name),
            size=len(desc["value"]),
            dtype=VARIABLE_TYPE_MAP[desc["type"]],
        )
        vars[name] = v

    # Operations
    for op in obj["operations"]:
        _deserialize_operation(seq, op, vars)

    # Measurement
    if obj["measurement"] is not None:
        seq.measure(obj["measurement"])

    return seq


def deserialize_device(obj_str: str) -> Device | VirtualDevice:
    """Deserialize a device from an abstract JSON object.

    Args:
        obj_str: the JSON string representing the device encoded
            in the abstract JSON format.

    Returns:
        The Pulser device.

    Raises:
        DeserializeDeviceError: Whenever the device deserialization
            fails due to an invalid 'obj_str'.
    """
    if not isinstance(obj_str, str):
        type_error = TypeError(
            f"'obj_str' must be a string, not {type(obj_str)}."
        )
        raise DeserializeDeviceError from type_error

    try:
        # Validate the format of the data against the JSON schema.
        validate_abstract_repr(obj_str, "device")
        return _deserialize_device_object(json.loads(obj_str))
    except (
        json.JSONDecodeError,  # From json.loads
        jsonschema.exceptions.ValidationError,  # From jsonschema.validate
        AbstractReprError,  # From _deserialize_device_object
    ) as e:
        raise DeserializeDeviceError from e


def deserialize_abstract_layout(obj_str: str) -> RegisterLayout:
    """Deserialize a layout from an abstract JSON object.

    Args:
        obj_str: the JSON string representing the layout encoded
            in the abstract JSON format.

    Returns:
        The RegisterLayout instance.
    """
    validate_abstract_repr(obj_str, "layout")
    return _deserialize_layout(json.loads(obj_str))


@overload
def deserialize_abstract_register(
    obj_str: str, expected_dim: Literal[2]
) -> Register:
    pass


@overload
def deserialize_abstract_register(
    obj_str: str, expected_dim: Literal[3]
) -> Register3D:
    pass


@overload
def deserialize_abstract_register(obj_str: str) -> Register | Register3D:
    pass


def deserialize_abstract_register(
    obj_str: str, expected_dim: Literal[None, 2, 3] = None
) -> Register | Register3D:
    """Deserialize a register from an abstract JSON object.

    Args:
        obj_str: The JSON string representing the register encoded
            in the abstract JSON format.
        expected_dim: If defined, ensures the register is of the
            specified dimensionality.

    Returns:
        The Register instance.
    """
    if expected_dim not in (None, 2, 3):
        raise ValueError(
            "When specified, 'expected_dim' must be 2 or 3, "
            f"not {expected_dim!s}."
        )
    validate_abstract_repr(obj_str, "register")
    obj = json.loads(obj_str)
    layout = _deserialize_layout(obj["layout"]) if "layout" in obj else None
    qubits = obj["register"]
    dim_ = len(set(qubits[0]) - {"name"})
    # These conditions should be enforced by the schema
    assert dim_ == 2 or dim_ == 3
    assert layout is None or layout.dimensionality == dim_
    if expected_dim is not None and expected_dim != dim_:
        raise ValueError(
            f"The provided register must be in {expected_dim}D, not {dim_}D."
        )
    if dim_ == 3:
        return _deserialize_register3d(qubits=qubits, layout=layout)
    return _deserialize_register(qubits=qubits, layout=layout)


def deserialize_abstract_noise_model(obj_str: str) -> NoiseModel:
    """Deserialize a noise model from an abstract JSON object.

    Args:
        obj_str: the JSON string representing the noise model encoded
            in the abstract JSON format.

    Returns:
        The NoiseModel instance.
    """
    validate_abstract_repr(obj_str, "noise")
    return _deserialize_noise_model(json.loads(obj_str))
