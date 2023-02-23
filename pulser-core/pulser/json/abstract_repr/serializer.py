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
"""Utility functions for JSON serialization to the abstract representation."""
from __future__ import annotations

import inspect
import json
from itertools import chain
from typing import TYPE_CHECKING, Any
from typing import Sequence as abcSequence
from typing import Union, cast

import numpy as np

from pulser.json.abstract_repr.signatures import SIGNATURES
from pulser.json.exceptions import AbstractReprError
from pulser.register.base_register import QubitId

if TYPE_CHECKING:
    from pulser.sequence import Sequence
    from pulser.sequence._call import _Call


class AbstractReprEncoder(json.JSONEncoder):
    """The custom encoder for abstract representation of Pulser objects."""

    def default(self, o: Any) -> Union[dict[str, Any], list[Any]]:
        """Handles JSON encoding of objects not supported by default."""
        if hasattr(o, "_to_abstract_repr"):
            return cast(dict, o._to_abstract_repr())
        elif isinstance(o, np.ndarray):
            return cast(list, o.tolist())
        elif isinstance(o, set):
            return list(o)
        else:
            return cast(dict, json.JSONEncoder.default(self, o))


def abstract_repr(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Generates the abstract repr of an object with a defined signature."""
    try:
        signature = SIGNATURES[name]
    except KeyError:
        raise ValueError(f"No signature found for '{name}'.")
    arg_as_kwarg: tuple[str, ...] = tuple()
    if len(args) < len(signature.pos):
        # If less arguments than those in the signature were given, that might
        # be because they were provided with a keyword and thus stored as
        # kwargs instead (unless var_pos is defined)
        arg_as_kwarg = signature.pos[len(args) :]
        if signature.var_pos is not None or not set(arg_as_kwarg) <= set(
            kwargs
        ):
            raise ValueError(
                f"Not enough arguments given for '{name}' (expected "
                f"{len(signature.pos)}, got {len(args)})."
            )
    res: dict[str, Any] = {}
    res.update(signature.extra)  # Starts with extra info ({} if undefined)
    # With PulseSignature.all_pos_args(), we safeguard against the opposite
    # case where an expected keyword argument is given as a positional argument
    res.update(
        {
            arg_name: arg_val
            for arg_name, arg_val in zip(signature.all_pos_args(), args)
        }
    )
    # Account for keyword arguments given as pos args
    max_pos_args = len(signature.pos) + len(
        set(signature.keyword) - set(kwargs)
    )
    if signature.var_pos:
        res[signature.var_pos] = args[len(signature.pos) :]
    elif len(args) > max_pos_args:
        raise ValueError(
            f"Too many positional arguments given for '{name}' (expected "
            f"{max_pos_args}, got {len(args)})."
        )
    for kw in kwargs:
        if kw in signature.keyword or kw in arg_as_kwarg:
            res[kw] = kwargs[kw]
        else:
            raise ValueError(
                f"Keyword argument '{kw}' is not in the signature of '{name}'."
            )
    return res


def serialize_abstract_sequence(
    seq: Sequence, seq_name: str = "pulser-exported", **defaults: Any
) -> str:
    """Serializes the Sequence into an abstract JSON object.

    Keyword Args:
        seq_name (str): A name for the sequence. If not defined, defaults
            to "pulser-exported".
        defaults: The default values for all the variables declared in this
            Sequence instance, indexed by the name given upon declaration.
            Check ``Sequence.declared_variables`` to see all the variables.
            When using a MappableRegister, the Qubit IDs to trap IDs
            mapping must also be provided under the `qubits` keyword.

    Note:
        Providing the `defaults` is optional but, when done, it is
        mandatory to give default values for all the expected parameters.

    Returns:
        str: The sequence encoded as an abstract JSON object.
    """
    res: dict[str, Any] = {
        "version": "1",
        "name": seq_name,
        "register": [],
        "channels": {},
        "variables": {},
        "operations": [],
        "measurement": None,
    }

    for var in seq._variables.values():
        res["variables"][var.name] = dict(type=var.dtype.__name__)

    qubits_default = defaults.pop("qubits", None)
    if defaults or qubits_default:
        seq._cross_check_vars(defaults)
        try:
            seq.build(qubits=qubits_default, **defaults)
        except Exception:
            raise ValueError(
                "The given 'defaults' produce an invalid sequence."
            )

        for var in seq._variables.values():
            value = var._validate_value(defaults[var.name])
            res["variables"][var.name]["value"] = value.tolist()
    else:
        # Still need to set a default value for the variables because the
        # deserializer uses it to infer the size of the variable
        for var in seq._variables.values():
            res["variables"][var.name]["value"] = [var.dtype()] * var.size

    def convert_targets(
        target_ids: Union[QubitId, abcSequence[QubitId]]
    ) -> Union[int, list[int]]:
        target_array = np.array(target_ids)
        og_dim = target_array.ndim
        if og_dim == 0:
            target_array = target_array[np.newaxis]
        indices = seq.register.find_indices(target_array.tolist())
        return indices[0] if og_dim == 0 else indices

    def get_kwarg_default(call_name: str, kwarg_name: str) -> Any:
        sig = inspect.signature(getattr(seq, call_name))
        return sig.parameters[kwarg_name].default

    def get_all_args(
        pos_args_signature: tuple[str, ...], call: _Call
    ) -> dict[str, Any]:
        params = {**dict(zip(pos_args_signature, call.args)), **call.kwargs}
        default_values = {
            p_name: get_kwarg_default(call.name, p_name)
            for p_name in pos_args_signature
            if p_name not in params
        }
        return {**default_values, **params}

    operations = res["operations"]
    for call in chain(seq._calls, seq._to_build_calls):
        if call.name == "__init__":
            data = get_all_args(("register", "device"), call)
            res["device"] = data["device"]
            res["register"] = data["register"]
            layout = data["register"].layout
            if layout is not None:
                res["layout"] = layout
            if qubits_default is not None:
                serial_reg = res["register"]._to_abstract_repr()
                for q_dict in serial_reg:
                    qid = q_dict["qid"]
                    if qid in qubits_default:
                        q_dict["default_trap"] = qubits_default[qid]
                res["register"] = serial_reg
        elif call.name == "declare_channel":
            data = get_all_args(
                ("channel", "channel_id", "initial_target"), call
            )
            res["channels"][data["channel"]] = data["channel_id"]
            if data["initial_target"] is not None:
                operations.append(
                    {
                        "op": "target",
                        "channel": data["channel"],
                        "target": convert_targets(data["initial_target"]),
                    }
                )
        elif "target" in call.name:
            data = get_all_args(("qubits", "channel"), call)
            if call.name == "target":
                target = convert_targets(data["qubits"])
            elif call.name == "target_index":
                target = data["qubits"]
            else:
                raise AbstractReprError(f"Unknown call '{call.name}'.")
            operations.append(
                {
                    "op": "target",
                    "channel": data["channel"],
                    "target": target,
                }
            )
        elif call.name == "align":
            operations.append({"op": "align", "channels": list(call.args)})
        elif call.name == "delay":
            data = get_all_args(("duration", "channel"), call)
            operations.append(
                {
                    "op": "delay",
                    "channel": data["channel"],
                    "time": data["duration"],
                }
            )
        elif call.name == "measure":
            data = get_all_args(("basis",), call)
            res["measurement"] = data["basis"]
        elif call.name == "add":
            data = get_all_args(("pulse", "channel", "protocol"), call)
            op_dict = {
                "op": "pulse",
                "channel": data["channel"],
                "protocol": data["protocol"],
            }
            op_dict.update(data["pulse"]._to_abstract_repr())
            operations.append(op_dict)
        elif "phase_shift" in call.name:
            targets = call.args[1:]
            if call.name == "phase_shift":
                targets = convert_targets(targets)
            elif call.name != "phase_shift_index":
                raise AbstractReprError(f"Unknown call '{call.name}'.")
            operations.append(
                {
                    "op": "phase_shift",
                    "phi": call.args[0],
                    "targets": targets,
                    "basis": call.kwargs.get(
                        "basis", get_kwarg_default(call.name, "basis")
                    ),
                }
            )
        elif call.name == "set_magnetic_field":
            res["magnetic_field"] = seq.magnetic_field.tolist()
        elif call.name == "config_slm_mask":
            res["slm_mask_targets"] = tuple(seq._slm_mask_targets)
        elif call.name == "enable_eom_mode":
            data = get_all_args(
                ("channel", "amp_on", "detuning_on", "optimal_detuning_off"),
                call,
            )
            operations.append({"op": "enable_eom_mode", **data})
        elif call.name == "add_eom_pulse":
            data = get_all_args(
                (
                    "channel",
                    "duration",
                    "phase",
                    "post_phase_shift",
                    "protocol",
                ),
                call,
            )
            operations.append({"op": "add_eom_pulse", **data})
        elif call.name == "disable_eom_mode":
            data = get_all_args(("channel",), call)
            operations.append(
                {"op": "disable_eom_mode", "channel": data["channel"]}
            )
        else:
            raise AbstractReprError(f"Unknown call '{call.name}'.")

    return json.dumps(res, cls=AbstractReprEncoder)
