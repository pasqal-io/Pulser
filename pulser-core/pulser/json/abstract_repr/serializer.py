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
"""Utility functions for JSON serializations."""

from __future__ import annotations

import json
from itertools import chain
from typing import Any
from typing import Sequence as abcSequence
from typing import Union, cast

import numpy as np

from pulser.json.exceptions import AbstractReprError
from pulser.json.signatures import SIGNATURES
from pulser.register.base_register import QubitId


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
    if len(args) < len(signature.pos):
        raise ValueError(
            f"Not enough positional arguments given for '{name}' (expected "
            f"{len(signature.pos)}, got {len(args)})."
        )
    res: dict[str, Any] = {}
    res.update(signature.extra)  # Starts with extra info ({} if undefined)
    res.update(
        {arg_name: arg_val for arg_name, arg_val in zip(signature.pos, args)}
    )
    if signature.var_pos:
        res[signature.var_pos] = args[len(signature.pos) :]
    elif len(args) > len(signature.pos):
        raise ValueError(
            f"Too many positional arguments given for '{name}' (expected "
            f"{len(signature.pos)}, got {len(args)})."
        )
    for kw in kwargs:
        if kw in signature.keyword:
            res[kw] = kwargs[kw]
        else:
            raise ValueError(
                f"Keyword argument '{kw}' is not in the signature of '{name}'."
            )
    return res


def serialize_abstract_sequence(
    seq, seq_name: str = "pulser-exported", **defaults: Any
) -> str:
    """Serializes the Sequence into an abstract JSON object.

    Keyword Args:
        seq_name (str): A name for the sequence. If not defined, defaults
            to "pulser-exported".
        defaults: The default values for all the variables declared in this
            Sequence instance, indexed by the name given upon declaration.
            Check ``Sequence.declared_variables`` to see all the variables.

    Returns:
        str: The sequence encoded as an abstract JSON object.
    """

    res: dict[str, Any] = {
        "version": "1",
        "name": seq_name,
        "register": {},
        "channels": {},
        "variables": {},
        "operations": [],
        "measurement": None,
    }

    seq._cross_check_vars(defaults)
    try:
        seq.build(**defaults)
    except Exception:
        raise ValueError("The given 'defaults' produce an invalid sequence.")

    for var in seq._variables.values():
        value = var._validate_value(defaults[var.name])
        res["variables"][var.name] = dict(
            type=var.dtype.__name__, value=value.tolist()
        )

    def convert_targets(
        target_ids: Union[QubitId, abcSequence[QubitId]]
    ) -> Union[int, list[int]]:
        target_array = np.array(target_ids)
        og_dim = target_array.ndim
        if og_dim == 0:
            target_array = target_array[np.newaxis]
        indices = seq.register.find_indices(target_array.tolist())
        return indices[0] if og_dim == 0 else indices

    operations = res["operations"]
    for call in chain(seq._calls, seq._to_build_calls):
        if call.name == "__init__":
            register, device = call.args
            res["device"] = device.name
            res["register"] = register
        elif call.name == "declare_channel":
            ch_name, ch_kind = call.args[:2]
            res["channels"][ch_name] = ch_kind
            initial_target = None
            if len(call.args) == 3:
                initial_target = call.args[2]
            elif "initial_target" in call.kwargs:
                initial_target = call.kwargs["initial_target"]
            if initial_target is not None:
                operations.append(
                    {
                        "op": "target",
                        "channel": ch_name,
                        "target": convert_targets(initial_target),
                    }
                )
        elif "target" in call.name:
            target_arg, ch_name = call.args
            if call.name == "target":
                target = convert_targets(target_arg)
            elif call.name == "_target_index":
                target = target_arg
            else:
                raise AbstractReprError(f"Unknown call '{call.name}'.")
            operations.append(
                {
                    "op": "target",
                    "channel": ch_name,
                    "target": target,
                }
            )
        elif call.name == "align":
            operations.append({"op": "align", "channels": list(call.args)})
        elif call.name == "delay":
            time, channel = call.args
            operations.append(
                {"op": "delay", "channel": channel, "time": time}
            )
        elif call.name == "measure":
            res["measurement"] = call.args[0]
        elif call.name == "add":
            pulse, ch_name = call.args[:2]
            if len(call.args) > 2:
                protocol = call.args[2]
            elif "protocol" in call.kwargs:
                protocol = call.kwargs["protocol"]
            else:
                protocol = "min-delay"
            op_dict = {
                "op": "pulse",
                "channel": ch_name,
                "protocol": protocol,
            }
            op_dict.update(pulse._to_abstract_repr())
            operations.append(op_dict)
        elif "phase_shift" in call.name:
            try:
                basis = call.kwargs["basis"]
            except KeyError:
                basis = "digital"
            targets = call.args[1:]
            if call.name == "phase_shift":
                targets = convert_targets(targets)
            elif call.name == "_phase_shift_index":
                pass
            else:
                raise AbstractReprError(f"Unknown call '{call.name}'.")
            operations.append(
                {
                    "op": "phase_shift",
                    "phi": call.args[0],
                    "targets": targets,
                    "basis": basis,
                }
            )
        else:
            raise AbstractReprError(
                f"Call name '{call.name}' is not supported."
            )

    undefined_str_vars = [
        var_name
        for var_name, var_dict in res["variables"].items()
        if var_dict["type"] == "str"
    ]
    if undefined_str_vars:
        raise AbstractReprError(
            "All 'str' type variables must be used to refer to a qubit. "
            f"Condition not respected for: {undefined_str_vars}"
        )

    return json.dumps(res, cls=AbstractReprEncoder)
