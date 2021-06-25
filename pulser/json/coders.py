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
"""Contains the custom Encoder and Decoder for JSON serialization."""

from __future__ import annotations

import importlib
import inspect
from json import JSONEncoder, JSONDecoder
from typing import Any, cast

import numpy as np

from pulser.json.utils import obj_to_dict
from pulser.parametrized import Variable


class PulserEncoder(JSONEncoder):
    """The custom encoder for Pulser objects."""

    def default(self, o: Any) -> dict[str, Any]:
        """Handles JSON encoding of objects not supported by default."""
        if hasattr(o, "_to_dict"):
            return cast(dict, o._to_dict())
        elif type(o) == type:
            return obj_to_dict(o, _build=False, _name=o.__name__)
        elif isinstance(o, np.ndarray):
            return obj_to_dict(o, o.tolist(), _name="array")
        elif isinstance(o, set):
            return obj_to_dict(o, list(o))
        else:
            return cast(dict, JSONEncoder.default(self, o))


class PulserDecoder(JSONDecoder):
    """The custom decoder for Pulser objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the decoder."""
        # TODO: Check version compatibility (stored at the Sequence level)
        self.vars: dict[str, Variable] = {}
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: dict[str, Any]) -> Any:
        """Enforces custom deserializations when decoding."""
        try:
            build = obj["_build"]
            obj_name = obj["__name__"]
            module_str = obj["__module__"]
        except KeyError:
            return obj

        if (
            obj_name == "Variable"
            and module_str == "pulser.parametrized.variable"
        ):
            var_name = obj["name"]
            try:
                var = self.vars[var_name]
                assert var.name == var_name, (
                    f"Variable {var.name} already "
                    + f"declared under {var_name}."
                )
                assert var.dtype == obj["dtype"], (
                    "Mismatching variable types for variables under the name "
                    + f"'{var_name}'."
                )
                assert var.size == obj["size"], (
                    "Mismatching sizes for variables under the name "
                    + f"'{var_name}'."
                )
            except KeyError:
                var = Variable(var_name, obj["dtype"], obj["size"])
                self.vars[var_name] = var

            return var

        module = importlib.import_module(module_str)
        if "__submodule__" in obj:
            submodule = getattr(module, obj["__submodule__"])
            cls = getattr(submodule, obj_name)
            if inspect.ismethod(cls):
                cls = cls.__func__  # Use the unbound function by default
        else:
            cls = getattr(module, obj_name)

        if not build:
            return cls

        if obj_name == "Sequence":
            seq = cls(*obj["__args__"], **obj["__kwargs__"])
            for name, args, kwargs in obj["calls"]:
                getattr(seq, name)(*args, **kwargs)
            seq._building = obj["vars"] == {}
            for name, var in obj["vars"].items():
                assert name not in seq._variables, (
                    "Multiples variables with" + f" the name '{name}'."
                )
                seq._variables[name] = var
            for name, args, kwargs in obj["to_build_calls"]:
                getattr(seq, name)(*args, **kwargs)
            return seq
        else:
            return cls(*obj["__args__"], **obj["__kwargs__"])
