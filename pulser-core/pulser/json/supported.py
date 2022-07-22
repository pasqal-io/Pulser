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
"""Supported modules and objects for JSON deserialization."""

from __future__ import annotations

from typing import Any, Mapping

import pulser.devices as devices
from pulser.json.exceptions import SerializationError

SUPPORTED_BUILTINS = ("float", "int", "str", "set")

SUPPORTED_OPERATORS = (
    "neg",
    "abs",
    "getitem",
    "add",
    "sub",
    "mul",
    "truediv",
    "pow",
    "mod",
)

SUPPORTED_NUMPY = (
    "array",
    "round_",
    "ceil",
    "floor",
    "sqrt",
    "exp",
    "log2",
    "log",
    "sin",
    "cos",
    "tan",
)

SUPPORTS_SUBMODULE = (
    "Pulse",
    "BlackmanWaveform",
    "KaiserWaveform",
    "Register",
    "Register3D",
)

SUPPORTED_MODULES = {
    "builtins": SUPPORTED_BUILTINS,
    "_operator": SUPPORTED_OPERATORS,
    "operator": SUPPORTED_OPERATORS,
    "numpy": SUPPORTED_NUMPY,
    "pulser.register.register": ("Register",),
    "pulser.register.register3d": ("Register3D",),
    "pulser.register.register_layout": ("RegisterLayout",),
    "pulser.register.special_layouts": (
        "SquareLatticeLayout",
        "TriangularLatticeLayout",
    ),
    "pulser.register.mappable_reg": ("MappableRegister",),
    "pulser.devices": tuple(
        [dev.name for dev in devices._valid_devices] + ["MockDevice"]
    ),
    "pulser.pulse": ("Pulse",),
    "pulser.waveforms": (
        "CompositeWaveform",
        "CustomWaveform",
        "ConstantWaveform",
        "RampWaveform",
        "BlackmanWaveform",
        "InterpolatedWaveform",
        "KaiserWaveform",
    ),
    "pulser.sequence.sequence": ("Sequence",),
    "pulser.parametrized.variable": ("Variable",),
    "pulser.parametrized.paramobj": ("ParamObj",),
}


def validate_serialization(obj_dict: Mapping[str, Any]) -> None:
    """Checks if 'obj_dict' can be serialized."""
    try:
        obj_dict["_build"]
        obj_str = obj_dict["__name__"]
        module_str = obj_dict["__module__"]
    except KeyError:
        raise TypeError("Invalid 'obj_dict'.")

    if module_str not in SUPPORTED_MODULES:
        raise SerializationError(
            f"No serialization support for module '{module_str}'."
        )

    if "__submodule__" in obj_dict:
        submodule_str = obj_dict["__submodule__"]
        if submodule_str not in SUPPORTS_SUBMODULE:
            raise SerializationError(
                "No serialization support for attributes of "
                f"'{module_str}.{submodule_str}'."
            )
        obj_str = submodule_str

    if obj_str not in SUPPORTED_MODULES[module_str]:
        raise SerializationError(
            f"No serialization support for '{module_str}.{obj_str}'."
        )
