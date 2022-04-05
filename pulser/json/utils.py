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

from typing import Any, Optional

import pulser
from pulser.json.signatures import SIGNATURES


def obj_to_dict(
    obj: object,
    *args: Any,
    _build: bool = True,
    _module: Optional[str] = None,
    _name: Optional[str] = None,
    _submodule: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Encodes an object in a dictionary for serialization.

    Args:
        obj: The object to encode in the dictionary.

    Other Parameters:
        _build (bool): Whether the object is to be built on deserialization.
        _module (str): Custom name for the module containing the object.
        _name (str): Custom name of the object.
        _submodule(str): Name of a submodule (e.g. the class holding a
                         classmethod). Only used when defined.
        args: If the object is to be built, the arguments to give on creation.
        kwargs: If the object is to be built, the keyword arguments to give on
            creation.

    Returns:
        dict: The dictionary encoding the object.
    """
    d = {
        "_build": _build,
        "__module__": _module if _module else obj.__class__.__module__,
        "__name__": _name if _name else obj.__class__.__name__,
    }
    if _build:
        d["__args__"] = args
        d["__kwargs__"] = kwargs
    if _submodule:
        d["__submodule__"] = _submodule

    pulser.json.supported.validate_serialization(d)
    return d


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
