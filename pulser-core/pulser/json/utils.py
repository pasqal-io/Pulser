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

import warnings
from typing import TYPE_CHECKING, Any, Optional, Sequence

import pulser
from pulser.json.exceptions import AbstractReprError

if TYPE_CHECKING:  # pragma: no cover
    from pulser.register import QubitId


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
        The dictionary encoding the object.
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


def stringify_qubit_ids(qubit_ids: Sequence[QubitId]) -> list[str]:
    """Converts all qubit IDs into strings and looks for conflicts."""
    not_str = [id for id in qubit_ids if not isinstance(id, str)]
    names = [str(id) for id in qubit_ids]
    if not_str:
        warnings.warn(
            "Register serialization to an abstract representation "
            "irreversibly converts all qubit ID's to strings.",
            stacklevel=2,
        )
        if len(set(names)) < len(names):
            collisions = [id for id in not_str if str(id) in qubit_ids]
            raise AbstractReprError(
                "Name collisions encountered when converting qubit IDs to "
                f"strings for IDs: {[(id, str(id)) for id in collisions]}"
            )
    return names
