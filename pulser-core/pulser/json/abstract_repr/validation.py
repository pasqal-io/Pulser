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
"""Function for validation of JSON serialization to abstract representation."""
import json
from importlib.metadata import version
from typing import Literal

import jsonschema
from packaging.version import InvalidVersion, Version
from referencing import Registry, Resource

import pulser
from pulser.exceptions.serialization import AbstractReprError
from pulser.json.abstract_repr import SCHEMAS, SCHEMAS_PATH

LEGACY_JSONSCHEMA = (
    Version("4.18") > Version(version("jsonschema")) >= Version("4.17.3")
)

REGISTRY: Registry = Registry(
    [
        ("device-schema.json", Resource.from_contents(SCHEMAS["device"])),
        ("layout-schema.json", Resource.from_contents(SCHEMAS["layout"])),
        ("register-schema.json", Resource.from_contents(SCHEMAS["register"])),
        ("noise-schema.json", Resource.from_contents(SCHEMAS["noise"])),
    ]
)

# Build one validator per schema at import time so that meta-schema
# checking and $ref resolution are paid once, not on every call.
_VALIDATORS: dict[str, jsonschema.Draft7Validator] = (
    {}
    if LEGACY_JSONSCHEMA
    else {
        name: jsonschema.Draft7Validator(schema, registry=REGISTRY)
        for name, schema in SCHEMAS.items()
    }
)


def validate_abstract_repr(
    obj_str: str,
    name: Literal[
        "sequence",
        "device",
        "layout",
        "register",
        "noise",
        "results",
        "config",
    ],
) -> None:
    """Validate the abstract representation of an object.

    Args:
        obj_str: A JSON-formatted string encoding the object.
        name: The type of object to validate (can be "sequence" or "device").
    """
    obj = json.loads(obj_str)
    try:
        if LEGACY_JSONSCHEMA:  # pragma: no cover
            jsonschema.validate(
                instance=obj,
                schema=SCHEMAS[name],
                resolver=jsonschema.validators.RefResolver(  # type: ignore[attr-defined] # noqa: E501
                    base_uri=f"{SCHEMAS_PATH.resolve().as_uri()}/",
                    referrer=SCHEMAS[name],
                ),
            )
        else:
            _VALIDATORS[name].validate(obj)
    except Exception as exc:
        try:
            ser_pulser_version = Version(obj.get("pulser_version", "0.0.0"))
        except InvalidVersion:
            # In case the serialized version is invalid
            raise exc
        if Version(pulser.__version__) < ser_pulser_version:
            raise AbstractReprError(
                "The provided object is invalid under the current abstract "
                "representation schema. It appears it was serialized with a "
                f"more recent version of pulser ({ser_pulser_version!s}) than "
                f"the one currently being used ({pulser.__version__}). "
                "It is possible validation failed because new features have "
                "since been added; consider upgrading your pulser "
                "installation and retrying."
            ) from exc
        raise exc
