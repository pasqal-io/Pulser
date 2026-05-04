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
from typing import Any, Callable

import jsonschema
import jsonschema.validators
from packaging.version import InvalidVersion, Version
from referencing import Registry, Resource

import pulser
from pulser.exceptions.serialization import AbstractReprError
from pulser.json.abstract_repr import SCHEMAS, SCHEMAS_PATH
from pulser.json.utils import ObjectType, get_filename

try:
    import fastjsonschema  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    fastjsonschema = None


LEGACY_JSONSCHEMA = (
    Version("4.18") > Version(version("jsonschema")) >= Version("4.17.3")
)


_SCHEMAS_BY_FILENAME = {
    get_filename(name): schema for name, schema in SCHEMAS.items()
}

_FAST_VALIDATORS: dict[str, Callable[[Any], Any]] = (
    {
        name: fastjsonschema.compile(  # type: ignore[misc]
            schema, handlers={"": _SCHEMAS_BY_FILENAME.__getitem__}
        )
        for name, schema in SCHEMAS.items()
    }
    if fastjsonschema is not None
    else {}
)

REGISTRY: Registry = Registry(
    [
        (get_filename(name), Resource.from_contents(SCHEMAS[name]))
        for name in (
            "device",
            "layout",
            "register",
            "noise",
        )
    ]
)

_VALIDATORS: dict[str, jsonschema.Draft7Validator] = (
    {}
    if LEGACY_JSONSCHEMA
    else {
        name: jsonschema.Draft7Validator(schema, registry=REGISTRY)
        for name, schema in SCHEMAS.items()
    }
)


def _validate_with_jsonschema(obj: Any, object_type: ObjectType) -> None:
    if LEGACY_JSONSCHEMA:  # pragma: no cover
        jsonschema.validate(
            instance=obj,
            schema=SCHEMAS[object_type],
            resolver=jsonschema.validators.RefResolver(
                base_uri=f"{SCHEMAS_PATH.resolve().as_uri()}/",
                referrer=SCHEMAS[object_type],
            ),
        )
    else:
        _VALIDATORS[object_type].validate(obj)


def validate_abstract_repr(
    obj_str: str,
    name: ObjectType,
) -> None:
    """Validate the abstract representation of an object.

    Args:
        obj_str: A JSON-formatted string encoding the object.
        name: The type of object to validate.
    """
    obj = json.loads(obj_str)
    try:
        if fastjsonschema is not None:
            try:
                # validation with fast library
                _FAST_VALIDATORS[name](obj)
            except fastjsonschema.JsonSchemaException:
                # in case of validation failure, we run validation with jsonschema
                # to have descriptive error reasons
                _validate_with_jsonschema(obj, name)
        else:
            _validate_with_jsonschema(obj, name)
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
