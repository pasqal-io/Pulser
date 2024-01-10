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
from referencing import Registry, Resource

from pulser.json.abstract_repr import SCHEMAS, SCHEMAS_PATH

LEGACY_JSONSCHEMA = "4.18" > version("jsonschema") >= "4.17.3"

REGISTRY: Registry = Registry(
    [
        ("device-schema.json", Resource.from_contents(SCHEMAS["device"])),
        ("layout-schema.json", Resource.from_contents(SCHEMAS["layout"])),
        ("register-schema.json", Resource.from_contents(SCHEMAS["register"])),
    ]
)


def validate_abstract_repr(
    obj_str: str, name: Literal["sequence", "device", "layout", "register"]
) -> None:
    """Validate the abstract representation of an object.

    Args:
        obj_str: A JSON-formatted string encoding the object.
        name: The type of object to validate (can be "sequence" or "device").
    """
    obj = json.loads(obj_str)
    validate_args = dict(instance=obj, schema=SCHEMAS[name])
    if LEGACY_JSONSCHEMA:  # pragma: no cover
        validate_args["resolver"] = jsonschema.validators.RefResolver(
            base_uri=f"{SCHEMAS_PATH.resolve().as_uri()}/",
            referrer=SCHEMAS[name],
        )
    else:  # pragma: no cover
        validate_args["registry"] = REGISTRY
    jsonschema.validate(**validate_args)
