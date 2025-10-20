# Copyright 2025 Pulser Development Team
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
"""Functions for setting and getting the serialized Sequence metadata."""
from __future__ import annotations

import contextvars
from typing import Any

_package_versions: contextvars.ContextVar[dict[str, str]] = (
    contextvars.ContextVar("_package_versions", default={})
)

_extra: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "_extra", default={}
)


def _get_metadata() -> dict[str, dict[str, Any]]:
    """Gets all the existing Sequence metadata."""
    package_versions = _package_versions.get()
    extra = _extra.get()
    if package_versions or extra:
        return {
            "package_versions": package_versions,
            "extra": extra,
        }
    return {}


def _reset_metadata() -> None:
    """Deletes all exisiting metadata."""
    _package_versions.set({})
    _extra.set({})


def store_package_version_metadata(
    package_name: str, package_version: str
) -> None:
    """Store the given package name and version in the Sequence metadata."""
    _package_versions.set(
        _package_versions.get() | {package_name: package_version}
    )


def store_extra_metadata(extra_metadata: dict) -> None:
    """Store any extra metadata in the Sequence metadata."""
    _extra.set(_extra.get() | extra_metadata)
