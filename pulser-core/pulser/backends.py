# Copyright 2024 Pulser Development Team
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
"""A module gathering all available backends.

This module is a single-point access to backends spread across different
packages. As long as the appropriate package is installed, the ``Backend``
instances defined within it should be importable via this module, like so::

    import pulser.backends as backends

    backends.QPUBackend  # Same as pulser.QPUBackend
    backends.QutipBackend  # Same as pulser_simulation.QutipBackend

Attributes:
    QPUBackend: See :py:class:`pulser.backend.QPUBackend`.
    QutipBackend: See :py:class:`pulser_simulation.QutipBackend`.
    EmuFreeBackend: See :py:class:`pulser_pasqal.EmuFreeBackend`.
    EmuTNBackend: See :py:class:`pulser_pasqal.EmuTNBackend`.

"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from pulser.backend import QPUBackend as QPUBackend
    from pulser.backend.abc import Backend
    from pulser_simulation import QutipBackend as QutipBackend

_BACKENDS = {
    "QPUBackend": "pulser.backend",
    "QutipBackend": "pulser_simulation",
    "QutipBackendV2": "pulser_simulation",
    "EmuFreeBackend": "pulser_pasqal",
    "EmuTNBackend": "pulser_pasqal",
}


# This prevents * imports to attempt importing unavailable backends
__all__: list[str] = []


def __getattr__(name: str) -> Type[Backend]:
    if name not in _BACKENDS:
        raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
    try:
        return getattr(  # type: ignore
            importlib.import_module(_BACKENDS[name]),
            name,
        )
    except ModuleNotFoundError:
        raise AttributeError(
            f"{name!r} requires the {_BACKENDS[name]!r} package. To install "
            f"it, run `pip install {_BACKENDS[name]}`."
        )
