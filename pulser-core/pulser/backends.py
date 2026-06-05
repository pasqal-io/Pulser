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
    backends.QutipBackendV2  # Same as pulser_simulation.QutipBackendV2

Attributes:
    QPUBackend: See :py:class:`pulser.backend.QPUBackend`.
    QutipBackend: See :py:class:`pulser_simulation.QutipBackend`.
    QutipBackendV2: See :py:class:`pulser_simulation.QutipBackendV2`.
    RemoteFreeBackend: See :py:class:`pasqal_cloud.RemoteFreeBackend`.
    RemoteMPSBackend: See :py:class:`pasqal_cloud.RemoteMPSBackend`.
    RemoteSVBackend: See :py:class:`pasqal_cloud.RemoteSVBackend`.
    MPSBackend: See `emu_mps.MPSBackend <https://pypi.org/project/emu-mps/>`_.
    SVBackend: See `emu_sv.SVBackend <https://pypi.org/project/emu-sv/>`_.

"""

from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from pulser.backend import QPUBackend as QPUBackend
    from pulser.backend.abc import Backend
    from pulser_simulation import QutipBackendV2 as QutipBackendV2


_BACKENDS = {
    "QPUBackend": "pulser.backend",
    "QutipBackend": "pulser_simulation",
    "QutipBackendV2": "pulser_simulation",
    "RemoteFreeBackend": "pasqal_cloud",
    "RemoteMPSBackend": "pasqal_cloud",
    "RemoteSVBackend": "pasqal_cloud",
    "MPSBackend": "emu_mps",
    "SVBackend": "emu_sv",
}

_DEPRECATED_REMOVED_BACKENDS = ["EmuFreeBackend", "EmuTNBackend"]
_RENAMED_BACKENDS = {
    "EmuFreeBackendV2": "RemoteFreeBackend",
    "EmuMPSBackend": "RemoteMPSBackend",
    "EmuSVBackend": "RemoteSVBackend",
}


# This prevents * imports to attempt importing unavailable backends
__all__: list[str] = []


def __getattr__(name: str) -> Type[Backend]:
    if name in _DEPRECATED_REMOVED_BACKENDS:
        raise AttributeError(
            f"{name!r} was deprecated and is now removed "
            f"from module {__name__!r}"
        )

    if name not in _BACKENDS and name not in _RENAMED_BACKENDS:
        raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")
    try:
        if name in _RENAMED_BACKENDS:
            new_name = _RENAMED_BACKENDS[name]

            warnings.warn(
                f"{name!r} was renamed to {new_name!r}. "
                f"Please use {new_name!r} from now on.",
                DeprecationWarning,
                stacklevel=2,
            )

            name = new_name

        return getattr(  # type: ignore
            importlib.import_module(_BACKENDS[name]),
            name,
        )
    except ModuleNotFoundError:
        raise AttributeError(
            f"{name!r} requires the {_BACKENDS[name]!r} package. To install "
            f"it, run `pip install {_BACKENDS[name]}`."
        )
