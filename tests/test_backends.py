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

import sys

import pytest

import pulser
from pulser.backend.abc import Backend
from pulser.backends import _BACKENDS


@pytest.mark.parametrize("backend, missing_package", list(_BACKENDS.items()))
def test_missing_package(monkeypatch, backend, missing_package):
    monkeypatch.setitem(sys.modules, missing_package, None)
    with pytest.raises(
        AttributeError,
        match=f"{backend!r} requires the {missing_package!r} package. "
        f"To install it, run `pip install {missing_package}`",
    ):
        getattr(pulser.backends, backend)


def test_missing_backend():
    with pytest.raises(
        AttributeError,
        match="Module 'pulser.backends' has no attribute 'SpecialBackend'",
    ):
        pulser.backends.SpecialBackend


@pytest.mark.parametrize("backend_name", list(_BACKENDS))
def test_succesful_imports(backend_name):
    backend = getattr(pulser.backends, backend_name)
    assert issubclass(backend, Backend)
