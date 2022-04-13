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
"""Valid devices for Pulser Sequence execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pulser.devices._device_datacls import Device
from pulser.devices._devices import Chadoq2, IroiseMVP
from pulser.devices._mock_device import MockDevice

# Registers which devices can be used to avoid definition of custom devices
_mock_devices: tuple[Device, ...] = (MockDevice,)
_valid_devices: tuple[Device, ...] = (Chadoq2, IroiseMVP)
