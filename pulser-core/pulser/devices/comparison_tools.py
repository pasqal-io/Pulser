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
"""Defines functions for comparison of Channels."""
from __future__ import annotations

import itertools
from dataclasses import asdict
from operator import gt, lt, ne
from typing import TYPE_CHECKING, cast

from pulser.channels.comparison_tools import (
    _validate_obj_from_best,
    validate_channel_from_best,
)
from pulser.devices import Device, VirtualDevice

if TYPE_CHECKING:
    from pulser.devices._device_datacls import BaseDevice


def _exist_good_configuration(
    possible_configurations: dict[str, list[str]]
) -> bool:
    # If one value is an empty list then no configuration can work
    if any(
        [
            len(possible_values) == 0
            for possible_values in possible_configurations.values()
        ]
    ):
        return False
    print(list(possible_configurations.values()))
    print(itertools.product(*list(possible_configurations.values())))
    for config in itertools.product(*list(possible_configurations.values())):
        # True if each value in the list is different
        print(config)
        set(config)
        if len(set(config)) == len(config):
            return True
    return False


def validate_device_from_best(
    device: BaseDevice, best_device: BaseDevice
) -> bool:
    """Checks that a device can be realized from another one.

    Attributes:
        device: The device to check.
        best_device: The device that should have better properties.
    """
    if type(device) != type(best_device):
        raise ValueError(
            "Devices do not have the same types, "
            f"{type(device)} and {type(best_device)}"
        )
    equivalent_channels: dict[str, list[str]] = {
        ch_name: [] for ch_name in device.channels.keys()
    }
    for ch_name, channel in device.channels.items():
        for best_ch_name, best_channel in best_device.channels.items():
            try:
                validate_channel_from_best(channel, best_channel)
            except ValueError:
                continue
            equivalent_channels[ch_name].append(best_ch_name)
    if not _exist_good_configuration(equivalent_channels):
        raise ValueError(
            "No configuration could be found where each channel of the device"
            " could be realized with one channel of the best device."
        )

    if isinstance(device, Device) and device.calibrated_register_layouts:
        equivalent_layouts: dict[str, list[str]] = {
            str(id): []
            for id in range(len(device.calibrated_register_layouts))
        }
        for id, layout in enumerate(device.calibrated_register_layouts):
            for id_best, best_layout in enumerate(
                cast(Device, best_device).calibrated_register_layouts
            ):
                if best_layout > layout:
                    equivalent_layouts[str(id)].append(str(id_best))
        if not _exist_good_configuration(equivalent_layouts):
            raise ValueError(
                "No configuration could be found where each calibrated layouts"
                " of the device could be realized with a calibrated layout of"
                " the best device."
            )

    best_device_att = asdict(best_device)
    device_att = asdict(device)

    # Error if attributes in device and best_device compare to True
    comparison_ops = {
        "dimensions": gt,
        "rydberg_level": ne,
        "min_atom_distance": lt,
        "max_atom_num": gt,
        "max_radial_distance": gt,
        "interaction_coeff_xy": ne,
        "supports_slm_mask": gt,
        "max_layout_filling": gt,
    }
    if isinstance(device, VirtualDevice):
        comparison_ops["reusable_channels"] = gt

    return _validate_obj_from_best(device_att, best_device_att, comparison_ops)
