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

import numbers
from collections.abc import Callable
from dataclasses import asdict
from operator import gt, lt, ne
from typing import TYPE_CHECKING, Any

import numpy as np

from pulser.channels.eom import RydbergEOM

if TYPE_CHECKING:
    from pulser.channels.base_channel import Channel
    from pulser.channels.eom import BaseEOM


def _compare_with_None(
    comparison_ops: dict[str, Callable],
    leftvalues: dict[str, float | None],
    rightvalues: dict[str, float | None],
) -> dict[str, Any]:
    """Compare dict of values using a dict of comparison operators.

    If comparison operator returns a boolean, the value returned is:
        - True if right value is None and left value is defined.
        - False if left value is None and right value is defined.
        - None if left and right values are None.
    Implemented for lt, gt, min, max.

    Args:
        comparison_ops: Associate keys to compare with comparison operator
        leftvalues: Dict of values on the left of the comparison operator
        rightvalues: Dict of values on the right of the comparison operator

    Returns:
        Dictionary having the keys of the comparison operator, associating
        None if both values are None for this key, and the result of the
        comparison otherwise.
    """
    # Error if some keys of comparison operators are not dict of left or right
    if not (
        comparison_ops.keys() <= leftvalues.keys()
        and comparison_ops.keys() <= rightvalues.keys()
    ):
        raise ValueError(
            "Keys in comparison_ops should be in left values and right values."
        )
    # Compare using +inf and -inf to replace None values
    return {
        key: comparison_op(
            *(
                value
                or (
                    float("+inf")
                    if comparison_op in [lt, min]
                    else float("-inf")
                )
                for value in (leftvalues[key], rightvalues[key])
            )
        )
        if (leftvalues[key], rightvalues[key]) != (None, None)
        else None
        for (key, comparison_op) in comparison_ops.items()
    }


def _validate_obj_from_best(
    obj_dict: dict, best_obj_dict: dict, comparison_ops: dict
) -> bool:
    """Validates an object by comparing it with a better one.

    Attributes:
        obj_dict: Dict of attributes and values of the object to compare.
        best_obj_dict: Dict of attributes and values of the best object.
        comparison_ops: Dict of attributes and comparison operators to
            use to compare the object and the best object.

    Returns:
        True if the comparison works, raises a ValueError otherwise.
    """
    # If the two values are almost equal then there is no need to compare them
    comparison_ops_keys = list(comparison_ops.keys())
    for key in comparison_ops_keys:
        if (
            obj_dict[key] is not None
            and best_obj_dict[key] is not None
            and isinstance(obj_dict[key], numbers.Number)
            and isinstance(best_obj_dict[key], numbers.Number)
            and np.isclose(obj_dict[key], best_obj_dict[key], rtol=1e-14)
        ):
            comparison_ops.pop(key)
    is_wrong_effective_ch = _compare_with_None(
        comparison_ops, obj_dict, best_obj_dict
    )
    # Validates if no True in the dictionary of the comparisons
    if not (True in is_wrong_effective_ch.values()):
        return True
    is_wrong_effective_index = list(is_wrong_effective_ch.values()).index(True)
    is_wrong_key = list(is_wrong_effective_ch.keys())[is_wrong_effective_index]
    raise ValueError(
        f"{is_wrong_key} cannot be"
        + (
            " below "
            if comparison_ops[is_wrong_key] == lt
            else (
                " above "
                if comparison_ops[is_wrong_key] == gt
                else " different than "
            )
        )
        + f"{best_obj_dict[is_wrong_key]}."
    )


def validate_channel_from_best(
    channel: Channel, best_channel: Channel
) -> bool:
    """Checks that a channel can be realized from another one.

    Attributes:
        channel: The channel to check.
        best_channel: The channel that should have better properties.
    """
    if type(channel) != type(best_channel):
        raise ValueError(
            "Channels do not have the same types, "
            f"{type(channel)} and {type(best_channel)}"
        )
    if channel.eom_config:
        if best_channel.eom_config:
            validate_eom_from_best(channel.eom_config, best_channel.eom_config)
        else:
            raise ValueError(
                "eom_config cannot be defined in channel as the best_channel"
                " does not have one."
            )
    best_ch_att = asdict(best_channel)
    ch_att = asdict(channel)

    # Error if attributes in channel and best_channel compare to True
    comparison_ops = {
        "addressing": ne,
        "max_abs_detuning": gt,
        "max_amp": gt,
        "min_retarget_interval": lt,
        "fixed_retarget_t": lt,
        "max_targets": gt,
        "clock_period": lt,
        "min_duration": lt,
        "max_duration": gt,
        "mod_bandwidth": gt,
    }
    return _validate_obj_from_best(ch_att, best_ch_att, comparison_ops)


def validate_eom_from_best(eom: BaseEOM, best_eom: BaseEOM) -> bool:
    """Checks that an EOM config can be realized from another one.

    Attributes:
        eom: The EOM config to check.
        best_eom: The EOM config that should have better properties.
    """
    best_eom_att = asdict(best_eom)
    eom_att = asdict(eom)

    # Error if attributes in eom and best_eom compare to True
    comparison_ops = {"mod_bandwidth": gt}
    if isinstance(eom, RydbergEOM):
        if isinstance(best_eom, RydbergEOM):
            comparison_ops.update(
                {
                    "limiting_beam": ne,
                    "max_limiting_amp": gt,
                    "intermediate_detuning": ne,
                    "controlled_beams": gt,
                }
            )
            best_eom_att["controlled_beams"] = set(
                best_eom_att["controlled_beams"]
            )
            eom_att["controlled_beams"] = set(eom_att["controlled_beams"])
        else:
            raise ValueError(
                "EOM config is RydbergEOM whereas best EOM config is not."
            )
    return _validate_obj_from_best(eom_att, best_eom_att, comparison_ops)
