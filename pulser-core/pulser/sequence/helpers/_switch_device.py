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
"""Function to switch the Device in a Sequence."""
from __future__ import annotations

import dataclasses
import itertools
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pulser.channels.base_channel import Channel
from pulser.channels.dmm import _get_dmm_name
from pulser.channels.eom import RydbergEOM
from pulser.devices._device_datacls import BaseDevice

if TYPE_CHECKING:
    from pulser.sequence.sequence import Sequence


def switch_device(
    seq: Sequence, new_device: BaseDevice, strict: bool = False
) -> Sequence:
    """Replicate the sequence with a different device.

    This method is designed to replicate the sequence with as few changes
    to the original contents as possible.
    If the `strict` option is chosen, the device switch will fail whenever
    it cannot guarantee that the new sequence's contents will not be
    modified in the process.

    Args:
        seq: The Sequence whose device should be switched.
        new_device: The target device instance.
        strict: Enforce a strict match between devices and channels to
            guarantee the pulse sequence is left unchanged.

    Returns:
        The sequence on the new device, using the match channels of
        the former device declared in the sequence.
    """
    # Check if the device is new or not

    if seq.device == new_device:
        warnings.warn(
            "Switching a sequence to the same device"
            + " returns the sequence unchanged.",
            stacklevel=2,
        )
        return seq

    if seq._in_xy:
        interaction_param = "interaction_coeff_xy"
        name_in_msg = "XY interaction coefficient"
    else:
        interaction_param = "rydberg_level"
        name_in_msg = "Rydberg level"

    if getattr(new_device, interaction_param) != getattr(
        seq._device, interaction_param
    ):
        if strict:
            raise ValueError(
                "Strict device match failed because the"
                f" devices have different {name_in_msg}s."
            )
        warnings.warn(
            f"Switching to a device with a different {name_in_msg},"
            " check that the expected interactions still hold.",
            stacklevel=2,
        )

    def check_retarget(ch_obj: Channel) -> bool:
        # Check the min_retarget_interval when it is is not
        # fully covered by the fixed_retarget_t
        return ch_obj.addressing == "Local" and cast(
            int, ch_obj.fixed_retarget_t
        ) < cast(int, ch_obj.min_retarget_interval)

    def check_channels_match(
        old_ch_name: str,
        new_ch_obj: Channel,
        active_eom_channels: list,
        strict: bool,
    ) -> tuple[str, str]:
        """Check whether two channels match.

        Returns a tuple that contains a non-strict error message and a
        strict error message. If the channel matches, the two error
        messages are empty strings. If strict=False, only non-strict
        conditions are checked, and only the non-strict error message
        will eventually be filled. If strict=True, all the conditions are
        checked - the returned error can either be non-strict or strict.
        """
        old_ch_obj = seq.declared_channels[old_ch_name]
        # We verify the channel class then
        # check whether the addressing is Global or Local
        type_match = type(old_ch_obj) is type(new_ch_obj)
        basis_match = old_ch_obj.basis == new_ch_obj.basis
        addressing_match = old_ch_obj.addressing == new_ch_obj.addressing
        if not (type_match and basis_match and addressing_match):
            # If there already is a message, keeps it
            return (" with the right type, basis and addressing.", "")
        if old_ch_name in active_eom_channels:
            # Uses EOM mode, so the new device needs a matching
            # EOM configuration
            if new_ch_obj.eom_config is None:
                return (" with an EOM configuration.", "")
            if strict:
                if not seq.is_parametrized():
                    if (
                        new_ch_obj.eom_config.mod_bandwidth
                        != cast(
                            RydbergEOM, old_ch_obj.eom_config
                        ).mod_bandwidth
                    ):
                        return (
                            "",
                            " with the same mod_bandwidth for the EOM.",
                        )
                else:
                    # Eom configs have to match is Sequence is parametrized
                    new_eom_config = dataclasses.asdict(
                        cast(RydbergEOM, new_ch_obj.eom_config)
                    )
                    old_eom_config = dataclasses.asdict(
                        cast(RydbergEOM, old_ch_obj.eom_config)
                    )
                    # However, multiple_beam_control only matters when
                    # the two beams are controlled
                    if len(old_eom_config["controlled_beams"]) == 1:
                        new_eom_config.pop("multiple_beam_control")
                        old_eom_config.pop("multiple_beam_control")
                        # Controlled beams only matter when only one beam
                        # is controlled by the new eom
                        if len(new_eom_config["controlled_beams"]) > 1:
                            new_eom_config.pop("controlled_beams")
                            old_eom_config.pop("controlled_beams")
                    # Controlled_beams doesn't matter if the two EOMs
                    # control two beams
                    elif set(new_eom_config["controlled_beams"]) == set(
                        old_eom_config["controlled_beams"]
                    ):
                        new_eom_config.pop("controlled_beams")
                        old_eom_config.pop("controlled_beams")

                    # And custom_buffer_time doesn't have to match as long
                    # as `Channel_eom_buffer_time`` does
                    if (
                        new_ch_obj._eom_buffer_time
                        == old_ch_obj._eom_buffer_time
                    ):
                        new_eom_config.pop("custom_buffer_time")
                        old_eom_config.pop("custom_buffer_time")
                    if new_eom_config != old_eom_config:
                        return ("", " with the same EOM configuration.")
        if not strict:
            return ("", "")

        params_to_check = [
            "mod_bandwidth",
            "fixed_retarget_t",
            "clock_period",
        ]
        if check_retarget(old_ch_obj) or check_retarget(new_ch_obj):
            params_to_check.append("min_retarget_interval")
        for param_ in params_to_check:
            if getattr(new_ch_obj, param_) != getattr(old_ch_obj, param_):
                return ("", f" with the same {param_}.")
        else:
            return ("", "")

    def is_good_match(
        channel_match: dict[str, str],
        reusable_channels: bool,
        all_channels_new_device: dict[str, Channel],
        active_eom_channels: list,
        strict: bool,
    ) -> bool:
        used_channels_new_device = list(channel_match.values())
        if not reusable_channels and len(set(used_channels_new_device)) < len(
            used_channels_new_device
        ):
            return False
        for old_ch_name, new_ch_name in channel_match.items():
            if check_channels_match(
                old_ch_name,
                all_channels_new_device[new_ch_name],
                active_eom_channels,
                strict,
            ) != ("", ""):
                return False
        return True

    def raise_error_non_matching_channel(
        reusable_channels: bool,
        all_channels_new_device: dict[str, Channel],
        active_eom_channels: list,
        strict: bool,
    ) -> None:
        strict_error_message = ""
        ch_match_err = ""
        channel_match: dict[str, Any] = {}
        for old_ch_name, old_ch_obj in seq.declared_channels.items():
            channel_match[old_ch_name] = None
            base_msg = f"No match for channel {old_ch_name}"
            # Find the corresponding channel on the new device
            for new_ch_id, new_ch_obj in all_channels_new_device.items():
                if (
                    not reusable_channels
                    and new_ch_id in channel_match.values()
                ):
                    # Channel already matched and can't be reused
                    continue
                (ch_match_err_suffix, strict_error_message_suffix) = (
                    check_channels_match(
                        old_ch_name,
                        new_ch_obj,
                        active_eom_channels,
                        strict,
                    )
                )
                if (ch_match_err_suffix, strict_error_message_suffix) == (
                    "",
                    "",
                ):
                    channel_match[old_ch_name] = new_ch_id
                    # Found a match, clear match error msg for this channel
                    if ch_match_err.startswith(base_msg):
                        ch_match_err = ""
                    if strict_error_message.startswith(base_msg):
                        strict_error_message = ""
                    break
                elif ch_match_err_suffix != "":
                    ch_match_err = (
                        ch_match_err or base_msg + ch_match_err_suffix
                    )
                else:
                    strict_error_message = (
                        base_msg + strict_error_message_suffix
                    )
        assert None in channel_match.values()
        if strict_error_message:
            raise ValueError(strict_error_message)
        raise TypeError(ch_match_err)

    def build_sequence_from_matching(
        new_device: BaseDevice,
        channel_match: dict[str, str],
        active_eom_channels: list,
        strict: bool,
    ) -> Sequence:
        # Initialize the new sequence (works for Sequence subclasses too)
        new_seq = type(seq)(register=seq._register, device=new_device)
        dmm_calls: list[str] = []
        # Copy the variables to the new sequence
        new_seq._variables = seq.declared_variables
        for call in seq._calls[1:] + seq._to_build_calls:
            # Switch the old id with the correct id
            sw_channel_args = list(call.args)
            sw_channel_kw_args = call.kwargs.copy()
            if not (
                call.name == "declare_channel"
                or call.name == "config_detuning_map"
                or call.name == "config_slm_mask"
                or call.name == "add_dmm_detuning"
            ):
                pass
            # if calling declare_channel
            elif "name" in sw_channel_kw_args:  # pragma: no cover
                sw_channel_kw_args["channel_id"] = channel_match[
                    sw_channel_kw_args["name"]
                ]
            elif "channel_id" in sw_channel_kw_args:  # pragma: no cover
                sw_channel_kw_args["channel_id"] = channel_match[
                    sw_channel_args[0]
                ]
            elif call.name == "declare_channel":
                sw_channel_args[1] = channel_match[sw_channel_args[0]]
            # if adding a detuning waveform to the dmm
            elif "dmm_name" in sw_channel_kw_args:  # program: no cover
                sw_channel_kw_args["dmm_name"] = channel_match[
                    sw_channel_kw_args["dmm_name"]
                ]
            elif call.name == "add_dmm_detuning":
                sw_channel_args[1] = channel_match[sw_channel_args[1]]
            # if configuring a detuning map or an SLM mask
            else:
                assert (
                    call.name == "config_detuning_map"
                    or call.name == "config_slm_mask"
                )
                if "dmm_id" in sw_channel_kw_args:  # pragma: no cover
                    dmm_called = _get_dmm_name(
                        sw_channel_kw_args["dmm_id"], dmm_calls
                    )
                    sw_channel_kw_args["dmm_id"] = channel_match[dmm_called]
                else:
                    dmm_called = _get_dmm_name(sw_channel_args[1], dmm_calls)
                    sw_channel_args[1] = channel_match[dmm_called]
                dmm_calls.append(dmm_called)
                channel_match[dmm_called] = _get_dmm_name(
                    channel_match[dmm_called],
                    list(new_seq.declared_channels.keys()),
                )
            getattr(new_seq, call.name)(*sw_channel_args, **sw_channel_kw_args)

        if strict:
            for eom_channel in active_eom_channels:
                current_samples = seq._schedule[eom_channel].get_samples()
                new_samples = new_seq._schedule[eom_channel].get_samples()
                if (
                    not np.all(
                        np.isclose(current_samples.amp, new_samples.amp)
                    )
                    or not np.all(
                        np.isclose(current_samples.det, new_samples.det)
                    )
                    or not np.all(
                        np.isclose(current_samples.phase, new_samples.phase)
                    )
                ):
                    raise ValueError(
                        f"No match for channel {eom_channel} with an"
                        " EOM configuration that does not change the"
                        " samples."
                    )
        return new_seq

    # Channel match
    active_eom_channels = [
        {**dict(zip(("channel",), call.args)), **call.kwargs}["channel"]
        for call in seq._calls + seq._to_build_calls
        if call.name == "enable_eom_mode"
    ]
    all_channels_new_device = {
        **new_device.channels,
        **new_device.dmm_channels,
    }
    possible_channel_match: list[dict[str, str]] = []
    for channels_comb in itertools.product(
        all_channels_new_device, repeat=len(seq.declared_channels)
    ):
        channel_match = dict(zip(seq.declared_channels, channels_comb))
        if is_good_match(
            channel_match,
            new_device.reusable_channels,
            all_channels_new_device,
            active_eom_channels,
            strict,
        ):
            possible_channel_match.append(channel_match)
    if not possible_channel_match:
        raise_error_non_matching_channel(
            new_device.reusable_channels,
            all_channels_new_device,
            active_eom_channels,
            strict,
        )
    err_channel_match = {}
    for channel_match in possible_channel_match:
        try:
            return build_sequence_from_matching(
                new_device, channel_match, active_eom_channels, strict
            )
        except ValueError as e:
            err_channel_match[tuple(channel_match.items())] = e.args
            continue
    raise ValueError(
        "No matching found between declared channels and channels in the "
        "new device that does not modify the samples of the Sequence. "
        "Here is a list of matchings tested and their associated errors: "
        f"{err_channel_match}"
    )
