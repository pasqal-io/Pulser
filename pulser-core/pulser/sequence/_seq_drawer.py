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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import CubicSpline

import pulser
from pulser import Register, Register3D
from pulser.channels.base_channel import Channel
from pulser.pulse import Pulse
from pulser.sampler.samples import ChannelSamples
from pulser.waveforms import InterpolatedWaveform

# Color scheme
COLORS = ["darkgreen", "indigo", "#c75000"]

CURVES_ORDER = ("amplitude", "detuning", "phase")

SIZE_PER_WIDTH = {1: 3, 2: 4, 3: 5}
LABELS = [
    r"$\Omega$ (rad/µs)",
    r"$\delta$ (rad/µs)",
    r"$\varphi$ / 2π",
]


class EOMSegment:
    """The class to mark an EOM slot."""

    def __init__(self, ti: int | None = None, tf: int | None = None) -> None:
        """Class is defined from its start and end value."""
        self.ti = ti
        self.tf = tf
        self.color = "steelblue"
        self.alpha = 0.3

    @property
    def isempty(self) -> bool:
        """Defines if the class is empty."""
        return self.ti is None or self.tf is None

    @property
    def nvspan(self) -> int:
        """Defines the number of points in the slot."""
        return cast(int, self.tf) - cast(int, self.ti)

    def draw(self, ax: Axes) -> None:
        """Draws a rectangle between the start and end value."""
        if not self.isempty:
            ax.axvspan(
                self.ti,
                self.tf,
                color=self.color,
                alpha=self.alpha,
                zorder=-100,
            )

    def smooth_draw(self, ax: Axes, decreasing: bool = False) -> None:
        """Draws a rectangle with an increasing/decreasing opacity."""
        if not self.isempty:
            for i in range(self.nvspan):
                ax.axvspan(
                    cast(int, self.ti) + i,
                    cast(int, self.ti) + i + 1,
                    facecolor=self.color,
                    alpha=self.alpha
                    * (
                        decreasing + (-1) ** decreasing * (i + 1) / self.nvspan
                    ),
                    zorder=-100,
                )
            ax.axvline(
                self.tf if decreasing else self.ti,
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                color=self.color,
                alpha=self.alpha / 2.0,
            )


@dataclass
class ChannelDrawContent:
    """The contents for drawing a single channel."""

    samples: ChannelSamples
    target: dict[Union[str, tuple[int, int]], Any]
    eom_intervals: list[EOMSegment]
    eom_start_buffers: list[EOMSegment]
    eom_end_buffers: list[EOMSegment]
    interp_pts: dict[str, list[list[float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples.phase = self.samples.phase / (2 * np.pi)
        self._samples_from_curves = {
            "amplitude": "amp",
            "detuning": "det",
            "phase": "phase",
        }
        self.curves_on = {"amplitude": True, "detuning": False, "phase": False}

    @property
    def n_axes_on(self) -> int:
        """The number of axes to draw for this channel."""
        return sum(self.curves_on.values())

    def get_input_curves(self) -> list[np.ndarray]:
        """The samples for the curves, as programmed."""
        return self._give_curves_from_samples(self.samples)

    def get_output_curves(self, ch_obj: Channel) -> list[np.ndarray]:
        """The modulated samples for the curves."""
        mod_samples = self.samples.modulate(ch_obj)
        return self._give_curves_from_samples(mod_samples)

    def interpolate_curves(
        self, curves: list[np.ndarray], sampling_rate: float
    ) -> list[np.ndarray]:
        """The curves with a fractional sampling rate."""
        indices = np.linspace(
            0,
            self.samples.duration,
            num=int(sampling_rate * self.samples.duration),
            endpoint=False,
            dtype=int,
        )
        sampled_curves = [curve[indices] for curve in curves]
        t = np.arange(self.samples.duration)
        return [CubicSpline(indices, sc)(t) for sc in sampled_curves]

    def curves_on_indices(self) -> list[int]:
        """The indices of the curves to draw."""
        return [i for i, qty in enumerate(CURVES_ORDER) if self.curves_on[qty]]

    def _give_curves_from_samples(
        self, samples: ChannelSamples
    ) -> list[np.ndarray]:
        return [
            getattr(samples, self._samples_from_curves[qty])
            for qty in CURVES_ORDER
        ]


def gather_data(seq: pulser.sequence.Sequence, gather_output: bool) -> dict:
    """Collects the whole sequence data for plotting.

    Args:
        seq: The input sequence of operations on a device.
        gather_output: Whether to gather the modulated output curves.

    Returns:
        The data to plot.
    """
    # The minimum time axis length is 100 ns
    total_duration = max(
        seq.get_duration(include_fall_time=gather_output), 100
    )
    data: dict[str, Any] = {}
    for ch, sch in seq._schedule.items():
        # List of interpolation points
        interp_pts: defaultdict[str, list[list[float]]] = defaultdict(list)
        target: dict[Union[str, tuple[int, int]], Any] = {}
        # Extracting the EOM Buffers
        eom_intervals = [
            EOMSegment(eom_interval[0], eom_interval[1])
            for eom_interval in sch.get_eom_mode_intervals()
        ]
        nb_eom_intervals = len(eom_intervals)
        eom_start_buffers = [EOMSegment() for _ in range(nb_eom_intervals)]
        eom_end_buffers = [EOMSegment() for _ in range(nb_eom_intervals)]
        in_eom_mode = False
        eom_block_n = -1
        # Last eom interval is extended if eom mode not disabled at the end
        if nb_eom_intervals > 0 and seq.get_duration() == eom_intervals[-1].tf:
            eom_intervals[-1].tf = total_duration
        # sampling the channel schedule
        samples = sch.get_samples()
        extended_samples = samples.extend_duration(total_duration)
        for slot in sch:
            if slot.ti == -1:
                target["initial"] = slot.targets
                continue
            else:
                # If slot is not the first element in schedule
                if sch.in_eom_mode(slot):
                    # EOM mode starts
                    if not in_eom_mode:
                        in_eom_mode = True
                        eom_block_n += 1
                elif in_eom_mode:
                    # Buffer when EOM mode is disabled and next slot has 0 amp
                    in_eom_mode = False
                    if extended_samples.amp[slot.ti] == 0:
                        eom_end_buffers[eom_block_n] = EOMSegment(
                            slot.ti, slot.tf
                        )
                if (
                    eom_block_n + 1 < nb_eom_intervals
                    and slot.tf == eom_intervals[eom_block_n + 1].ti
                    and extended_samples.det[slot.tf - 1]
                    == sch.eom_blocks[eom_block_n + 1].detuning_off
                ):
                    # Buffer if next is eom and final det matches det_off
                    eom_start_buffers[eom_block_n + 1] = EOMSegment(
                        slot.ti, slot.tf
                    )

            if slot.type == "target":
                target[(slot.ti, slot.tf - 1)] = slot.targets
                continue
            if slot.type == "delay":
                continue
            pulse = cast(Pulse, slot.type)
            for wf_type in ["amplitude", "detuning"]:
                wf = getattr(pulse, wf_type)
                if isinstance(wf, InterpolatedWaveform):
                    pts = wf.data_points
                    pts[:, 0] += slot.ti
                    interp_pts[wf_type] += pts.tolist()

        # Store everything
        data[ch] = ChannelDrawContent(
            extended_samples,
            target,
            eom_intervals,
            eom_start_buffers,
            eom_end_buffers,
        )
        if interp_pts:
            data[ch].interp_pts = dict(interp_pts)
    if hasattr(seq, "_measurement"):
        data["measurement"] = seq._measurement
    data["total_duration"] = total_duration
    return data


def draw_sequence(
    seq: pulser.sequence.Sequence,
    sampling_rate: Optional[float] = None,
    draw_phase_area: bool = False,
    draw_interp_pts: bool = True,
    draw_phase_shifts: bool = False,
    draw_register: bool = False,
    draw_input: bool = True,
    draw_modulation: bool = False,
    draw_phase_curve: bool = False,
) -> tuple[Figure | None, Figure]:
    """Draws the entire sequence.

    Args:
        seq: The input sequence of operations on a device.
        sampling_rate: Sampling rate of the effective pulse used by
            the solver. If present, plots the effective pulse alongside the
            input pulse.
        draw_phase_area: Whether phase and area values need to be shown
            as text on the plot, defaults to False. If `draw_phase_curve=True`,
            phase values are ommited.
        draw_interp_pts: When the sequence has pulses with waveforms of
            type InterpolatedWaveform, draws the points of interpolation on
            top of the respective waveforms (defaults to True).
        draw_phase_shifts: Whether phase shift and reference information
            should be added to the plot, defaults to False.
        draw_register: Whether to draw the register before the pulse
            sequence, with a visual indication (square halo) around the qubits
            masked by the SLM, defaults to False.
        draw_input: Draws the programmed pulses on the channels, defaults
            to True.
        draw_modulation: Draws the expected channel output, defaults to
            False. If the channel does not have a defined 'mod_bandwidth', this
            is skipped unless 'draw_input=False'.
        draw_phase_curve: Draws the changes in phase in its own curve (ignored
            if the phase doesn't change throughout the channel).
    """

    def phase_str(phi: float) -> str:
        """Formats a phase value for printing."""
        value = (((phi + np.pi) % (2 * np.pi)) - np.pi) / np.pi
        if value == -1:
            return r"$\pi$"
        elif value == 0:
            return "0"  # pragma: no cover - just for safety
        else:
            return rf"{value:.2g}$\pi$"

    n_channels = len(seq.declared_channels)
    if not n_channels:
        raise RuntimeError("Can't draw an empty sequence.")
    data = gather_data(seq, gather_output=draw_modulation)
    total_duration = data["total_duration"]
    time_scale = 1e3 if total_duration > 1e4 else 1
    for ch in seq._schedule:
        if np.count_nonzero(data[ch].samples.det) > 0:
            data[ch].curves_on["detuning"] = True
        if draw_phase_curve and np.count_nonzero(data[ch].samples.phase) > 0:
            data[ch].curves_on["phase"] = True

    # Boxes for qubit and phase text
    q_box = dict(boxstyle="round", facecolor="orange")
    ph_box = dict(boxstyle="round", facecolor="ghostwhite")
    area_ph_box = dict(boxstyle="round", facecolor="ghostwhite", alpha=0.7)
    slm_box = dict(boxstyle="round", alpha=0.4, facecolor="grey", hatch="//")
    eom_box = dict(boxstyle="round", facecolor="lightsteelblue")

    # Draw masked register
    if draw_register:
        pos = np.array(seq.register._coords)
        if isinstance(seq.register, Register3D):
            labels = "xyz"
            fig_reg, axes_reg = seq.register._initialize_fig_axes_projection(
                pos,
                blockade_radius=35,
                draw_half_radius=True,
            )
            fig_reg.tight_layout(w_pad=6.5)

            for ax_reg, (ix, iy) in zip(
                axes_reg, combinations(np.arange(3), 2)
            ):
                seq.register._draw_2D(
                    ax=ax_reg,
                    pos=pos,
                    ids=seq.register._ids,
                    plane=(ix, iy),
                    masked_qubits=seq._slm_mask_targets,
                )
                ax_reg.set_title(
                    "Masked register projected onto\n the "
                    + labels[ix]
                    + labels[iy]
                    + "-plane"
                )

        elif isinstance(seq.register, Register):
            fig_reg, ax_reg = seq.register._initialize_fig_axes(
                pos,
                blockade_radius=35,
                draw_half_radius=True,
            )
            seq.register._draw_2D(
                ax=ax_reg,
                pos=pos,
                ids=seq.register._ids,
                masked_qubits=seq._slm_mask_targets,
            )
            ax_reg.set_title("Masked register", pad=10)

    ratios = [
        SIZE_PER_WIDTH[data[ch].n_axes_on] for ch in seq.declared_channels
    ]
    fig = plt.figure(
        constrained_layout=False,
        figsize=(20, sum(ratios)),
    )
    gs = fig.add_gridspec(n_channels, 1, hspace=0.075, height_ratios=ratios)

    ch_axes = {}
    for i, (ch, gs_) in enumerate(zip(seq.declared_channels, gs)):
        ax = fig.add_subplot(gs_)
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_color("none")
        ax.tick_params(
            labelcolor="w", top=False, bottom=False, left=False, right=False
        )
        ax.set_ylabel(ch, labelpad=40, fontsize=18)
        subgs = gs_.subgridspec(data[ch].n_axes_on, 1, hspace=0.0)
        ch_axes[ch] = [
            fig.add_subplot(subgs[i, :]) for i in range(data[ch].n_axes_on)
        ]
        for j, ax in enumerate(ch_axes[ch]):
            ax.axvline(0, linestyle="--", linewidth=0.5, color="grey")
            if j > 0:
                ax.spines["top"].set_visible(False)
            if j < len(ch_axes[ch]) - 1:
                ax.spines["bottom"].set_visible(False)
            if i < n_channels - 1 or j < len(ch_axes[ch]) - 1:
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=True,
                    top=False,
                    labelbottom=False,
                    direction="in",
                )
            else:
                unit = "ns" if time_scale == 1 else r"$\mu s$"
                ax.set_xlabel(f"t ({unit})", fontsize=12)

    # The time axis of all channels is the same
    t = np.arange(total_duration) / time_scale
    final_t = t[-1]
    t_min = -final_t * 0.03
    t_max = final_t * 1.05

    for ch, axes in ch_axes.items():
        ch_obj = seq.declared_channels[ch]
        ch_data = data[ch]
        ch_eom_intervals = data[ch].eom_intervals
        ch_eom_start_buffers = data[ch].eom_start_buffers
        ch_eom_end_buffers = data[ch].eom_end_buffers
        basis = ch_obj.basis
        ys = ch_data.get_input_curves()
        ys_mod = [()] * 3
        yseff = [()] * 3
        draw_output = draw_modulation and (
            ch_obj.mod_bandwidth or not draw_input
        )
        if draw_output:
            ys_mod = ch_data.get_output_curves(ch_obj)

        if sampling_rate:
            curves = ys_mod if draw_output else ys
            yseff = ch_data.interpolate_curves(curves, sampling_rate)
        ref_ys = [
            list(chain.from_iterable(all_ys))
            for all_ys in zip(ys, ys_mod, yseff)
        ]
        max_amp = np.max(ref_ys[0])
        max_amp = 1 if max_amp == 0 else max_amp
        amp_top = max_amp * 1.2
        amp_bottom = min(0.0, *ref_ys[0])
        # Makes sure that [-1, 1] range is always represented
        det_max = max(*ref_ys[1], 1)
        det_min = min(*ref_ys[1], -1)
        det_range = det_max - det_min
        det_top = det_max + det_range * 0.15
        det_bottom = det_min - det_range * 0.05
        ax_lims = [
            (amp_bottom, amp_top),
            (det_bottom, det_top),
            (min(0.0, *ref_ys[2]), max(1.1, *ref_ys[2])),
        ]
        ax_lims = [ax_lims[i] for i in ch_data.curves_on_indices()]
        for ax, ylim in zip(axes, ax_lims):
            ax.set_xlim(t_min, t_max)
            ax.set_ylim(*ylim)

        for i, ax in zip(ch_data.curves_on_indices(), axes):
            if draw_input:
                ax.plot(t, ys[i], color=COLORS[i], linewidth=0.8)
            if sampling_rate:
                ax.plot(
                    t,
                    yseff[i],
                    color=COLORS[i],
                    linewidth=0.8,
                )
                ax.fill_between(t, 0, yseff[i], color=COLORS[i], alpha=0.3)
            elif draw_input:
                ax.fill_between(t, 0, ys[i], color=COLORS[i], alpha=0.3)
            if draw_output:
                if not sampling_rate:
                    ax.fill_between(
                        t,
                        0,
                        ys_mod[i][:total_duration],
                        color=COLORS[i],
                        alpha=0.3,
                        hatch="////",
                    )
                else:
                    ax.plot(
                        t,
                        ys_mod[i][:total_duration],
                        color=COLORS[i],
                        linestyle="dotted",
                    )
            special_kwargs = dict(labelpad=10) if i == 0 else {}
            ax.set_ylabel(LABELS[i], fontsize=14, **special_kwargs)

        if draw_phase_area:
            top = False  # Variable to track position of box, top or center.
            print_phase = not draw_phase_curve and any(
                seq_.type.phase != 0
                for seq_ in seq._schedule[ch]
                if isinstance(seq_.type, Pulse)
            )
            for pulse_num, seq_ in enumerate(seq._schedule[ch]):
                # Select only `Pulse` objects
                if isinstance(seq_.type, Pulse):
                    if sampling_rate:
                        area_val = (
                            np.sum(yseff[0][seq_.ti : seq_.tf]) * 1e-3 / np.pi
                        )
                    else:
                        area_val = seq_.type.amplitude.integral / np.pi
                    phase_val = seq_.type.phase
                    x_plot = (seq_.ti + seq_.tf) / 2 / time_scale
                    if (
                        seq._schedule[ch][pulse_num - 1].type == "target"
                        or not top
                    ):
                        y_plot = np.max(seq_.type.amplitude.samples) / 2
                        top = True  # Next box at the top.
                    elif top:
                        y_plot = np.max(seq_.type.amplitude.samples)
                        top = False  # Next box at the center.
                    area_fmt = (
                        r"A: $\pi$"
                        if round(area_val, 2) == 1
                        else rf"A: {area_val:.2g}$\pi$"
                    )
                    if not print_phase:
                        txt = area_fmt
                    else:
                        phase_fmt = rf"$\phi$: {phase_str(phase_val)}"
                        txt = "\n".join([phase_fmt, area_fmt])
                    axes[0].text(
                        x_plot,
                        y_plot,
                        txt,
                        fontsize=10,
                        ha="center",
                        va="center",
                        bbox=area_ph_box,
                    )

        target_regions = []  # [[start1, [targets1], end1],...]
        for coords in ch_data.target:
            targets = list(ch_data.target[coords])
            tgt_strs = [str(q) for q in targets]
            tgt_txt_y = max_amp * 1.1 - 0.25 * (len(targets) - 1)
            tgt_str = "\n".join(tgt_strs)
            if coords == "initial":
                x = t_min + final_t * 0.005
                target_regions.append([0, targets])
                if seq.declared_channels[ch].addressing == "Global":
                    axes[0].text(
                        x,
                        amp_top * 0.98,
                        "GLOBAL",
                        fontsize=13,
                        rotation=90,
                        ha="left",
                        va="top",
                        bbox=q_box,
                    )
                else:
                    axes[0].text(
                        x,
                        tgt_txt_y,
                        tgt_str,
                        fontsize=12,
                        ha="left",
                        bbox=q_box,
                    )
                    phase = seq._basis_ref[basis][targets[0]].phase[0]
                    if phase and draw_phase_shifts:
                        msg = r"$\phi=$" + phase_str(phase)
                        axes[0].text(
                            0,
                            max_amp * 1.1,
                            msg,
                            ha="left",
                            fontsize=12,
                            bbox=ph_box,
                        )
            else:
                ti, tf = np.array(coords) / time_scale
                target_regions[-1].append(ti)  # Closing previous regions
                target_regions.append(
                    [tf + 1 / time_scale, targets]
                )  # New one
                phase = seq._basis_ref[basis][targets[0]].phase[
                    tf * time_scale + 1
                ]
                for ax in axes:
                    ax.axvspan(ti, tf, alpha=0.4, color="grey", hatch="//")
                axes[0].text(
                    tf + final_t * 5e-3,
                    tgt_txt_y,
                    tgt_str,
                    ha="left",
                    fontsize=12,
                    bbox=q_box,
                )
                if phase and draw_phase_shifts:
                    msg = r"$\phi=$" + phase_str(phase)
                    wrd_len = len(max(tgt_strs, key=len))
                    x = tf + final_t * 0.01 * (wrd_len + 1)
                    axes[0].text(
                        x,
                        max_amp * 1.1,
                        msg,
                        ha="left",
                        fontsize=12,
                        bbox=ph_box,
                    )
        # Terminate the last open regions
        if target_regions:
            target_regions[-1].append(final_t)
        for start, targets_, end in (
            target_regions if draw_phase_shifts else []
        ):
            start = cast(float, start)
            targets_ = cast(list, targets_)
            end = cast(float, end)
            # All targets have the same ref, so we pick
            q = targets_[0]
            ref = seq._basis_ref[basis][q].phase
            if end != total_duration - 1 or "measurement" in data:
                end += 1 / time_scale
            for t_, delta in ref.changes(start, end, time_scale=time_scale):
                conf = dict(linestyle="--", linewidth=1.5, color="black")
                for ax in axes:
                    ax.axvline(t_, **conf)
                msg = "\u27F2 " + phase_str(delta)
                axes[0].text(
                    t_ - final_t * 8e-3,
                    max_amp * 1.1,
                    msg,
                    ha="right",
                    fontsize=14,
                    bbox=ph_box,
                )

        # Draw the EOM intervals
        for ch_eom_start_buffer, ch_eom_interval, ch_eom_end_buffer in zip(
            ch_eom_start_buffers, ch_eom_intervals, ch_eom_end_buffers
        ):
            for ax in axes:
                ch_eom_start_buffer.smooth_draw(ax, decreasing=False)
                ch_eom_interval.draw(ax)
                ch_eom_end_buffer.smooth_draw(ax, decreasing=True)
            tgt_txt_x = ch_eom_start_buffer.ti or ch_eom_interval.ti
            tgt_txt_y = axes[0].get_ylim()[1]
            axes[0].text(
                tgt_txt_x,
                tgt_txt_y,
                "EOM",
                fontsize=12,
                ha="left",
                va="top",
                bbox=eom_box,
            )
        # Draw the SLM mask
        if seq._slm_mask_targets and seq._slm_mask_time:
            tf_m = seq._slm_mask_time[1]
            for ax in axes:
                ax.axvspan(0, tf_m, color="black", alpha=0.1, zorder=-100)
            tgt_strs = [str(q) for q in seq._slm_mask_targets]
            tgt_txt_x = final_t * 0.005
            tgt_txt_y = axes[-1].get_ylim()[0]
            tgt_str = "\n".join(tgt_strs)
            axes[-1].text(
                tgt_txt_x,
                tgt_txt_y,
                tgt_str,
                fontsize=12,
                ha="left",
                bbox=slm_box,
            )

        hline_kwargs = dict(linestyle="-", linewidth=0.5, color="grey")
        if "measurement" in data:
            msg = f"Basis: {data['measurement']}"
            if len(axes) == 1:
                mid_ax = axes[0]
                mid_point = (amp_top + amp_bottom) / 2
                fontsize = 12
            else:
                mid_ax = axes[-1]
                mid_point = (
                    ax_lims[-1][1]
                    if len(axes) == 2
                    else ax_lims[-1][0] + sum(ax_lims[-1]) * 1.5
                )
                fontsize = 14

            for ax in axes:
                ax.axvspan(final_t, t_max, color="midnightblue", alpha=1)

            mid_ax.text(
                final_t * 1.025,
                mid_point,
                msg,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white",
                rotation=90,
            )
            hline_kwargs["xmax"] = 0.95

        for i, ax in enumerate(axes):
            if i > 0:
                ax.axhline(ax_lims[i][1], **hline_kwargs)
            if ax_lims[i][0] < 0:
                ax.axhline(0, **hline_kwargs)

        if draw_interp_pts:
            for qty in ("amplitude", "detuning"):
                if qty in ch_data.interp_pts and ch_data.curves_on[qty]:
                    ind = CURVES_ORDER.index(qty)
                    pts = np.array(ch_data.interp_pts[qty])
                    axes[ind].scatter(pts[:, 0], pts[:, 1], color=COLORS[ind])

    return (fig_reg if draw_register else None, fig)
