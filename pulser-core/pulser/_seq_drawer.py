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
from itertools import combinations
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.interpolate import CubicSpline

import pulser
from pulser import Register, Register3D
from pulser.pulse import Pulse
from pulser.waveforms import ConstantWaveform, InterpolatedWaveform

# Color scheme
COLORS = ["darkgreen", "indigo", "#c75000"]

SIZE_PER_WIDTH = {1: 3, 2: 4, 3: 5}
LABELS = [
    r"$\Omega$ (rad/µs)",
    r"$\delta$ (rad/µs)",
    r"$\varphi$ / 2π",
]


def gather_data(seq: pulser.sequence.Sequence) -> dict:
    """Collects the whole sequence data for plotting.

    Args:
        seq: The input sequence of operations on a device.

    Returns:
        The data to plot.
    """
    # The minimum time axis length is 100 ns
    total_duration = max(seq.get_duration(), 100)
    data: dict[str, Any] = {}
    for ch, sch in seq._schedule.items():
        time = [-1]  # To not break the "time[-1]" later on
        amp = []
        detuning = []
        phase = []
        # List of interpolation points
        interp_pts: defaultdict[str, list[list[float]]] = defaultdict(list)
        target: dict[Union[str, tuple[int, int]], Any] = {}
        # phase_shift = {}
        for slot in sch:
            if slot.ti == -1:
                target["initial"] = slot.targets
                time += [0]
                amp += [0.0]
                detuning += [0.0]
                phase += [0.0]
                continue
            if slot.type in ["delay", "target"]:
                time += [
                    slot.ti,
                    slot.tf - 1 if slot.tf > slot.ti else slot.ti,
                ]
                amp += [0.0, 0.0]
                detuning += [0.0, 0.0]
                phase += [phase[-1]] * 2
                if slot.type == "target":
                    target[(slot.ti, slot.tf - 1)] = slot.targets
                continue
            pulse = cast(Pulse, slot.type)
            if isinstance(pulse.amplitude, ConstantWaveform) and isinstance(
                pulse.detuning, ConstantWaveform
            ):
                time += [slot.ti, slot.tf - 1]
                amp += [float(pulse.amplitude[0])] * 2
                detuning += [float(pulse.detuning[0])] * 2
                phase += [float(pulse.phase) / (2 * np.pi)] * 2
            else:
                time += list(range(slot.ti, slot.tf))
                amp += pulse.amplitude.samples.tolist()
                detuning += pulse.detuning.samples.tolist()
                phase += [float(pulse.phase) / (2 * np.pi)] * pulse.duration
                for wf_type in ["amplitude", "detuning"]:
                    wf = getattr(pulse, wf_type)
                    if isinstance(wf, InterpolatedWaveform):
                        pts = wf.data_points
                        pts[:, 0] += slot.ti
                        interp_pts[wf_type] += pts.tolist()

        if time[-1] < total_duration - 1:
            time += [time[-1] + 1, total_duration - 1]
            amp += [0, 0]
            detuning += [0, 0]
            phase += [phase[-1] if len(phase) else 0] * 2
        # Store everything
        time.pop(0)  # Removes the -1 in the beginning
        data[ch] = {
            "time": time,
            "amp": amp,
            "detuning": detuning,
            "phase": phase,
            "target": target,
        }
        if hasattr(seq, "_measurement"):
            data[ch]["measurement"] = seq._measurement
        if interp_pts:
            data[ch]["interp_pts"] = interp_pts
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
) -> tuple[Figure, Figure]:
    """Draws the entire sequence.

    Args:
        seq: The input sequence of operations on a device.
        sampling_rate: Sampling rate of the effective pulse used by
            the solver. If present, plots the effective pulse alongside the
            input pulse.
        draw_phase_area: Whether phase and area values need to be shown
            as text on the plot, defaults to False.
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

    n_channels = len(seq._channels)
    if not n_channels:
        raise RuntimeError("Can't draw an empty sequence.")
    data = gather_data(seq)
    total_duration = data["total_duration"]
    time_scale = 1e3 if total_duration > 1e4 else 1
    curves_per_ch = {}
    for ch in seq._schedule:
        curves_per_ch[ch] = [True, False, False]
        if np.nonzero(data[ch]["detuning"])[0].size > 0:
            curves_per_ch[ch][1] = True
        if draw_phase_curve and np.nonzero(data[ch]["phase"])[0].size > 0:
            curves_per_ch[ch][2] = True
    axes_per_ch = {
        ch: sum(curves_on) for ch, curves_on in curves_per_ch.items()
    }

    # Boxes for qubit and phase text
    q_box = dict(boxstyle="round", facecolor="orange")
    ph_box = dict(boxstyle="round", facecolor="ghostwhite")
    area_ph_box = dict(boxstyle="round", facecolor="ghostwhite", alpha=0.7)
    slm_box = dict(boxstyle="round", alpha=0.4, facecolor="grey", hatch="//")

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

    ratios = [SIZE_PER_WIDTH[n_axes] for n_axes in axes_per_ch.values()]
    fig = plt.figure(
        constrained_layout=False,
        figsize=(20, sum(ratios)),
    )
    gs = fig.add_gridspec(n_channels, 1, hspace=0.075, height_ratios=ratios)

    ch_axes = {}
    for i, (ch, gs_) in enumerate(zip(seq._channels, gs)):
        ax = fig.add_subplot(gs_)
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_color("none")
        ax.tick_params(
            labelcolor="w", top=False, bottom=False, left=False, right=False
        )
        ax.set_ylabel(ch, labelpad=40, fontsize=18)
        subgs = gs_.subgridspec(axes_per_ch[ch], 1, hspace=0.0)
        ch_axes[ch] = [
            fig.add_subplot(subgs[i, :]) for i in range(axes_per_ch[ch])
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

    if sampling_rate:
        indexes = np.linspace(
            0,
            total_duration - 1,
            int(sampling_rate * total_duration),
            dtype=int,
        )
        times = np.arange(total_duration, dtype=np.double) / time_scale
        solver_time = times[indexes]
        delta_t = np.diff(solver_time)[0]
        # Compare pulse with an interpolated pulse with 100 times more samples
        teff = np.arange(0, max(solver_time), delta_t / 100)

    # Make sure the time axis of all channels are aligned
    final_t = total_duration / time_scale
    if draw_modulation:
        for ch, ch_obj in seq._channels.items():
            final_t = max(
                final_t,
                (seq.get_duration(ch) + 2 * ch_obj.rise_time) / time_scale,
            )
    t_min = -final_t * 0.03
    t_max = final_t * 1.05

    for ch, axes in ch_axes.items():
        ch_obj = seq._channels[ch]
        basis = ch_obj.basis
        times = np.array(data[ch]["time"])
        t = times / time_scale
        ys = [data[ch][qty] for qty in ("amp", "detuning", "phase")]
        if sampling_rate:
            cubic_splines = []
            yseff = []
            t2 = 1
            t2s = []
            for t_solv in solver_time:
                # Find the interval [t[t2],t[t2+1]] containing t_solv
                while t_solv > t[t2]:
                    t2 += 1
                t2s.append(t2)
            for i, y_ in enumerate(ys):
                y2 = [y_[t_] for t_ in t2s]
                cubic_splines.append(CubicSpline(solver_time, y2))
                yseff.append(cubic_splines[i](teff))

        draw_output = draw_modulation and (
            ch_obj.mod_bandwidth or not draw_input
        )
        if draw_output:
            ys_mod = []
            t_diffs = np.diff(times)
            end_index = int(final_t * time_scale)
            for i, y_ in enumerate(ys):
                input = np.repeat(y_[1:], t_diffs)
                ys_mod.append(
                    ch_obj.modulate(input, keep_ends=i > 0)[:end_index]
                )

        ref_ys = yseff if sampling_rate else ys
        max_amp = np.max(ref_ys[0])
        max_amp = 1 if max_amp == 0 else max_amp
        amp_top = max_amp * 1.2
        amp_bottom = min(0.0, *ref_ys[0])
        det_max = np.max(ref_ys[1])
        det_min = np.min(ref_ys[1])
        det_range = det_max - det_min
        if det_range == 0:
            det_min, det_max, det_range = -1, 1, 2
        det_top = det_max + det_range * 0.15
        det_bottom = det_min - det_range * 0.05
        ax_lims = [
            (amp_bottom, amp_top),
            (det_bottom, det_top),
            (min(0.0, *ref_ys[2]) / (2 * np.pi), 1.1),
        ]
        ax_lims = [
            lim for i, lim in enumerate(ax_lims) if curves_per_ch[ch][i]
        ]
        for ax, ylim in zip(axes, ax_lims):
            ax.set_xlim(t_min, t_max)
            ax.set_ylim(*ylim)

        selected_inds = [
            i for i, curve_on in enumerate(curves_per_ch[ch]) if curve_on
        ]
        for i, ax in zip(selected_inds, axes):
            if draw_input:
                ax.plot(t, ys[i], color=COLORS[i], linewidth=0.8)
            if sampling_rate:
                ax.plot(
                    teff,
                    yseff[i],
                    color=COLORS[i],
                    linewidth=0.8,
                )
                ax.fill_between(teff, 0, yseff[i], color=COLORS[i], alpha=0.3)
            elif draw_input:
                ax.fill_between(t, 0, ys[i], color=COLORS[i], alpha=0.3)
            if draw_output:
                ax.fill_between(
                    np.arange(ys_mod[i].size),
                    0,
                    ys_mod[i],
                    color=COLORS[i],
                    alpha=0.3,
                    hatch="////",
                )
            special_kwargs = dict(labelpad=10) if i == 0 else {}
            ax.set_ylabel(LABELS[i], fontsize=14, **special_kwargs)

        if draw_phase_area:
            top = False  # Variable to track position of box, top or center.
            draw_phase = any(
                seq_.type.phase != 0
                for seq_ in seq._schedule[ch]
                if isinstance(seq_.type, Pulse)
            )
            for pulse_num, seq_ in enumerate(seq._schedule[ch]):
                # Select only `Pulse` objects
                if isinstance(seq_.type, Pulse):
                    if sampling_rate:
                        area_val = (
                            np.sum(
                                cubic_splines[0](
                                    np.arange(seq_.ti, seq_.tf) / time_scale
                                )
                            )
                            * 1e-3
                            / np.pi
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
                    if not draw_phase:
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
        for coords in data[ch]["target"]:
            targets = list(data[ch]["target"][coords])
            tgt_strs = [str(q) for q in targets]
            tgt_txt_y = max_amp * 1.1 - 0.25 * (len(targets) - 1)
            tgt_str = "\n".join(tgt_strs)
            if coords == "initial":
                x = t_min + final_t * 0.005
                target_regions.append([0, targets])
                if seq._channels[ch].addressing == "Global":
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
                    phase = seq._phase_ref[basis][targets[0]][0]
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
                phase = seq._phase_ref[basis][targets[0]][tf * time_scale + 1]
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
            ref = seq._phase_ref[basis][q]
            if end != total_duration - 1 or "measurement" not in data[ch]:
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
        if "measurement" in data[ch]:
            msg = f"Basis: {data[ch]['measurement']}"
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

        if "interp_pts" in data[ch] and draw_interp_pts:
            all_points = data[ch]["interp_pts"]
            if "amplitude" in all_points:
                pts = np.array(all_points["amplitude"])
                axes[0].scatter(pts[:, 0], pts[:, 1], color=COLORS[0])
            if "detuning" in all_points:
                pts = np.array(all_points["detuning"])
                axes[1].scatter(pts[:, 0], pts[:, 1], color=COLORS[1])

    return (fig_reg if draw_register else None, fig)
