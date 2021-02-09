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

import matplotlib.pyplot as plt
import numpy as np

from pulser.waveforms import ConstantWaveform


def gather_data(seq):
    """Collects the whole sequence data for plotting.

    Args:
        seq (pulser.Sequence): The input sequence of operations on a device.

    Returns:
        dict: The data to plot.
    """
    # The minimum time axis length is 100 ns
    seq._total_duration = max([seq._last(ch).tf for ch in seq._schedule
                               if seq._schedule[ch]] + [100])
    data = {}
    for ch, sch in seq._schedule.items():
        time = [-1]     # To not break the "time[-1]" later on
        amp = []
        detuning = []
        target = {}
        # phase_shift = {}
        for slot in sch:
            if slot.ti == -1:
                target['initial'] = slot.targets
                time += [0]
                amp += [0]
                detuning += [0]
                continue
            if slot.type in ['delay', 'target']:
                time += [slot.ti, slot.tf-1]
                amp += [0, 0]
                detuning += [0, 0]
                if slot.type == 'target':
                    target[(slot.ti, slot.tf-1)] = slot.targets
                continue
            pulse = slot.type
            if (isinstance(pulse.amplitude, ConstantWaveform) and
                    isinstance(pulse.detuning, ConstantWaveform)):
                time += [slot.ti, slot.tf-1]
                amp += [pulse.amplitude._value] * 2
                detuning += [pulse.detuning._value] * 2
            else:
                time += list(range(slot.ti, slot.tf))
                amp += pulse.amplitude.samples.tolist()
                detuning += pulse.detuning.samples.tolist()
        if time[-1] < seq._total_duration - 1:
            time += [time[-1]+1, seq._total_duration-1]
            amp += [0, 0]
            detuning += [0, 0]
        # Store everything
        time.pop(0)     # Removes the -1 in the beginning
        data[ch] = {'time': time, 'amp': amp, 'detuning': detuning,
                    'target': target}
        if hasattr(seq, "_measurement"):
            data[ch]['measurement'] = seq._measurement
    return data


def draw_sequence(seq):
    """Draw the entire sequence.

    Args:
        seq (pulser.Sequence): The input sequence of operations on a device.
    """

    def phase_str(phi):
        """Formats a phase value for printing."""
        value = (((phi + np.pi) % (2*np.pi)) - np.pi) / np.pi
        if value == -1:
            return r"$\pi$"
        elif value == 0:
            return "0"
        else:
            return r"{:.2g}$\pi$".format(value)

    n_channels = len(seq._channels)
    if not n_channels:
        raise SystemError("Can't draw an empty sequence.")
    data = gather_data(seq)
    time_scale = 1e3 if seq._total_duration > 1e4 else 1

    # Boxes for qubit and phase text
    q_box = dict(boxstyle="round", facecolor='orange')
    ph_box = dict(boxstyle="round", facecolor='ghostwhite')

    fig = plt.figure(constrained_layout=False, figsize=(20, 4.5*n_channels))
    gs = fig.add_gridspec(n_channels, 1, hspace=0.075)

    ch_axes = {}
    for i, (ch, gs_) in enumerate(zip(seq._channels, gs)):
        ax = fig.add_subplot(gs_)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False,
                       right=False)
        ax.set_ylabel(ch, labelpad=40, fontsize=18)
        subgs = gs_.subgridspec(2, 1, hspace=0.)
        ax1 = fig.add_subplot(subgs[0, :])
        ax2 = fig.add_subplot(subgs[1, :])
        ch_axes[ch] = (ax1, ax2)
        for j, ax in enumerate(ch_axes[ch]):
            ax.axvline(0, linestyle='--', linewidth=0.5, color='grey')
            if j == 0:
                ax.spines['bottom'].set_visible(False)
            else:
                ax.spines['top'].set_visible(False)

            if i < n_channels - 1 or j == 0:
                ax.tick_params(axis='x', which='both', bottom=True,
                               top=False, labelbottom=False, direction='in')
            else:
                unit = 'ns' if time_scale == 1 else r'$\mu s$'
                ax.set_xlabel(f't ({unit})', fontsize=12)

    for ch, (a, b) in ch_axes.items():
        basis = seq._channels[ch].basis
        t = np.array(data[ch]['time']) / time_scale
        ya = data[ch]['amp']
        yb = data[ch]['detuning']

        t_min = -t[-1]*0.03
        t_max = t[-1]*1.05
        a.set_xlim(t_min, t_max)
        b.set_xlim(t_min, t_max)

        max_amp = np.max(ya)
        max_amp = 1 if max_amp == 0 else max_amp
        amp_top = max_amp * 1.2
        a.set_ylim(-0.02, amp_top)
        det_max = np.max(yb)
        det_min = np.min(yb)
        det_range = det_max - det_min
        if det_range == 0:
            det_min, det_max, det_range = -1, 1, 2
        det_top = det_max + det_range * 0.15
        det_bottom = det_min - det_range * 0.05
        b.set_ylim(det_bottom, det_top)

        a.plot(t, ya, color="darkgreen", linewidth=0.8)
        b.plot(t, yb, color='indigo', linewidth=0.8)
        a.fill_between(t, 0, ya, color="darkgreen", alpha=0.3)
        b.fill_between(t, 0, yb, color="indigo", alpha=0.3)
        a.set_ylabel(r'$\Omega$ (rad/µs)', fontsize=14, labelpad=10)
        b.set_ylabel(r'$\delta$ (rad/µs)', fontsize=14)

        target_regions = []     # [[start1, [targets1], end1],...]
        for coords in data[ch]['target']:
            targets = list(data[ch]['target'][coords])
            tgt_strs = [str(q) for q in targets]
            tgt_txt_y = max_amp*1.1-0.25*(len(targets)-1)
            tgt_str = "\n".join(tgt_strs)
            if coords == 'initial':
                x = t_min + t[-1]*0.005
                target_regions.append([0, targets])
                if seq._channels[ch].addressing == 'Global':
                    a.text(x, amp_top*0.98, "GLOBAL", fontsize=13,
                           rotation=90, ha='left', va='top', bbox=q_box)
                else:
                    a.text(x, tgt_txt_y, tgt_str, fontsize=12, ha='left',
                           bbox=q_box)
                    phase = seq._phase_ref[basis][targets[0]][0]
                    if phase:
                        msg = r"$\phi=$" + phase_str(phase)
                        a.text(0, max_amp*1.1, msg, ha='left', fontsize=12,
                               bbox=ph_box)
            else:
                ti, tf = np.array(coords) / time_scale
                target_regions[-1].append(ti)   # Closing previous regions
                target_regions.append([tf + 1/time_scale, targets])  # New one
                phase = seq._phase_ref[basis][targets[0]][tf * time_scale + 1]
                a.axvspan(ti, tf, alpha=0.4, color='grey', hatch='//')
                b.axvspan(ti, tf, alpha=0.4, color='grey', hatch='//')
                a.text(tf + t[-1]*5e-3, tgt_txt_y, tgt_str, ha='left',
                       fontsize=12, bbox=q_box)
                if phase:
                    msg = r"$\phi=$" + phase_str(phase)
                    wrd_len = len(max(tgt_strs, key=len))
                    x = tf + t[-1]*0.01*(wrd_len+1)
                    a.text(x, max_amp*1.1, msg, ha='left',
                           fontsize=12, bbox=ph_box)
        # Terminate the last open regions
        if target_regions:
            target_regions[-1].append(t[-1])
        for start, targets, end in target_regions:
            q = targets[0]  # All targets have the same ref, so we pick
            ref = seq._phase_ref[basis][q]
            if end != seq._total_duration - 1 or 'measurement' not in data[ch]:
                end += 1 / time_scale
            for t_, delta in ref.changes(start, end, time_scale=time_scale):
                conf = dict(linestyle='--', linewidth=1.5, color='black')
                a.axvline(t_, **conf)
                b.axvline(t_, **conf)
                msg = u"\u27F2 " + phase_str(delta)
                a.text(t_-t[-1]*8e-3, max_amp*1.1, msg, ha='right',
                       fontsize=14, bbox=ph_box)

        if 'measurement' in data[ch]:
            msg = f"Basis: {data[ch]['measurement']}"
            b.text(t[-1]*1.025, det_top, msg, ha='center', va='center',
                   fontsize=14, color='white', rotation=90)
            a.axvspan(t[-1], t_max, color='midnightblue', alpha=1)
            b.axvspan(t[-1], t_max, color='midnightblue', alpha=1)
            a.axhline(0, xmax=0.95, linestyle='-', linewidth=0.5,
                      color='grey')
            b.axhline(0, xmax=0.95, linestyle=':', linewidth=0.5,
                      color='grey')
        else:
            a.axhline(0, linestyle='-', linewidth=0.5, color='grey')
            b.axhline(0, linestyle=':', linewidth=0.5, color='grey')

    plt.show()
