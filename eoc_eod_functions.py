import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_multi_axis(df, column_map, legend_names, figsize=(12, 7), x=None, color_cycle=None):
    """
    Plot multiple DataFrame columns on one left y-axis and multiple right-side twin y-axes.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    column_map : dict
        Ordered mapping like {1: 'col_a', 2: ('col_b', 'col_c'), 3: 'col_d'}.
        The first entry uses the left y-axis. Each later entry gets its own right-side twin axis.
        If a value is a tuple, those columns are plotted on the same axis.
    legend_names : tuple | list
        Legend labels for every plotted line, in plotting order.
    figsize : tuple, default (12, 7)
        Figure size.
    x : str | array-like | None, default None
        X values. If None, the DataFrame index is used.
    color_cycle : list | None, default None
        Colors cycled through across all plotted lines.
    """
    if not column_map:
        raise ValueError('column_map cannot be empty.')

    ordered_items = sorted(column_map.items(), key=lambda item: item[0])
    grouped_columns = []

    for _, value in ordered_items:
        if isinstance(value, str):
            cols = (value,)
        elif isinstance(value, tuple):
            cols = value
        else:
            raise TypeError('Each column_map value must be a column name or a tuple of column names.')

        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f'Missing columns in df: {missing_cols}')

        grouped_columns.append(cols)

    total_lines = sum(len(cols) for cols in grouped_columns)
    if len(legend_names) != total_lines:
        raise ValueError(
            f'legend_names must have {total_lines} entries, received {len(legend_names)}.'
        )

    if color_cycle is None:
        color_cycle = [
            'black', 'tab:red', 'tab:purple', 'tab:green',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        ]

    x_values = df.index if x is None else (df[x] if isinstance(x, str) else x)

    fig, ax_left = plt.subplots(figsize=figsize)
    axes = [ax_left]
    lines = []
    labels = []
    legend_iter = iter(legend_names)
    color_index = 0

    for axis_idx, cols in enumerate(grouped_columns):
        if axis_idx == 0:
            ax = ax_left
            ax.grid(True, alpha=0.3)
        else:
            ax = ax_left.twinx()
            ax.spines['right'].set_position(('axes', 1 + 0.1 * (axis_idx - 1)))
            axes.append(ax)

        axis_colors = []
        for col in cols:
            color = color_cycle[color_index % len(color_cycle)]
            color_index += 1
            label = next(legend_iter)
            line, = ax.plot(x_values, df[col], color=color, label=label)
            lines.append(line)
            labels.append(label)
            axis_colors.append(color)

        axis_label = ', '.join(cols)
        ax.set_ylabel(axis_label, color=axis_colors[0])
        ax.tick_params(axis='y', colors=axis_colors[0])

    ax_left.set_xlabel(x if isinstance(x, str) else 'Index')
    ax_left.legend(lines, labels, loc='best')
    fig.tight_layout()

    return fig, axes


def cell_level_scaling(systems_metadata, cell_id):
    cell = systems_metadata[systems_metadata['ID'] == cell_id].reset_index(drop=True)
    n_s = cell['Cell_number_in_series'].iloc[0]
    n_p = cell['Cell_number_in_parallel'].iloc[0]
    V_nom = cell['Voltage_nominal_in_V'].iloc[0] / n_s
    Cell_ah = cell['Capacity_nominal_in_Ah'].iloc[0] / n_p
    return V_nom, Cell_ah


def detect_relaxation_after_throughput(
    seconds,
    current,
    voltage,
    capacity_ah,
    c_rate_threshold=0.02,          # ~C/50
    peak_current_threshold=None,
    peak_duration_threshold=10.0,   # seconds
    min_duration=120.0,             # seconds
    min_throughput_soc=0.2          # 20% SOC
):
    """
    Returns:
        charge_relaxations: [(start_time, end_time), ...]
        discharge_relaxations: [(start_time, end_time), ...]
    """

    seconds = np.asarray(seconds)
    current = np.asarray(current)
    voltage = np.asarray(voltage)

    dt = np.diff(seconds, prepend=seconds[0])

    # -----------------------------
    # 1. THROUGHPUT SEGMENTATION
    # -----------------------------
    charge_segments = []
    discharge_segments = []

    seg_start = 0
    cumulative_ah = 0.0

    def finalize_segment(start, end, total_ah):
        soc = abs(total_ah) / capacity_ah
        if soc >= min_throughput_soc:
            if total_ah > 0:
                charge_segments.append((start, end))
            else:
                discharge_segments.append((start, end))

    for i in range(1, len(seconds)):
        cumulative_ah += current[i] * dt[i] / 3600.0

        # Detect sign change → end of event
        if np.sign(current[i]) != np.sign(current[i-1]):
            finalize_segment(seg_start, i-1, cumulative_ah)
            seg_start = i
            cumulative_ah = 0.0

    # finalize last segment
    finalize_segment(seg_start, len(seconds)-1, cumulative_ah)

    # -----------------------------
    # Helper: apply remaining filters
    # -----------------------------
    def process_segment(s, e):
        seg_time = seconds[s:e+1]
        seg_current = current[s:e+1]

        # --- CURRENT FILTER ---
        current_threshold = c_rate_threshold * capacity_ah
        relax_mask = np.abs(seg_current) < current_threshold

        # find subsegments
        subsegments = []
        start = None
        for i, val in enumerate(relax_mask):
            if val and start is None:
                start = i
            elif not val and start is not None:
                subsegments.append((start, i-1))
                start = None
        if start is not None:
            subsegments.append((start, len(relax_mask)-1))

        # --- PEAK FILTER ---
        if peak_current_threshold is None:
            peak_thr = current_threshold * 3
        else:
            peak_thr = peak_current_threshold

        peak_filtered = []
        for (ss, ee) in subsegments:
            sub_i = np.abs(seg_current[ss:ee+1])
            sub_t = seg_time[ss:ee+1]

            peak_mask = sub_i > peak_thr

            peak_start = None
            reject = False

            for k, val in enumerate(peak_mask):
                if val and peak_start is None:
                    peak_start = k
                elif not val and peak_start is not None:
                    duration = sub_t[k-1] - sub_t[peak_start]
                    if duration >= peak_duration_threshold:
                        reject = True
                        break
                    peak_start = None

            if peak_start is not None:
                duration = sub_t[-1] - sub_t[peak_start]
                if duration >= peak_duration_threshold:
                    reject = True

            if not reject:
                peak_filtered.append((ss, ee))

        # --- DURATION FILTER ---
        final = []
        for (ss, ee) in peak_filtered:
            duration = seg_time[ee] - seg_time[ss]
            if duration >= min_duration:
                final.append((seconds[s + ss], seconds[s + ee]))

        return final

    # -----------------------------
    # 2–4. Apply filters per segment
    # -----------------------------
    charge_relaxations = []
    discharge_relaxations = []

    for (s, e) in charge_segments:
        charge_relaxations.extend(process_segment(s, e))

    for (s, e) in discharge_segments:
        discharge_relaxations.extend(process_segment(s, e))

    return charge_relaxations, discharge_relaxations