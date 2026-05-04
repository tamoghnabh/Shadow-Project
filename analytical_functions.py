import numpy as np
import pandas as pd

def detect_relaxation_after_throughput(
    df,
    capacity_ah,
    c_rate_threshold=0.02,          # ~C/50
    peak_current_threshold=None,
    peak_duration_threshold=10.0,   # seconds
    min_duration=120.0,             # seconds
    min_throughput_soc=0.2          # 20% SOC
):
    """
    Detect relaxation periods after significant charge/discharge events.

    Returns:
        charge_relaxations: [(start_time, end_time), ...]
        discharge_relaxations: [(start_time, end_time), ...]
    """

    seconds = df['Seconds'].values
    current = df['I_in_A'].values
    voltage = df['V_in_V'].values

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

def coulomb_count_bidirectional(
    df,
    capacity_ah,
    anchor_time,
    anchor_soc,
    clip_soc=True
):
    """
    Perform bidirectional Coulomb Counting from an anchor SOC.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time and current data
    capacity_ah : float
        Battery capacity in Ah
    anchor_time : float
        Timestamp where SOC is known
    anchor_soc : float
        SOC at anchor_time (0 to 1)
    clip_soc : bool
        Whether to clip SOC to [0,1]

    Returns
    -------
    soc : np.ndarray
        SOC profile over entire time range
    """
    time_s = df['Seconds'].values
    current_a = -df['I_cell_in_A'].values

    time_s = np.asarray(time_s)
    current_a = np.asarray(current_a)

    if len(time_s) != len(current_a):
        raise ValueError("time and current must have same length")

    # Convert capacity to Coulombs
    capacity_c = capacity_ah * 3600.0

    # Find anchor index
    anchor_idx = np.argmin(np.abs(time_s - anchor_time))

    soc = np.zeros_like(current_a, dtype=float)
    soc[anchor_idx] = anchor_soc

    # -------------------
    # Forward propagation
    # -------------------
    for i in range(anchor_idx + 1, len(time_s)):
        dt = time_s[i] - time_s[i - 1]
        dQ = current_a[i - 1] * dt  # Coulombs
        soc[i] = soc[i - 1] - dQ / capacity_c

    # -------------------
    # Backward propagation
    # -------------------
    for i in range(anchor_idx - 1, -1, -1):
        dt = time_s[i + 1] - time_s[i]
        dQ = current_a[i] * dt
        soc[i] = soc[i + 1] + dQ / capacity_c

    if clip_soc:
        soc = np.clip(soc, 0.0, 1.0)

    return soc

def vmax_vmin(df):
    max_V = df['V_cell_in_V'].max()
    min_V = df['V_cell_in_V'].min()
    max_V_time = df[df['V_cell_in_V'] == max_V]['Seconds'].iloc[0]
    min_V_time = df[df['V_cell_in_V'] == min_V]['Seconds'].iloc[0]
    return max_V, min_V, max_V_time, min_V_time

def compute_ah_throughput(
    df,
    time_col="Seconds",
    start_time=None,
    end_time=None,
    current_col="I_cell_in_A",
    method="trapezoidal"
):
    """
    Compute cumulative Ah throughput between two timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain time and current columns
    start_time : float or datetime-like
    end_time : float or datetime-like
    time_col : str
        Name of time column
    current_col : str
        Name of current column (A)
    method : str
        'trapezoidal' (recommended) or 'rectangular'

    Returns
    -------
    throughput_ah : float
    """

    # Sort just in case
    df = df.sort_values(time_col)
    if start_time is None or end_time is None:
        _, _, start_time, end_time = vmax_vmin(df)
    # Filter time window
    # Detect direction
    reverse = end_time < start_time

    t1, t2 = (end_time, start_time) if reverse else (start_time, end_time)

    mask = (df[time_col] >= t1) & (df[time_col] <= t2)
    sub_df = df.loc[mask]

    if len(sub_df) < 2:
        raise ValueError("Not enough data points in the selected interval")

    t = sub_df[time_col].values
    i = sub_df[current_col].values

    # Compute dt
    dt = np.diff(t)

    if method == "rectangular":
        # Left Riemann sum
        ah = np.sum(i[:-1] * dt) / 3600.0

    elif method == "trapezoidal":
        # Better accuracy
        ah = np.sum(0.5 * (i[:-1] + i[1:]) * dt) / 3600.0

    else:
        raise ValueError("method must be 'trapezoidal' or 'rectangular'")

    return np.abs(ah)

# --------------------------------------------------
# 1. DCIR ESTIMATION FUNCTION
# --------------------------------------------------
def estimate_dcir(year: int, month: int) -> float:
    start_year, start_month = 2016, 1
    end_year, end_month = 2022, 12

    dcir_start = 10e-3  # 10 mΩ
    dcir_end = 40e-3    # 40 mΩ

    def to_month_index(y, m):
        return y * 12 + (m - 1)

    t_start = to_month_index(start_year, start_month)
    t_end = to_month_index(end_year, end_month)
    t = to_month_index(year, month)

    if t <= t_start:
        return dcir_start
    if t >= t_end:
        return dcir_end

    alpha = (t - t_start) / (t_end - t_start)

    # Optional nonlinear aging (uncomment if needed)
    # alpha = alpha**1.5

    return dcir_start + alpha * (dcir_end - dcir_start)


# --------------------------------------------------
# 2. MAIN PIPELINE FUNCTION
# --------------------------------------------------
def detect_soc_window_segments(
    df: pd.DataFrame,
    year: int,
    month: int,
    voltage_col: str = "V_cell_in_V",
    current_col: str = "I_cell_in_A",
    time_col: str = "Seconds",
    upper_bounds=(3.95, 4.05),
    lower_bounds=(3.55, 3.65),
    min_segment_length: int = 5,
    seed: int = None,
):
    """
    Full pipeline:
    - Estimate DCIR
    - Compute pseudo-OCV
    - Detect upper/lower SOC segments

    Returns:
        dict with:
            {
                "upper_segments": [(t_start, t_end), ...],
                "lower_segments": [(t_start, t_end), ...],
                "dcir": value used,
                "upper_threshold": value,
                "lower_threshold": value
            }
    """

    # -------------------------------
    # 1. Estimate DCIR
    # -------------------------------
    R = estimate_dcir(year, month)

    # -------------------------------
    # 2. Compute pseudo-OCV
    # -------------------------------
    V = df[voltage_col].values
    I = df[current_col].values
    t = df[time_col].values

    # V_ocv = V_terminal + I * R
    V_ocv = V + I * R

    # -------------------------------
    # 3. Random threshold sampling
    # -------------------------------
    rng = np.random.default_rng(seed)

    upper_thresh = rng.uniform(*upper_bounds)
    lower_thresh = rng.uniform(*lower_bounds)

    if lower_thresh >= upper_thresh:
        raise ValueError("Invalid thresholds")

    # -------------------------------
    # 4. Create masks
    # -------------------------------
    upper_mask = V_ocv >= upper_thresh
    lower_mask = V_ocv <= lower_thresh

    # -------------------------------
    # 5. Helper to extract segments
    # -------------------------------
    def extract_segments(mask):
        segments = []
        in_segment = False
        start_idx = None

        for i in range(len(mask)):
            if mask[i]:
                if not in_segment:
                    in_segment = True
                    start_idx = i
            else:
                if in_segment:
                    end_idx = i - 1

                    if (end_idx - start_idx + 1) >= min_segment_length:
                        segments.append((t[start_idx], t[end_idx]))

                    in_segment = False

        # Handle tail case
        if in_segment:
            end_idx = len(mask) - 1
            if (end_idx - start_idx + 1) >= min_segment_length:
                segments.append((t[start_idx], t[end_idx]))

        return segments

    # -------------------------------
    # 6. Extract segments
    # -------------------------------
    upper_segments = extract_segments(upper_mask)
    lower_segments = extract_segments(lower_mask)

    # -------------------------------
    # 7. Return structured output
    # -------------------------------
    return {
        "upper_segments": upper_segments,
        "lower_segments": lower_segments,
        "dcir": R,
        "upper_threshold": upper_thresh,
        "lower_threshold": lower_thresh,
    }