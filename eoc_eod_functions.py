import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def load_and_operate(folder_path, start_year, end_year, cellID, month = None, print_missing=False, operation=None):
    results = {}
    """Load CSV files for a given cell and year range, apply an optional operation, and return results.
    Parameters:
    - folder_path: str, path to the folder containing CSV files
    - start_year: int, starting year (inclusive)
    - end_year: int, ending year (inclusive)
    - cellID: str, cell ID to filter files (e.g., '06')
    - month: int or None, if specified, only load files for this month (1-12)
    - print_missing: bool, if True, print names of missing files
    - operation: function or None, if provided, apply this function to each loaded DataFrame
    Returns:
    - dict, keys are 'YYYY_MM' or 'YYYY' (if month is None), values are results of the operation or the DataFrame itself if operation is None
    """
    
    for year in range(start_year, end_year + 1):
        if month is not None:
            file_name = f"{year}_{month:02d}_System_ID_{cellID:02d}.csv"
            file_path = os.path.join(folder_path, file_name)

            try:
                df = pd.read_csv(file_path)
                
                df = process_time(df)
                
                if operation:
                    result = operation
                    results[f"{year}_{month:02d}"] = result

                # print(f"Loaded: {file_name}")

            except FileNotFoundError:
                if print_missing:
                    print(f"Missing: {file_name}")
                    continue

            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue
        else:
            for month in range(1, 13):
                file_name = f"{year}_{month:02d}_System_ID_{cellID:02d}.csv"
                file_path = os.path.join(folder_path, file_name)

                try:
                    df = pd.read_csv(file_path)
                    df = process_time(df)
                    
                    if operation:
                        result = operation
                        results[f"{year}_{month:02d}"] = result

                    # print(f"Loaded: {file_name}")

                except FileNotFoundError:
                    if print_missing:
                        print(f"Missing: {file_name}")
                        continue

                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
                    continue
    print(year ,month)
    return results

def process_time(df):
    df['Time'] = pd.to_datetime(df['Time'])
    time_delta = (df['Time'] - df['Time'].iloc[0])
    seconds = time_delta.dt.total_seconds()
    df['Seconds'] = seconds
    return df


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

def cell_level_scaling(df, metadata, cell_id):
    cell = metadata[metadata['ID'] == cell_id].reset_index(drop=True)
    n_s = cell['Cell_number_in_series'].iloc[0]
    n_p = cell['Cell_number_in_parallel'].iloc[0]
    df['V_cell_in_V'] = df['V_in_V'] / n_s
    df['I_cell_in_A'] = df['I_in_A'] / n_p
    return df

def cell_level_info(systems_metadata, cell_id):
    cell = systems_metadata[systems_metadata['ID'] == cell_id].reset_index(drop=True)
    n_s = cell['Cell_number_in_series'].iloc[0]
    n_p = cell['Cell_number_in_parallel'].iloc[0]
    V_nom = cell['Voltage_nominal_in_V'].iloc[0] / n_s
    Cell_ah = cell['Capacity_nominal_in_Ah'].iloc[0] / n_p
    return V_nom, Cell_ah

def dict_to_df(data_dict, columns=None, index_name='key'):
    """Convert a dict of values to a DataFrame.

    Parameters
    ----------
    data_dict : dict
        Mapping from key to values (tuple or dict).
    columns : list[str] | None
        Optional column names. If None, uses dict keys or auto-generates names.
    index_name : str
        Name for the index column. Default is 'key'.

    Returns
    -------
    pd.DataFrame
        Rows are the dict keys and columns are the values.
    """
    if len(data_dict) == 0:
        return pd.DataFrame(columns=columns)

    first_value = next(iter(data_dict.values()))
    
    if isinstance(first_value, dict):
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        if columns is not None and list(df.columns) != columns:
            df = df.reindex(columns=columns)
    else:
        df = pd.DataFrame.from_dict(data_dict, orient='index', columns=columns)

    df.index.name = index_name
    return df