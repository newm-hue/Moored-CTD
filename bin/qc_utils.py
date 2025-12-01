import numpy as np
import matplotlib.pyplot as plt
import yaml

def load_flagged_arrays(filepath, variables):
    with open(filepath, 'r') as f:
        flagged_config = yaml.safe_load(f)

    print("Loaded keys:", flagged_config.keys())

    flag_arrays = {}

    for var_name, arr in variables.items():
         #Default flag value = 2
        flag_array = np.full_like(arr, 2, dtype=int)

        if var_name in flagged_config:
            flagged_t_data = []

            # data point flags
            flags = flagged_config[var_name].get('flags', [])
            flagged_t_data.extend(flags)

            # data range flags
            ranges = flagged_config[var_name].get('ranges', [])
            for entry in ranges:
                if len(entry) == 3:
                    start, end, value = entry
                elif len(entry) == 2:
                    start, end = entry
                    value = 2  # default flag
                else:
                    raise ValueError(f"Invalid range entry: {entry}")
                flagged_t_data.extend([(i, value) for i in range(start, end)])

            # Apply flags
            if flagged_t_data:
                indices, values = zip(*flagged_t_data)
                flag_array[np.array(indices, dtype=int)] = np.array(values, dtype=int)

        # assign NaNs
        flag_array[np.isnan(arr)] = 5

        flag_arrays[var_name] = flag_array

    print("YAML keys:", flagged_config.keys())
    print("Variable keys:", variables.keys())

    return flag_arrays

#create data frame
def merge_flags(trimmed, flag_arrays):
    flagged_df = trimmed.copy()
    flagged_df['flag_t'] = flag_arrays['temperature']
    flagged_df['flag_c'] = flag_arrays['conductivity']
    flagged_df['flag_p'] = flag_arrays['pressure']
    return flagged_df

#Plot flagged data
# Define flag colors and labels
FLAG_COLORS = {
    3: '#DAA520',   # Questionable
    4: '#B22222',   # Bad
    5: '#A9A9A9',   # Not Reported
}
FLAG_LABELS = {
    3: "Flag 3: Questionable",
    4: "Flag 4: Bad",
    5: "Flag 5: Not Reported",
}

def plot_flagged_data(flagged_df, include_do=False):

    var_labels = {
        't': 'Temperature',
        'c': 'Conductivity',
        'p': 'Pressure',
        's': 'Salinity',
        'rho': "Density (kg/m³)"
    }

    plot_vars = {
        't': flagged_df['t'],
        'c': flagged_df['c'],
        'p': flagged_df['p'],
        's': flagged_df['s'],
        'rho': flagged_df['rho']
    }

    flag_vars = {
        't': flagged_df['flag_t'],
        'c': flagged_df['flag_c'],
        'p': flagged_df['flag_p'],
        's': flagged_df.get('flag_s'),
        'rho': flagged_df.get('flag_rho')
    }

    if include_do and 'do' in flagged_df and 'flag_do' in flagged_df:
        var_labels['do'] = 'Dissolved Oxygen'
        plot_vars['do'] = flagged_df['do']
        flag_vars['do'] = flagged_df['flag_do']

    fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.04)
    fig.align_ylabels()

    for i, (var_name, data_array) in enumerate(plot_vars.items()):
        flags = flag_vars[var_name]

        # Base line
        axes[i].plot(data_array, lw=1, color='k')
        axes[i].set_ylabel(var_labels[var_name])

        # Overlay flagged points
        if flags is not None:
            for flag_value, color in FLAG_COLORS.items():
                idx = np.where(flags == flag_value)[0]
                if len(idx) > 0:
                    axes[i].scatter(idx, data_array.iloc[idx] if hasattr(data_array, "iloc") else data_array[idx],
                                    color=color, s=10, label=FLAG_LABELS[flag_value])

    # Add legend only once
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right')

    plt.suptitle("Flagged In-Place Variables")
    plt.show(block=True)

    return fig, axes