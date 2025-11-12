#Set function for plotting untrimmed data
import matplotlib.pyplot as plt

def plot_available_vars(available_vars, data_dict, include_do=False):
    # Setup Plotting Variable Structure
    available_vars = [var for var in available_vars if var != 'time']
    if 'p' not in available_vars:
         available_vars.append('p')

    var_labels = {
        't': 'Temp.',
        'c': 'Cond.',
        'do': 'Dissolved Oxygen',
        'p': 'Pressure'  # Ensure pressure is included
    }
    if include_do and 'do' in available_vars:
        var_labels['do'] = 'Dissolved Oxygen'

    # Set up plotting structure
    num_vars = len(available_vars)
    fig, axes = plt.subplots(num_vars, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.04)
    fig.align_ylabels()

    for i, var_name in enumerate(available_vars):
        if var_name in var_labels and var_name in data_dict:
            axes[i].plot(data_dict[var_name], lw=1, color='k')
            axes[i].set_ylabel(var_labels[var_name])

    plt.suptitle("Available Variables (Pre-Trim)")
    plt.show(block=True)
