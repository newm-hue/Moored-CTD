#Set function for plotting untrimmed data
import matplotlib.pyplot as plt
import gsw
import yaml
import os



#yaml directory
base_dir = os.path.dirname(__file__)
yaml_path = os.path.join(base_dir, "config_process.yaml")


#Load in variable map with trim start and finish indices
with open(yaml_path, "r") as f:
    variable_map = yaml.safe_load(f)

#Name start and finish indices as data
start = variable_map["trim"]["start"]
finish = variable_map["trim"]["finish"]


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

    return fig


#Function for trimming data
def plot_trimmed_var (data_dict, start_index = start, finish_index = finish, include_do=False):
    raw_var_labels = {
        't': 'Temperature',
        'c': 'Conductivity',
        'p': 'Pressure',
    }
    if include_do and 'do' in data_dict:
        raw_var_labels['do'] = 'Dissolved Oxygen'

    fig, axes = plt.subplots(len(raw_var_labels), 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.04)
    fig.align_ylabels()

    for i, (var_name, label) in enumerate(raw_var_labels.items()):
        if var_name in data_dict:
            axes[i].plot(data_dict[var_name], lw=1, color='gray')
            axes[i].axvline(start_index, color='g', linestyle='--', label='Start Trim')
            axes[i].axvline(finish_index, color='r', linestyle='--', label='Finish Trim')
            axes[i].set_ylabel(label)
            if i == 0:
                axes[i].legend(loc='best')

    plt.suptitle('Untrimmed Variables with Trim Indices')
    plt.show(block=True)

    return fig

#trim the data
def trim_data(data_dict, start, finish, time_trim = False, include_do=False):
    trimmed = {}

    time = data_dict['time']
    if not time_trim:
        dt = time[start:finish]
    else:
        dt = time[:(finish - start)].copy()

    t_trim = data_dict['t'][start:finish]  #Changed variable names so not overwriting previously defined variables
    c_trim = data_dict['c'][start:finish]
    p_trim = data_dict['p'][start:finish]
    do_trim = None
    if include_do and data_dict.get('do') is not None:
        do_trim = data_dict['do'][start:finish]

    # Calculate salinity after trimming
    s_trim = gsw.SP_from_C(10 * c_trim, t_trim, p_trim)

    #Calculate density after trimming
    SA = gsw.SA_from_SP(s_trim, p_trim, lon=0, lat=0)  # supply lon/lat if known
    CT = gsw.CT_from_t(SA, t_trim, p_trim)
    rho_trim = gsw.rho(SA, CT, p_trim)  # in-situ density [kg/m³]

 #Plot trimmed data
    var_labels = {
        't': 'Temperature',
        'c': 'Conductivity',
        'p': 'Pressure',
        's': 'Salinity',
        'rho': "Density (kg/m³)"
    }


    plot_vars = {
        't': t_trim,
        'c': c_trim,
        'p': p_trim,
        's': s_trim,
        'rho':rho_trim
    }

    if do_trim is not None:
        var_labels['do'] = 'Dissolved Oxygen'
        plot_vars['do'] = do_trim

    fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.04)
    fig.align_ylabels()

    for i, (var_name, data_array) in enumerate(plot_vars.items()):
        axes[i].plot(data_array, lw=1, color='k')
        axes[i].set_ylabel(var_labels[var_name])

    plt.suptitle('Trimmed In-Place Variables')
    plt.show(block=True)

    # Log trim indices
    print(f"Trimmed in-water indices: start={start}, finish={finish}")

    # Return trimmed arrays
    result = {
        'time': dt,
        't': t_trim,
        'c': c_trim,
        'p': p_trim,
        's': s_trim,
        'rho':rho_trim
    }
    if do_trim is not None:
        result['do'] = do_trim

    return result