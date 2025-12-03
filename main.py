# First version of creating condensed, function based script
### %%Section 1:  Import packages and setup environment

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr
import gsw
import ctd
import os
from pathlib import Path
import plotly.express as px
from pyrsktools import RSK

# Set interactive backend for matplotlib
plt.switch_backend('Qt5Agg')  # Alternatives: 'WebAgg', 'TkAgg', etc.

# Change to the script's directory
script_dir = Path(__file__).parent
os.chdir(script_dir)
print(os.getcwd())

###
#Section 2 Import metadata from yaml and calculate any derived variables
import yaml

with open(f"example/config_metadata.yaml", "r") as f:
    meta = yaml.safe_load(f)

#read out filename and directory for data upload
filename=meta["filename"]
directory=meta["directory_data"]


# Derived values
meta["year_1"] = meta["year_n"] + 1
meta["year_str"] = str(meta["year_n"])
meta["year_2str"] = str(meta["year_n"] - 2000)
meta["lat"] = round(meta["latdeg"] + meta["latdec"] / 60, 6)
meta["latstr"] = f'{meta["latdeg"]} {meta["latdec"]}'
meta["lon"] = round(-(meta["londeg"] + meta["londec"] / 60), 6)
meta["lonstr"] = f'{-meta["londeg"]} {meta["londec"]}'
meta["dataset_id"] = f'{meta["instrument_type"]}_{meta["cruise_number"]}_{meta["mooring_number"]}_{meta["serial_number"]}_{meta["year_n"]}'

print("Latitude:", meta["lat"])
print("Longitude:", meta["lon"])

#read out site and sitename from yaml
site=meta["site"]
subsite=meta["subsite"]
serial=meta["serial_number"]
year_str= str(meta["year_n"])
pres=meta["pres"]
figDir = meta["directory_figures"]
figRoot = f"{site}{year_str}{subsite}{pres}_{serial}"

#read in data  Altered Shannon and Annie's to incorporate rsk files
cnv = filename.endswith(".cnv")
asc = filename.endswith(".asc")
rbr = filename.endswith(".rsk")
data = None  # Initialize empty dataset
available_vars = []  # List to store available variable names

if cnv:
    data = ctd.from_cnv(rf'{directory}{filename}')  # Load .cnv data
    raw_keys = data.keys()  # Get column names
elif asc:
    data = pd.read_csv(
	   rf'{directory}{filename}',
	   header=None,
	   names=['temperature', 'conductivity', 'pressure', 'dates', 'times'],
	   skiprows=60, #skip metadata
	   encoding='utf-8',
	   skip_blank_lines=True
   )
    raw_keys = ['temperature', 'conductivity', 'pressure', 'dates', 'times']
elif rsk:
    rsk_obj = RSK(rf'{directory}{filename}')
    rsk_obj.open()
    rsk_obj.readdata()
    df = pd.DataFrame(rsk_obj.data)
    timestamps = df['timestamp']
    dt_array = pd.to_datetime(df['timestamp'])
    df['dates'] = dt_array.dt.strftime('%d %b %Y')
    df['times'] = dt_array.dt.strftime('%H:%M:%S')
    data = df[['temperature', 'conductivity', 'pressure', 'dates', 'times']]

    raw_keys = data.columns.tolist()



print(data.head())

print("Raw keys:", raw_keys)

###
#%%Section 4: Save raw data as NetCDF altered Shannon and Annie's to incorporate rsk files
raw_ds = xr.Dataset()

if cnv:
    dim = 'depth'
    desc = 'Raw CNV data'
    comment = "Converted from CNV to NetCDF; no QC, trimming, or calibration applied."
    source_type = 'CNV'
elif asc:
    dim = 'time'
    desc = 'Raw ASC data'
    comment = "Converted from ASC to NetCDF; no QC, trimming, or calibration applied."
    source_type = 'ASC'
elif rsk:
    dim = 'time'
    desc = 'Raw RSK data'
    comment = "Converted from RSK to NetCDF; no QC, trimming, or calibration applied."
    source_type = 'RSK'

for key in raw_keys:
    safe_key = key.replace("/", "_")
    if cnv:
        raw_ds[safe_key] = ([dim], data[key].values)
    elif asc:
        raw_ds[safe_key] = ([dim], data[key])
    elif rsk:
        raw_ds[safe_key] = ([dim], data[key].to_numpy())

raw_ds.attrs.update({
    'source_file': filename,
    'description': f'{desc} before any trimming or drift correction',
    'comment': comment,
    'source_type': source_type
})

raw_output_path = os.path.join(directory, filename.replace('.cnv', '_raw.nc').replace('.asc', '_raw.nc').replace('.rsk', '_raw.nc')
)
raw_ds.to_netcdf(raw_output_path)
print(f"Raw data saved to {raw_output_path}")

###
#%%Section 5: Extract Variables
import yaml
with open("bin/variable_map.yaml", "r") as f:
    var_map = yaml.safe_load(f)

data_ = data.copy()

vars_dict = {} # Extract available variables using the mapping
for raw_var, new_var in var_map.items():
    if raw_var in raw_keys:
        vars_dict[new_var] = data_[raw_var].values if isinstance(data_, pd.DataFrame) else data_[raw_var]
        if new_var not in available_vars:
            available_vars.append(new_var)

#Assign core variables
t = vars_dict.get('t')
c = vars_dict.get('c')
do = vars_dict.get('do', None)
time = vars_dict.get('time')

#Pressure: for CNV, use index; for ASC, get from vars_dict
p = data_.index.values if cnv else vars_dict.get('p')
#print("Extracted Pressure Data:", p)

#entire group below is only for asc files  MN
# --- Time Conversion for ASC Files ---
if asc or rsk and 'dates' in raw_keys and 'times' in raw_keys:  #added dates MN
    dates = data_['dates']
    times = data_['times']


datetime_array = np.array([
		dt.datetime.strptime(f"{str(date).strip()} {str(time_str).strip()}", "%d %b %Y %H:%M:%S")
		for date, time_str in zip(dates, times)
	])

# Extract year from first date    added MN
year_n = datetime_array[0].year


# Convert to Julian Day   new naming conventions  MN
reference_date = dt.datetime(year_n - 1, 12, 31)
julian_array = np.array([
    	(dt_obj - reference_date).days + (dt_obj - reference_date).seconds / 86400
   		for dt_obj in datetime_array
	])
#assign time to Julian Day  added MN
time=julian_array

print("Available Variables:", available_vars)

###
#%%Section 6: Time Data Check
fig, ax0 = plt.subplots(1,1,figsize=(12,8)); #fig.tight_layout();
ax0.plot(range(0, len(time)), time, lw=1, color='k'); #change second time to timej
ax0.set_xlabel('Index'); ax0.set_ylabel('Time')
plt.title("Time Check!")
plt.show(block=True)

#need to make sure plot is closed before moving to next section

###
#Section 7 Time Correction data

no_time_data = False        # if time data does not exist or is entirely incorrect, and have initialisation time from data or metadata
format_time_data = False    # if time data in wrong format e.g. separate year, month, day, hour, min, sec values
time_trim = False           # if totally incorrect time data on either end of record (e.g. data present from previous deployment); time out-of-water can still be correct!
time_spike = False          # if bad time data somewhere in record
fix_irregular_data = False  # if cnv jd time data exists but incorrect, and need to manufacture NEW time data (NOTE: can typically just use 'no_time_data' case)
UTC_offset = False          # e.g. add 7 hours if obviously not synced to UTC upon deployment; must already be in datetime format
time_offset = False         # if need to add or remove regular time offset throughout record; must already be in datetime format

from bin.timecorr_utils import *

# Generate time from burst sampling
if no_time_data:
    time = generate_time_from_bursts(data, '2019-10-04 13:00:03.000000') #Initial time
    print(f'Insturment started (not deployed): {time[0]}, Instrument stopped (not recovered): {time[-1]}')

# Format time from separate columns
if format_time_data:
    time = format_time_columns(data)
    print(f'Instrument started (not deployed): {time[0]}, Instrument stopped (not recovered): {time[-1]}')

# Trim time  Unchanged from original code
if time_trim:
    time = time[3849:].copy() # check time, p, t, c for proper trim indices
    print(f'Instrument started (not deployed): {time[0]}')
    print(f'Instrument stopped (not recovered): {time[-1]}')

# Interpolate over spikes
if time_spike:
    time = interpolate_spikes(time, np.r_[35232:35235]) # indices of spike; use len(time) for end of record if necessary
    print(f'Instrument started (not deployed): {time[0]}, Instrument stopped (not recovered): {time[-1]}')

if UTC_offset:   #unchanged from original script
    time = time.copy() + dt.timedelta(hours=7)
    print(f'UTC adjusted instrument started (not deployed): {time[0]}')
    print(f'UTC adjusted instrument stopped (not recovered): {time[-1]}')

if time_offset:   #unchanged from original script
	offset = dt.timedelta(days=-365)
	time = time.copy() + offset
	print(f'Offset adjusted instrument started (not deployed): {time[0]}')
	print(f'Offset adjusted instrument stopped (not recovered): {time[-1]}')

###
#Section 8 Trimming data
# #Plot untrimmed data
from bin.trim_utils import plot_available_vars, finish, start

data_dict = {
        't': t,
        'c': c,
        'p': p
    }
fig = plot_available_vars(
      available_vars=['t','c','p'],
      data_dict=data_dict,
      include_do=False
 )

fig.savefig(f"{figDir}{figRoot}_raw_data.png", dpi=300, bbox_inches="tight")

##
#%%Section 9 Plot trim indices

from bin.trim_utils import plot_trimmed_var

fig = plot_trimmed_var(
    data_dict=data_dict,
    start_index=start,
    finish_index=finish,
    include_do=False
)

fig.savefig(f"{figDir}{figRoot}_trim_indices.png", dpi=300, bbox_inches="tight")

##
#Section 10
#Trim data and plot trimmed data
from bin.trim_utils import trim_data
data_dict = {'time': time, 't': t, 'c': c, 'p': p, 'do': do}
trimmed = trim_data(data_dict, start, finish, time_trim=False, include_do=False)
print({k: v.shape for k, v in trimmed.items()})


# CF-compliant unit mapping for common oceanographic variables
unit_map = {
    "time": "seconds since 2021-12-31",           # since deployment NEEDS TO BE EDITED EACH TIME
    "t": "degree_Celsius",                        # temperature
    "c": "S/m",                                   # conductivity
    "p": "dbar",                                  # pressure
    "do": "ml/l",                                 # dissolved oxygen
    "depth": "meter",                             # depth below sea surface
    "lat": "degrees_north",                       # latitude
    "lon": "degrees_east"                         # longitude
}

#save trimmed and metadata as NetCDF, try to mimic Shannon and Annie's format for creation
#added in NetCDF compliant cooridnate variable creation
# Choose dimension name
dim = "time"

# Create trimmed data set from trimmed variables
trimmed_ds = xr.Dataset()

# Add coordinate variables
if "time" in trimmed:
    trimmed_ds["time"] = ("time", trimmed["time"])
    trimmed_ds["time"].attrs.update({
        "standard_name": "time",
        "long_name": "time",
        "units": "seconds since start"  # adjust to actual units
    })

# Create NetCDF compliant depth variable
if "p" in trimmed:  # assuming 'p' is pressure in dbar
    depth = gsw.z_from_p(trimmed["p"], meta["lat"]) * -1 #Gibbs Seawater depth conversion from pressure
    trimmed_ds["depth"] = ("time", depth)
    trimmed_ds["depth"].attrs.update({
        "standard_name": "depth",
        "long_name": "depth below sea surface",
        "units": "meter",
        "positive": "down"
    })

# Latitude and longitude as scalar coordinates
trimmed_ds["lat"] = meta["lat"]
trimmed_ds["lat"].attrs.update({
    "standard_name": "latitude",
    "long_name": "latitude",
    "units": "degrees_north"
})

trimmed_ds["lon"] = meta["lon"]
trimmed_ds["lon"].attrs.update({
    "standard_name": "longitude",
    "long_name": "longitude",
    "units": "degrees_east"
})

for key, arr in trimmed.items():
    if key in ["time", "p"]:  # already handled as coordinates
        continue
    safe_key = key.replace("/", "_")
    trimmed_ds[safe_key] = ("time", arr)
    trimmed_ds[safe_key].attrs.update({
        "long_name": safe_key,
        "units":  unit_map.get(safe_key, "unknown")
    })
# Add metadata as global attributes
trimmed_ds.attrs.update({
    "source_file": meta["filename"],
    "description": "Trimmed dataset after applying start/finish indices",
    "comment": "Converted to NetCDF; trimming applied, no QC or calibration",
    **meta  # include all YAML + derived metadata
})

# Output path
outpath = os.path.join(meta["directory_data"], meta["filename"].replace(".cnv", "_trimmed.nc").replace(".asc", "_trimmed.nc"))

# Save to NetCDF
trimmed_ds.to_netcdf(outpath)
print(f"Trimmed data saved to {outpath}")
###
#%%Section 11A: Temperature Salinity Plot
#need to pull t and s data from previous sections
s = trimmed['s']
t = trimmed['t']

fig_ts, ax_ts = plt.subplots(figsize=(8, 6))
sc = ax_ts.scatter(
    s, t,
    c=np.arange(len(t)), cmap="viridis",
    edgecolor='k', alpha=0.7
)
ax_ts.set_xlabel('Salinity')
ax_ts.set_ylabel('Temperature')
ax_ts.set_title('T-S Diagram')
cbar = plt.colorbar(sc, label='Index')



plt.savefig(f"{figDir}/{figRoot}_TS_plot.png", dpi=300, bbox_inches='tight')
plt.show(block=True)
###
#%%Section 11B: Temperature Salinity Plot, Interactive

# Assuming s and t are your salinity and temperature arrays
df = pd.DataFrame({
    'Salinity': trimmed['s'],
    'Temperature': trimmed['t'],
    'Index': np.arange(len(t))  # This is what you'll see on hover
})

# Create interactive scatter plot
fig = px.scatter(
    df,
    x='Salinity',
    y='Temperature',
    color='Index',
    color_continuous_scale='viridis',
    title='T-S Diagram',
    hover_data={'Index': True}  # Show index on hover
)

fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))

# Show the plot
fig.show()

###
#%%Section 12: Manual Data Inspection with Spike Suggestions

detect_spikes_t = True    # Detect spikes in Temperature
detect_spikes_c = True    # Detect spikes in Conductivity
detect_spikes_do = False  # Optional: Detect in DO
detect_spikes_p = False   # Optional: Detect in Pressure
detect_spikes_rho = False #Optional: Detect spikes in Density

# Spike detection parameters
#n1 = level 1 std error, n2 = level 2 std error, block = number of points in assesment block
n1 = 2; n2 = 20; block = 400


#NOTE: These plots are for manual review only.
# Spikes are algorithmically suggested, but no data is removed or altered.
# Use this as a guide to record spike locations in your lab notebook per WHOCE QC flagging.

#need to pull trimmed data from previous sections
s = trimmed['s']
t = trimmed['t']
c = trimmed['c']
#do = trimmed['do']
p = trimmed['p']
rho= trimmed['rho']

def detect_spikes(data, n1, n2, block):
    original = pd.Series(data)
    processed = ctd.processing.despike(original.copy(), n1=n1, n2=n2, block=block)
    return np.where(~np.isclose(original, processed, equal_nan=True))[0]

# Run spike detection (as guidance only)
spike_indices_t = detect_spikes(t, n1, n2, block) if detect_spikes_t else []
spike_indices_c = detect_spikes(c, n1, n2, block) if detect_spikes_c else []
spike_indices_do = detect_spikes(do, n1, n2, block) if detect_spikes_do and 'do' in trimmed_data else []
spike_indices_p = detect_spikes(p, n1, n2, block) if detect_spikes_p and 'p' in trimmed_data else []
spike_indices_rho = detect_spikes(rho, n1, n2, block) if detect_spikes_rho and 'rho' in trimmed_data else []

# Plotting
var_labels = {
    't': 'Temperature',
    'c': 'Conductivity',
    'do': 'Dissolved Oxygen',
    'p': 'Pressure',
    'rho': 'Density'
}

plot_vars = [var for var in ['t', 'c', 'do', 'p','rho'] if eval(var) is not None]
spike_indices = {
    't': spike_indices_t,
    'c': spike_indices_c,
    'do': spike_indices_do,
    'p': spike_indices_p,
    'rho': spike_indices_rho
}

fig, axes = plt.subplots(len(plot_vars), 1, figsize=(12, 8), sharex=True)
fig.subplots_adjust(hspace=0.04)
fig.align_ylabels()

if len(plot_vars) == 1:
    axes = [axes]

for i, var in enumerate(plot_vars):
    data = eval(var)
    axes[i].plot(data, lw=1, color='k')

    idx = spike_indices[var]
    if len(idx) > 0:
        axes[i].scatter(idx, data[idx], color='red', zorder=5)

    axes[i].set_ylabel(var_labels[var])

plt.suptitle('Manual Data Inspection with Spike Suggestions')
plt.savefig(f'{figDir}{figRoot}_ManualSpikeReview.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

###
#%%Section 13: Apply WHOCE CTD Flags to variables (flags stored in YAML and read in)
from bin.qc_utils import load_flagged_arrays

from bin.qc_utils import merge_flags
variables = {
    'temperature': trimmed['t'],
    'conductivity': trimmed['c'],
    'pressure': trimmed['p']
}

flag_arrays = load_flagged_arrays(rf'example/config_process.yaml', variables)
flagged_df = merge_flags(trimmed, flag_arrays)
flagged_df = pd.DataFrame(flagged_df)

#plot flagged data
from bin.qc_utils import plot_flagged_data
fig, axes = plot_flagged_data(flagged_df, include_do=False)




## section 14  Create final NetCDF
#Load in metadata from YAML and merge with NetCDF attributes
def load_metadata(yaml_file):
    with open(yaml_file, 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata

def apply_metadata(ds, metadata):
    ds.attrs.update(metadata)
    return ds

 #Load variable attributes for t,c,p   from Shannon and Annie code
def apply_variable_attributes(ds, metadata, t, c, p):
    instrument_depth = metadata.get("instrument_depth")
    serial_number = metadata.get("serial_number")

    # Temperature
    if "temperature" in ds:
        ds["temperature"].attrs.update({
            "units": "degree_C",
            "long_name": "temperature",
            "standard_name": "sea_water_temperature",
            "sdn_parameter_urn": "SDN:P01::TEMPPR01",
            "sdn_parameter_name": "Temperature of the water body",
            "sdn_uom_urn": "UPAA",
            "sdn_uom_name": "Degrees Celsius",
            "generic_name": "temperature",
            "coverage_content_type": "physicalMeasurement",
            "reference_scale": "ITS-90",
            "sensor_type": "SBE37-SM",
            "sensor_depth": instrument_depth,
            "serial_number": serial_number,
            "_FillValue": 1e35,
            "data_min": float(np.nanmin(t)),
            "data_max": float(np.nanmax(t)),
        })

    # Conductivity
    if "conductivity" in ds:
        ds["conductivity"].attrs.update({
            "units": "S m-1",
            "long_name": "conductivity",
            "standard_name": "sea_water_electrical_conductivity",
            "sdn_parameter_urn": "SDN:P01::CNDCZZ01",
            "sdn_uom_urn": "UECA",
            "sdn_uom_name": "Siemens per metre",
            "generic_name": "conductivity",
            "coverage_content_type": "physicalMeasurement",
            "sensor_type": "SBE37-SM",
            "sensor_depth": instrument_depth,
            "serial_number": serial_number,
            "_FillValue": 1e35,
            "data_min": float(np.nanmin(c)),
            "data_max": float(np.nanmax(c)),
        })

    # Pressure
    if "pressure" in ds:
        ds["pressure"].attrs.update({
            "units": "decibars",
            "long_name": "pressure",
            "standard_name": "sea_water_pressure",
            "sdn_parameter_urn": "SDN:P01::PRESPR01",
            "sdn_uom_urn": "UPDB",
            "sdn_uom_name": "decibars",
            "generic_name": "pressure",
            "coverage_content_type": "physicalMeasurement",
            "sensor_type": "SBE37-SM",
            "sensor_depth": instrument_depth,
            "serial_number": serial_number,
            "_FillValue": 1e35,
            "data_min": float(np.nanmin(p)),
            "data_max": float(np.nanmax(p)),
        })

    return ds

#Load functions and merge
metadata = load_metadata("example/config_metadata.yaml")
ds = xr.Dataset.from_dataframe(flagged_df)
ds = apply_metadata(ds, metadata)
ds = apply_variable_attributes(ds, metadata, flagged_df['t'].values, flagged_df['c'].values, flagged_df['p'].values)

#QC flag attributes  from Shannon and Annies code
ds['flag_t'].attrs.update({
    "long_name": "quality flag for temperature",
    "standard_name": "sea_water_temperature status_flag",
    "flag_values": [2, 3, 4, 5],
    "flag_meanings": "good questionable bad missing",
})
ds['flag_c'].attrs.update({
    "long_name": "quality flag for conductivity",
    "standard_name": "sea_water_electrical_conductivity status_flag",
    "flag_values": [2, 3, 4, 5],
    "flag_meanings": "good questionable bad missing",
})
ds['flag_p'].attrs.update({
    "long_name": "quality flag for pressure",
    "standard_name": "sea_water_pressure status_flag",
    "flag_values": [2, 3, 4, 5],
    "flag_meanings": "good questionable bad missing",
})

# Output path
outpath = os.path.join(metadata["directory_data"],
                       metadata["filename"].replace(".cnv", "_flagged.nc")
                                           .replace(".asc", "_flagged.nc")
                                           .replace(".rsk", "_flagged.nc")
)

# Save to NetCDF
ds.to_netcdf(outpath)
print(f"Flagged data saved to {outpath}")



