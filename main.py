# First version of creating condensed, function based script
### %%Section 1:  Import packages and setup environment
from fileinput import filename

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr
import gsw
import ctd
import isodate
from scipy import stats


# Set interactive backend for matplotlib
plt.switch_backend('Qt5Agg')  # Alternatives: 'WebAgg', 'TkAgg', etc.

#Read in the Utility module created for netCDF initialization
from netcdf_utils import
# (
#     (set_plot_style,
#     note,
#     update_comment,
#     OrdinalToDatetime,
#     processing_notes)
# )
###
#Section 2 Import metadata from yaml and calculate any derived variables
import yaml

with open("test.yaml", "r") as f:
    meta = yaml.safe_load(f)

#read out filename and directory for data upload
filename=meta["filename"]
directory=meta["directory"]


    # Derived values
    meta["year_1"] = meta["year_n"] + 1
    meta["year_str"] = str(meta["year_n"])
    meta["year_2str"] = str(meta["year_n"] - 2000)
    meta["lat"] = round(meta["latdeg"] + meta["latdec"] / 60, 6)
    meta["latstr"] = f'{meta["latdeg"]} {meta["latdec"]}'
    meta["lon"] = round(-(meta["londeg"] + meta["londec"] / 60), 6)
    meta["lonstr"] = f'{-meta["londeg"]} {meta["londec"]}'
    meta["dataset_id"] = f'{meta["instrument_type"]}_{meta["cruise"]}_{meta["mooring"]}_{meta["serial"]}_{meta["year_n"]}'

###
#read in data
cnv = filename.endswith(".cnv")
asc = filename.endswith(".asc")
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
###
   print(data.head())  # Check the first few rows


    raw_keys = ['temperature', 'conductivity', 'pressure', 'dates', 'times']

print("Raw keys:", raw_keys)