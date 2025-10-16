# First version of creating condensed, function based script
### %%Section 1:  Import packages
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

