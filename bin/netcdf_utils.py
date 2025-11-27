# creating a utility module for the initialization of netcdf creation
###
import datetime as dt
import matplotlib.pyplot as plt


# Set interactive backend for matplotlib
plt.switch_backend('Qt5Agg')  # Alternatives: 'WebAgg', 'TkAgg', etc.

def set_plot_style(font="Times New Roman", size=14):
    plt.ion()
    plt.rcParams["font.family"] = font
    plt.rcParams["font.size"] = size

# Function to append processing to comment attribute in the final NetCDF
processing_notes = []


def update_comment(ds, message: str):
    # append to the global 'comment' attribute, creating it if needed
    prev = str(ds.attrs.get("comment", "")).strip()
    ds.attrs["comment"] = message if not prev else f"{prev}; {message}"


def note(msg: str):
    processing_notes.append(msg)


# Function to convert time ordinal to datetime
def OrdinalToDatetime(ordinal):
    plaindate = dt.date.fromordinal(int(ordinal))
    date_time = dt.datetime.combine(plaindate, dt.datetime.min.time())
    return date_time + dt.timedelta(days=ordinal - int(ordinal))