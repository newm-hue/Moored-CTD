
def generate_time_from_bursts(data, time0_str, burstn=10, sample_sec=1, burst_min=1):
    time0 = dt.datetime.strptime(time0_str, '%Y-%m-%d %H:%M:%S.%f')
    sample_rate = dt.timedelta(seconds=sample_sec) #time between samples
    burst_rate = dt.timedelta(minutes=burst_min) #time between bursts
    timen = len(data[:, 0]) #length of time data
    time_data = np.zeros_like(data[:, 0], dtype=dt.datetime) # empty array for time data
    burst_count = 0  #tracker for samples within burst
    for i in range(timen):
        if i == 0: # first time datapoint
            time_data[i] = time0
            burst_count += 1
        else:  # subsequent datapoints
            if burst_count == 0:
                time_data[i] = time_data[i - burstn] + burst_rate  # add burst interval
                burst_count += 1
            elif burst_count > 0 and burst_count != (burstn - 1):
                time_data[i] = time_data[i - 1] + sample_rate  # add sample interval
                burst_count += 1
            elif burst_count > 0 and burst_count == (burstn - 1):
                time_data[i] = time_data[i - 1] + sample_rate  # add sample interval
                burst_count = 0  # reset burst tracker
    return time_data

def format_time_columns(data):
    time0 = dt.datetime(int(data[0, 0]), int(data[0, 1]), int(data[0, 2]), int(data[0, 3]), int(data[0, 4]), int(data[0, 5]))  # initial time
    timen = len(data[:, 0])  #length of time data
    time_data = np.zeros_like(data[:, 0], dtype=dt.datetime) # empty array for time data
    for i in range(timen):
        time_data[i] = dt.datetime(int(data[i, 0]), int(data[i, 1]), int(data[i, 2]), int(data[i, 3]), int(data[i, 4]), int(data[i, 5]))  # format time at each step
    time = time_data.copy()

def interpolate_spikes(time, spike_indices, limit=10):
    time[spike_indices] = np.nan # set bad time data to NaN
    time_series = pd.Series(time)
    interpolated = time_series.interpolate(method="linear", limit=limit, limit_direction='forward') # interpolate over the time gap limit set in main code
    return np.array(interpolated) # set interpolated data to original array

