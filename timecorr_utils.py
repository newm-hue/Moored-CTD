
def generate_time_from_bursts(data, time0_str, burstn=10, sample_sec=1, burst_min=1):
    time0 = dt.datetime.strptime(time0_str, '%Y-%m-%d %H:%M:%S.%f')
    sample_rate = dt.timedelta(seconds=sample_sec)
    burst_rate = dt.timedelta(minutes=burst_min)
    timen = len(data[:, 0])
    time_data = np.zeros_like(data[:, 0], dtype=dt.datetime)
    burst_count = 0
    for i in range(timen):
        if i == 0:
            time_data[i] = time0
            burst_count += 1
        elif burst_count == 0:
            time_data[i] = time_data[i - burstn] + burst_rate
            burst_count += 1
        elif burst_count != (burstn - 1):
            time_data[i] = time_data[i - 1] + sample_rate
            burst_count += 1
        else:
            time_data[i] = time_data[i - 1] + sample_rate
            burst_count = 0
    return time_data

def format_time_columns(data):
    timen = len(data[:, 0])
    time_data = np.zeros_like(data[:, 0], dtype=dt.datetime)
    for i in range(timen):
        time_data[i] = dt.datetime(*map(int, data[i, :6]))
    return time_data

def trim_time(time, start_idx):
    return time[start_idx:].copy()

def interpolate_spikes(time, spike_indices, limit=10):
    time[spike_indices] = np.nan
    time_series = pd.Series(time)
    interpolated = time_series.interpolate(method="linear", limit=limit, limit_direction='forward')
    return np.array(interpolated)

def generate_regular_time(time0, sample_rate, length):
    return np.array([time0 + dt.timedelta(seconds=i * sample_rate) for i in range(length)])

def generate_irregular_time(time0, intervals, rates):
    time_new = []
    for idx_range, rate in zip(intervals, rates):
        for i in idx_range:
            time_new.append(time0 + dt.timedelta(seconds=int(i) * rate))
    return np.array(time_new)

def apply_utc_offset(time, hours):
    return time + dt.timedelta(hours=hours)

def apply_time_offset(time, offset_days):
    return time + dt.timedelta(days=offset_days)
