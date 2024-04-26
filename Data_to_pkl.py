import numpy as np
import pickle
import time
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from datetime import datetime

start_run = 11
end_run = 11

output_file_name = 'Run11_list.pkl'
boards = 3

Tstamp_us = []
Brd = []
Ch = []
LG = []
HG = []
runNum = []

file_start_time = None

for run_number in range(start_run, end_run+1):
    input_file_name = f"Run{run_number}_list.txt"

    with open(input_file_name) as f:
        lines = f.read().split('\n')

    last_tstamp = 0  # Initialize last timestamp for continuity across files

    for i, line in enumerate(lines):
        if i == 6:  # Line containing the start time
            start_time_str = ' '.join(line.split()[4:-1])  # Extract start time string, excluding "UTC"
            file_start_time = datetime.strptime(start_time_str, "%a %b %d %H:%M:%S %Y")
            file_start_time = file_start_time.timestamp() * 1e6  # Convert to microseconds
        elif i > 8:
            data = line.split()

            if len(data) == 6:
                Tstamp_us.append(float(data[0]) + file_start_time)
                last_tstamp = float(data[0]) + file_start_time
                Brd.append(int(data[2]))
                Ch.append(int(data[3]))
                LG.append(int(data[4]))
                HG.append(int(data[5]))
                runNum.append(run_number)
            elif len(data) == 4:
                if last_tstamp is not None:
                    Tstamp_us.append(last_tstamp)
                Brd.append(int(data[0]))
                Ch.append(int(data[1]))
                LG.append(int(data[2]))
                HG.append(int(data[3]))
                runNum.append(run_number)

# Convert timestamps to datetime objects
Tstamp_utc = [datetime.utcfromtimestamp(tstamp_us / 1e6) for tstamp_us in Tstamp_us]

window_size = 100
current_event_id = 1
current_timestamp = Tstamp_us[0]
event_ids = []
for timestamp in Tstamp_us:
    if timestamp - current_timestamp <= window_size:
        event_ids.append(current_event_id)
    else:
        current_event_id += 1
        event_ids.append(current_event_id)
        current_timestamp = timestamp
        
event_counts = Counter(event_ids)
mask = [event_counts[event_id] == (boards*64) for event_id in event_ids]


Ch = [value for i, value in enumerate(Ch) if mask[i]]
LG = [value for i, value in enumerate(LG) if mask[i]]
HG = [value for i, value in enumerate(HG) if mask[i]]
Tstamp_us = [value for i, value in enumerate(Tstamp_us) if mask[i]]
Tstamp_utc = [value for i, value in enumerate(Tstamp_utc) if mask[i]]
event_ids = [value for i, value in enumerate(event_ids) if mask[i]]

events = pd.DataFrame({
    'Brd': Brd,
    'Ch': Ch,
    'LG': LG,
    'HG': HG,
    'Tstamp_us': Tstamp_us,
    'Tstamp_utc': Tstamp_utc,
    'event_ids': event_ids
})

with open(output_file_name, 'wb') as handle:
    pickle.dump(events, handle, protocol=pickle.HIGHEST_PROTOCOL)



