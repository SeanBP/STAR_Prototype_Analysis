import numpy as np
import pickle
import time
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from datetime import datetime
from iminuit import Minuit, cost
from iminuit.cost import LeastSquares
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta

start_run = 156
end_run = 156

output_file_name = 'Run156'
geometry = 'Prototype_Geometry_5-3-24.txt'
ptrig = pd.read_pickle(r'PTRIG_154.pkl')
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

Brd = [value for i, value in enumerate(Brd) if mask[i]]
Ch = [value for i, value in enumerate(Ch) if mask[i]]
LG = [value for i, value in enumerate(LG) if mask[i]]
HG = [value for i, value in enumerate(HG) if mask[i]]
Tstamp_us = [value for i, value in enumerate(Tstamp_us) if mask[i]]
Tstamp_utc = [value for i, value in enumerate(Tstamp_utc) if mask[i]]
event_ids = [value for i, value in enumerate(event_ids) if mask[i]]

# Define empty lists for x, y, and z positions
x_pos = []
y_pos = []
z_pos = []

# Read coordinates from the text file
with open(geometry, "r") as file:
    coordinates = [eval(line.strip()) for line in file]

# Calculate positions based on Ch and Brd values
for i in range(len(Ch)):
    total_channel = Ch[i] + (Brd[i] * 64)
    if total_channel < len(coordinates):
        x, y, z = coordinates[total_channel]
        x_pos.append(x)
        y_pos.append(y)
        z_pos.append(z)
    else:
        print(f"Error: total_channel value {total_channel} is out of range.")
        
Brd = np.array(Brd)
Ch = np.array(Ch)
LG = np.array(LG)
HG = np.array(HG)
Tstamp_us = np.array(Tstamp_us)
Tstamp_utc = np.array(Tstamp_utc)
event_ids = np.array(event_ids)
x_pos = np.array(x_pos)
y_pos = np.array(y_pos)
z_pos = np.array(z_pos)
ptrig_Brd = np.array(ptrig["board"])
ptrig_Ch = np.array(ptrig["channel"])
ptrig_LG_mean = np.array(ptrig["LG_mean"])
ptrig_LG_sigma = np.array(ptrig["LG_sigma"])
ptrig_HG_mean = np.array(ptrig["HG_mean"])
ptrig_HG_sigma = np.array(ptrig["HG_sigma"])


# Get unique event IDs and their indices
unique_event_ids, event_indices = np.unique(event_ids, return_index=True)

# Calculate average Tstamp_us for each event ID
average_timestamps = []
for i, event_id in enumerate(unique_event_ids):
    # Calculate the range of indices for the current event ID
    start_index = event_indices[i]
    end_index = event_indices[i+1] if i < len(event_indices) - 1 else len(Tstamp_us)
    
    # Calculate average timestamp for the current event ID
    timestamps_for_event = Tstamp_us[start_index:end_index]
    average_timestamp = np.mean(timestamps_for_event)
    average_timestamps.append(average_timestamp)
average_timestamps_utc = [datetime.utcfromtimestamp(tstamp_us / 1e6) for tstamp_us in average_timestamps]
average_timestamps = np.array(average_timestamps) - average_timestamps[0]

fig = plt.figure(figsize=(8, 8))
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'

# Calculate the range of your data
data_range = max(average_timestamps) - min(average_timestamps)

# Calculate the number of bins needed to represent 1-minute intervals
num_mins = data_range / (60 * 1e6)  # Convert microseconds to minutes
num_bins = int(np.ceil(num_mins))


# Calculate the bin edges
bin_edges = np.linspace(min(average_timestamps), max(average_timestamps), num_bins + 1 )

# Compute histogram
h, bins = np.histogram(average_timestamps, bins=bin_edges)

bin_centers_us = 0.5 * (bins[:-1] + bins[1:])


# Convert bin centers to hours
bin_centers_mins = bin_centers_us / (60 * 1e6)  # Convert microseconds to minutes

# Convert counts to rates (Hz)
bin_width_mins = (bins[1] - bins[0]) / (60 * 1e6)  # Convert microseconds to minutes

h_rate = h / 60 #convert counts per minute to counts per second

# Define the start time of the file
start_time = average_timestamps_utc[0]

# Calculate the bin centers in datetime format
bin_centers_datetime = [start_time + timedelta(microseconds=us) for us in bin_centers_us]

# Plot
plt.errorbar(bin_centers_datetime, h_rate, yerr=np.zeros(len(h_rate)), fmt='o', ms=3, ecolor='tab:blue', color='tab:blue', capsize=0, elinewidth=1, markeredgewidth=0)

date_format = DateFormatter("%m-%d %H:%M")
plt.gca().xaxis.set_major_formatter(date_format)

# Set x-limits to be within a certain time period
#plt.xlim(datetime(2024, 4, 24, 8, 0), datetime(2024, 4, 24, 12, 0))  # Example range from 6:30 to 7:00

#plt.ylim(0, 3)
plt.xlabel("Time [UTC]")
plt.ylabel("Rate [Hz]")
plt.xticks(rotation=45)
plt.savefig(output_file_name+"_timestamps.pdf", format="pdf", bbox_inches="tight")

def plot_histograms(board_number, events, ch, brd, title, high):
    # Create subplots for each channel
    fig, axs = plt.subplots(4, 16, figsize=(32, 8), sharey=True, sharex=True)
    fig.suptitle(title, fontsize=40, y=1)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.bbox'] = 'tight'

    for i in range(4):
        for j in range(16):
            channel_number = i * 16 + j
            ax = axs[i, j] if len(axs.shape) > 1 else axs[j]  # For 1D array indexing
            mask = (ch == channel_number) & (brd == board_number) & (events != 0)

            if len(events[mask]) != 0:
                # Create histogram
                h, bins = np.histogram(events[mask], bins=50, range=(0, high))
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                errors = np.sqrt(h)

                # Plot error bar plot
                ax.errorbar(bin_centers, h, yerr=errors, fmt='o', ecolor='tab:blue', color='tab:blue', capsize=0,
                            elinewidth=1, markeredgewidth=0)

                initial_params = [max(h), np.mean(events[mask]), np.std(events[mask])]
                mask = h > 0

                ax.set_yscale("log")
                ax.set_ylim(0.1, 10e4)

            # Set labels
            ax.set_title(f'Ch {channel_number}', fontsize=20, pad=-18, loc='center')
            if i == 3:  # Bottom most row
                ax.set_xlabel('Value [ADC]', fontsize=20)
            if j == 0:  # Left most column
                ax.set_ylabel('Count', fontsize=20)

    # Adjust layout
    plt.tight_layout(pad=0.0)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.savefig(title+".pdf", format="pdf", bbox_inches="tight")

plot_histograms(0, HG, Ch, Brd, output_file_name+"_Raw_HG_Board_0", 10000)
plot_histograms(1, HG, Ch, Brd, output_file_name+"_Raw_HG_Board_1", 10000)
plot_histograms(2, HG, Ch, Brd, output_file_name+"_Raw_HG_Board_2", 10000)
plot_histograms(0, LG, Ch, Brd, output_file_name+"_Raw_LG_Board_0", 10000)
plot_histograms(1, LG, Ch, Brd, output_file_name+"_Raw_LG_Board_1", 10000)
plot_histograms(2, LG, Ch, Brd, output_file_name+"_Raw_LG_Board 2", 10000)

LG_pedsub = np.zeros_like(LG)  # Initialize LG_pedsub array with zeros
HG_pedsub = np.zeros_like(HG)  # Initialize HG_pedsub array with zeros

for i in range(len(LG)):
    # Find corresponding board and channel
    board = Brd[i]
    channel = Ch[i]
    
    # Find index of board and channel in ptrig arrays
    index = np.where((ptrig_Brd == board) & (ptrig_Ch == channel))[0]
    if len(index) > 0:  # If a matching index is found
        index = index[0]
        # Calculate the difference
        differenceLG = LG[i] - ptrig_LG_mean[index]
        differenceHG = HG[i] - ptrig_HG_mean[index]
        # Check if the difference is less than the corresponding sigma value
        if ptrig_LG_sigma[index] == 100:
            if np.abs(differenceLG) < 4 * 10:
                LG_pedsub[i] = 0
            else:
                LG_pedsub[i] = differenceLG
        elif np.abs(differenceLG) < 5 * ptrig_LG_sigma[index]:
            LG_pedsub[i] = 0
        else:
            # Subtract corresponding LG_mean value from LG
            LG_pedsub[i] = differenceLG
            
        if ptrig_HG_sigma[index] == 100:
            if np.abs(differenceHG) < 4 * 10:
                HG_pedsub[i] = 0
            else:
                HG_pedsub[i] = differenceHG
        elif np.abs(differenceHG) < 5 * ptrig_HG_sigma[index]:
            HG_pedsub[i] = 0
        else:
            # Subtract corresponding LG_mean value from LG
            HG_pedsub[i] = differenceHG
    else:
        # Handle cases where no matching index is found
        print(f"No matching index found for board {board} and channel {channel}")

        
unique_event_ids, event_counts = np.unique(event_ids, return_counts=True)

# Use numpy's bincount to efficiently calculate the sum of LG values for each unique event_id
LG_sums = np.bincount(event_ids, weights=LG_pedsub)
HG_sums = np.bincount(event_ids, weights=HG_pedsub)

# Since bincount will include zeros for event_ids that are not present, 
# we need to trim the resulting LG_sums array to match the length of unique_event_ids
LG_sums = LG_sums[unique_event_ids]
HG_sums = HG_sums[unique_event_ids]


fig, axs = plt.subplots(1, 2, figsize=(12, 4))
h, bins = np.histogram(LG_sums, bins=50, range=(0,130000))
bin_centers = 0.5 * (bins[:-1] + bins[1:])
errors = np.sqrt(h)
axs[0].errorbar(bin_centers, h, yerr=errors, fmt='o', ecolor='tab:blue', color='tab:blue', capsize=0, elinewidth=1, markeredgewidth=0)
axs[0].set_yscale("log")
axs[0].set_ylabel("Count")
axs[0].set_xlabel("ADC Sum")
axs[0].set_title(output_file_name+" LG ADC Event Sum")
h, bins = np.histogram(HG_sums, bins=50, range=(0,130000))
bin_centers = 0.5 * (bins[:-1] + bins[1:])
errors = np.sqrt(h)
axs[1].errorbar(bin_centers, h, yerr=errors, fmt='o', ecolor='tab:blue', color='tab:blue', capsize=0, elinewidth=1, markeredgewidth=0)
axs[1].set_yscale("log")
axs[1].set_ylabel("Count")
axs[1].set_xlabel("ADC Sum")
axs[1].set_title(output_file_name+" HG ADC Event Sum")
plt.savefig(output_file_name+"_ADCSums.pdf", format="pdf", bbox_inches="tight")

highHGEvts = [np.zeros(64), np.zeros(64), np.zeros(64)]
highLGEvts = [np.zeros(64), np.zeros(64), np.zeros(64)]
numEvts = len(list(set(event_ids)))

for brd in range(3):
    Brd_mask = (Brd == brd)
    LG_brd = LG[Brd_mask]
    HG_brd = HG[Brd_mask]
    Ch_brd = Ch[Brd_mask]
    for ch in range(64):
        Ch_mask = (Ch_brd == ch)
        LG_ch = LG_brd[Ch_mask]
        HG_ch = HG_brd[Ch_mask]
            
        LGPed_mean = ptrig_LG_mean[(ptrig_Brd == brd) & (ptrig_Ch == ch)][0]
        LGPed_sigma = ptrig_LG_sigma[(ptrig_Brd == brd) & (ptrig_Ch == ch)][0]
        HGPed_mean = ptrig_HG_mean[(ptrig_Brd == brd) & (ptrig_Ch == ch)][0]
        HGPed_sigma = ptrig_HG_sigma[(ptrig_Brd == brd) & (ptrig_Ch == ch)][0]
            
        LG_above_ped = LG_ch[(LG_ch - LGPed_mean) > (5 * LGPed_sigma)]
        HG_above_ped = HG_ch[(HG_ch - HGPed_mean) > (5 * HGPed_sigma)]
        highHGEvts[brd][ch] = len(HG_above_ped)
        highLGEvts[brd][ch] = len(LG_above_ped)
        
for brd in range(3):
    for ch in range(64):
        highHGEvts[brd][ch] = highHGEvts[brd][ch] / numEvts
        highLGEvts[brd][ch] = highLGEvts[brd][ch] / numEvts

# Create a figure and axes for subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)

# Plot each set of data in a subplot
for i in range(3):
    axs[0, i].scatter(range(64), highLGEvts[i])
    axs[0, i].set_title(output_file_name + " LG Events >5*sigma Board " + str(i))
    axs[0, i].set_xlabel("Ch#")
    axs[0, i].set_ylabel("Count")
    axs[0, i].set_ylim(0, None)

    axs[1, i].scatter(range(64), highHGEvts[i])
    axs[1, i].set_title(output_file_name + " HG Events >5*sigma Board " + str(i))
    axs[1, i].set_xlabel("Ch#")
    axs[1, i].set_ylabel("Count")
    axs[1, i].set_ylim(0, None)

# Adjust layout
plt.tight_layout()
plt.savefig(output_file_name+"_Ch_Hit_Freq.pdf", format="pdf", bbox_inches="tight")

LG_hits = []
HG_hits = []

for evt in set(event_ids):
    LG_evt_hits = 0
    HG_evt_hits = 0
    
    LGevt_mask = (event_ids == evt) & (LG_pedsub != 0)
    HGevt_mask = (event_ids == evt) & (HG_pedsub != 0)
    
    LG_hits.append(len(LG_pedsub[LGevt_mask]))
    HG_hits.append(len(HG_pedsub[HGevt_mask]))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

h, bins = np.histogram(LG_hits, bins=32, range=(0,64*3))
bin_centers = 0.5 * (bins[:-1] + bins[1:])
errors = np.sqrt(h)
axs[0].errorbar(bin_centers, h / numEvts, yerr=errors / numEvts, fmt='o', ecolor='tab:blue', color='tab:blue', capsize=0,
                            elinewidth=1, markeredgewidth=0)
axs[0].set_title(output_file_name+" LG Hit Mult. > 5*sigma")
axs[0].set_xlabel("Hit Multiplicity")
axs[0].set_ylabel("Count")
axs[0].set_yscale("log")

h, bins = np.histogram(HG_hits, bins=32, range=(0,64*3))
bin_centers = 0.5 * (bins[:-1] + bins[1:])
errors = np.sqrt(h)
axs[1].errorbar(bin_centers, h / numEvts, yerr=errors / numEvts, fmt='o', ecolor='tab:blue', color='tab:blue', capsize=0,
                            elinewidth=1, markeredgewidth=0)
axs[1].set_title(output_file_name+" HG Hit Mult. > 5*sigma")
axs[1].set_xlabel("Hit Multiplicity")
axs[1].set_ylabel("Count")
axs[1].set_yscale("log")
plt.savefig(output_file_name+"_Hit_Mult.pdf", format="pdf", bbox_inches="tight")

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot on the first subplot
quadmesh_0 = axs[0].hist2d(x_pos[LG_pedsub != 0], y_pos[LG_pedsub != 0], bins=(20,20))
axs[0].set_xlabel("X Position [mm]")
axs[0].set_ylabel("Y Position [mm]")
axs[0].set_title(output_file_name+" LG Hits >5*sigma")
fig.colorbar(quadmesh_0[3], ax=axs[0])

# Plot on the second subplot
quadmesh_1 = axs[1].hist2d(x_pos[LG_pedsub != 0], z_pos[LG_pedsub != 0], bins=(20,20))
axs[1].set_xlabel("X Position [mm]")
axs[1].set_ylabel("Z Position [mm]")
axs[1].set_title(output_file_name+" LG Hits >5*sigma")
fig.colorbar(quadmesh_1[3], ax=axs[1])

# Plot on the third subplot
quadmesh_2 = axs[2].hist2d(y_pos[LG_pedsub != 0], z_pos[LG_pedsub != 0], bins=(20,20))
axs[2].set_xlabel("Y Position [mm]")
axs[2].set_ylabel("Z Position [mm]")
axs[2].set_title(output_file_name+" LG Hits >5*sigma")
fig.colorbar(quadmesh_2[3], ax=axs[2])

plt.tight_layout()
plt.savefig(output_file_name+"_Spatial_Plots.pdf", format="pdf", bbox_inches="tight")