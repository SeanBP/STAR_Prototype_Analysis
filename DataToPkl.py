import numpy as np
import pickle
import time

input_file_name = 'Run5_list.txt'
output_file_name = 'Run5_list.pickle'
filter_incomplete_evts = True

start = time.time()
with open(input_file_name) as f:
    lines = f.read().split('\n')
    
Tstamp_us = []
Brd = []
Ch = []
LG = []
HG = []
last_tstamp = 0


for i, line in enumerate(lines):
    if i > 8:
        data = line.split()
    
        if len(data) == 6:
            Tstamp_us.append(float(data[0]))
            last_tstamp = float(data[0])
            Brd.append(int(data[2]))
            Ch.append(int(data[3]))
            LG.append(int(data[4]))
            HG.append(int(data[5]))
        if len(data) == 4:
            Tstamp_us.append(last_tstamp)
            Brd.append(int(data[0]))
            Ch.append(int(data[1]))
            LG.append(int(data[2]))
            HG.append(int(data[3]))
            
middle = time.time()

events = []
current_event = {'Tstamp_us': [], 'Brd': [], 'Ch': [], 'LG': [], 'HG': []}
for i in range(len(Tstamp_us)):
    if i == 0 or Tstamp_us[i] - current_event['Tstamp_us'][-1] < 5:
        current_event['Tstamp_us'].append(Tstamp_us[i])
        current_event['Brd'].append(Brd[i])
        current_event['Ch'].append(Ch[i])
        current_event['LG'].append(LG[i])
        current_event['HG'].append(HG[i])
    else:
        events.append(current_event)
        current_event = {'Tstamp_us': [Tstamp_us[i]], 'Brd': [Brd[i]], 'Ch': [Ch[i]], 'LG': [LG[i]], 'HG': [HG[i]]}

events.append(current_event)

if(filter_incomplete_evts):
    events = [event for event in events if len(event.get('LG', [])) >= 192]

end = time.time()

with open(output_file_name, 'wb') as handle:
    pickle.dump(events, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print((middle - start)/60)
print((end - middle)/60)
print((end - start)/60)
print(len(events))