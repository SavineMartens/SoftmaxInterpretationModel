import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from Hamacher_utils import *
from create_data import *
import re
import matplotlib as mpl

# To do
# [ ] NH with 1903 fibers and TP2 cut-off at 500 Hz --> running on cluster rn
# [X] EH with 1903 fibers and TP2 cut-off at 500 Hz
# [X] Check height of IRs for NH and EH 


if __name__ == "__main__":
    test = 'MP'  # 'AM' or 'MP'
    hearing = 'EH'  # 'NH' or 'EH'
    folder_IR = f'./{test}/{hearing}/IR/'
    num_fibers = 1903# 952
    TP2_cut_off_Hz = 500
    Fs_down = 5000 # Downsampled frequency of neurograms and IRs

    if test == 'AM':
        if hearing == 'NH':
            wildcard_dB_start = '91_'
            wildcard_dB_end = 'dB_IR'
        if hearing == 'EH':
            wildcard_dB_start = 'reference1_'
            wildcard_dB_end = 'dB_relscale'
    if test == 'MP':
        if hearing == 'NH':
            wildcard_dB_start = 'probe_'
            wildcard_dB_end = 'dB_IR'
        if hearing == 'EH':
            wildcard_dB_start = 'probe_'
            wildcard_dB_end = 'dB_relscale'

wildcard = f'*{num_fibers}CFs*{TP2_cut_off_Hz}Hz.npy'

single_file = glob.glob(f'{folder_IR}*0dB{wildcard}' )[0] # to get num_bands and num_time_pointsnum_bands, num_time_points = single_file.shape
single_file = np.load(single_file)
max_band, _ = np.where(single_file == np.max(single_file))
max_band = max_band[0]

num_bands, num_time_points = single_file.shape
time_vector = np.linspace(0, num_time_points/Fs_down, num_time_points)
rows, columns = closestDivisors(num_bands)

files = sorted(glob.glob(folder_IR + wildcard))

# Extract dB values using regex (handles negative numbers too)
def extract_db(filename):
    match = re.search(r'([-]?\d+)dB', filename)
    return int(match.group(1)) if match else None

# Filter out files that don’t match the pattern
files_with_db = [(f, extract_db(f)) for f in files if extract_db(f) is not None]

# Sort by dB value
files_sorted = sorted(files_with_db, key=lambda x: x[1])
sorted_files = [f for f, db in files_sorted]

# create color map
color_map = plt.get_cmap('viridis', len(files_sorted))
custom_palette = [mpl.colors.rgb2hex(color_map(i)) for i in range(color_map.N)]

plt.figure(figsize=(10, 10))
for f, file in enumerate(sorted_files):
    print(f'Loading IR {f+1}: {file}')
    IR = np.load(file)
    try:
        label_str = str(float(file.split(wildcard_dB_start)[1].split(wildcard_dB_end)[0])) + ' dB'
    except:
        label_str = 'Reference signal'

    plt.plot(time_vector, IR[max_band, :], label=label_str, color=custom_palette[f], linewidth=2)

plt.suptitle(f'Internal Representations for {test} - {hearing} in band {max_band+1}', fontsize=16)
plt.legend(ncol=2)
plt.xlim((0, time_vector[-1]))
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Internal Representation', fontsize=14)

plt.show()
        
