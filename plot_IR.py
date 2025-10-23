import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from Hamacher_utils import *
from create_data import *
import re
import matplotlib as mpl

# To do
# [X] NH with 1903 fibers and TP2 cut-off at 500 Hz --> running on cluster rn
# [X] EH with 1903 fibers and TP2 cut-off at 500 Hz
# [X] Check height of IRs for NH and EH 


if __name__ == "__main__":
    test = 'MP'  # 'AM' or 'MP'
    hearing = 'EH'  # 'NH' or 'EH'
    folder_IR = f'S:/python/SoftmaxInterpretationModel/{test}/{hearing}/IR/' #f'./{test}/{hearing}/IR/'
    num_fibers = 1903# 952
    TP2_cut_off_Hz = 500
    Fs_down = 5000 # Downsampled frequency of neurograms and IRs

    NH_dB = 55

    if test == 'AM':
        if hearing == 'NH':
            folder_IR += f'seed42/*91_{NH_dB}dB_*'
            wildcard_dB_start = f'91_{NH_dB}dB_'
            wildcard_R = f'*unmodulated*reference91_{NH_dB}dB*'
            wildcard_dB_end = 'dB_IR'
            wildcard_RT_max = f'*modulated*reference91_{NH_dB}dB_0dB*'
        if hearing == 'EH':
            wildcard_dB_start = 'reference1_'
            wildcard_dB_end = 'dB_relscale'
            wildcard_R = '*unmodulated*reference1*'
            wildcard_RT_max = f'*modulated*reference1*_0dB*'
    if test == 'MP':
        if hearing == 'NH':
            folder_IR += f'seed42/*91_{NH_dB}dB_'
            wildcard_dB_start = f'probe_'
            wildcard_R = f'IR*' #*f'*masker_reference91_{NH_dB}dB*'
            wildcard_dB_end = 'dB_IR'
            wildcard_RT_max = f'probe_{NH_dB}dB_*' #f'*masker_reference91_{NH_dB}dB_probe_{NH_dB}dB*'
        if hearing == 'EH':
            wildcard_dB_start = 'probe_'
            wildcard_R = '*masker_reference1_rel*'
            wildcard_dB_end = 'dB_relscale'
            wildcard_RT_max = f'*masker_reference1_*probe_0*'
 
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

try:
    sorted_files.remove(glob.glob(folder_IR + wildcard_R)[0])
except:
    pass
rows, columns = closestDivisors(len(sorted_files))


dB_values = []

IR_R = np.load(glob.glob(folder_IR + wildcard_R)[0])

fig, axes = plt.subplots(rows, columns, figsize=(15, 10), sharex=True, sharey=True)
axes= axes.flatten()
for f, file in enumerate(sorted_files):
    print(f'Loading IR {f+1}: {file}')
    IR = np.load(file)
    try:
        label_str = str(float(file.split(wildcard_dB_start)[1].split(wildcard_dB_end)[0])) + ' dB'
        dB_values.append(float(label_str.split(' dB')[0]))
    except:
        label_str = 'Cannot read dB value'

    axes[f].plot(time_vector, IR[max_band, :], label=label_str, linewidth=2, color='blue' )
    axes[f].plot(time_vector, IR_R[max_band, :], label='Reference signal', color='r', linestyle='--', linewidth=2)
    axes[f].set_title(label_str)
    axes[f].set_xlim((0, time_vector[-1]))
    axes[f].set_xlabel('Time (s)', fontsize=12)
    axes[f].set_ylabel('Internal Representation', fontsize=12)
    axes[f].legend()

unique_dB_values = set(dB_values)
# print which dB values are duplicated
for db in unique_dB_values:
    count = dB_values.count(db)
    if count > 1:
        print(f'dB value {db} is duplicated {count} times.')

duration = 0.25 # seconds
time_vector = np.linspace(0, duration, int(Fs_down * duration), endpoint=False)

if test == 'MP':
    location = 'upper center'
if test == 'AM':
    location = 'lower center'


# figure A2
plt.figure(figsize=(7, 5))
IR_RT = np.load(glob.glob(folder_IR + wildcard_RT_max)[0])
plt.subplot(2,1,1)
plt.plot(time_vector, IR_RT[max_band, :len(time_vector)], label='IR(RT)', color='green', linewidth=2)
plt.plot(time_vector, IR_R[max_band, :len(time_vector)], label='IR(R)', color='r', linestyle='--', linewidth=2)
plt.ylabel('Internal \n Representation (IR)', fontsize=14)
plt.legend()
plt.subplot(2,1,2)
S = IR_RT - IR_R
plt.plot(time_vector, S[max_band, :len(time_vector)], label='S = IR(RT) - IR(R)', color='blue', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Detector (S)', fontsize=14)
if test == 'MP' and hearing == 'NH':
    plt.suptitle('Masker (R) and probe (RT) in critical band 960Hz-1080Hz', fontsize=16)
plt.legend()

# figure with NIR
# plt.figure(figsize=(12, 5))
figA2, axes = plt.subplots(2,2, figsize=(12, 5), sharex=True, sharey='row')
plt.subplots_adjust(left=0.08, right = 0.983)
plt.subplot(2,2,1)
plt.plot(time_vector, IR_RT[max_band, :len(time_vector)], label='IR(RT)', color='green', linewidth=2)
plt.plot(time_vector, IR_R[max_band, :len(time_vector)], label='IR(R)', color='r', linestyle='--', linewidth=2)
plt.ylabel('Internal \n Representation (IR)', fontsize=14)
plt.subplot(2,2,3)
S = IR_RT - IR_R
plt.plot(time_vector, S[max_band, :len(time_vector)], label='S = IR(RT) - IR(R)', color='blue', linewidth=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Detector (S)', fontsize=14)
sigma = 0.6
sigma_w = np.std(IR_R)*sigma
NIR_R = get_Hamacher_NIR(IR_R, sigma=sigma_w)
NIR_RT = get_Hamacher_NIR(IR_RT, sigma=sigma_w)
plt.subplot(2,2,2)
plt.plot(time_vector, NIR_RT[max_band, :len(time_vector)], label='NIR(RT)', color='green', linewidth=2)
plt.plot(time_vector, NIR_R[max_band, :len(time_vector)], label='NIR(R)', color='r', linestyle='--', linewidth=2)
plt.ylabel(f'Internal Representation \n with noise (N$_σ$= {sigma})', fontsize=14)
plt.subplot(2,2,4)
plt.plot(time_vector, NIR_RT[max_band, :len(time_vector)] - NIR_R[max_band, :len(time_vector)], label='NIR(RT) - NIR(R)', color='blue', linewidth=2)
plt.plot(time_vector, S[max_band, :len(time_vector)], label='S = IR(RT) - IR(R)', color='orange', linestyle='--', linewidth=2)
plt.ylabel('Detector', fontsize=14)
plt.xlabel('Time (s)', fontsize=14)
import matplotlib.transforms as mtransforms # labeling axes
trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)

letters = ['A', 'B', 'C', 'D']

for a, ax in zip(np.arange(1,5), axes.flatten()) :
    plt.subplot(2,2,a)
    plt.xlim((0, duration))
    if test == 'MP':
        ax.text(-0.001, 0.98, letters[a-1], transform=ax.transAxes + trans,
                fontsize=16, verticalalignment='top', color='black')
    if test == 'AM':
        ax.text(-0.003, 0.16, letters[a-1], transform=ax.transAxes + trans,
                fontsize=16, verticalalignment='top', color='black')
    plt.legend(loc='best')
    

if test == 'MP' and hearing == 'NH':
    plt.suptitle(f'Masker probe detection: masker (R) and masker-probe (RT)', fontsize=16)
    figA2.savefig('./paper/figure_A2_MP_NH.png', dpi=300)
if test == 'AM' and hearing == 'NH':
    plt.suptitle(f'Amplitude modulation: unmodulated (R) and modulated (RT)', fontsize=16)
    figA2.savefig('./paper/figure_A2_AM_NH.png', dpi=300)
if test == 'MP' and hearing == 'EH':
    plt.suptitle(f'Masker probe detection: masker (R) and masker-probe (RT)', fontsize=16)
    figA2.savefig('./paper/figure_A2_MP_EH.png', dpi=300)
if test == 'AM' and hearing == 'EH':
    plt.suptitle(f'Amplitude modulation: unmodulated (R) and modulated (RT)', fontsize=16)
    figA2.savefig('./paper/figure_A2_AM_EH.png', dpi=300)

plt.show()
        
