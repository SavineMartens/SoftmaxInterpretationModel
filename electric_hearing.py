import numpy as np
import os
import platform
import glob
from utilities import *
from Hamacher_utils import *
import platform
import matplotlib.pyplot as plt

test = 'AM'

# raw data folder
if platform.system() == 'Linux':
    raw_data_folder = '/exports/kno-shark/users/Savine/python/temporal-phast-plus/output/'
else:
    raw_data_folder = './'+ test +'/EH/RawData/'

# load frequencies and fiber IDs
frequencies_EH = np.load('./data/EH_freq_vector_electrode_allocation_logspaced.npy')
fiber_ids = np.load('./data/fiber_ID_list_FFT.npy')
# select half of the fibers
frequencies_EH = frequencies_EH[::2]
fiber_ids = fiber_ids[::2]
num_fibers = len(fiber_ids)


for f, file in enumerate(sorted(glob.glob(raw_data_folder + '/*trains*.npy'))):
    print(file, '\n', f+1, 'out of', len(glob.glob(raw_data_folder + '/*trains*.npy')))
    # get neurogram and IR
    neurogram, IR = get_Hamacher_IR_from_numpy(file, fiber_IDs=fiber_ids, frequencies=frequencies_EH, plot_IR=True, band_type='adapted')
    save_dir_neurograms = './'+ test +'/EH/neurograms/'
    save_dir_IR = './'+ test +'/EH/IR/'
    for save_dir in [save_dir_neurograms, save_dir_IR]:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    np.save(os.path.join(save_dir_neurograms, os.path.basename(file).replace('spike_trains_F120', 'neurogram').replace('.npy', '') + str(num_fibers) + 'CFs.npy'), neurogram)
    np.save(os.path.join(save_dir_IR, os.path.basename(file).replace('spike_trains_F120', 'neurogram').replace('.npy', '') + str(num_fibers) + 'CFs_' + str(IR.shape[0]) + 'bands.npy'), IR)





plt.show()