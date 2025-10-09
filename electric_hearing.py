import numpy as np
import os
import platform
import glob
from utilities import *
from Hamacher_utils import *

# raw data folder
if platform.system() == 'Linux':
    raw_data_folder = '/exports/kno-shark/users/Savine/python/temporal-phast-plus/output/'
else:
    raw_data_folder = './MP/EH/RawData/'

# get fiber id
# select half of the fibers

for file in sorted(glob.glob(raw_data_folder + '/*trains*.npy')):
    print(file)
    spike_trains = np.load(file)
    # create neurograms

    # saving neurogram
    now = get_time_str(seconds=False)
    save_dir = './MP/EH/neurograms/'
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir)) 
    save_path = os.path.join(save_dir, now + '_' + sound_name.replace('.wav', '_neurogram_' + str(num_fibers) + 'CFs.npy'))
    
    # get internal representations

    # save IR
    save_dir = './MP/EH/IR/'