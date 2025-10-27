import numpy as np 
import os
from brucezilany import stimulus, Neurogram, Species
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        fname = './sounds/AM/modulated_reference1_-3dB.wav'
    else:
        fname = sys.argv[1]
    print(f'Processing {os.path.basename(fname)}')
    
    # Load the sound file
    stim = stimulus.from_file(fname, verbose=False)
    stim = stimulus.normalize_db(stim, stim_db=65)   
    
    cfs = np.load('./data/AB_MS_based_on_min_filtered_thresholdsfreq_x_fft.npy')
    # create neurograms
    ng = Neurogram(cfs=cfs, n_low=10, n_med=10, n_high=30)
    ng.create(sound_wave=stim, species=Species.HUMAN_SHERA, n_trials=5)

    output = ng.get_output()  # 3D array: [fiber, trial, time]
    # save neurogram
    output_dir = './SMRT_neurograms/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, os.path.basename(fname).replace('.wav', '_neurogram.npy')), output)
