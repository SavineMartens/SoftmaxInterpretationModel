import numpy as np
import matplotlib.pyplot as plt
from brucezilany import stimulus, Neurogram, Species
import brucezilany
import os
import glob
import librosa

# To do
# [ ] create pipeline
# [ ] check if RT max in memory causes not to reach 100% accuracy

frequencies_EH = np.load('./data/EH_freq_vector_electrode_allocation_logspaced')

def get_stimulus(data_dir, sound_name):
    """
    Load the stimulus from the specified directory and file name.
    """
    audio, audio_fs = librosa.load(os.path.join(data_dir, sound_name), sr=44100, mono=True)
    audio = audio[:len(audio)//2] # use half only for sounds with reference in it
    duration = (1 / audio_fs) * len(audio)
    stim = stimulus.Stimulus(audio, audio_fs, duration)
    return stim


def create_neurogram(stim, plot_neurogram=False, n_trials=10):
    """
    Create a Neurogram object based on the stimulus.
    """
   
    # Create Neurogram instance
    seed = np.random.randint(0, 100)
    brucezilany.set_seed(seed)
    np.random.seed(seed)
    ng = Neurogram(frequencies_EH, n_low=10, n_med=10, n_high=30) #  n_low=10, n_med=10, n_high=30
    ng.bin_width =1/Fs
    Fs = 1e5
    # Create neurogram output
    ng.create(sound_wave=stim, species=Species.HUMAN_SHERA, n_trials=n_trials)    
    output = ng.get_output()  # 3D array: [fiber, trial, time]
    if plot_neurogram:
        plt.figure()  
        t = np.arange(output.shape[2]) * ng.bin_width
        plt.pcolormesh(t, frequencies_EH, output.mean(axis=1), cmap='viridis', shading='auto')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
    return ng