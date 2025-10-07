import numpy as np
import matplotlib.pyplot as plt
from brucezilany import stimulus, Neurogram, Species
import brucezilany
import os
import glob
import librosa
from utilities import *
import platform

# To do
# [ ] create pipeline
# [ ] check if RT max in memory causes not to reach 100% accuracy

if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg') 

frequencies_EH = np.load('./data/EH_freq_vector_electrode_allocation_logspaced.npy')
# use half for less computation
frequencies_EH = frequencies_EH[::8]


def get_stimulus_wo_reference(data_dir, sound_name, timing_wo_reference=0.3):
    """
    Load the stimulus from the specified directory and file name.
    """
    audio, audio_fs = librosa.load(os.path.join(data_dir, sound_name), sr=44100, mono=True)
    samples_wo_reference = int(timing_wo_reference * audio_fs)
    audio = audio[:samples_wo_reference] # remove reference part
    duration = (1 / audio_fs) * len(audio)
    stim = stimulus.Stimulus(audio, audio_fs, duration)
    return stim


def save_neurogram(ng, save_path):
    """
    Save the neurogram data to a .npy file.
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, ng.get_data())
    print(f'Neurogram saved to {save_path}')


def create_neurogram(stim, plot_neurogram=False, n_trials=5):
    """
    Create a Neurogram object based on the stimulus.
    """
   
    # Create Neurogram instance
    # seed = np.random.randint(0, 100)
    # brucezilany.set_seed(seed)
    # np.random.seed(seed)
    ng = Neurogram(frequencies_EH, n_low=5, n_med=5, n_high=15) #  n_low=10, n_med=10, n_high=30
    Fs = 1e4
    ng.bin_width =1/Fs
    print(f'Number of trials: {n_trials}, Bin width: {ng.bin_width*1e3} ms, number of fibers: {len(frequencies_EH)}')
    # Create neurogram output
    print('Am I going to crash?')
    ng.create(sound_wave=stim, species=Species.HUMAN_SHERA, n_trials=n_trials)   
    print('I did not crash') 
    bin_ratio = Fs/5e3 # downsample to 5kHz
    output = ng.get_output() # 3D array: [fiber, trial, time]
    if plot_neurogram:
        plt.figure()  
        t = np.arange(output.shape[2]) * ng.bin_width
        plt.pcolormesh(t, frequencies_EH, output.mean(axis=1), cmap='viridis', shading='auto')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
    return ng

for file in sorted(glob.glob('./sounds/MP/*.wav')):
    sound_name = os.path.basename(file)
    print(f'Processing {sound_name}...')
    stim = get_stimulus_wo_reference('./sounds/MP', sound_name, timing_wo_reference=0.25)
    ng = create_neurogram(stim, plot_neurogram=True, n_trials=1)
    now = get_time_str(seconds=False)
    num_fibers = len(frequencies_EH)
    save_dir = './data/neurograms/NH/MP/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, now + '_' + sound_name.replace('.wav', '_neurogram_' + str(num_fibers) + 'CFs.npy'))
    save_neurogram(ng, save_path)