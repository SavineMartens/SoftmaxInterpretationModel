import os
import glob
import platform
import numpy as np
import matplotlib.pyplot as plt
import argparse

from utilities import *
from Hamacher_utils import *

# Optional import for NH (Bruce–Zilany model)
try:
    from brucezilany import stimulus, Neurogram, Species
    import librosa
except ImportError:
    stimulus = None

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def get_frequencies_and_fibers(hearing_type):
    """Load frequency and fiber arrays depending on hearing type."""
    frequencies = np.load('./data/EH_freq_vector_electrode_allocation_logspaced.npy')
    fiber_ids = np.load('./data/fiber_ID_list_FFT.npy')

    # Select half of the fibers for EH
    # if hearing_type == 'EH':
    #     frequencies = frequencies[::2]
    #     fiber_ids = fiber_ids[::2]
    if hearing_type == 'NH':
        # NH only needs frequencies, Bruce-Zilany handles fibers internally
        fiber_ids = None

    return frequencies, fiber_ids


def load_stimulus(file_path, trim_reference=0.3):
    """Load and optionally trim a stimulus file."""
    import librosa
    audio, fs = librosa.load(file_path, sr=44100, mono=True)
    samples = int(trim_reference * fs)
    audio = audio[:samples]
    duration = len(audio) / fs
    return stimulus.Stimulus(audio, fs, duration)


def save_numpy(data, path):
    ensure_dirs(os.path.dirname(path))
    np.save(path, data)
    print(f"✅ Saved: {path}")


# ------------------------------------------------------------------------------
# Processing functions
# ------------------------------------------------------------------------------

def process_electric_hearing(test, create_files=False, plot_files=True, TP2_cut_off_Hz=500):
    """Process data for electric hearing (EH)."""
    save_dir_neuro = f'./{test}/EH/neurograms/'
    save_dir_IR = f'./{test}/EH/IR/'
    ensure_dirs(save_dir_neuro, save_dir_IR)

    freqs, fiber_ids = get_frequencies_and_fibers('EH')
    num_fibers = len(fiber_ids)

    # --- Step 1: Create neurograms and IRs ---
    if create_files:
        raw_data_folder = (
            '/exports/kno-shark/users/Savine/python/temporal-phast-plus/output/'
            if platform.system() == 'Linux'
            else f'./{test}/EH/RawData/'
        )
        search_pattern = '*trains*dulated*reference1*.npy' if test == 'AM' else '*trains*masker*reference1*.npy'
        files = sorted(glob.glob(os.path.join(raw_data_folder, search_pattern)))

        print(f'Found {len(files)} files for EH.')

        for f, file in enumerate(files):
            print(f'[{f+1}/{len(files)}] Processing {file}')
            neurogram, IR = get_Hamacher_IR_from_numpy(
                file, fiber_IDs=fiber_ids, frequencies=freqs, plot_IR=False, band_type='adapted'
            )

            neurogram_path = os.path.join(save_dir_neuro,
                                          os.path.basename(file).replace('spike_trains_F120', 'neurogram')
                                          + f'_{num_fibers}CFs.npy')
            IR_path = os.path.join(save_dir_IR,
                                   os.path.basename(file).replace('spike_trains_F120', 'neurogram')
                                   + f'_{num_fibers}CFs_{IR.shape[0]}bands_TP2_{TP2_cut_off_Hz}Hz.npy')

            save_numpy(neurogram, neurogram_path)
            save_numpy(IR, IR_path)

    # --- Step 2: Plot created files ---
    if plot_files:
        for f, file in enumerate(sorted(glob.glob(save_dir_IR + '*.npy'))):
            print(f'Plotting IR {f+1}: {file}')
            IR = np.load(file)
            t = np.linspace(0, IR.shape[1]/1000, IR.shape[1])
            fig = plot_single_internal_representation(IR, t, freqs, font_size=14)
            plt.suptitle(os.path.basename(file).replace('.npy', ''), fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.savefig(file.replace('.npy', '.png'))
        plt.show()


def process_normal_hearing(test, create_files=False, plot_files=True, TP2_cut_off_Hz=500):
    """Process data for normal hearing (NH) using Bruce–Zilany model."""
    if stimulus is None:
        raise ImportError("Bruce–Zilany model not installed or imported correctly.")

    fixed_seed = True
    save_dir_neuro = f'./{test}/NH/neurograms/'
    save_dir_IR = f'./{test}/NH/IR/'
    ensure_dirs(save_dir_neuro, save_dir_IR)

    frequencies, _ = get_frequencies_and_fibers('NH')
    num_fibers = len(frequencies)
    Fs = 1e4

    if fixed_seed:
        seed = 42
        import brucezilany
        brucezilany.set_seed(seed)
        np.random.seed(seed)
        save_dir_neuro += f'seed{seed}/'
        save_dir_IR += f'seed{seed}/'

    sound_files = sorted(glob.glob(f'./sounds/{test}/*reference91_*17*.wav'))
    print(f'Found {len(sound_files)} sound files for NH.')


    for i, file_path in enumerate(sound_files):
        sound_name = os.path.basename(file_path).replace('.wav', '')
        print(f'[{i+1}/{len(sound_files)}] Processing {sound_name}')

        # Build expected neurogram filename pattern
        neurogram_pattern = f"*{sound_name}*_{num_fibers}CFs.npy"
        existing_neurograms = glob.glob(os.path.join(save_dir_neuro, neurogram_pattern))
        IR_pattern = f"*{sound_name}*_{num_fibers}CFs_*bands_TP2_{TP2_cut_off_Hz}Hz.npy"
        existing_IRs = glob.glob(os.path.join(save_dir_IR, IR_pattern))

        if existing_neurograms:
            # if existing_IRs:
            #     print(f"🧠 IR already exists: {os.path.basename(existing_IRs[0])}, skipping.")
            #     continue
            # else:
            neurogram_path = existing_neurograms[0]
            print(f"🧠 Found existing neurogram: {os.path.basename(neurogram_path)}")
            time_stamp = os.path.basename(neurogram_path).split(sound_name)[0]
            neurogram = np.load(neurogram_path)
            IR = compute_internal_representation_from_numpy(neurogram=neurogram,
                                                            Fs_neurogram=Fs,
                                                            fiber_frequencies=frequencies,
                                                            TP2_cut_off_Hz=TP2_cut_off_Hz,
                                                            plot_IR=plot_files)
            num_bands, _ = IR.shape
            IR_filename = f"{sound_name}_IR_{num_fibers}CFs_{num_bands}bands_TP2_{TP2_cut_off_Hz}Hz.npy"
            IR_path = os.path.join(save_dir_IR, time_stamp + IR_filename)
            save_numpy(IR, IR_path)

        else:
            print("🧠 Generating new neurogram...")
            stim = load_stimulus(file_path, trim_reference=0.25)
            ng = Neurogram(frequencies, n_low=10, n_med=10, n_high=30)
            ng.bin_width = 1 / Fs
            ng.create(sound_wave=stim, species=Species.HUMAN_SHERA, n_trials=1)

            # Save neurogram
            now = get_time_str(seconds=True)
            neuro_path = os.path.join(save_dir_neuro, f'{now}_{sound_name}_neurogram_{num_fibers}CFs.npy')
            save_numpy(ng.get_output(), neuro_path)

            # Compute and save IR
            IR = compute_internal_representation_from_object(ng, frequencies)
            num_bands, _ = IR.shape
            IR_path = os.path.join(save_dir_IR, f'{now}_{sound_name}_IR_{num_fibers}CFs_{num_bands}bands_TP2_{TP2_cut_off_Hz}Hz.npy')
            save_numpy(IR, IR_path)
            

            if plot_files:
                plt.figure()
                t = np.arange(ng.get_output().shape[2]) * ng.bin_width
                plt.pcolormesh(t, frequencies, ng.get_output().mean(axis=1), cmap='viridis', shading='auto')
                plt.title(sound_name)
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.colorbar(label='Rate')
            del ng, stim, IR





        

    if plot_files:
        plt.show()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create data for normal or electric hearing.")
    parser.add_argument("--hearing", type=str, choices=["NH", "EH"], default="NH", help="Type of hearing to simulate")
    parser.add_argument("--test", type=str, choices=["AM", "MP"], default="MP", help="Test type")
    parser.add_argument("--create-files", type=lambda x: x.lower() == "true", default=True, help="Whether to generate data")
    parser.add_argument("--plot-files", type=lambda x: x.lower() == "true", default=False, help="Whether to plot IRs/neurograms")
    args = parser.parse_args()

    TP2_cut_off_Hz = 500  # Hz

    print(f"\n Running {args.test} for {args.hearing} (create_files={args.create_files}, plot_files={args.plot_files})\n")

    if args.hearing == "EH":
        process_electric_hearing(args.test, args.create_files, args.plot_files, TP2_cut_off_Hz)
    else:
        process_normal_hearing(args.test, args.create_files, args.plot_files, TP2_cut_off_Hz)