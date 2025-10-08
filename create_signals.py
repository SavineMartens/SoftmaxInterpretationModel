import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile

Fs=44100

def create_sine(frequency, Fs, sound_duration, amplitude = 1):
    num_samples = sound_duration*Fs
    t = np.linspace(0, sound_duration, int(num_samples))
    amplitude *= np.iinfo(np.int16).max
    sine_wave = amplitude * np.sin(2*np.pi * frequency * t)
    return sine_wave

def create_gaussian(Fs, SD=0.5e-3):
    duration_s = 6*SD # s
    duration_i = int(duration_s*Fs) # discrete
    SD_i = int(SD*Fs)
    n = np.arange(duration_i)-duration_i/2
    gaussian = np.exp(-0.5*(n/SD_i)**2)
    return gaussian

def apply_gaussian_ramp(stimulus, Fs):
    # ramp
    gaussian = create_gaussian(Fs)
    duration_i = len(gaussian)
    # ramp at beginning
    stimulus[:int(duration_i/2)] *= gaussian[:int(duration_i/2)]
    # ramp at ending
    stimulus[-1*int(duration_i/2):] *= gaussian[-1*int(duration_i/2):]
    return stimulus

def create_masker(frequency=1e3, amplitude = 1, masker_duration=100e-3, total_ramp_duration=6*0.5e-3, Fs=44100, plot=False):
    stimulus = create_sine(frequency, Fs, masker_duration+total_ramp_duration, amplitude=amplitude)
    stimulus = apply_gaussian_ramp(stimulus, Fs)
    if plot:
        plt.figure()
        plt.plot(stimulus)
        plt.title('masker')
        # plt.show()
    masker_fname = './sounds/MP/gaussian_probe_' + str(frequency) + 'Hz_' + str(masker_duration)+'s_gaussian_ramp' + str(total_ramp_duration)+'s_amplitude_' + str(amplitude) + '.wav'
    wavfile.write(masker_fname, Fs, stimulus.astype(np.int16))
    return masker_fname

def fix_lengths(masker_probe, masker):
    if len(masker_probe.get_array_of_samples()) > len(masker.get_array_of_samples()):
        samples = masker_probe.get_array_of_samples()  
        shorter = masker_probe._spawn(samples[:len(masker.get_array_of_samples())])
        masker_probe = shorter
    else:
        samples = masker.get_array_of_samples()  
        shorter = masker._spawn(samples[:len(masker_probe.get_array_of_samples())])
        masker = shorter
    return masker_probe, masker


# Amplitude modulation
carrier_frequency = 1500 # Hz
modulation_frequency = 40 # Hz
modulation_duration = 300e-3

def create_AM_tone(carrier_frequency=carrier_frequency, modulation_frequency=modulation_frequency, sound_duration=300e-3, modulation_dB=0, Fs=44100, plot=False):
    # create reference tone
    reference_dB = 91 # [dB], amplitude of 1 seems to be 
    amplitude_reference = 1 
    reference_stimulus = create_sine(carrier_frequency, Fs, 200e-3, amplitude=amplitude_reference)
    reference_stimulus = apply_gaussian_ramp(reference_stimulus, Fs)

    # create unmodulated tone
    unmodulated_dB = 65 # dB
    t = np.linspace(0, sound_duration, int(sound_duration*Fs))
    unmodulated_amplitude_dB_reduction =  unmodulated_dB -reference_dB 
    unmodulated_amplitude = amplitude_reference*10**(unmodulated_amplitude_dB_reduction/20) # A = A_ref * 10 ^(dB/20)

    # unmodulated_stimulus = create_sine(frequency, Fs, sound_duration, amplitude=unmodulated_amplitude)
    # unmodulated_stimulus = apply_gaussian_ramp(unmodulated_stimulus, Fs)
    unmodulated_stimulus = unmodulated_amplitude * np.sin(2 * np.pi * carrier_frequency * t)

    # create modulated tone
    modulation_depth = 1* 10**(modulation_dB/20) # m = 10^(dB/20)
    starting_phase = -np.pi/2 # start at 0 amplitude
    
    modulation_stimulus = np.sin(2 * np.pi * modulation_frequency * t + starting_phase)
    modulated_stimulus = (1 + modulation_depth * modulation_stimulus) * unmodulated_stimulus
    
    x=3

    modulated_stimulus *= np.iinfo(np.int16).max

    # concatenate with reference tone
    modulated_stimulus = np.concatenate((modulated_stimulus, reference_stimulus))
    unmodulated_stimulus = np.concatenate((unmodulated_stimulus, reference_stimulus))

    t_full = np.linspace(0, len(modulated_stimulus)/Fs, len(modulated_stimulus))

    audio_out_file_unmodulated = './sounds/AM/unmodulated_reference91_' + str(unmodulated_dB) +'.wav'
    # audio_out_file = './sounds/AM/masker_reference91_' + str(masker_dB) + 'dB_probe_'+ str(probe_dB) +'dB.wav'
    # masker_probe.export(audio_out_file, format="wav")
    # masker.export(audio_out_file_masker, format="wav")

    if plot:
        plt.figure()
        plt.plot(t_full, unmodulated_stimulus, label='unmodulated')
        plt.plot(t_full, modulated_stimulus, label='modulated')
        plt.xlim(0, 0.3)
        plt.legend()
        plt.title('Unmodulated (' + str(unmodulated_dB) + ' dB) and modulated (' + str(modulation_dB) + ' dB) tones at ' + str(carrier_frequency) + ' Hz')
# 

# for dB in range(21):
#     modulation_dB = -3*dB
#     print(modulation_dB)
#     create_AM_tone(carrier_frequency=carrier_frequency, modulation_frequency=modulation_frequency, sound_duration=300e-3, modulation_dB=modulation_dB, Fs=44100, plot=True)



# Masker probe
frequency = 1e3 # 1kHz
masker_duration=100e-3 # s
probe_duration = 10e-3 # s
masker_dB = 65 # dB



# create_masker()
def create_probe(frequency=1e3, amplitude = 1, probe_duration=probe_duration, Fs=44100, plot=False):  
    # amplitude *= np.iinfo(np.int16).max
    stimulus = create_sine(frequency, Fs, probe_duration, amplitude=amplitude)
    stimulus = apply_gaussian_ramp(stimulus, Fs)
    if plot:
        plt.figure()
        plt.plot(stimulus)
        # plt.show()
    probe_fname = './sounds/MP/masker_' + str(frequency) + 'Hz_' + str(probe_duration)+'s_amplitude_' + str(amplitude) + '.wav'
    wavfile.write(probe_fname, Fs, stimulus.astype(np.int16))
    return probe_fname

def create__varying_amplitude_masker_probe_stimuli_w_reference(masker_dB, probe_dB, frequency, plot=False):
    # Experiment Hamacher masker probe
    # frequency = 1e3 # 2kHz 
    amplitude_masker = 1 
    reference_dB = 91 # [dB], amplitude of 1 seems to be 
    reference_fname = create_masker(amplitude=amplitude_masker)

    probe_amplitude_dB_reduction =  probe_dB -reference_dB 
    probe_amplitude = amplitude_masker*10**(probe_amplitude_dB_reduction/20) # A = A_ref * 10 ^(dB/20)
    masker_amplitude_dB_reduction =  masker_dB -reference_dB 
    masker_amplitude = amplitude_masker*10**(masker_amplitude_dB_reduction/20) # A = A_ref * 10 ^(dB/20)
    masker_fname = create_masker(amplitude=masker_amplitude)
    probe_fname = create_probe(amplitude=probe_amplitude)

    # 'gap t between the probe-tone and the termination of the forward-masker to be meaningfully
    # specified, i.e., as the interval between the beginning of the forward- masker’s terminal
    # decline (ramping-down) and the peak of the probe-tone’s envelope.'
    total_duration = 250 # ms
    masker_duration = 100 # ms
    duration_second_segment_masker = total_duration - masker_duration # ms
    duration_MPI = 100 # ms
    duration_second_segment_probe = 40  # ms

    # create silence audio segments
    masker_probe_silent_segment = AudioSegment.silent(duration=duration_second_segment_probe, frame_rate=Fs)  #duration in milliseconds
    silent_MPI = AudioSegment.silent(duration=duration_MPI, frame_rate=Fs)  #duration in milliseconds
    silent_segment_masker = AudioSegment.silent(duration=duration_second_segment_masker, frame_rate=Fs)  #duration in milliseconds

    #read wav file to an audio segment
    masker = AudioSegment.from_wav(masker_fname)
    probe = AudioSegment.from_wav(probe_fname)
    reference = AudioSegment.from_wav(reference_fname)

    #Add above two audio segments    
    masker_probe = masker + silent_MPI + probe + masker_probe_silent_segment + reference
    masker = masker + silent_segment_masker + reference

    if len(masker_probe.get_array_of_samples()) != len(masker.get_array_of_samples()):
        masker_probe, masker = fix_lengths(masker_probe, masker)
        print('After fixing lengths:')
        print('length masker', len(masker.get_array_of_samples()))
        print('length masker_probe', len(masker_probe.get_array_of_samples()))

    if plot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(np.linspace(0,len(masker_probe.get_array_of_samples())/44100, len(masker_probe.get_array_of_samples())), masker_probe.get_array_of_samples())
        plt.title('Masker ( '+ str(masker_dB) +' dB) +  probe (' + str(probe_dB) + 'dB)')
        plt.subplot(2,1,2)
        plt.plot(np.linspace(0,len(masker.get_array_of_samples())/44100, len(masker.get_array_of_samples())), masker.get_array_of_samples())
        plt.title('Masker ( '+ str(masker_dB) +' dB) +  reference (' + str(reference_dB) + 'dB)')
        print(len(masker_probe.get_array_of_samples()))
        print(len(masker.get_array_of_samples()))

    # Either save modified audio
    audio_out_file_masker = './sounds/MP/masker_reference91dB_' + str(masker_dB) +'.wav'
    audio_out_file = './sounds/MP/masker_reference91dB_' + str(masker_dB) + 'dB_probe_'+ str(probe_dB) +'dB.wav'
    masker_probe.export(audio_out_file, format="wav")
    masker.export(audio_out_file_masker, format="wav")

# for dB in range(21):
#     probe_dB = masker_dB - 3*dB
#     print(-3*dB, 'dB:', probe_dB, 'dB')
#     create__varying_amplitude_masker_probe_stimuli_w_reference(masker_dB=masker_dB, probe_dB=probe_dB, frequency=frequency, plot=True)


def create_varying_probe_amplitude_stimuli(probe_amplitude_dB_reduction, frequency, plot=False):
    # Experiment Hamacher masker probe
    amplitude_masker = 1 
    reference_dB = 109.6 # [dB]

    probe_amplitude = amplitude_masker*10**(probe_amplitude_dB_reduction/20) # A = A_ref * 10 ^(dB/20)
    masker_fname = create_masker(amplitude=amplitude_masker)
    probe_fname = create_probe(amplitude=probe_amplitude)

    total_duration = 250 # ms
    masker_duration = 100 # ms
    duration_second_segment_masker = total_duration - masker_duration # ms
    duration_MPI = 100 # ms
    duration_second_segment_probe = 40  # ms

    # create silence audio segments
    masker_probe_silent_segment = AudioSegment.silent(duration=duration_second_segment_probe, frame_rate=Fs)  #duration in milliseconds
    silent_MPI = AudioSegment.silent(duration=duration_MPI, frame_rate=Fs)  #duration in milliseconds
    silent_segment_masker = AudioSegment.silent(duration=duration_second_segment_masker, frame_rate=Fs)  #duration in milliseconds

    #read wav file to an audio segment
    masker = AudioSegment.from_wav(masker_fname)
    probe = AudioSegment.from_wav(probe_fname)

    #Add above two audio segments    
    masker_probe = masker + silent_MPI + probe + masker_probe_silent_segment 
    masker = masker + silent_segment_masker 

    if len(masker_probe.get_array_of_samples()) != len(masker.get_array_of_samples()):
        masker_probe, masker = fix_lengths(masker_probe, masker)
        print('After fixing lengths:')
        print('length masker', len(masker.get_array_of_samples()))
        print('length masker_probe', len(masker_probe.get_array_of_samples()))

    if plot:
        plt.figure()
        plt.plot(np.linspace(0,len(masker_probe.get_array_of_samples())/44100, len(masker_probe.get_array_of_samples())), masker_probe.get_array_of_samples())
        plt.plot(np.linspace(0,len(masker.get_array_of_samples())/44100, len(masker.get_array_of_samples())), masker.get_array_of_samples(), '--')
        plt.title('Masker (full scale) +  probe reduction (' + str(probe_amplitude_dB_reduction) + 'dB)')
        print(len(masker_probe.get_array_of_samples()))
        print(len(masker.get_array_of_samples()))

    # Either save modified audio
    audio_out_file_masker = './sounds/MP/masker_reference1.wav'
    audio_out_file = './sounds/MP/masker_reference1_probe_'+ str(probe_amplitude_dB_reduction) +'dB.wav'
    masker_probe.export(audio_out_file, format="wav")
    masker.export(audio_out_file_masker, format="wav")

for dB in range(21):
    probe_dB = - 3*dB
    print('probe reduction:', -3*dB, 'dB' )
    create_varying_probe_amplitude_stimuli(probe_amplitude_dB_reduction=probe_dB, frequency=frequency, plot=True)



plt.show()