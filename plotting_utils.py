import matplotlib.pyplot as plt
import numpy as np
from utilities import *

def plot_single_internal_representation(IR, t_filtered_resampled, frequency_bands, font_size = 20):
    num_remaining_crit_bands, _ = IR.shape
    number_bands_prime = num_remaining_crit_bands
    row_plot = 0
    while row_plot<3:
        # while is_prime(number_bands_prime):
        number_bands_prime += 1
        if not is_prime(number_bands_prime):
            row_plot, column_plot = closestDivisors(number_bands_prime)
            # print(row_plot, column_plot)
    fig, ax = plt.subplots(row_plot, column_plot, sharey=True, sharex=True)
    axes = ax.flatten()
    for n in range(num_remaining_crit_bands):
        axes[n].plot(t_filtered_resampled, IR[n,:])
        if len(frequency_bands) == num_remaining_crit_bands:
            axes[n].set_title('Critical band ' + str(n+1) + '\n (CF=' + str(round(frequency_bands[n])) + ' Hz)')
        else:    
            axes[n].set_title('Critical band ' + str(n+1) + '\n (' + str(round(frequency_bands[n])) + '-' + str(round(frequency_bands[n+1])) + 'Hz)')
        axes[n].set_xlim((t_filtered_resampled[0], t_filtered_resampled[-1]))

    fig.text(0.08, 0.35, 'Internal Representations (IR)', ha='center', rotation='vertical', fontsize=font_size)
    fig.text(0.5, 0.04, 'Time [s]', ha='center', fontsize=font_size)

    return fig