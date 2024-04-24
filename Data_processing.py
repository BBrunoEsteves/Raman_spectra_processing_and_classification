import matplotlib.pyplot as plt
import numpy as np
#import pywt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import os

#change list to define wich peaks are considered
peaks_used = np.array([855, 875, 936, 1004, 1070, 1218, 1265, 1302, 1335, 1445, 1576, 1618, 1655, 1745])

global_peak_values = []
ratio_number = 0;


#
def baseline_als(y, lam, p, niter=10):  # Asymmetric Least Squares Smoothing (P.Eilers e H.Boelens)
    """
    Perform asymmetric least squares smoothing on input data.

    Args:
    - y: Input data array.
    - lam: Lambda parameter controlling smoothness.
    - p: Asymmetry parameter, controlling the balance between
         minimizing residuals above and below the data.
    - niter: Number of iterations for the optimization process.

    Returns:
    - Smoothed data array.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def spike_detector(intensity):
    """
    Detect spikes in intensity data.

    Args:
    - intensity: Array containing intensity values over time.

    Returns:
    - Array of modified Z-scores indicating spike presence.
    """
    dist = 0
    delta_intensity = []  # variation of intensity (It - It-1)
    for i in np.arange(len(intensity) - 1):
        dist = intensity[i + 1] - intensity[i]
        delta_intensity.append(dist)
    delta_intensity.append(0)
    delta_int = np.array(delta_intensity)
    median_int = np.median(delta_int)  # mediana
    mad_int = np.median([np.abs(delta_int - median_int)])  # desvio absoluto medio
    modified_z_scores = 0.6745 * (delta_int - median_int) / mad_int
    return modified_z_scores


def modified_z_score(intensity):
    #improved spike_detector()
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return modified_z_scores



# the function uses modified_z_scores to determine if a point is a cosmic spike. After detecting spikes the function
# selects an area around for averaging and getting a substitute value.
# In case all points around are spikes, the function progressively expands the search area
def while_spike_remover(intensity, m=5, threshold=5,max_tries=100):
    """
    The function uses modified_z_scores to determine if a point is a cosmic spike. After detecting spikes the function
    selects an area around for averaging and getting a substitute value.
    In case all points around are spikes, the function progressively expands the search area

    Args:
    - intensity: Array containing intensity values.
    - m: Size of the window around spikes for averaging.
    - threshold: Threshold for spike detection.
    - max_tries: Maximum number of attempts to remove spikes.

    Returns:
    - Smoothed intensity data with spikes removed.
    """
    print("Removing spikes")
    has_spikes = True
    spikes = abs(np.array(modified_z_score(np.diff(intensity)))) > threshold
    tester = intensity.copy()
    intensity_out = intensity.copy()
    tries = 0
    while has_spikes and tries<max_tries:
        for i in np.arange(len(spikes)):
            if spikes[i] != 0:
                w = np.arange(i - m, i + 1 + m)  # select area around spike
                w = w[w >= 0]
                w = w[w < len(spikes)]
                w2 = w[spikes[w] == 0]  # From such interval, we choose the ones which are not spikes
                while len(w2) == 0:
                    m += 1
                    w = np.arange(i - m, i + 1 + m)
                    w = w[w >= 0]
                    w = w[w < len(spikes)]
                    w2 = w[spikes[w] == 0]
                intensity_out[i] = np.mean(intensity[w2])  # and we average their values
        spikes = abs(np.array(modified_z_score(np.diff(intensity_out)))) > threshold
        if 1 not in spikes:
            has_spikes = False
            print("no spikes left")
        elif np.array_equal(intensity_out, tester):
            threshold += 0.1
            print("increasing threshold by 0.1, new one = ", threshold)
        else:
            print("trying same threshold again")
        tester = intensity_out.copy()
        tries+=1
        if tries == max_tries-1:
            print("Coudn't remove all spikes")
    return intensity_out

def normal_spike_remover(intensity, m=5, threshold=5):
    #same without recursion and expansion
    has_spikes = True
    spikes = abs(np.array(modified_z_score(np.diff(intensity)))) > threshold
    tester = intensity.copy()
    intensity_out = intensity.copy()
    tries = 0
    for i in np.arange(len(spikes)):
        if spikes[i] != 0:
            w = np.arange(i - m, i + 1 + m)  # select area around spike
            w = w[w >= 0]
            w = w[w < len(spikes)]
            w2 = w[spikes[w] == 0]  # From such interval, we choose the ones which are not spikes
            while len(w2) == 0:
                m += 1
                w = np.arange(i - m, i + 1 + m)
                w = w[w >= 0]
                w = w[w < len(spikes)]
                w2 = w[spikes[w] == 0]
            intensity_out[i] = np.mean(intensity[w2])  # and we average their values
    spikes = abs(np.array(modified_z_score(np.diff(intensity_out)))) > threshold
    return intensity_out


def white_noise_remover(intensity, wl=7, po=3):
    """
    Remove white noise from intensity data using Savitzky–Golay filter.

    Args:
    - intensity: Array containing intensity values.
    - wl: Window length parameter for the Savitzky–Golay filter.
    - po: Polynomial order parameter for the Savitzky–Golay filter.

    Returns:
    - Smoothed intensity data with white noise removed.
    """
    intensity_out = savgol_filter(intensity, wl, po)  # Savitzky–Golay filter
    return intensity_out



def find_local_maximum(intensity, radius=3):
    """
    Find local maxima in intensity data within a given radius.

    Args:
    - intensity: Array containing intensity values.
    - radius: Radius for considering local maxima.

    Returns:
    - Indices of local maxima in the intensity array.
    """
    local_maxima = argrelextrema(intensity, np.greater, 0, radius)
    return local_maxima


def scale(intensity):
    """
    Scale intensity data to the range [0, 1] based on a specified range.

    Args:
    - intensity: Array containing intensity values.

    Returns:
    - Scaled intensity data.
    """
    maximum = np.amax(intensity[604:667])
    minimum = np.amin(intensity[257:860])
    scaled = np.interp(intensity, (minimum, maximum), (0, 1))
    # mean = np.mean(scaled)
    # scaled = scaled - mean + 0.3

    return scaled


def find_peak_values(intensity, frequencies, peak_list, radius=2, use_peak_proximity = True):
    """
    Find peak values in intensity data at specified frequencies.
    Corrects for /0 for ratio calculation convenience

    Args:
    - intensity: Array containing intensity values.
    - frequencies: Array containing frequency values corresponding to intensity.
    - peak_list: List of frequencies at which peaks are detected.
    - radius: Radius for considering peaks when using mean.
    - use_peak_proximity: Flag indicating whether to use peak proximity for intensity estimation.

    Returns:
    - Array containing peak frequencies and corresponding intensity values.
    """
    peak_values = np.array([peak_list, np.zeros(len(peak_list))])
    for i in range(len(peak_list)):
        idx = (np.abs(frequencies - peak_list[i])).argmin()
        max = np.amax(intensity[idx - radius: idx + 1 + radius])
        mean = np.mean(intensity[idx - radius: idx + 1 + radius])
        real = intensity[idx]
        if real == 0:
            real == mean
        if real == 0:
            real += 0.001
        if use_peak_proximity:
            peak_values[1, i] = mean
        else:
            peak_values[1, i] = real

    return peak_values


def peak_ratios(peak_values, use_unique_ratios=True):
    """
    Calculate ratios between peak intensities.

    Args:
    - peak_values: Array containing peak frequencies and corresponding intensities.
    - use_unique_ratios: Flag indicating whether to exclude inverse of existing ratios or include all.

    Returns:
    - Array containing peak ratios and corresponding peak frequency pairs.
    """
    ratios = []
    peak_list = peak_values[0, :].astype(int).astype(str)
    peak_keys = []
    for i in range(len(peak_values[1, :])):
        if use_unique_ratios:
            for j in range(i + 1, len(peak_values[1, :])):
                ratios.append(peak_values[1, i] / peak_values[1, j])
                peak_keys.append('/'.join([peak_list[i], peak_list[j]]))
        else:
            for j in range(len(peak_values[1, :])):
                if i != j:
                    ratios.append(peak_values[1, i] / peak_values[1, j])
                    peak_keys.append('/'.join([peak_list[i], peak_list[j]]))
            #ratios.append(peak_values[1, :] / peak_values[1, i])
            #peak_keys.append(np.core.defchararray.add(peak_list, np.full([np.shape(peak_list)[0]], '/' + peak_list[i])))
    peak_ratios = np.array([peak_keys, ratios])
    return peak_ratios


def process_one(filename, use_unique_ratios = True, use_peak_proximity=True):
    """
    Process a single spectrum file.

    Args:
    - filename: Name of the spectrum file.
    - use_unique_ratios: Flag indicating whether to use unique ratios.
    - use_peak_proximity: Flag indicating whether to use peak proximity for intensity estimation.

    Returns:
    - List containing frequency array, scaled intensity data, and peak ratios.
    """
    if filename.endswith(".txt"):
        print(filename + " started")
        spectrum = np.genfromtxt('espectros\\' + filename)
        baseline = baseline_als(spectrum[:, 1], 70000, 0.02, 20)  # 100000, 0.02, 20
        corrected = spectrum - np.c_[np.zeros(len(baseline)), baseline]
        threshold = 4
        spikes = spike_detector(corrected[:, 1])
        spike_info = while_spike_remover(corrected[:, 1], threshold=threshold)
        no_spikes = spike_info
        filtered = no_spikes#white_noise_remover(no_spikes, 11, 2)
        # filtered = wavelet_noise_remover(no_spikes)
        scaled = scale(filtered)

        peak_list = peaks_used
        peak_values = find_peak_values(scaled, spectrum[:, 0], peak_list, use_peak_proximity=use_peak_proximity)

        #neural_data = np.concatenate((peak_values[1, :], peak_values[2, :]), 0)
        # np.savetxt("processing_results\\peaks\\" + filename, neural_data)

        save_file_peaks = np.zeros(peak_list.size, dtype=[('key_name', 'U9'), ('value', float)])
        save_file_peaks['key_name'] = peak_values[0, :].astype(int).astype(str)
        save_file_peaks['value'] = peak_values[1, :]

        np.savetxt("processing_results\\peaks\\" + filename[:-4] + " peaks.txt", save_file_peaks,
                   fmt="%10s , %10.15f")

        ratios = peak_ratios(peak_values, use_unique_ratios=use_unique_ratios)

        # neural_data_ratios = np.concatenate((ratios[1].flatten(), mean_ratios.flatten()))
        # np.savetxt("processing_results\\neural_data\\ratios\\" + filename[:-4] + " ratios.txt", neural_data_ratios)
        save_file_ratios = np.zeros(ratios[0].flatten().size, dtype=[('key_name', 'U9'), ('value', float)])
        save_file_ratios['key_name'] = ratios[0].flatten()
        save_file_ratios['value'] = ratios[1].flatten()

        np.savetxt("processing_results\\ratios\\" + filename[:-4] + " ratios.txt", save_file_ratios,
                   fmt="%10s , %10.15f")

        main_ratios = np.where(ratios[0] == "")

        plt.figure(figsize=(15, 20), dpi=300)

        plt.suptitle(filename[:-4], fontsize=16)

        plt.subplot(611, )
        plt.plot(spectrum[:, 0], spectrum[:, 1], "k", label="raw data")
        plt.plot(spectrum[:, 0], baseline, "k", label="baseline")
        plt.ylabel("Original spectrum", fontsize=10)

        plt.subplot(612)
        plt.plot(spectrum[:, 0], corrected[:, 1], "k", label="corrected signal")
        plt.ylabel("Subtracted baseline", fontsize=10)

        plt.subplot(613)
        plt.plot(spectrum[:, 0], spikes, "b-", label="corrected signal")
        plt.ylabel("spikes", fontsize=10)

        plt.subplot(614)
        plt.plot(spectrum[:, 0], no_spikes, "k", label="corrected signal")
        plt.ylabel("corrected spikes", fontsize=10)

        plt.subplot(615)
        plt.plot(spectrum[:, 0], no_spikes, "r-", label="filtered signal")
        plt.plot(spectrum[:, 0], filtered, "g-", label="corrected signal")
        plt.legend()
        plt.ylabel("filtered", fontsize=10)

        plt.subplot(616)
        plt.plot(spectrum[:, 0], filtered, "g-", label="final signal")
        plt.ylabel("final", fontsize=10)

        '''plt.subplot(616)
        plt.plot(spectrum[:, 0], scaled, "k", label="corrected signal")
        plt.scatter(peak_list, peak_values[1, :], cmap='b-')
        plt.scatter(peak_list, peak_values[2, :], cmap='g-')
        plt.ylabel("scaled", fontsize=10)
        plt.xlabel("raman shift", fontsize=10)
        points = peaks_used
        plt.vlines(points, 0, 1, colors='k', linestyles='dotted')
        for i in points:
            plt.text(i + 0.1, 0, str(i))
        plt.xlim([800, 1800])
        # plt.ylim([-3, 5])'''

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        plt.savefig('processing_results\\' + filename[:-4] + ' processed', bbox_inches='tight')
        plt.clf()
        plt.close()

        global global_peak_values
        global_peak_values.append([filename[:-4],peak_values[1,:]])


        # wavelet_noise_remover(corrected[:,1] ,spectrum[:, 0], filename)
        print(filename + " concluded")

    return [spectrum[:, 0], scaled, ratios]


def process_all_data(use_unique_ratios = True, use_peak_proximity = True):
    """
    Process all spectrum data.

    Args:
    - use_unique_ratios: Flag indicating whether to use unique ratios.
    - use_peak_proximity: Flag indicating whether to use peak proximity for intensity estimation.
    """
    print("Processing all data")
    counter_healthy = 0
    counter_cancer = 0
    counter_healthy_best = 0
    counter_cancer_best = 0

    init_shape = np.genfromtxt("espectros\\NCM1a.txt").shape
    mean_healthy = np.zeros([init_shape[1], init_shape[0]])
    mean_cancer = np.zeros([init_shape[1], init_shape[0]])
    mean_healthy_best = np.zeros([init_shape[1], init_shape[0]])
    mean_cancer_best = np.zeros([init_shape[1], init_shape[0]])

    if use_unique_ratios:
        mean_ratios_healthy = np.zeros([int((len(peaks_used) * len(peaks_used) - len(peaks_used)) / 2)])
        mean_ratios_cancer = np.zeros([int((len(peaks_used) * len(peaks_used) - len(peaks_used)) / 2)])
    else:
        mean_ratios_healthy = np.zeros([len(peaks_used) * len(peaks_used) - len(peaks_used)])
        mean_ratios_cancer = np.zeros([len(peaks_used) * len(peaks_used) - len(peaks_used)])
        # mean_ratios_healthy = np.zeros([len(peaks_used) * len(peaks_used)])
        # mean_ratios_cancer = np.zeros([len(peaks_used) * len(peaks_used)])

    peak_keys = np.zeros([len(peaks_used)*len(peaks_used)])
    plt.figure(figsize=(40, 10), dpi=300)

    for filename in os.listdir("espectros"):
        new_plot = process_one(filename, use_unique_ratios, use_peak_proximity)
        ratios = new_plot[2]
        peak_keys = ratios[0].flatten()
        if filename[0] == 'N':
            counter_healthy += 1
            mean_healthy += new_plot[:2]
            mean_ratios_healthy += ratios[1].flatten().astype(float)
            if filename[-5] != 'a':
                counter_healthy_best += 1
                mean_healthy_best += new_plot[:2]
                plt.plot(new_plot[0], new_plot[1], "b-")
        else:
            counter_cancer += 1
            mean_cancer += new_plot[:2]
            mean_ratios_cancer += ratios[1].flatten().astype(float)
            if filename[-5] != 'a':
                counter_cancer_best += 1
                mean_cancer_best += new_plot[:2]
                plt.plot(new_plot[0], new_plot[1], "r-")
    plt.savefig('processing_results\\' + 'all best data' + ' processed', bbox_inches='tight')
    plt.clf()
    plt.close()

    mean_healthy = mean_healthy / counter_healthy
    mean_cancer = mean_cancer / counter_cancer

    mean_healthy_best = mean_healthy_best / counter_healthy_best
    mean_cancer_best = mean_cancer_best / counter_cancer_best

    mean_ratios_healthy /= counter_healthy
    mean_ratios_cancer /= counter_cancer

    variation = np.abs(
        np.divide(mean_ratios_healthy - mean_ratios_cancer, (mean_ratios_healthy + mean_ratios_cancer) / 2))
    index_data = np.argwhere(variation >= 0.2)

    largest_peak_ratios = np.take(peak_keys, index_data)
    np.savetxt("processing_results\\ratios\\" + "largest_peak_ratios_mean.txt",
               np.concatenate((index_data, largest_peak_ratios), 1), fmt="%s")

    mean_difference = mean_healthy[1] - mean_cancer[1]
    mean_difference_regions = white_noise_remover(mean_difference, 11, 1)
    mean_cancer_filtered = white_noise_remover(mean_cancer[1], 11, 1)
    mean_healthy_filtered = white_noise_remover(mean_healthy[1], 11, 1)

    peak_list = peaks_used
    peak_values_healthy = find_peak_values(mean_healthy[1], mean_healthy[0], peak_list)
    peak_values_cancer = find_peak_values(mean_cancer[1], mean_healthy[0], peak_list)

    plt.figure(figsize=(40, 10), dpi=300)
    plt.plot(mean_cancer[0], mean_cancer[1], "r-")
    plt.plot(mean_healthy[0], mean_healthy[1], "b-")
    plt.savefig('processing_results\\' + 'all mean' + ' processed', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 10), dpi=300)
    plt.plot(mean_cancer[0], mean_cancer[1], "r-")
    plt.plot(mean_healthy[0], mean_healthy[1], "b-")
    plt.scatter(peak_list, peak_values_healthy[1, :], cmap='g-')
    plt.scatter(peak_list, peak_values_cancer[1, :], cmap='y-')
    plt.xlim([800, 1800])
    plt.savefig('processing_results\\' + 'all mean' + ' processed from 800 to 1800', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 10), dpi=300)
    plt.plot(mean_cancer_best[0], mean_cancer_best[1], "r-")
    plt.plot(mean_healthy_best[0], mean_healthy_best[1], "b-")
    plt.savefig('processing_results\\' + 'best mean' + ' processed', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 10), dpi=300)
    plt.plot(mean_cancer_best[0], mean_cancer_best[1], "r-")
    plt.plot(mean_healthy_best[0], mean_healthy_best[1], "b-")
    plt.xlim([800, 1800])
    points = peaks_used
    plt.vlines(points, 0, 1, colors='k', linestyles='dotted')
    for i in points:
        plt.text(i+0.1,0,str(i))
    plt.savefig('processing_results\\' + 'best mean' + ' processed' + 'from 800 to 1800', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 10), dpi=300)
    plt.plot(mean_healthy[0], mean_difference, "r-")
    plt.annotate("cancer", (1300, -0.15))
    plt.annotate("healthy", (1300, 0.15))
    plt.plot([800, 1800], [0, 0], "g-")
    plt.xlim([800, 1800])
    plt.savefig('processing_results\\' + 'mean_difference' + ' processed', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 10), dpi=300)
    plt.plot(mean_healthy[0], -mean_difference_regions, "r-")
    plt.plot([800, 1800], [0, 0], "g-")
    plt.annotate("healthy", (1300, -0.15))
    plt.annotate("cancer", (1300, 0.15))
    plt.xlim([800, 1800])
    plt.savefig('processing_results\\' + 'mean_difference_regions_inverted' + ' processed', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 10), dpi=300)
    plt.plot(variation)
    plt.savefig('processing_results\\ratios\\' + 'abs_ratios_differences', bbox_inches='tight')
    plt.clf()
    plt.close()

    healthy_list = []
    cancer_list = []
    healthy_mean_list = []
    cancer_mean_list = []
    first_healthy = True
    first_cancer = True
    counter_H = 0
    counter_C = 0
    for i in global_peak_values:
        if i[0][0] == "N":
            counter_H += 1
            healthy_list.append(i)
            if first_healthy:
                healthy_mean_list = i[1]
                first_healthy = False
            else:
                healthy_mean_list = [x + y for (x, y) in zip(i[1], healthy_mean_list)]
        if i[0][0]=="R":
            counter_C += 1
            cancer_list.append(i)
            if first_cancer:
                cancer_mean_list = i[1]
                first_cancer = False
            else:
                cancer_mean_list = [x + y for (x, y) in zip(i[1], cancer_mean_list)]
    healthy_mean_list = [x / counter_H for x in healthy_mean_list]
    cancer_mean_list = [x / counter_C for x in cancer_mean_list]


    plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(peak_list, healthy_mean_list, label= "healthy_mean", marker="<")
    plt.scatter(peak_list, cancer_mean_list, label="cancer_mean",  marker=">")
    for i in healthy_list:
        plt.scatter(peak_list, i[1], label=i[0], marker="_")
    plt.ylabel("scaled", fontsize=10)
    plt.xlabel("raman shift", fontsize=10)
    points = peaks_used
    plt.vlines(points, 0, 1, colors='k', linestyles='dotted')
    for i in points:
        plt.text(i + 0.1, 0, str(i))
    plt.xlim([800, 1800])
    plt.legend()
    plt.savefig('processing_results\\' + ' only_peaks_healthy', bbox_inches='tight')
    plt.clf()
    plt.close()


    plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(peak_list, healthy_mean_list, label= "healthy_mean", marker="<")
    plt.scatter(peak_list, cancer_mean_list, label="cancer_mean",  marker=">")
    for i in cancer_list:
        plt.scatter(peak_list, i[1], label=i[0], marker="_")
    plt.ylabel("scaled", fontsize=10)
    plt.xlabel("raman shift", fontsize=10)
    points = peaks_used
    plt.vlines(points, 0, 1, colors='k', linestyles='dotted')
    for i in points:
        plt.text(i + 0.1, 0, str(i))
    plt.xlim([800, 1800])
    plt.legend()
    plt.savefig('processing_results\\' + ' only_peaks_cancer', bbox_inches='tight')
    plt.clf()
    plt.close()


    plt.figure(figsize=(8, 8), dpi=300)
    for i in global_peak_values:
        key = i[0][0]+i[0][3]
        if key in  ['R2']:
            plt.scatter(peak_list, i[1], label=i[0])
    plt.scatter(peak_list, healthy_mean_list, label= "healthy_mean", marker="_")
    plt.scatter(peak_list, cancer_mean_list, label="cancer_mean",  marker="_")
    plt.ylabel("scaled", fontsize=10)
    plt.xlabel("raman shift", fontsize=10)
    points = peaks_used
    plt.vlines(points, 0, 1, colors='k', linestyles='dotted')
    for i in points:
        plt.text(i + 0.1, 0, str(i))
    plt.xlim([800, 1800])
    plt.legend()
    plt.savefig('processing_results\\' + ' only_peaks_RKO2', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.figure(figsize=(8, 8), dpi=300)
    for i in global_peak_values:
        key = i[0][0] + i[0][3]
        if key in ['N4']:
            plt.scatter(peak_list, i[1], label=i[0])
    plt.scatter(peak_list, healthy_mean_list, label="healthy_mean", marker="_")
    plt.scatter(peak_list, cancer_mean_list, label="cancer_mean", marker="_")
    plt.ylabel("scaled", fontsize=10)
    plt.xlabel("raman shift", fontsize=10)
    points = peaks_used
    plt.vlines(points, 0, 1, colors='k', linestyles='dotted')
    for i in points:
        plt.text(i + 0.1, 0, str(i))
    plt.xlim([800, 1800])
    plt.legend()
    plt.savefig('processing_results\\' + ' only_peaks_N4', bbox_inches='tight')
    plt.clf()
    plt.close()


    np.savetxt("processing_results\\means\\" + "mean_healthy.txt", np.transpose((mean_healthy[0, 256:860], mean_healthy[1, 256:860])))
    np.savetxt("processing_results\\means\\" + "mean_healthy_filtered.txt", np.transpose((mean_healthy[0, 256:860], mean_healthy_filtered[256:860])))
    np.savetxt("processing_results\\means\\" + "mean_cancer.txt", np.transpose((mean_cancer[0, 256:860], mean_cancer[1, 256:860])))
    np.savetxt("processing_results\\means\\" + "mean_cancer_filtered.txt", np.transpose((mean_cancer[0, 256:860], mean_cancer_filtered[256:860])))
    np.savetxt("processing_results\\means\\" + "mean_diference.txt", np.transpose((mean_cancer[0, 256:860], -mean_difference[256:860])))
    np.savetxt("processing_results\\means\\" + "mean_diference_filtered.txt", np.transpose((mean_cancer[0, 256:860], -mean_difference_regions [256:860])))

    print("All data processed")

process_all_data(use_unique_ratios=True, use_peak_proximity = True)
