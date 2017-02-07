import logging

import numpy as np
from matplotlib import pyplot as plt
from publication.plotting import adjust_position, without_spines_and_ticks, label_subplot

logging.basicConfig(level=logging.DEBUG)

# Butterworth filter code

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut=100, highcut=3500, fs=20000, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Final figure 3

def figure03(test=True):
    if test:
        s, p  = get_distributions()
    else:
        s,p  = get_distributions(N_all=1000000)

    # Making figure
    fig = plt.figure('Figure 3', figsize=(18, 9))
    fig.suptitle('Figure 3. Distinguishing signals from noise using neighboring electrodes in high-density electrode arrays', fontsize=14,
                 fontweight='bold')

    ax1 = plt.subplot(231)
    for N in p.keys():
        ax1.plot (s ,p[N], label = 'N=%d' % N)
    sqrt_twelth = np.sqrt(1.0 / 12)
    sqrt_half = np.sqrt(0.5)
    plt.vlines(sqrt_twelth,0,10,linestyles=':')
    plt.xticks((0, 0.2, sqrt_twelth, 0.4, 0.6, sqrt_half, 0.8, 1.0), ('0.1', '0.2', r'$\frac{1}{\sqrt{12}}$', '0.4', '0.6', r'$\frac{1}{\sqrt{2}}$', '0.8', '1.0'))
    ax1.legend()
    ax1.set_xlabel (r'standard deviation $s$')
    ax1.set_ylabel (r'probability $P(s)$')
    adjust_position(ax1, yshrink=0.01)
    without_spines_and_ticks(ax1)
    label_subplot(ax1, 'A', xoffset=-0.045, yoffset=-0.015)


    ax2 = plt.subplot(232)
    img = plt.imread('data/hdmea_neighborhoods.png')
    plt.imshow(img)
    plt.axis('off')
    label_subplot(ax2, 'B', xoffset=-0.025, yoffset=-0.015)


    # Potential distribution around point source (axon)
    # Assuming the HDMEA surface to be an insulator and the extracellular space to be homogeneous and isotropic, the
    # potential distribution is the same as for two charges in free medium, one mirrored at the surface
    ax3 = plt.subplot(233)
    r_range = np.linspace(-50,50,101)
    z_range = np.linspace(0,75,75)
    r, z = np.meshgrid(r_range, z_range)
    z0 = 10
    z0_mirror = -z0
    phi = 1.0/np.sqrt(np.power(r, 2) + np.power(z-z0, 2)) + 1.0/np.sqrt(np.power(r, 2) + np.power(z-z0_mirror, 2))
    ax3.contour(r,z,phi, levels=np.linspace(0,0.6,50), colors='gray', linestyles='solid')
    plt.plot(0,z0,'r.', markersize=10)
    plt.annotate('point source',(0,z0),(10,60), size=15, arrowprops={'arrowstyle': '->', 'color': 'black'})
    ax3.set_xlabel (r'radius $r$ [$\mu$m]')
    ax3.set_ylabel (r'distance from surface $z$ [$\mu$m]')
    without_spines_and_ticks(ax3)
    ax3.annotate('', (35, 0), (35, 10), arrowprops=dict(shrinkB=0, shrinkA=0, arrowstyle='<->'))
    plt.hlines(10,0,35, color='black', linestyle=':', zorder=10)
    ax3.text(35, 3, r' z', size=15)
    ax3.text(-45, 65, r'Potential $\Phi(d,r)$', color='gray', size=15, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    label_subplot(ax3, 'C', xoffset=-0.035, yoffset=-0.015)

    # Potential at HDMEA surface
    ax4 = plt.subplot(234)
    for z in (2, 3, 5, 10, 20):
        r = np.linspace(-50,50,101)
        y = attenuation(r, z)
        plt.plot(r,y,label='z=%d$\mu$m' % z)
    plt.legend()
    ax4.text(-48, 0.88, r'$A \approx \frac{z}{\sqrt{r^2+z^2}}$', size=20, bbox=dict(facecolor='white', pad=5, edgecolor='none'))
    annotate_r_arrow(ax4, -18)
    annotate_r_arrow(ax4, +18)
    ax4.set_xlabel (r'radius $r$ [$\mu$m]')
    ax4.set_ylabel (r'signal attenuation $A$')
    ax4.set_xlim((-50,50))
    ax4.set_ylim((0,1))
    adjust_position(ax4, yshrink=0.01)
    without_spines_and_ticks(ax4)
    label_subplot(ax4, 'D', xoffset=-0.045, yoffset=-0.015)

    # Define hdmea neighborhood geometry
    z = 10 # um
    r = 10 # um
    N7 = [0, r, r, r, r, r, r]  # hidens, neighborhood
    N5 = [0, r, r, r, r]  # mea1k Neumann neighborhood
    r2 = np.sqrt(0.5)*r
    N9 = [0, r, r, r, r, r2, r2, r2, r2]  # mea1k Moore neighborhood

    # Define recording parameters
    n_samples = 161  # number of samples spike triggered average

    # Example
    ax5 = plt.subplot(235)
    snr = 4   # signal to noise ratio for the othe axon signal, simulated as a single (!) negative peak
    std_delay, delays, signal_time, time, traces = peak_sd(N7, n_samples, snr, z, signal_index=43)
    ax5.plot(time, traces, color='gray')
    ax5.scatter(delays, snr*np.ones_like(delays), marker='v', s=100, edgecolor='None', facecolor='black')
    ax5.set_ylim((-snr-1,+snr+1))
    ax5.set_xlim((0,1))
    plt.hlines((-1,1),0,1,linestyles=':',colors='red', linewidth=2, zorder=10)
    plt.vlines(signal_time,0,-snr,linestyles='solid',colors='red', linewidth=2, zorder=10)
    ax5.set_ylabel('signal $y$')
    ax5.set_xlabel('time $t$')
    without_spines_and_ticks(ax5)
    adjust_position(ax5, xshrink=0.005)
    label_subplot(ax5, 'E', xoffset=-0.035, yoffset=-0.015)

    # Statistics
    ax6 = plt.subplot(236)
    n_bins=100
    n_signals=1000 if test else 100000
    snr_range = (3,4,5,6)
    p_signal = get_signal_distributions(N7, n_bins, n_samples, n_signals, snr_range, z)
    ax6.plot(s, p[7], label='only noise')
    for snr in snr_range:
        ax6.plot(s, p_signal[snr], label='signal, snr=%d' % snr)
    ax6.set_xlim((0,1))
    ax6.set_ylim((0,10))
    plt.legend()
    ax6.set_xlabel (r'standard deviation $s$')
    ax6.set_ylabel (r'probability $P(s)$')
    without_spines_and_ticks(ax6)
    label_subplot(ax6, 'F', xoffset=-0.035, yoffset=-0.015)

    plt.show()


def get_signal_distributions(N7, n_bins, n_samples, n_signals, snr_range, z):
    p_signal = {}
    for snr in snr_range:
        std_delays = np.zeros(n_signals)
        for i in range(n_signals):
            std_delays[i], _, _, _, _ = peak_sd(N7, n_samples, snr, z)
        count, bins = np.histogram(std_delays, bins=np.linspace(0, 1, n_bins))
        p_signal[snr] = 1.0 * np.array(count) * n_bins / n_signals
    return p_signal


def peak_sd(radii, n_samples, snr, z, signal_index=None):
    sampling_rate = 1.0/n_samples
    time = np.arange(0,1,sampling_rate)
    y = attenuation(radii, z)
    N = len(radii)

    # Background noise and randomly timed signal
    traces = np.random.normal(0, 1, (n_samples, N))
    if not signal_index:
        signal_index = np.random.randint(0, n_samples, 1)
    traces[signal_index, :] = - snr * y

    signal_time = signal_index * sampling_rate
    delays = np.argmin(traces, axis=0).astype(np.float) * sampling_rate
    std_delay = np.std(delays)
    return std_delay, delays, signal_time, time, traces


def attenuation(r, z):
    y = z / np.sqrt(np.power(r, 2) + np.power(z, 2))
    return y


def annotate_r_arrow(ax4, radius):
    ax4.annotate('', (0, 0.05), (radius, 0.05), arrowprops=dict(shrinkB=0, shrinkA=0, arrowstyle='<->'))
    ax4.text(radius/2, 0.07, r'r', size=15)


def get_distributions(N_max=9, N_all=11016, n_bins=100):
    bins = np.linspace(0, 1, n_bins)
    p = {}
    for N in range(2, N_max+1):
        sample = np.random.rand(N_all, N)
        sample_sd = np.std(sample, axis=1, ddof=1)
        count, _ = np.histogram(sample_sd, bins=bins)
        p[N] = 1.0 * np.array(count) * n_bins / N_all
    s = bins[0:-1]
    return s, p


figure03(test=False)


