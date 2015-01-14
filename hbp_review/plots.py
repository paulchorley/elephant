import numpy as np
import quantities as pq
import matplotlib.pyplot as plt


def spike_raster(sts):
    for i in range(len(sts)):
        t = sts[i].rescale(pq.s).magnitude
        plt.plot(t, i * np.ones_like(t), 'k.', ms=2)
    plt.axis('tight')
    plt.xlim(t[0], t[-1])
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Unit Index', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)


def population_rate(phist, col='k'):
    t = phist.times.rescale(pq.s).magnitude
    h = phist.rescale(pq.Hz).magnitude
    w = phist.sampling_period.rescale(pq.s).magnitude
    x = np.zeros((len(h) * 2))
    y = np.zeros((len(h) * 2))
    x[::2] = t
    x[1::2] = t + w
    y[::2] = y[1::2] = h
    plt.plot(x, y, color=col)
    plt.xlim(x[0], x[-1])

    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Rate (Hz)', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)


def rate_distribution(rates, bin_width=2 * pq.Hz, col='k', opacity=1.):
    w = bin_width.rescale(pq.Hz).magnitude
    x = [rate.rescale(pq.Hz).magnitude for rate in rates]

    bins = np.arange(np.min(x), np.max(x) + w, w)
    h, b = np.histogram(x, bins, density=True)

    plt.bar(left=b[:-1], height=h, width=w, fc=col, alpha=opacity)

    m = np.mean(x)
    plt.plot(m, plt.ylim()[1] * 0.1, 's', color='w', markersize=10)
    plt.plot(m, plt.ylim()[1] * 0.1, 'o', color=col, markersize=6)
    plt.axis('tight')
    plt.xlim(0, 20)

    plt.xlabel('Spike Rate (Hz)', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)


def isi_distribution(isi, bin_width=2 * pq.ms, col='k', opacity=1.):
    w = bin_width.rescale(pq.ms).magnitude
    x = isi.rescale(pq.ms).magnitude

    bins = np.arange(np.min(x), np.max(x) + w, w)
    h, b = np.histogram(x, bins, density=True)

    plt.bar(left=b[:-1], height=h, width=w, fc=col, alpha=opacity)

    m = np.mean(x)
    plt.plot(m, plt.ylim()[1] * 0.1, 's', color='w', markersize=10)
    plt.plot(m, plt.ylim()[1] * 0.1, 'o', color=col, markersize=6)
    plt.axis('tight')

    plt.xlabel('ISI (ms)', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    plt.xlim(0, 200)


def cv_distribution(cvs, bins=20, col='k', opacity=1.):
    x = cvs
    if type(bins) is int:
        bins, bin_width = np.linspace(np.min(x), np.max(x), bins, retstep=True)
    else:
        bins = bins
        bin_width = bins[1] - bins[0]
    h, b = np.histogram(x, bins, density=True)

    plt.bar(left=b[:-1], height=h, width=bin_width, fc=col, alpha=opacity)

    m = np.mean(x)
    plt.plot(m, plt.ylim()[1] * 0.1, 's', color='w', markersize=10)
    plt.plot(m, plt.ylim()[1] * 0.1, 'o', color=col, markersize=6)
    plt.axis('tight')

    plt.xlabel('CV', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    plt.xlim(bins[0], bins[-1] + bin_width)


def lv_distribution(lvs, bins=20, col='k', opacity=1.):
    x = lvs
    if type(bins) is int:
        bins, bin_width = np.linspace(np.min(x), np.max(x), bins, retstep=True)
    else:
        bins = bins
        bin_width = bins[1] - bins[0]
    h, b = np.histogram(x, bins, density=True)

    plt.bar(left=b[:-1], height=h, width=bin_width, fc=col, alpha=opacity)

    m = np.mean(x)
    plt.plot(m, plt.ylim()[1] * 0.1, 's', color='w', markersize=10)
    plt.plot(m, plt.ylim()[1] * 0.1, 'o', color=col, markersize=6)
    plt.axis('tight')

    plt.xlabel('LV', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    plt.xlim(bins[0], bins[-1] + bin_width)


def cc_distribution(cc_matrix, bins=20, col='k', opacity=1.):
    x = cc_matrix[np.triu_indices(cc_matrix.shape[0], 1)]

    if type(bins) is int:
        bins, bin_width = np.linspace(np.min(x), np.max(x), bins, retstep=True)
    else:
        bins = bins
        bin_width = bins[1] - bins[0]
    h, b = np.histogram(x, bins, density=True)

    plt.bar(
        left=b[:-1], height=h, width=bin_width,
        fc=col, alpha=opacity)

    m = np.mean(x)
    plt.plot(m, plt.ylim()[1] * 0.1, 's', color='w', markersize=10)
    plt.plot(m, plt.ylim()[1] * 0.1, 'o', color=col, markersize=6)
    plt.axis('tight')
    plt.xlim(bins[0], bins[-1] + bin_width)

    plt.xlabel('Correlation Coefficient', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)


def cc_sig_matrix(cc_matrix, bins=20, color='k', opacity=1.):
    x = cc_matrix[np.triu_indices(cc_matrix.shape[0], 1)]

    if type(bins) is int:
        bins, bin_width = np.linspace(np.min(x), np.max(x), bins, retstep=True)
    else:
        bins = bins
        bin_width = bins[1] - bins[0]
    h, b = np.histogram(x, bins, density=True)

    plt.bar(
        left=b[:-1], height=h, width=bin_width,
        fc=color, alpha=opacity)

    m = np.mean(x)
    plt.plot([m, m], plt.ylim(), color)
    plt.axis('tight')

    plt.xlabel('Correlation Coefficient', fontsize=16)
    plt.ylabel('PDF', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    plt.xlim(bins[0], bins[-1] + bin_width)
