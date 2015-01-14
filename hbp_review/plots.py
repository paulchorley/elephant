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


def cc_examples(cc, ex_i, num_ex):
    # randomizers to select random examples
    rdm = 341
    rdo = 3

    y = cc['edge_time_series']['cch'][ex_i * rdm + rdo, :]
    x = cc['edge_time_series']['times_ms'][ex_i * rdm + rdo, :]
    s1 = cc['edge_time_series']['sig_upper_975'][ex_i * rdm + rdo, :]

    n1 = cc['edges']['id_i'][ex_i * rdm + rdo]
    n2 = cc['edges']['id_j'][ex_i * rdm + rdo]
    x1 = cc['neuron_topo']['x'][n1]
    y1 = cc['neuron_topo']['y'][n1]
    x2 = cc['neuron_topo']['x'][n2]
    y2 = cc['neuron_topo']['y'][n2]

    # find zero index in CCH
    ind = np.argmin(np.abs(x))
    plt.gca().fill_between(
        x[ind - 5:ind + 5], s1[ind - 5:ind + 5], color=[0.6, 0.6, 0.6])
    plt.plot(x, y, 'k-')
    plt.ylabel(
        "El: %i - %i \n (%i,%i) - (%i,%i)" %
        (n1, n2, x1, y1, x2, y2),
        fontsize=16)
    if ex_i == num_ex - 1:
        plt.xlabel("time (ms)", fontsize=16)
    else:
        plt.gca().set_xticklabels([])
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.axis('tight')
    plt.xlim((-100, 100))


def cc_sig_matrix(cc_matrix, num_neurons):
    plt.pcolor(np.arange(num_neurons), np.arange(num_neurons), cc_matrix)
    plt.xlabel("Neuron ID i", fontsize=16)
    plt.ylabel("Neuron ID j", fontsize=16)
    plt.axis('tight')
    plt.gca().tick_params(axis='both', which='major', labelsize=14)
    plt.clim(0, 1)
    cb = plt.colorbar()
    cb.set_label('significance (1-p)', fontsize=14, labelpad=10, y=0.45)
