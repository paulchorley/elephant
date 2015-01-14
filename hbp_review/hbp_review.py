# coding: utf-8

# ## Initialization

import sys

# to find our "special" elephant
sys.path.insert(1, '..')

# change this to point to your reachgrasp IO
sys.path.insert(1, '../../dataset_repos/reachgrasp/python')
sys.path.insert(1, '../../toolboxes/py/python-neo')
sys.path.insert(1, '../../toolboxes/py/python-odml')
sys.path.insert(1, '../../toolboxes/py/csn_toolbox')

import pickle

import numpy as np
import scipy
import quantities as pq
import matplotlib.pyplot as plt

# provides neo framework and I/Os to load exp and mdl data
import neo
import rg.restingstateio
import mesocircuitio

# provides core analysis library component
import elephant

# provides plots of the results
import plots

# for ipython notebook
# get_ipython().magic(u'matplotlib inline')

# useful in notebook -- delete for final
# reload(elephant.conversion)
# reload(elephant.statistics)
# reload(elephant.spikecorr)
# reload(elephant.surrogates)
# reload(plots)


# ## Global variables

# duration of recording to load
rec_start = 10.*pq.s
duration = 50.*pq.s


# ## Load experimental data

# data should be in a subdirectory 'data' relative to this notebook's location
# Load only first unit (ID: 1) of each channel
session_exp = rg.restingstateio.RestingStateIO(
    "data/i140701-004", print_diagnostic=False)
block_exp = session_exp.read_block(
    n_starts=[rec_start], n_stops=[rec_start + duration],
    channel_list=[], units=[1])

# select spike trains (min. 2 spikes, SUA only)
sts_exp = [
    st for st in
    block_exp.filter(sua=True, object="SpikeTrain") if len(st) > 2]

print("Number of experimental spike trains: " + str(len(sts_exp)))

# create binned spike trains
sts_exp_bin = elephant.conversion.Binned(
    sts_exp, binsize=20 * pq.ms,
    t_start=rec_start, t_stop=rec_start + duration)


# ## Load simulation data

# data should be in a subdirectory 'data' relative to this notebook's location
# Load only first unit (ID: 0) of each channel (one exc., one inh.)
# Load layer 5
session_mdl = mesocircuitio.MesoCircuitIO(
    "data/utah_array_spikes_60s.h5", print_diagnostic=False)
block_mdl = session_mdl.read_block(
    n_starts=[10 * pq.s], n_stops=[10 * pq.s + duration],
    channel_list=[], layer_list=['L5'],
    units=[], unit_type=['excitatory', 'inhibitory'])

sts_mdl = block_mdl.filter(
    targdict=[{'unit_type': 'excitatory'}, {'unit_id': 0}])
sts_mdl = [
    sts_mdl[i] for i in np.linspace(
        0, len(sts_mdl) - 1, len(sts_exp), dtype=int)]

print("Number of model spike trains: " + str(len(sts_mdl)))

# create binned spike trains
sts_mdl_bin = elephant.conversion.Binned(
    sts_mdl, binsize=20 * pq.ms,
    t_start=rec_start, t_stop=rec_start + duration)


# ## Plot population activity rasters

plt.figure(figsize=(16, 8))
plt.suptitle('Population Spike Activity', fontsize=14)

plt.subplot(2, 1, 1)
plt.title('Experiment')
sts_exp_cut = [st.time_slice(10 * pq.s, 20 * pq.s) for st in sts_exp]
plots.spike_raster(sts_exp_cut)

plt.subplot(2, 1, 2)
plt.title('Model')
sts_mdl_cut = [st.time_slice(10 * pq.s, 20 * pq.s) for st in sts_mdl]
plots.spike_raster(sts_mdl_cut)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Population histograms

phist_exp = elephant.statistics.peth(sts_exp_cut, 100 * pq.ms, output='rate')
phist_mdl = elephant.statistics.peth(sts_mdl_cut, 100 * pq.ms, output='rate')


# TODO: Show difference in rates?
# Note: Emphasize shape of data rather than absolute values
# Make scale axes more prominent (larger font size?)
plt.figure(figsize=(16, 6))
plt.suptitle('Population Spike Rate', fontsize=14)

plt.subplot(2, 1, 1)
plt.title('Experiment')
plots.population_rate(phist_exp)

plt.subplot(2, 1, 2)
plt.title('Model')
plots.population_rate(phist_mdl)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Rate distributions

rates_exp = [elephant.statistics.mean_firing_rate(st) for st in sts_exp]
rates_mdl = [elephant.statistics.mean_firing_rate(st) for st in sts_mdl]

# Note: For all following plots: can we compare the distributions via KS tests
# or other mechanisms?
plt.figure(figsize=(16, 4))
plt.suptitle('Mean Firing Rate Distribution', fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.rate_distribution(rates_exp, bin_width=0.5 * pq.Hz)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.rate_distribution(rates_mdl, bin_width=0.5 * pq.Hz)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Pooled inter-spike interval (ISI) distributions

isis_exp = [elephant.statistics.isi(st) for st in sts_exp]
isis_mdl = [elephant.statistics.isi(st) for st in sts_mdl]

# plot ISI distribution of unit with highest rate
plt.figure(figsize=(16, 4))
plt.suptitle('Example ISI Distribution', fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.isi_distribution(
    isis_exp[np.argsort(rates_exp)[-1]], bin_width=5 * pq.ms)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.isi_distribution(
    isis_mdl[np.argsort(rates_mdl)[-1]], bin_width=5 * pq.ms)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Distribution of the coefficient of variation (CV)

cvs_exp = [elephant.statistics.cv(isi) for isi in isis_exp]
cvs_mdl = [elephant.statistics.cv(isi) for isi in isis_mdl]

cvbins = np.linspace(0, 1.5, 50)

plt.figure(figsize=(16, 4))
plt.suptitle('CV Distribution', fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.cv_distribution(cvs_exp, bins=cvbins)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.cv_distribution(cvs_mdl, bins=cvbins)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Distribution of the local coefficient of variation (LV)

lvs_exp = [elephant.statistics.lv(isi) for isi in isis_exp]
lvs_mdl = [elephant.statistics.lv(isi) for isi in isis_mdl]

lvbins = np.linspace(0, 1.5, 50)

plt.figure(figsize=(16, 4))
plt.suptitle('LV Distribution', fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.lv_distribution(lvs_exp, bins=lvbins)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.lv_distribution(lvs_mdl, bins=lvbins)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Distribution of correlation coefficients

# TODO: Too slow!
import time
a = time.time()
cc_matrix_exp_prec = elephant.spikecorr.corrcoef_continuous(
    sts_exp, coinc_width=20 * pq.ms)
cc_matrix_mdl_prec = elephant.spikecorr.corrcoef_continuous(
    sts_mdl, coinc_width=20 * pq.ms)
print "Continuous:", time.time() - a

a = time.time()
sts_exp_bin = elephant.conversion.Binned(
    sts_exp, binsize=20 * pq.ms, t_start=10 * pq.s, t_stop=60 * pq.s)
sts_mdl_bin = elephant.conversion.Binned(
    sts_mdl, binsize=20 * pq.ms, t_start=10 * pq.s, t_stop=60 * pq.s)
print "Binning:", time.time() - a

a = time.time()
cc_matrix_exp_bc = elephant.spikecorr.corrcoef(sts_exp_bin, clip=True)
cc_matrix_mdl_bc = elephant.spikecorr.corrcoef(sts_mdl_bin, clip=True)
print "Binned Clipped:", time.time() - a

a = time.time()
cc_matrix_exp_bu = elephant.spikecorr.corrcoef(sts_exp_bin, clip=False)
cc_matrix_mdl_bu = elephant.spikecorr.corrcoef(sts_mdl_bin, clip=False)
print "Binned Unclipped:", time.time() - a

ccbins = np.linspace(-0.5, 0.5, 50)

# TODO Fix y-ticks
plt.figure(figsize=(16, 4))
plt.suptitle(
    "Binned Unclipped Pairwise Correlation Coefficient Distribution",
    fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.cc_distribution(cc_matrix_exp_bu, bins=ccbins)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.cc_distribution(cc_matrix_mdl_bu, bins=ccbins)

plt.subplots_adjust(hspace=0.5)


plt.figure(figsize=(16, 4))
plt.suptitle(
    "Binned Clipped Pairwise Correlation Coefficient Distribution",
    fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.cc_distribution(cc_matrix_exp_bc, bins=ccbins)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.cc_distribution(cc_matrix_mdl_bc, bins=ccbins)

plt.subplots_adjust(hspace=0.5)


plt.figure(figsize=(16, 4))
plt.suptitle(
    "Continuous Pairwise Correlation Coefficient Distribution",
    fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.cc_distribution(cc_matrix_exp_prec, bins=ccbins)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.cc_distribution(cc_matrix_mdl_prec, bins=ccbins)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Cross-correlograms

filename = [
    '../results/hbp_review_task/viz_output_exp.pkl',
    '../results/hbp_review_task/viz_output_mdl.pkl']
titles = ['Experiment', 'Model']

plt.figure()
for i in range(2):
    # load results
    f = open(filename[i], 'r')
    cc = pickle.load(f)
    f.close()

    # example: build correlation matrix
    num_neurons = cc['meta']['num_neurons']
    num_edges = cc['meta']['num_edges']

    C = np.zeros((num_neurons, num_neurons))
    x = cc['edges']['id_i'].astype(int)
    y = cc['edges']['id_j'].astype(int)
    p = cc['func_conn']['cch_peak']['pvalue'] / 1000.
    for edge_i in range(num_edges):
        C[x[edge_i], y[edge_i]] = C[y[edge_i], x[edge_i]] = p[edge_i]

    ax = plt.subplot(1, 2, i + 1)
    plt.pcolor(np.arange(num_neurons), np.arange(num_neurons), 1. - C)
    plt.title(titles[i])
    plt.xlabel("Neuron ID i")
    plt.ylabel("Neuron ID j")
    plt.axis('tight')
    ax.set_aspect('equal', 'datalim')
    plt.clim(0, 1)
    plt.colorbar()

plt.figure()
num_cc_plots = 5
for i in range(2):
    # load results
    f = open(filename[i], 'r')
    cc = pickle.load(f)
    f.close()

    # example: build correlation matrix
    num_neurons = cc['meta']['num_neurons']
    num_edges = cc['meta']['num_edges']

    for j in range(num_cc_plots):
        y = cc['edge_time_series']['cch'][j * 111 + 3, :]
        x = cc['edge_time_series']['times_ms'][j * 111 + 3, :]
        s1 = cc['edge_time_series']['sig_upper_975'][j * 111 + 3, :]
        s2 = cc['edge_time_series']['sig_lower_25'][j * 111 + 3, :]

        n1 = cc['edges']['id_i'][i]
        n2 = cc['edges']['id_j'][j]
        x1 = cc['neuron_topo']['x'][n1]
        y1 = cc['neuron_topo']['y'][n1]
        x2 = cc['neuron_topo']['x'][n2]
        y2 = cc['neuron_topo']['y'][n2]

        ax = plt.subplot(num_cc_plots, 2, i + 1 + j * 2)
        ind = np.argmin(np.abs(x))
        plt.plot(x[ind - 5:ind + 5], s1, 'g:')
        plt.plot(x[ind - 5:ind + 5], s2, 'r:')
        plt.plot(x, y, 'k-')
        if j == 0:
            plt.title(titles[i])
        plt.ylabel(
            "Cross-correlogram %i (%i,%i) - %i (%i,%i)" %
            (n1, x1, y1, n2, x2, y2))
        if j == num_cc_plots - 1:
            plt.xlabel("time (ms)")
        else:
            ax.set_xticklabels([])

        plt.axis('tight')

plt.show()
