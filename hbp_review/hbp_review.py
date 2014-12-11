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

import numpy as np
import quantities as pq

# remove -- for cluster calculations only
# import matplotlib
# matplotlib.use('agg')

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

# select random excitatory and inhibitory neurons
# idx = np.random.permutation(range(len(block_mdl.segments[0].spiketrains)))
# sts_mdl = [block_mdl.segments[0].spiketrains[i] for i in idx[:len(sts_exp)]]

# select neuron
sts_mdl = block_mdl.filter(
    unit_type='excitatory', unit_id=0, object="SpikeTrain")[:len(sts_exp)]

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
plots.rate_distribution(rates_exp, bin_width=2 * pq.Hz)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.rate_distribution(rates_mdl, bin_width=0.25 * pq.Hz)

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
    isis_mdl[np.argsort(rates_mdl)[-1]], bin_width=20 * pq.ms)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Distribution of the coefficient of variation (CV)

cvs_exp = [elephant.statistics.cv(isi) for isi in isis_exp]
cvs_mdl = [elephant.statistics.cv(isi) for isi in isis_mdl]

plt.figure(figsize=(16, 4))
plt.suptitle('CV Distribution', fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.cv_distribution(cvs_exp, n_bins=25)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.cv_distribution(cvs_mdl, n_bins=25)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Distribution of the local coefficient of variation (LV)

lvs_exp = [elephant.statistics.lv(isi) for isi in isis_exp]
lvs_mdl = [elephant.statistics.lv(isi) for isi in isis_mdl]

plt.figure(figsize=(16, 4))
plt.suptitle('LV Distribution', fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.lv_distribution(lvs_exp, n_bins=25)

plt.subplot(1, 2, 2)
plt.title('Model')
plots.lv_distribution(lvs_mdl, n_bins=25)

plt.subplots_adjust(hspace=0.5)
# plt.show()


# ## Distribution of correlation coefficients

# TODO: Too slow!
import time
a = time.time()
cc_matrix_exp_prec = elephant.spikecorr.corrcoef(
    sts_exp, bin_size=10 * pq.ms)
cc_matrix_mdl_prec = elephant.spikecorr.corrcoef(
    sts_mdl, bin_size=10 * pq.ms)
print "Continuous:", time.time() - a

a = time.time()
sts_exp_bin = elephant.conversion.Binned(
    sts_exp, binsize=20 * pq.ms, t_start=10 * pq.s, t_stop=60 * pq.s)
sts_mdl_bin = elephant.conversion.Binned(
    sts_mdl, binsize=20 * pq.ms, t_start=10 * pq.s, t_stop=60 * pq.s)
cc_matrix_exp = elephant.spikecorr.corrcoef_binned(sts_exp_bin, clip=False)
cc_matrix_mdl = elephant.spikecorr.corrcoef_binned(sts_mdl_bin, clip=False)
print "Binned:", time.time() - a

# TODO Fix y-ticks
plt.figure(figsize=(16, 4))
plt.suptitle(
    "Binned Pairwise Correlation Coefficient Distribution", fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.cc_distribution(cc_matrix_exp, n_bins=np.linspace(-0.5, 0.5, 50))

plt.subplot(1, 2, 2)
plt.title('Model')
plots.cc_distribution(cc_matrix_mdl, n_bins=np.linspace(-0.5, 0.5, 50))

plt.subplots_adjust(hspace=0.5)
plt.show()
plt.figure(figsize=(16, 4))
plt.suptitle(
    "Continuous Pairwise Correlation Coefficient Distribution", fontsize=14)

plt.subplot(1, 2, 1)
plt.title('Experiment')
plots.cc_distribution(cc_matrix_exp_prec, n_bins=np.linspace(-0.5, 0.5, 50))

plt.subplot(1, 2, 2)
plt.title('Model')
plots.cc_distribution(cc_matrix_mdl_prec, n_bins=np.linspace(-0.5, 0.5, 50))

plt.subplots_adjust(hspace=0.5)
plt.show()


# ## Cross-correlograms

num_surrs = 100
max_lag_bins = 100
lag_res = 1 * pq.ms
max_lag = max_lag_bins * lag_res
smoothing = 10 * pq.ms

num_neurons_exp = len(sts_exp)
num_ccs = (num_neurons_exp ** 2 - num_neurons_exp) / 2

cc = {}
cc['original'] = []
cc['surr'] = []
cc['original_measure'] = []
cc['surr_measure'] = []
cc['pos'] = []
cc['unit_i'] = []
cc['unit_j'] = []

sts = sts_mdl

for ni in [0, 1, 2, 3]:  # range(num_neurons_exp):
    for nj in [0, 1, 2, 3]:  # range(ni, num_neurons_exp):
        cc['unit_i'].append(ni)
        cc['unit_j'].append(nj)

        print "Cross-correlating ", ni, " and ", nj
        cco = elephant.spikecorr.cch(
            sts[ni], sts[nj], w=lag_res, lag=max_lag, smooth=smoothing)
        cc['original'].append(cco)

        ind = np.argmin(np.abs(cco.times))
        ccom = np.sum(cco[ind - 5:ind + 5].magnitude)
        cc['original_measure'].append(ccom)

        surr_i = elephant.surrogates.spike_dithering(
            sts[ni], dither=50 * pq.ms, n=num_surrs)
        surr_j = elephant.surrogates.spike_dithering(
            sts[nj], dither=50 * pq.ms, n=num_surrs)

        ccs = []
        ccsm = []
        for surrogate in range(num_surrs):
            scc = elephant.spikecorr.cch(
                surr_i[surrogate], surr_j[surrogate],
                w=lag_res, lag=max_lag, smooth=smoothing)
            ccs.append(scc)
            ccsm.append(np.sum(scc[ind - 5:ind + 5].magnitude))
        cc['surr'].append(ccs)
        cc['surr_measure'].append(np.sort(ccsm))
        cc['pos'].append(np.count_nonzero(np.array(ccsm) >= ccom))

# write parameters to disk
import h5py_wrapper.wrapper
h5py_wrapper.wrapper.add_to_h5(
    'correlation_output.h5', cc, write_mode='w', overwrite_dataset=True)

# plot example CC's
for selected_unit in range(len(cc['original'])):
    plt.subplot2grid((4, 4), (cc['unit_i'][selected_unit], cc['unit_j'][selected_unit]))
    surr_matrix = np.sort(np.array(cc['surr'][selected_unit]), axis=0)
    plt.plot(cc['original'][selected_unit].times.magnitude, surr_matrix[int(num_surrs * 0.05)], color=[0.3, 0.3, 0.3])
    plt.plot(cc['original'][selected_unit].times.magnitude, surr_matrix[int(num_surrs * 0.95)], color=[0.3, 0.3, 0.3])
    plt.plot(cc['original'][selected_unit].times.magnitude, cc['original'][selected_unit].magnitude)
    plt.axis('tight')

    plt.subplot2grid((4, 4), (cc['unit_j'][selected_unit], cc['unit_i'][selected_unit]))
    plt.plot(cc['original'][selected_unit].times.magnitude, surr_matrix[int(num_surrs * 0.05)], color=[0.3, 0.3, 0.3])
    plt.plot(cc['original'][selected_unit].times.magnitude, surr_matrix[int(num_surrs * 0.95)], color=[0.3, 0.3, 0.3])
    plt.plot(cc['original'][selected_unit].times.magnitude, cc['original'][selected_unit].magnitude)
    plt.axis('tight')
plt.show()


# ## Higher-order analysis (CuBIC) ?

# In[ ]:
plt.show()
