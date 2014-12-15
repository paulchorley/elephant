#==============================================================================
# Initialization
#==============================================================================

# paths
import sys
# to find our "special" elephant
sys.path.insert(1, '..')
# change this to point to your reachgrasp IO
sys.path.insert(1, '../../dataset_repos/reachgrasp/python')
sys.path.insert(1, '../../toolboxes/py/python-neo')
sys.path.insert(1, '../../toolboxes/py/python-odml')
sys.path.insert(1, '../../toolboxes/py/csn_toolbox')

import os
import glob

import numpy as np
import quantities as pq

# provides neo framework and I/Os to load exp and mdl data
import rg.restingstateio
import mesocircuitio

# provides core analysis library component
import elephant

import h5py_wrapper.wrapper


#==============================================================================
# Global variables
#==============================================================================

# duration of recording to load
rec_start = 10.*pq.s
duration = 50.*pq.s


#==============================================================================
# Load experimental data
#==============================================================================

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


#==============================================================================
# Load simulation data
#==============================================================================

# data should be in a subdirectory 'data' relative to this notebook's location
# Load only first unit (ID: 0) of each channel (one exc., one inh.)
# Load layer 5
session_mdl = mesocircuitio.MesoCircuitIO(
    "data/utah_array_spikes_60s.h5", print_diagnostic=False)
block_mdl = session_mdl.read_block(
    n_starts=[10 * pq.s], n_stops=[10 * pq.s + duration],
    channel_list=[], layer_list=['L5'],
    units=[], unit_type=['excitatory', 'inhibitory'])

# select neuron
sts_mdl = block_mdl.filter(
    unit_type='excitatory', unit_id=0, object="SpikeTrain")[:len(sts_exp)]

print("Number of model spike trains: " + str(len(sts_mdl)))

# create binned spike trains
sts_mdl_bin = elephant.conversion.Binned(
    sts_mdl, binsize=20 * pq.ms,
    t_start=rec_start, t_stop=rec_start + duration)

num_neurons = len(sts_exp)


#==============================================================================
# Calculate measures
#==============================================================================

rates = {}
rates['exp'] = [elephant.statistics.mean_firing_rate(st) for st in sts_exp]
rates['mdl'] = [elephant.statistics.mean_firing_rate(st) for st in sts_mdl]

isis_exp = [elephant.statistics.isi(st) for st in sts_exp]
isis_mdl = [elephant.statistics.isi(st) for st in sts_mdl]

cvs = {}
cvs['exp'] = [elephant.statistics.cv(isi) for isi in isis_exp]
cvs['mdl'] = [elephant.statistics.cv(isi) for isi in isis_mdl]

lvs = {}
lvs['exp'] = [elephant.statistics.lv(isi) for isi in isis_exp]
lvs['mdl'] = [elephant.statistics.lv(isi) for isi in isis_mdl]


#==============================================================================
# Rewrite files
#==============================================================================

num_total_pairs = 0
for ni in range(num_neurons):
    for nj in range(ni, num_neurons):
        num_total_pairs += 1

cc = {}
for dta in ['exp', 'mdl']:
    cc[dta] = {}

    cc[dta]['meta'] = {}

    cc[dta]['neuron_topo'] = {}
    cc[dta]['neuron_topo']['x'] = {}
    cc[dta]['neuron_topo']['y'] = {}

    cc[dta]['func_conn'] = {}
    cc[dta]['func_conn']['cch_peak'] = {}
    cc[dta]['func_conn']['cch_peak']['pvalue'] = {}

    cc[dta]['edges'] = {}
    cc[dta]['edges']['id_i'] = {}
    cc[dta]['edges']['id_j'] = {}

    cc[dta]['neuron_single_values'] = {}
    cc[dta]['neuron_single_values']['rate'] = {}
    cc[dta]['neuron_single_values']['cv'] = {}
    cc[dta]['neuron_single_values']['lv'] = {}
    cc[dta]['neuron_single_values']['behavior'] = {}

    cc[dta]['edge_time_series'] = {}
    cc[dta]['edge_time_series']['cch'] = {}
    cc[dta]['edge_time_series']['sig_upper_975'] = {}
    cc[dta]['edge_time_series']['sig_lower_25'] = {}

    cc[dta]['meta']['num_neurons'] = num_neurons
    cc[dta]['meta']['num_edges'] = num_total_pairs

# values per neuron
for dta, sts in zip(['exp', 'mdl'], [sts_exp, sts_mdl]):
    for neuron_i in range(num_neurons):
        cc[dta]['neuron_topo']['x'][neuron_i] = \
            int(neuron_i) / 10
        cc[dta]['neuron_topo']['y'][neuron_i] = \
            int(neuron_i) % 10

        if dta == 'exp':
            cc[dta]['neuron_single_values']['behavior'][neuron_i] = np.array([
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 0, 0, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 0, 0, 0, 1, 1, 1, 1,
                2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                2, 2, 2, 0, 0, 3, 3, 3, 3, 3,
                2, 2, 2, 0, 0, 3, 3, 3, 3, 3,
                2, 2, 2, 0, 0, 0, 3, 3, 3, 3,
                2, 2, 2, 0, 0, 0, 3, 3, 3, 3,
                2, 2, 2, 0, 0, 0, 3, 3, 3, 3])[neuron_i]
        else:
            cc[dta]['neuron_single_values']['behavior'][neuron_i] = np.zeros(
                (10, 10))

        cc[dta]['neuron_single_values']['rate'][neuron_i] = rates[dta][neuron_i]
        cc[dta]['neuron_single_values']['cv'][neuron_i] = cvs[dta][neuron_i]
        cc[dta]['neuron_single_values']['lv'][neuron_i] = lvs[dta][neuron_i]

# values per edge
num_tasks = len(glob.glob(
    '../results/hbp_review_task/correlation_output_*.h5'))
for job_parameter in range(num_tasks):
    filename = \
        '../results/hbp_review_task/correlation_output_' + \
        str(job_parameter) + '.h5'
    if not os.path.exists(filename):
        raise IOError('Cannot find file %s.', filename)
    print("Assembly of : %s" % filename)

    cc_part = h5py_wrapper.wrapper.load_h5(filename)

    for dta, sts in zip(['exp', 'mdl'], [sts_exp, sts_mdl]):
        for calc_i in cc_part[dta]['pvalue']:
            cc[dta]['func_conn']['cch_peak']['pvalue'][calc_i] = \
                cc_part[dta]['pvalue'][calc_i]

            cc[dta]['edges']['id_i'][calc_i] = cc_part[dta]['unit_i'][calc_i]
            cc[dta]['edges']['id_j'][calc_i] = cc_part[dta]['unit_j'][calc_i]

            cc[dta]['edge_time_series']['cch'][calc_i] = \
                cc_part[dta]['original'][calc_i]
            cc[dta]['edge_time_series']['sig_upper_975'][calc_i] = \
                cc_part[dta]['surr'][calc_i][975, :]
            cc[dta]['edge_time_series']['sig_lower_25'][calc_i] = \
                cc_part[dta]['surr'][calc_i][25, :]

    del cc_part

# write parameters to disk
h5py_wrapper.wrapper.add_to_h5(
    '../results/hbp_review_task/viz_output_exp.h5',
    cc['exp'], write_mode='w', overwrite_dataset=True)
h5py_wrapper.wrapper.add_to_h5(
    '../results/hbp_review_task/viz_output_mdl.h5',
    cc['mdl'], write_mode='w', overwrite_dataset=True)