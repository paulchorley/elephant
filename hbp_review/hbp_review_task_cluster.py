#!/usr/bin/python
# PBS -N HBP_REVIEW_DEMO
# PBS -d /users/denker/Projects/hbp_review/hbp_review
# PBS -o /scratch/denker/logs/output.${PBS_JOBID}
# PBS -e /scratch/denker/logs/error.${PBS_JOBID}
# PBS -t 0-199
# PBS -l mem=2GB,walltime=4:00:00


#==============================================================================
# Initialization
#==============================================================================

# this number relates to the "-t" parameter:
#   -t 0-X => num_tasks=X+1
num_tasks = 200

# get job parameter
import os
PBS_value = os.getenv('PBS_ARRAYID')
if PBS_value is not None:
    job_parameter = int(PBS_value)
else:
    job_parameter = 0

# paths
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

# provides neo framework and I/Os to load exp and mdl data
import rg.restingstateio
import mesocircuitio

# provides core analysis library component
import elephant



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

# select random excitatory and inhibitory neurons
# idx = np.random.permutation(range(len(block_mdl.segments[0].spiketrains)))
# sts_mdl = [block_mdl.segments[0].spiketrains[i] for i in idx[:len(sts_exp)]]

# select neuron
sts_mdl = block_mdl.filter(
    unit_type='excitatory', unit_id=0, object="SpikeTrain")[:len(sts_exp)]

print("Number of model spike trains: " + str(len(sts_mdl)))


# ## Cross-correlograms
num_surrs = 1000
max_lag_bins = 200
lag_res = 1 * pq.ms
max_lag = max_lag_bins * lag_res
smoothing = 10 * pq.ms

num_neurons_exp = len(sts_exp)
num_ccs = (num_neurons_exp ** 2 - num_neurons_exp) / 2

cc = {}
for dta in ['exp', 'mdl']:
    cc[dta] = {}
    cc[dta]['unit_i'] = {}
    cc[dta]['unit_j'] = {}
    cc[dta]['times_ms'] = {}
    cc[dta]['original'] = {}
    cc[dta]['surr'] = {}
    cc[dta]['original_measure'] = {}
    cc[dta]['surr_measure'] = {}
    cc[dta]['pvalue'] = {}

# create all combinations of tasks
num_total_pairs = 0
all_combos_unit_i = []
all_combos_unit_j = []
for ni in range(num_neurons_exp - 1):
    for nj in range(ni, num_neurons_exp):
        all_combos_unit_i.append(ni)
        all_combos_unit_j.append(nj)
        num_total_pairs += 1

# distribute number of calculations to each task
max_calc_per_task = num_total_pairs / num_tasks
num_calc_last_task = num_total_pairs % num_tasks

# calculate indices in cc['unit_i'] list which to calculate for each task
task_starts_idx = range(0, num_total_pairs, max_calc_per_task)
if num_calc_last_task == 0:
    task_stop_idx = [_ + max_calc_per_task for _ in task_starts_idx]
else:
    task_stop_idx = [_ + max_calc_per_task for _ in task_starts_idx[0:-1]]
    task_stop_idx.append(task_starts_idx[-1] + num_calc_last_task)

print("Task Nr.: %i" % job_parameter)
print("Number of tasks: %i" % num_tasks)
print("Max. calcs per task: %i" % max_calc_per_task)
print("Calcs for last task: %i" % num_calc_last_task)


def cch_measure(cch):
    return np.sum(cch[ind - 5:ind + 5].magnitude)

for dta, sts in zip(['exp', 'mdl'], [sts_exp, sts_mdl]):
    for calc_i in range(
            task_starts_idx[job_parameter], task_stop_idx[job_parameter]):
        # save neuron i,j index
        ni = all_combos_unit_i[calc_i]
        nj = all_combos_unit_j[calc_i]

        cc[dta]['unit_i'][calc_i] = ni
        cc[dta]['unit_j'][calc_i] = nj

        print("Cross-correlating %i and %i" % (ni, nj))

        # original CCH
        cco = elephant.spikecorr.cch(
            sts[ni], sts[nj], w=lag_res, lag=max_lag, smooth=smoothing)
        cc[dta]['original'][calc_i] = cco.magnitude
        cc[dta]['times_ms'][calc_i] = cco.times.rescale(pq.ms).magnitude

        # extract measure
        ind = np.argmin(np.abs(cco.times))
        ccom = cch_measure(cco)
        cc[dta]['original_measure'][calc_i] = ccom

        surr_i = elephant.surrogates.spike_dithering(
            sts[ni], dither=50. * pq.ms, n=num_surrs)
        surr_j = elephant.surrogates.spike_dithering(
            sts[nj], dither=50. * pq.ms, n=num_surrs)

        ccs = []
        ccsm = []
        for surrogate in range(num_surrs):
            scc = elephant.spikecorr.cch(
                surr_i[surrogate], surr_j[surrogate],
                w=lag_res, lag=max_lag, smooth=smoothing)
            ccs.append(scc.magnitude)
            ccsm.append(cch_measure(scc))
        cc[dta]['surr'][calc_i] = np.array(ccs)
        cc[dta]['surr_measure'][calc_i] = np.sort(ccsm)
        cc[dta]['pvalue'][calc_i] = np.count_nonzero(np.array(ccsm) >= ccom)

# write parameters to disk
import h5py_wrapper.wrapper
h5py_wrapper.wrapper.add_to_h5(
    '../results/hbp_review_task/correlation_output_' +
    str(job_parameter) + '.h5',
    cc, write_mode='w', overwrite_dataset=True)
