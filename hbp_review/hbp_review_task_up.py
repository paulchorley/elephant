#!/usr/bin/python

#==============================================================================
# Initialization
#==============================================================================

# this number relates to the "-t" parameter:
#   -t 0-X => num_tasks=X+1
num_tasks = 200

# get job parameter: a number between 0 and num_tasks-1
import os
PBS_value = os.getenv('PBS_ARRAYID')
if PBS_value is not None:
    job_parameter = int(PBS_value)
else:
    job_parameter = 0

import numpy as np
import quantities as pq
import neo
import elephant


#==============================================================================
# Load data
#==============================================================================

filename = 'data/model.h5'  # <- this is the task input
session = neo.NeoHdf5IO(filename=filename)
block = session.read_block()

# select spike trains
sts = block.filter(use_st=True)

print("Number of spike trains: " + str(len(sts)))

#==============================================================================
# Cross-correlograms
#==============================================================================

num_surrs = 1000
max_lag_bins = 200
lag_res = 1 * pq.ms
max_lag = max_lag_bins * lag_res
smoothing = 10 * pq.ms

num_neurons = len(sts)

cc = {}
cc = {}
cc['unit_i'] = {}
cc['unit_j'] = {}
cc['times_ms'] = {}
cc['original'] = {}
cc['surr'] = {}
cc['original_measure'] = {}
cc['surr_measure'] = {}
cc['pvalue'] = {}

# create all combinations of tasks
num_total_pairs = 0
all_combos_unit_i = []
all_combos_unit_j = []
for ni in range(num_neurons):
    for nj in range(ni, num_neurons):
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

for calc_i in range(
        task_starts_idx[job_parameter], task_stop_idx[job_parameter]):
    # save neuron i,j index
    ni = all_combos_unit_i[calc_i]
    nj = all_combos_unit_j[calc_i]

    cc['unit_i'][calc_i] = ni
    cc['unit_j'][calc_i] = nj

    print("Cross-correlating %i and %i" % (ni, nj))

    # original CCH
    cco = elephant.spikecorr.cch(
        sts[ni], sts[nj], w=lag_res, lag=max_lag, smooth=smoothing)
    cc['original'][calc_i] = cco.magnitude
    cc['times_ms'][calc_i] = cco.times.rescale(pq.ms).magnitude

    # extract measure
    ind = np.argmin(np.abs(cco.times))
    ccom = cch_measure(cco)
    cc['original_measure'][calc_i] = ccom

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
    cc['surr'][calc_i] = np.array(ccs)
    cc['surr_measure'][calc_i] = np.sort(ccsm)
    cc['pvalue'][calc_i] = np.count_nonzero(np.array(ccsm) >= ccom)

# write parameters to disk
import h5py_wrapper.wrapper
h5py_wrapper.wrapper.add_to_h5(
    'correlation_output_' + filename + '_' + str(job_parameter) + '.h5',
    cc, write_mode='w', overwrite_dataset=True)
