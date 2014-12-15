#==============================================================================
# Initialization
#==============================================================================

# paths
import os
import sys
# to find our "special" elephant
sys.path.insert(1, '..')
# change this to point to your reachgrasp IO
sys.path.insert(1, '../../dataset_repos/reachgrasp/python')
sys.path.insert(1, '../../toolboxes/py/python-neo')
sys.path.insert(1, '../../toolboxes/py/python-odml')
sys.path.insert(1, '../../toolboxes/py/csn_toolbox')

import quantities as pq

# provides neo framework and I/Os to load exp and mdl data
import neo
import rg.restingstateio
import mesocircuitio


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

for st in sts_exp:
    st.annotate(use_st=True)

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

# select neuron
sts_mdl = block_mdl.filter(
    unit_type='excitatory', unit_id=0, object="SpikeTrain")[:len(sts_exp)]

for st in sts_mdl:
    st.annotate(use_st=True)

print("Number of model spike trains: " + str(len(sts_mdl)))

# save experimental
filename = 'data/experiment.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_exp)
session.close()

# save model
filename = 'data/model.h5'
if os.path.exists(filename):
    os.remove(filename)
session = neo.NeoHdf5IO(filename=filename)
session.save(block_mdl)
session.close()