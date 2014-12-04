
# coding: utf-8

## DO NOT WORK ON THIS NOTEBOOK WORK ON THE `hbp_review.py` FILE

### Initialization

# In[13]:

import sys
sys.path.append('..')

# change this to point to your reachgrasp IO
sys.path.append('../../dataset_repos/reachgrasp/python')

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

# provides data model for ephys data
import neo

# provides core analysis library component
import elephant

# provides plots of the results
import rg.restingstateio
import plots

get_ipython().magic(u'matplotlib inline')


# In[14]:

reload(elephant.conversion)
reload(elephant.statistics)
reload(elephant.xcorr)
reload(elephant.surrogates)
reload(plots)


### Load experimental data

# In[17]:

# data should be in a subdirectory 'data' relative to this notebook's location
# TODO: Make sure this loads only single units -- it's not sorted
# Note: Use this data sets because it is more stationary than 
session = rg.restingstateio.RestingStateIO("data/i140701-004")
block = session.read_block(n_starts=[10*pq.s],n_stops=[20*pq.s],channel_list=[], units=[1,2,3])

# Is this too much of a hack (some below)?
sts_exp = [ st for st in block.segments[0].spiketrains if len(st) > 2 and st.annotations['sua']==True ]


# In[67]:

print("Number of experimental spiketrains: "+str(len(sts_exp)))


### Load simulation data (still Renato's data, not yet Espen and Johanna's!)

# In[68]:

# data should be in a subdirectory 'data' relative to this notebook's location
# TODO: Convert data to a hdf5 file and load it via neo.io.hdf5io
sts = np.load('data/sts_mdl.npy')
mdl_t_start = 0*pq.s
mdl_t_stop = 10*pq.s
n_mdl = 150
s_start = int(mdl_t_start.rescale(pq.ms).magnitude)
s_stop = int(mdl_t_stop.rescale(pq.ms).magnitude)
sts_mdl = [ neo.SpikeTrain(st[(st>=s_start)&(st<s_stop)],
                           t_start = mdl_t_start, t_stop = mdl_t_stop,
                           units = pq.ms) for st in sts[:n_mdl] ]


# In[69]:

print("Number of simulation spiketrains: "+str(len(sts_mdl)))


### Plot population activity rasters

# In[70]:

plt.figure(figsize=(16,8))
plt.suptitle('Population Spike Activity',fontsize=14)

plt.subplot(2,1,1)
plt.title('Experiment')
plots.spike_raster(sts_exp)

plt.subplot(2,1,2)
plt.title('Model')
plots.spike_raster(sts_mdl)

plt.subplots_adjust(hspace=0.5)


### Population histograms

# In[71]:

phist_exp = elephant.statistics.peth(sts_exp, 100*pq.ms, output='rate')
phist_mdl = elephant.statistics.peth(sts_mdl, 100*pq.ms, output='rate')


# In[72]:

#TODO: Show difference in rates?
# Note: Emphasize shape of data rather than absolute values
# Make scale axes more prominent (larger font size?)
plt.figure(figsize=(16,6))
plt.suptitle('Population Spike Rate',fontsize=14)

plt.subplot(2,1,1)
plt.title('Experiment')
plots.population_rate(phist_exp)

plt.subplot(2,1,2)
plt.title('Model')
plots.population_rate(phist_mdl)

plt.subplots_adjust(hspace=0.5)


### Rate distributions

# In[73]:

rates_exp = [ elephant.statistics.mean_firing_rate(st) for st in sts_exp ]
rates_mdl = [ elephant.statistics.mean_firing_rate(st) for st in sts_mdl ]


# In[74]:

# Note: For all following plots: can we compare the distributions via KS tests or other mechanisms?
plt.figure(figsize=(16,4))
plt.suptitle('Mean Firing Rate Distribution',fontsize=14)

plt.subplot(1,2,1)
plt.title('Experiment')
plots.rate_distribution(rates_exp,bin_width=2*pq.Hz)

plt.subplot(1,2,2)
plt.title('Model')
plots.rate_distribution(rates_mdl,bin_width=0.25*pq.Hz)

plt.subplots_adjust(hspace=0.5)


### Pooled inter-spike interval (ISI) distributions

# In[14]:

isis_exp = [ elephant.statistics.isi(st) for st in sts_exp ]
isis_mdl = [ elephant.statistics.isi(st) for st in sts_mdl ]


# In[15]:

# Plot ISI distribution of unit with highest rate

plt.figure(figsize=(16,4))
plt.suptitle('Example ISI Distribution',fontsize=14)

plt.subplot(1,2,1)
plt.title('Experiment')
plots.isi_distribution(isis_exp[np.argsort(rates_exp)[-1]],bin_width=5*pq.ms)

plt.subplot(1,2,2)
plt.title('Model')
plots.isi_distribution(isis_mdl[np.argsort(rates_mdl)[-1]],bin_width=20*pq.ms)

plt.subplots_adjust(hspace=0.5)


### Distribution of the coefficient of variation (CV)

# In[16]:

cvs_exp = [ elephant.statistics.cv(isi) for isi in isis_exp ]
cvs_mdl = [ elephant.statistics.cv(isi) for isi in isis_mdl ]


# In[17]:

plt.figure(figsize=(16,4))
plt.suptitle('CV Distribution',fontsize=14)

plt.subplot(1,2,1)
plt.title('Experiment')
plots.cv_distribution(cvs_exp,n_bins=25)

plt.subplot(1,2,2)
plt.title('Model')
plots.cv_distribution(cvs_mdl,n_bins=25)

plt.subplots_adjust(hspace=0.5)


### Distribution of the local coefficient of variation (LV)

# In[18]:

lvs_exp = [ elephant.statistics.lv(isi) for isi in isis_exp ]
lvs_mdl = [ elephant.statistics.lv(isi) for isi in isis_mdl ]


# In[19]:

plt.figure(figsize=(16,4))
plt.suptitle('LV Distribution',fontsize=14)

plt.subplot(1,2,1)
plt.title('Experiment')
plots.lv_distribution(lvs_exp,n_bins=25)

plt.subplot(1,2,2)
plt.title('Model')
plots.lv_distribution(lvs_mdl,n_bins=25)

plt.subplots_adjust(hspace=0.5)


### Distribution of correlation coefficients

# In[75]:

cc_matrix_exp = elephant.statistics.corrcoef(sts_exp, binsize = 20*pq.ms, clip=True)
cc_matrix_mdl = elephant.statistics.corrcoef(sts_mdl, binsize = 20*pq.ms, clip=True)


# In[77]:

# TODO Fix y-ticks
plt.figure(figsize=(16,4))
plt.suptitle('Pairwise Correlation Coefficient Distribution',fontsize=14)

plt.subplot(1,2,1)
plt.title('Experiment')
plots.cc_distribution(cc_matrix_exp,n_bins=50)

plt.subplot(1,2,2)
plt.title('Model')
plots.cc_distribution(cc_matrix_mdl,n_bins=50)

plt.subplots_adjust(hspace=0.5)


### Cross-correlograms

# In[64]:

num_surrs=100
max_lag_bins=100
lag_res=1*pq.ms
max_lag=max_lag_bins*lag_res
smoothing=10*pq.ms

num_neurons_exp=len(sts_exp)
num_ccs = (num_neurons_exp**2 - num_neurons_exp)/2

cc_original = []
cc_surrs = []
unit_i=[]
unit_j=[]

sts=sts_exp

for ni in [0,1,2,3]:
    for nj in [0,1,2,3]:
        unit_i.append(ni)
        unit_j.append(nj)
        
        print "Cross-correlating ", ni, " and ", nj
        cc_original.append(elephant.xcorr.cch(sts[ni],sts[nj],w=lag_res,lag=max_lag,smooth=smoothing))

        surr_i = elephant.surrogates.spike_dithering(sts[ni],dither=50*pq.ms,n=num_surrs)
        surr_j = elephant.surrogates.spike_dithering(sts[nj],dither=50*pq.ms,n=num_surrs)
        
        cc_surrs.append([])
        for surrogate in range(num_surrs):
            cc_surrs[-1].append(elephant.xcorr.cch(surr_i[surrogate],surr_j[surrogate],w=lag_res,lag=max_lag,smooth=smoothing))


# In[65]:

for selected_unit in range(len(cc_original)):   
    plt.subplot2grid((4,4),(unit_i[selected_unit],unit_j[selected_unit]))

    surr_matrix=np.sort(np.array(cc_surrs[selected_unit]),axis=0)
    plt.plot(cc_original[selected_unit].times.magnitude,surr_matrix[int(num_surrs*0.05)],color=[0.3,0.3,0.3])
    plt.plot(cc_original[selected_unit].times.magnitude,surr_matrix[int(num_surrs*0.95)],color=[0.3,0.3,0.3])
    plt.plot(cc_original[selected_unit].times.magnitude,cc_original[selected_unit].magnitude)
    plt.axis('tight')
    plt.subplot2grid((4,4),(unit_j[selected_unit],unit_i[selected_unit]))
    surr_matrix=np.sort(np.array(cc_surrs[selected_unit]),axis=0)
    plt.plot(cc_original[selected_unit].times.magnitude,surr_matrix[int(num_surrs*0.05)],color=[0.3,0.3,0.3])
    plt.plot(cc_original[selected_unit].times.magnitude,surr_matrix[int(num_surrs*0.95)],color=[0.3,0.3,0.3])
    plt.plot(cc_original[selected_unit].times.magnitude,cc_original[selected_unit].magnitude)
    plt.axis('tight')    


### Higher-order analysis (CuBIC) ?

# In[ ]:



