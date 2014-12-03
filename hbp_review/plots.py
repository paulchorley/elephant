import sys

import numpy as np
import quantities as pq
import matplotlib.pyplot as plt

from neo.io import BlackrockIO



def load_data(exp_t_start = 0.0*pq.s, exp_t_stop = 10.0*pq.s,
              mdl_t_start = 0.0*pq.s, mdl_t_stop = 10.0*pq.s, n_mdl = 100):

    session = BlackrockIO("../../data/i140701-004")
    block = session.read_block(n_starts=[exp_t_start],n_stops=[exp_t_stop],
                               channel_list=[], nsx=None, units=[],
                               events=False, waveforms=False)
    sts_exp = [ st for st in block.segments[0].spiketrains
            if st.annotations['unit_id'] not in [0,255]
            and len(st) > 2 ]

    import neo
    sts = np.load('../../data/sts_mdl.npy')
    s_start = int(mdl_t_start.rescale(pq.ms).magnitude)
    s_stop = int(mdl_t_stop.rescale(pq.ms).magnitude)
    sts_mdl = [ neo.SpikeTrain(st[(st>=s_start)&(st<s_stop)],
                               t_start = mdl_t_start, t_stop = mdl_t_stop,
                               units = pq.ms) for st in sts[:n_mdl] ]

    return sts_exp, sts_mdl


def spike_raster(sts):
    for i in range(len(sts)):
        t = sts[i].rescale(pq.s).magnitude
        plt.plot(t,i*np.ones_like(t),'k.',ms=2)
    plt.axis('tight')
    plt.xlim(t[0],t[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Unit Index')


def population_rate(phist):
    t = phist.times.rescale(pq.s).magnitude
    h = phist.rescale(pq.Hz).magnitude
    w = phist.sampling_period.rescale(pq.s).magnitude
    x = np.zeros((len(h)*2))
    y = np.zeros((len(h)*2))
    x[::2]  = t
    x[1::2] = t+w
    y[::2] = y[1::2] = h
    plt.plot(x,y,'k')
    plt.xlim(x[0],x[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Rate (Hz)')


def rate_distribution(rates,bin_width=2*pq.Hz):

    w = bin_width.rescale(pq.Hz).magnitude
    x = [ rate.rescale(pq.Hz).magnitude for rate in rates ]

    bins = np.arange(np.min(x),np.max(x)+w,w)
    h,b = np.histogram(x,bins,density=True)

    plt.bar(left=b[:-1],height=h,width=w,fc='k')

    m = np.mean(x)
    plt.plot([m,m],plt.ylim(),'b')
    plt.axis('tight')

    plt.xlabel('Spike Rate (Hz)')
    plt.ylabel('PDF')


def isi_distribution(isi,bin_width=2*pq.ms):

    w = bin_width.rescale(pq.ms).magnitude
    x = isi.rescale(pq.ms).magnitude

    bins = np.arange(np.min(x),np.max(x)+w,w)
    h,b = np.histogram(x,bins,density=True)

    plt.bar(left=b[:-1],height=h,width=w,fc='k')

    m = np.mean(x)
    plt.plot([m,m],plt.ylim(),'b')
    plt.axis('tight')

    plt.xlabel('ISI (ms)')
    plt.ylabel('PDF')


def cv_distribution(cvs,n_bins=20):

    x = cvs
    bins,bin_width = np.linspace(np.min(x),np.max(x),n_bins,retstep=True)
    h,b = np.histogram(x,bins,density=True)

    plt.bar(left=b[:-1],height=h,width=bin_width,fc='k')

    m = np.mean(x)
    plt.plot([m,m],plt.ylim(),'b')
    plt.axis('tight')

    plt.xlabel('CV')
    plt.ylabel('PDF')


def lv_distribution(lvs,n_bins=20):

    x = lvs
    bins,bin_width = np.linspace(np.min(x),np.max(x),n_bins,retstep=True)
    h,b = np.histogram(x,bins,density=True)

    plt.bar(left=b[:-1],height=h,width=bin_width,fc='k')

    m = np.mean(x)
    plt.plot([m,m],plt.ylim(),'b')
    plt.axis('tight')

    plt.xlabel('LV')
    plt.ylabel('PDF')


def cc_distribution(cc_matrix,n_bins=20):

    x = cc_matrix[np.triu_indices(cc_matrix.shape[0],1)]

    bins,bin_width = np.linspace(np.min(x),np.max(x),n_bins,retstep=True)
    h,b = np.histogram(x,bins,density=True)

    plt.bar(left=b[:-1],height=h,width=bin_width,fc='k')

    m = np.mean(x)
    plt.plot([m,m],plt.ylim(),'b')
    plt.axis('tight')

    plt.xlabel('Correlation Coeficient')
    plt.ylabel('PDF')
