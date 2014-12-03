# -*- coding: utf-8 -*-
"""
docstring goes here.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""

from __future__ import division, print_function

import numpy as np
import quantities as pq
import scipy.stats
import neo.core
import conversion


def isi(spiketrain, axis=-1):
    """
    Return an array containing the inter-spike intervals of the SpikeTrain.

    Accepts a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    If either a SpikeTrain or Quantity array is provided, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the same as spiketrain.

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy ndarray
                 The spike times.
    axis : int, optional
           The axis along which the difference is taken.
           Default is the last axis.

    Returns
    -------

    NumPy array or quantities array.

    """
    if axis is None:
        axis = -1
    intervals = np.diff(spiketrain, axis=axis)
    if hasattr(spiketrain, 'waveforms'):
        intervals = pq.Quantity(intervals.magnitude, units=spiketrain.units)
    return intervals


def mean_firing_rate(spiketrain, t_start=None, t_stop=None, axis=None):
    """
    Return the firing rate of the SpikeTrain.

    Accepts a Neo SpikeTrain, a Quantity array, or a plain NumPy array.
    If either a SpikeTrain or Quantity array is provided, the return value will
    be a quantities array, otherwise a plain NumPy array. The units of
    the quantities array will be the inverse of the spiketrain.

    The interval over which the firing rate is calculated can be optionally
    controlled with `t_start` and `t_stop`

    Parameters
    ----------

    spiketrain : Neo SpikeTrain or Quantity array or NumPy ndarray
                 The spike times.
    t_start : float or Quantity scalar, optional
              The start time to use for the inveral.
              If not specified, retrieved from the``t_start`
              attribute of `spiketrain`.  If that is not present, default to
              `0`.  Any value from `spiketrain` below this value is ignored.
    t_stop : float or Quantity scalar, optional
             The stop time to use for the time points.
             If not specified, retrieved from the `t_stop`
             attribute of `spiketrain`.  If that is not present, default to
             the maximum value of `spiketrain`.  Any value from
             `spiketrain` above this value is ignored.
    axis : int, optional
           The axis over which to do the calculation.
           Default is `None`, do the calculation over the flattened array.

    Returns
    -------

    float, quantities scalar, NumPy array or quantities array.

    Notes
    -----

    If `spiketrain` is a Quantity or Neo SpikeTrain and `t_start` or `t_stop`
    are not, `t_start` and `t_stop` are assumed to have the same units as
    `spiketrain`.

    Raises
    ------

    TypeError
        If `spiketrain` is a NumPy array and `t_start` or `t_stop`
        is a quantity scalar.

    """
    if t_start is None:
        t_start = getattr(spiketrain, 't_start', 0)

    found_t_start = False
    if t_stop is None:
        if hasattr(spiketrain, 't_stop'):
            t_stop = spiketrain.t_stop
        else:
            t_stop = np.max(spiketrain, axis=axis)
            found_t_start = True

    # figure out what units, if any, we are dealing with
    if hasattr(spiketrain, 'units'):
        units = spiketrain.units
    else:
        units = None

    # convert everything to the same units
    if hasattr(t_start, 'units'):
        if units is None:
            raise TypeError('t_start cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_start = t_start.rescale(units)
    elif units is not None:
        t_start = pq.Quantity(t_start, units=units)
    if hasattr(t_stop, 'units'):
        if units is None:
            raise TypeError('t_stop cannot be a Quantity if '
                            'spiketrain is not a quantity')
        t_stop = t_stop.rescale(units)
    elif units is not None:
        t_stop = pq.Quantity(t_stop, units=units)

    if not axis or not found_t_start:
        return np.sum((spiketrain >= t_start) & (spiketrain <= t_stop),
                      axis=axis) / (t_stop-t_start)
    else:
        # this is needed to handle broadcasting between spiketrain and t_stop
        t_stop_test = np.expand_dims(t_stop, axis)
        return np.sum((spiketrain >= t_start) & (spiketrain <= t_stop_test),
                      axis=axis) / (t_stop-t_start)


# we make `cv` an alias for scipy.stats.variation for the convenience
# of former NeuroTools users
cv = scipy.stats.variation


def fanofactor(spiketrains):
    """
    Evaluates the empirical Fano factor F of the spike counts of
    a list of `neo.core.SpikeTrain` objects.

    Given the vector v containing the observed spike counts (one per
    spike train) in the time window [t0, t1], F is defined as:

                        F := var(v)/mean(v).

    The Fano factor is typically computed for spike trains representing the
    activity of the same neuron over different trials. The higher F, the larger
    the cross-trial non-stationarity. In theory for a time-stationary Poisson
    process, F=1.

    Parameters
    ----------
    spiketrains : list of neo.core.SpikeTrain objects, quantity array,
                  numpy array or list
        Spike trains for which to compute the Fano factor of spike counts.

    Returns
    -------
    fano : float or nan
        The Fano factor of the spike counts of the input spike trains. If an
        empty list is specified, or if all spike trains are empty, F:=nan.
    """
    # Build array of spike counts (one per spike train)
    spike_counts = np.array([len(t) for t in spiketrains])

    # Compute FF
    if all([count == 0 for count in spike_counts]):
        fano = np.nan
    else:
        fano = spike_counts.var() / spike_counts.mean()
    return fano


def lv(isi):
    """ Calculate LV Shinomoto (2003) from the given ISI distribution """
    if type(isi) is pq.Quantity:
        isi = isi.magnitude

    return np.mean(
        (3. * np.power(isi[:-1] - isi[1:], 2)) /
        np.power(isi[:-1] + isi[1:], 2))


def corrcoef(spiketrains, binsize, clip=True):
    '''
    Matrix of pairwise Pearson's correlation coefficients for a list of
    spike trains.

    For each spike trains i,j in the list, the correlation coefficient
    C[i, j] is given by the correlation coefficient between the vectors
    obtained by binning i and j at the desired bin size. Called b_i, b_j
    such vectors and m_i, m_j their respective averages:

    C[i,j] = <b_i-m_i, b_j-m_j> / sqrt{<b_i-m_i, b_i-m_i>*<b_j-m_j,b_j-m_j>},

    where <.,.> is the scalar product of two vectors.
    If spiketrains is a list of n spike trains, a n x n matrix is returned.
    Each entry in the matrix is a real number ranging between -1 (perfectly
    anticorrelated spike trains) and +1 (perfectly correlated spike trains).
    If clip is True, the spike trains are clipped before computing the
    correlation coefficients, so that the binned vectors b_i, b_j are binary.

    Parameters
    ----------
    spiketrains : list
        a list of SpikeTrains with same t_start and t_stop values
    binsize : Quantity
        the bin size used to bin the spike trains
    clip : bool, optional
        whether to clip spikes of the same spike train falling in the same
        bin (True) or not (False). If True, the binned spike trains are
        binary arrays

    Output
    ------
    M : ndarrray
        the sqaure matrix of correlation coefficients. M[i,j] is the
        correlation coefficient between spiketrains[i] and spiketrains[j]

    '''

    # Check that all spike trains have same t_start and t_stop
    tstart_0 = spiketrains[0].t_start
    tstop_0 = spiketrains[0].t_stop
    assert(all([st.t_start == tstart_0 for st in spiketrains[1:]]))
    assert(all([st.t_stop == tstop_0 for st in spiketrains[1:]]))

    # Bin the spike trains
    t_start = spiketrains[0].t_start
    t_stop = spiketrains[0].t_stop
    binned_sts = conversion.Binned(
        spiketrains, binsize=binsize, t_start=t_start, t_stop=t_stop)

    # Create the binary matrix M of binned spike trains
    if clip is True:
        M = binned_sts.matrix_clipped()
    else:
        M = binned_sts.matrix_unclipped()

    # Return the matrix of correlation coefficients
    return np.corrcoef(M)


def peth(sts, w, t_start=None, t_stop=None, output='counts', clip=False):
    """
    Peri-Event Time Histogram (PETH) of a list of spike trains.

    Parameters
    ----------
    sts : list of neo.core.SpikeTrain objects
        Spiketrains with a common time axis (same t_start and t_stop)
    w : Quantity
        width of the histogram's time bins.
    t_start, t_stop : Quantity (optional)
        Start and stop time of the histogram. Only events in the input
        spike trains falling between t_start and t_stop (both included) are
        considered in the histogram. If t_start and/or t_stop are not
        specified, the maximum t_start of all Spiketrains is used as t_start,
        and the minimum t_stop is used as t_stop.
        Default: t_start=t_stop=None
    output : str (optional)
        Normalization of the histogram. Can be one of:
        * 'counts': spike counts at each bin (as integer numbers)
        * 'mean': mean spike counts per spike train
        * 'rate': mean spike rate per spike train. Like 'mean', but the
          counts are additionally normalized by the bin width.

    Returns
    -------
    analogSignal : neo.core.AnalogSignal
        neo.core.AnalogSignal object containing the PETH values.
        AnalogSignal[j] is the PETH computed between
        t_start + j * w and t_start + (j + 1) * w.

    """

    # Find the internal range t_start, t_stop where all spike trains are
    # defined; cut all spike trains taking that time range only
    max_tstart = max([t.t_start for t in sts])
    min_tstop = min([t.t_stop for t in sts])

    if t_start is None:
        t_start = max_tstart
        if not all([max_tstart == t.t_start for t in sts]):
            warnings.warn(
                "Spiketrains have different t_start values -- "
                "using maximum t_start as t_start.")

    if t_stop is None:
        t_stop = min_tstop
        if not all([min_tstop == t.t_stop for t in sts]):
            warnings.warn(
                "Spiketrains have different t_stop values -- "
                "using minimum t_stop as t_stop.")

    sts_cut = [st.time_slice(t_start=t_start, t_stop=t_stop) for st in sts]

    # Bin the spike trains and sum across columns
    bs = conversion.Binned(sts_cut, t_start=t_start, t_stop=t_stop, binsize=w)

    if clip is True:
        bin_hist = np.sum(bs.matrix_clipped(), axis=0)
    else:
        bin_hist = np.sum(bs.matrix_unclipped(), axis=0)

    # Renormalise the histogram
    if output == 'counts':
        # Raw
        bin_hist = bin_hist * pq.dimensionless
    elif output == 'mean':
        # Divide by number of input spike trains
        bin_hist = bin_hist * 1. / len(sts) * pq.dimensionless
    elif output == 'rate':
        # Divide by number of input spike trains and bin width
        bin_hist = bin_hist * 1. / len(sts) / w
    else:
        raise ValueError('Parameter output is not valid.')

    return neo.AnalogSignal(
        signal=bin_hist, sampling_period=w, t_start=t_start)
