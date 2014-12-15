# -*- coding: utf-8 -*-
"""
Spike train correlation

This modules provides functions to calculate correlations between spike trains.

:copyright: Copyright 2014 by the Elephant team, see AUTHORS.txt.
:license: Modified BSD, see LICENSE.txt for details.
"""
from __future__ import division
import numpy as np
import scipy.sparse
import quantities as pq
import neo
import elephant.conversion as rep


def corrcoef(binned_sts, clip=False):
    '''
    Calculate the NxN matrix of pairwise Pearson's correlation coefficients
    between all combinations given a N binned spike trains.

    For each pair of spike trains $(i,j)$ in the list, the correlation
    coefficient $C[i, j]$ is given by the correlation coefficient between the
    vectors obtained by binning $i$ and $j$ at the desired bin size. Let $b_i$
    and $b_j$ denote the binary vectors and $m_i$ and  $m_j$ their respective
    averages. Then

    .math $$ C[i,j] = <b_i-m_i, b_j-m_j> /
                      \sqrt{<b_i-m_i, b_i-m_i>*<b_j-m_j,b_j-m_j>} $$

    where <..,.> is the scalar product of two vectors.

    For an input of n spike trains, a n x n matrix is returned.
    Each entry in the matrix is a real number ranging between -1 (perfectly
    anti-correlated spike trains) and +1 (perfectly correlated spike trains).

    If clip is True, the binned spike trains are clipped before computing the
    correlation coefficients, so that the binned vectors b_i, b_j are binary.

    Parameters
    ----------
    binned_sts : elephant.conversion.Binned
        A binned spike train class containing the spike trains to be evaluated.
    clip : bool, optional
        If True, two spikes of the a particular spike train falling in the same
        bin are counted as 1, resulting in binary binned vectors b_i. If False,
        the binned vectors $b_i$ contain the actually spike counts.
        Default: False

    Returns
    -------
    C : ndarrray
        The square matrix of correlation coefficients. The element
        $C[i,j]=C[j,i]$ is the Pearson's correlation coefficient between
        binned_sts[i] and binned_sts[j]. If binned_sts contains only one
        SpikeTrain, C=1.0.

    Notes
    -----
    * The spike trains in the binned structure are assumed to all cover the
      complete time span of binned_sts [t_start,t_stop).

    Example
    -------
    >>>
    '''
    num_neurons = binned_sts.matrix_rows

    # Pre-allocate correlation matrix
    C = np.zeros((num_neurons, num_neurons))

    # Retrieve unclipped matrix
    spmat = binned_sts.sparse_mat_unclip

    # For each row, extract the nonzero column indices and the corresponding
    # data in the matrix (for performance reasons)
    bin_idx_unique = []
    bin_counts_unique = []
    if clip:
        for s in spmat:
            bin_idx_unique.append(s.nonzero()[1])
    else:
        for s in spmat:
            bin_counts_unique.append(s.data)

    # All combinations of spike trains
    for i in range(num_neurons):
        for j in range(i, num_neurons):
            # Number of spikes in i and j
            if clip:
                n_i = len(bin_idx_unique[i])
                n_j = len(bin_idx_unique[j])
            else:
                n_i = np.sum(bin_counts_unique[i])
                n_j = np.sum(bin_counts_unique[j])

            # Enumerator:
            # $$ <b_i-m_i, b_j-m_j>
            #      = <b_i, b_j> + l*m_i*m_j - <b_i, M_j> - <b_j, M_i>
            #      =:    ij     + l*m_i*m_j - n_i * m_j  - n_j * m_i
            #      =     ij     - n_i*n_j/l                         $$
            # where $n_i$ is the spike count of spike train $i$,
            # $l$ is the number of bins used (i.e., length of $b_i$ or $b_j$),
            # and $M_i$ is a vector [m_i, m_i,..., m_i].
            if clip:
                # Intersect indices to identify number of coincident spikes in
                # i and j (more efficient than directly using the dot product)
                ij = len(np.intersect1d(
                    bin_idx_unique[i], bin_idx_unique[j], assume_unique=True))
            else:
                # Calculate dot product b_i*b_j between unclipped matrices
                ij = spmat[i].dot(spmat[j].transpose()).toarray()[0][0]

            cc_enum = ij - n_i * n_j / binned_sts.num_bins

            # Denominator:
            # $$ <b_i-m_i, b_i-m_i>
            #      = <b_i, b_i> + m_i^2 - 2 <b_i, M_i>
            #      =:    ii     + m_i^2 - 2 n_i * m_i
            #      =     ii     - n_i^2 /               $$
            if clip:
                # Here, b_i*b_i is just the number of filled bins (since each
                # filled bin of a clipped spike train has value equal to 1)
                ii = len(bin_idx_unique[i])
                jj = len(bin_idx_unique[j])
            else:
                # directly calculate the dot product based on the counts of all
                # filled entries (more efficient than using the dot product of
                # the rows of the sparse matrix)
                ii = np.dot(bin_counts_unique[i], bin_counts_unique[i])
                jj = np.dot(bin_counts_unique[j], bin_counts_unique[j])

            cc_denom = np.sqrt(
                (ii - (n_i ** 2) / binned_sts.num_bins) *
                (jj - (n_j ** 2) / binned_sts.num_bins))

            # Fill entry of correlation matrix
            C[i, j] = C[j, i] = cc_enum / cc_denom
    return C

    # ---- Alternate version: does not work since sparse matrix has no
    #      addition of scalars implemented --> workaround possible?
    if clip:
        mat = scipy.sparse.csr_matrix(binned_sts.matrix_clipped())
    else:
        mat = scipy.sparse.csr_matrix(binned_sts.matrix_unclipped())

    num_neurons = binned_sts.matrix_rows

    # Pre-allocate correlation matrix
    C = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(i, num_neurons):
            mean_i = mat[i].mean()
            mean_j = mat[j].mean()

            # Fill entry of correlation matrix
            C[i, j] = C[j, i] = \
                np.dot(mat[i] - mean_i, mat[j] - mean_j) / \
                np.sqrt(
                    np.dot(mat[i] - mean_i, mat[i] - mean_i) *
                    np.dot(mat[j] - mean_j, mat[j] - mean_j))

    return C


def corrcoef_continuous(sts, coinc_width):
    '''
    Calculate the NxN matrix of pairwise Pearson's correlation coefficients
    between all combinations given a list of N spike trains.

    For each pair of spike trains $(i,j)$ in the list, the correlation
    coefficient $C[i, j]$ is given by the correlation coefficient between the
    vectors obtained by binning $i$ and $j$ at the desired bin size. Let $b_i$
    and $b_j$ denote the binary vectors and $m_i$ and  $m_j$ their respective
    averages. Then

    .math $$ C[i,j] = <b_i-m_i, b_j-m_j> /
                      \sqrt{<b_i-m_i, b_i-m_i>*<b_j-m_j,b_j-m_j>} $$

    where <..,.> is the scalar product of two vectors.

    If spike trains is a list of n spike trains, a n x n matrix is returned.
    Each entry in the matrix is a real number ranging between -1 (perfectly
    anti-correlated spike trains) and +1 (perfectly correlated spike trains).

    If clip is True, the spike trains are clipped before computing the
    correlation coefficients, so that the binned vectors b_i, b_j are binary.

    Parameters
    ----------
    sts : list of SpikeTrain
        A list of SpikeTrain objects with common t_start and t_stop.
    coinc_width : Quantity
        The coincidence width for which to calculate the correlation
        coefficient. Any two spikes $t_i, t_j$ with
            $$|t_i-t_j|<coinc_width$$ are
        treated as coincident.

    Returns
    -------
    C : ndarrray
        The square matrix of correlation coefficients. The element
        $C[i,j]=C[j,i]$ is the Pearson's correlation coefficient between sts[i]
        and sts[j]. If sts contains only one SpikeTrain, C=1.0.

    '''
    # Check that all spike trains have same t_start and t_stop
    t_start = sts[0].t_start
    t_stop = sts[0].t_stop
    if not all([st.t_start == t_start for st in sts[1:]]) or \
            not all([st.t_stop == t_stop for st in sts[1:]]):
        raise ValueError(
            "All spike trains must have common t_start and t_stop.")

    num_bins = np.ceil(
        (t_stop.rescale(coinc_width.units) -
            t_start.rescale(coinc_width.units)) /
        coinc_width).magnitude
    num_neuron = len(sts)

    # Pre-allocate correlation matrix
    C = np.zeros((num_neuron, num_neuron))

    # Rescale all spike trains the resolution of the bin width
    sts_rescale = [st.rescale(coinc_width.units).magnitude for st in sts]

    # All combinations of spike trains
    for i, sts_i in enumerate(sts_rescale):
        for j, sts_j in enumerate(sts_rescale):
            # Calculate only half the correlation coefficient matrix
            if j >= i:
                # Enumerator:
                # $$ <b_i-m_i, b_j-m_j>
                #      = <b_i, b_j> + l*m_i*m_j - <b_i, M_j> - <b_j, M_i>
                #      =:    ij     + l*m_i*m_j - n_i * m_j  - n_j * m_i    $$
                # where $n_i$ is the spike count of spike train $i$,
                # $l$ is the number of bins used (i.e., length of $b_i$),
                # and $M_i$ is a vector [m_i, m_i,..., m_i].
                ij = np.count_nonzero(np.abs(np.subtract.outer(
                    sts_i, sts_j)) < coinc_width.magnitude)

                # Number of spikes in i and j
                n_i = len(sts_i)
                n_j = len(sts_j)

                cc_enum = ij - n_i * n_j / num_bins

                # Denominator:
                # $$ <b_i-m_i, b_i-m_i>
                #       = <b_i, b_i> + m_i^2 - 2 <b_i, M_i>
                #       =:    ii     + m_i^2 - 2 n_i * m_i   $$
                ii = np.count_nonzero(np.abs(np.subtract.outer(
                    sts_i, sts_i)) < coinc_width.magnitude)
                jj = np.count_nonzero(np.abs(np.subtract.outer(
                    sts_j, sts_j)) < coinc_width.magnitude)

                cc_denom = np.sqrt(
                    (ii - n_i ** 2 / num_bins) *
                    (jj - n_j ** 2 / num_bins))

                # Fill entry of correlation matrix
                C[i, j] = C[j, i] = cc_enum / cc_denom
    return C


def cchb(
        x, y, hlen=None, corrected=False, smooth=0, clip=False,
        normed=False, kernel='boxcar'):
    """
    Computes the cross-correlation histogram (CCH) between two binned
    spike trains x and y.

    Parameters
    ----------
    x,y : binned_st
        binned spike trains. If ndarrays, they are interpreted as sequences
        of zero (no spike) and non-zero (one or more spikes) values.
    hlen : int or None (optional)
        histogram half-length. If specified, the cross-correlation histogram
        has a number of bins equal to 2*hlen+1 (up to the maximum length).
        If not specified, the full crosscorrelogram is returned
        Default: None
    corrected : bool (optional)
        whether to correct for the border effect. If True, the value of the
        CCH at bin b (for b=-H,-H+1, ...,H, where H is the CCH half-length)
        is multiplied by the correction factor:
                            (H+1)/(H+1-|b|),
        which linearly corrects for loss of bins at the edges.
        Default: False
    smooth : Quantity or None (optional)
        if smooth is a positive time, each bin in the raw cross-correlogram
        is averaged over a window (-smooth/2, +smooth/2) with the values in
        the neighbouring bins. If smooth <= w, no smoothing is performed.
        Default: None
    clip : bool (optional)
        whether to clip spikes from the same spike train falling in the
        same bin. If True, such spikes are considered as a single spike;
        otherwise they are considered as different spikes.
        Default: False.
    normed : bool (optional)
        whether to normalize the central value (corresponding to time lag
        0 s) to 1; the other values are rescaled accordingly.
        Default: False
    kernel : str or array (optional)
        kernel used for smoothing (see parameter smooth above). Can be:
        * list or array of floats defining the kernel weights
        * one of the following strings:
          * 'boxcar' : normalized boxcar window;
          * 'hamming': normalized hamming window;
          * 'hanning': normalized hanning window;
          * 'bartlett': normalized bartlett window;
        Default: 'boxcar'

   Returns
   -------
      returns the cross-correlation histogram between x and y. The central
      bin of the histogram represents correlation at zero delay. Offset bins
      correspond to correlations at a delay equivalent to the difference
      between the spike times of x and those of y. I.e., bins to the right
      correspond to spike of y following spikes of x, and viceversa for bins
      to the left.

    Example
    -------
    TODO: make example!

    """
    if isinstance(x, rep.Binned):
        x_filled = x.spike_indices[0]  # Take the indices of the spike_indices bins in x
        x_mat = x.matrix_clipped()[0] if clip else  x.matrix_unclipped()[0]
        x_filled_howmany = x_mat[x_filled]
        del(x_mat)  # Delete big unnecessary object
    else:
        # TODO: return error instead
        if type(x) != np.ndarray:
            x = np.array(x)
        if clip == True:
            x = 1 * (x > 0)
        x_filled = np.where(x > 0)[0]
        x_filled_howmany = x[x_filled]

    if isinstance(y, rep.Binned):
        y_filled = y.spike_indices[0]  # Take the indices of the spike_indices bins in y
        y_mat = y.matrix_clipped()[0] if clip else  y.matrix_unclipped()[0]
        y_filled_howmany = y_mat[y_filled]
        del(y_mat)  # Delete big unnecessary object
    else:
        # TODO: return error instead
        if type(y) != np.ndarray:
            y = np.array(y)
        if clip == True:
            y = 1 * (y > 0)
        y_filled = np.where(y > 0)[0]
        y_filled_howmany = y[y_filled]

    # Take the indices of the spike_indices bins in x and y
    x_filled = x.spike_indices[0]
    y_filled = y.spike_indices[0]

    # Compute the binned spike trains
    x_mat = x.matrix_clipped()[0]
    y_mat = y.matrix_clipped()[0]

    # Select the spike_indices bins of x and y into a smaller array
    x_filled_howmany = x_mat[x_filled]
    y_filled_howmany = y_mat[y_filled]

    # Delete big unnecessary objects
    del(x_mat, y_mat)

    # Define the half-length of the full crosscorrelogram.
    Len = x.num_bins + y.num_bins - 1
    Hlen = Len // 2
    Hbins = Hlen if hlen is None else min(hlen, Hlen)

    # Initialize the counts to an array of zeroes, and the bin ids to
    counts = np.zeros(2 * Hbins + 1)
    bin_ids = np.arange(-Hbins, Hbins + 1)

    # Compute the cch at lags in -Hbins,...,Hbins only
    for r, i in enumerate(x_filled):
        timediff = y_filled - i
        timediff_in_range = np.all(
            [timediff >= -Hbins, timediff <= Hbins], axis=0)
        timediff = (timediff[timediff_in_range]).reshape((-1,))
        counts[timediff + Hbins] += x_filled_howmany[r] * \
            y_filled_howmany[timediff_in_range]

    # Correct the values taking into account lacking contributes at the edges
    if corrected == True:
        correction = float(Hlen + 1) / np.array(
            Hlen + 1 - abs(np.arange(-Hlen, Hlen + 1)), float)
        counts = counts * correction

    # Define the kernel for smoothing as an ndarray
    if hasattr(kernel, '__iter__'):
        kernel = np.array(kernel, dtype=float)
    elif isinstance(kernel, str) and smooth > 1:
        smooth_Nbin = min(int(smooth), Len)
        if kernel == 'hamming':
            win = np.hamming(smooth_Nbin)
        elif kernel == 'bartlett':
            win = np.bartlett(smooth_Nbin)
        elif kernel == 'hanning':
            win = np.hanning(smooth_Nbin)
        elif kernel == 'boxcar':
            win = np.ones(smooth_Nbin)
        else:
            raise ValueError(
                'kernel (%s) can be either an array or one of the following '
                'strings: "boxcar", "hamming", "hanning", "bartlett".'
                % str(kernel))
        kernel = 1. * win / sum(win)
    else:
        kernel = np.array(1)

    # Smooth the cross-correlation histogram with the kernel
    counts = np.convolve(counts, kernel, mode='same')

    # Rescale the histogram so that the central bin has height 1, if requested
    if normed:
        counts = np.array(counts, float) / float(counts[Hlen])

    # Return only the Hbins bins and counts before and after the central one
    return counts, bin_ids


def ccht(x, y, w, window=None, start=None, stop=None, corrected=False,
    smooth=None, clip=False, normed=False, xaxis='time', kernel='boxcar'):
    """
    Computes the cross-correlation histogram (CCH) between two spike trains.

    Given a reference spike train x and a target spike train y, their CCH
    at time lag t is computed as the number of spike pairs (s1, s2), with s1
    from x and s2 from y, such that s2 follows s1 by a time lag \tau in the
    range - or time bin - [t, t+w):

        CCH(\tau; x,y) := #{(s1,s2) \in x \times y:  t= < s2-s1 < t+w}

    Therefore, the times associated to the CCH are the left ends of the
    corresponding time bins.

    Note: CCH(\tau; x,y) = CCH(-\tau; y,x).

    This routine computes CCH(x,y) at the times
                     ..., -1.5*w, -0.5*w, 0.5*w, ...
    corresponding to the time bins
            ..., [-1.5*w, -0.5*w), [-0.5*w, 0.5*w), [0.5*w, 1.5*w), ...
    The second one is the central bin, corresponding to synchronous spiking
    of x and y.

    Parameters
    ----------
    x,y : neo.SpikeTrains or lists of neo.SpikeTrains
        If x and y are both SpikeTrains, computes the CCH between them.
        If x and y are both lists of SpikeTrains (with same length, say l),
        computes for each i=1,2,...,l the CCH between x[i] and y[i] , and
        returns their average CCH.
        All input SpikeTrains must have same t_start and t_stop.
    w : Quantity
        time width of the CCH time bin.
    lag : Quantity (optional).
        positive time, specifying the range of the CCH: the CCH is computed
        in [-lag, lag]. This interval is automatically extended to contain
        an integer, odd number of bins. If None, the range extends to
        the maximum lag possible, i.e.[start-stop, stop-start].
        Default: None
    start, stop : Quantities (optional)
        If not None, the CCH is computed considering only spikes from x and y
        in the range [start, stop]. Spikes outside this range are ignored.
        If start (stop) is None, x.t_start (x.t_stop) is used instead
        Default: None
    corrected : bool (optional)
        whether to correct for the border effect. If True, the value of the
        CCH at bin b (for b=-H,-H+1, ...,H, where H is the CCH half-length)
        is multiplied by the correction factor:
                            (H+1)/(H+1-|b|),
        which linearly corrects for loss of bins at the edges.
        Default: False
    smooth : Quantity or None (optional)
        if smooth is a positive time, each bin in the raw cross-correlogram
        is averaged over a window (-smooth/2, +smooth/2) with the values in
        the neighbouring bins. If smooth <= w, no smoothing is performed.
        Default: None
    clip : bool (optional)
        whether to clip spikes from the same spike train falling in the
        same bin. If True, such spikes are considered as a single spike;
        otherwise they are considered as different spikes.
        Default: False.
    normed : bool (optional)
        whether to normalize the central value (corresponding to time lag
        0 s) to 1; the other values are rescaled accordingly.
        Default: False
    kernel : str or array (optional)
        kernel used for smoothing (see parameter smooth above). Can be:
        * list or array of floats defining the kernel weights
        * one of the following strings:
          * 'boxcar' : normalized boxcar window;
          * 'hamming': normalized hamming window;
          * 'hanning': normalized hanning window;
          * 'bartlett': normalized bartlett window;
        Default: 'boxcar'
    xaxis : str (optional)
        whether to return the times or the bin ids as the first output.
        Can be one of:
        * 'time' (default): returns the actul times of the cch
        * 'ids': returns the bin ids of the cch.
        Default: 'time'

    Returns
    -------
    counts : array of float
        array of lagged coincidences counts, representing the number of
        spike pairs (t1, t2), with t1 from x and t2 from y, such that
        t2-t1 lies in a given range (possibly smoothed or normalized).
    times : Quantity array or array of floats
        array of spike times (or array of bin ids) associated to the
        calculated counts in CCH.

    Example
    -------
    >>> import neo, quantities as pq
    >>> st1 = neo.SpikeTrain([1.2, 3.5, 8.7, 10.1] * pq.ms, t_stop=15*pq.ms)
    >>> st2 = neo.SpikeTrain([1.9, 5.2, 8.4] * pq.ms, t_stop=15*pq.ms)
    >>> print ccht(st1, st2, w=3*pq.ms)
    (array([ 0.,  0.,  1.,  2.,  3.,  3.,  2.,  1.,  0.]),
     array([-12.,  -9.,  -6.,  -3.,   0.,   3.,   6.,   9.,  12.]) * ms)
    >>> print ccht(st1, st2, w=3*pq.ms, window=3*pq.ms)
    (array([ 2.,  3.,  3.]), array([-3.,  0.,  3.]) * ms)
    >>> print ccht(st1, st2, w=3*pq.ms, window=3*pq.ms, xaxis='ids')
    (array([ 2.,  3.,  3.]), array([-1,  0,  1]))

    """

    # Raise errors if x.t_start != y.t_start or x.t_stop != y.t_stop
    if x.t_start != y.t_start:
        raise ValueError('x and y must have the same t_start attribute')
    if x.t_stop != y.t_stop:
        raise ValueError('x and y must have the same t_stop attribute')

    # Set start. Only spike times >= start will be considered for the CCH
    if start is None:
        start = x.t_start

    # Set stop to end of spike trains if None
    if stop is None:
        if len(x) * len(y) == 0:
            stop = 0 * x.units if window is None else 2 * window
        else:
            stop = x.t_stop if window is None else max(x.t_stop, window)

    # By default, set smoothing to 0 ms (no smoothing)
    if smooth is None:
        smooth = 0 * pq.ms

    # Set the window for the CCH
    win = (stop - start) if window is None else min(window, (stop - start))

    # Cut the spike trains, keeping the spikes between start and stop only
    x_cut = x if len(x) == 0 else x.time_slice(t_start=start, t_stop=stop)
    y_cut = y if len(y) == 0 else y.time_slice(t_start=start, t_stop=stop)

    # Bin the spike trains
    x_binned = rep.Binned(x_cut, t_start=start, t_stop=stop, binsize=w)
    y_binned = rep.Binned(y_cut, t_start=start, t_stop=stop, binsize=w)

    # Evaluate the CCH for the binned trains with cchb()
    counts, bin_ids = cchb(
        x_binned, y_binned, corrected=corrected, clip=clip, normed=normed,
        smooth=int((smooth / w).rescale(pq.dimensionless)), kernel=kernel,
        hlen=int((win / w).rescale(pq.dimensionless).magnitude))

    # Convert bin ids to times if the latter were requested
    if xaxis == 'time':
        bin_ids = bin_ids * w

    # Return the CCH and the bins used to compute it
    return counts, bin_ids


def cch(x, y, w, lag=None, start=None, stop=None, corrected=False,
    smooth=None, clip=False, normed=False, kernel='boxcar'):
    """
    Computes the cross-correlation histogram (CCH) between two spike trains,
    or the average CCH between the spike trains in two spike train lists.

    Given a reference spike train x and a target spike train y, their CCH
    at time lag t is computed as the number of spike pairs (s1, s2), with s1
    from x and s2 from y, such that s2 follows s1 by a time lag \tau in the
    range - or time bin - [t, t+w):

        CCH(\tau; x,y) := #{(s1,s2) \in x \times y:  t <= s2-s1 < t+w}

    Therefore, the times associated to the CCH are the left ends of the
    corresponding time bins.

    Note: CCH(\tau; x,y) = CCH(-\tau; y,x).

    This routine computes CCH(x,y) at the times
                     ..., -1.5*w, -0.5*w, 0.5*w, ...
    corresponding to the time bins
            ..., [-1.5*w, -0.5*w), [-0.5*w, 0.5*w), [0.5*w, 1.5*w), ...
    The second one is the central bin, corresponding to synchronous spiking
    of x and y.

    Parameters
    ----------
    x,y : neo.SpikeTrains or lists of neo.SpikeTrains
        If x and y are both SpikeTrains, computes the CCH between them.
        If x and y are both lists of SpikeTrains (with same length, say l),
        computes for each i=1,2,...,l the CCH between x[i] and y[i] , and
        returns their average CCH.
        All input SpikeTrains must have same t_start and t_stop.
    w : Quantity
        time width of the CCH time bin.
    lag : Quantity (optional).
        positive time, specifying the range of the CCH: the CCH is computed
        in [-lag, lag]. This interval is automatically extended to contain
        an integer, odd number of bins. If None, the range extends to
        the maximum lag possible, i.e.[start-stop, stop-start].
        Default: None
    start, stop : Quantities (optional)
        If not None, the CCH is computed considering only spikes from x and y
        in the range [start, stop]. Spikes outside this range are ignored.
        If start (stop) is None, x.t_start (x.t_stop) is used instead
        Default: None
    corrected : bool (optional)
        whether to correct for the border effect. If True, the value of the
        CCH at bin b (for b=-H,-H+1, ...,H, where H is the CCH half-length)
        is multiplied by the correction factor:
                            (H+1)/(H+1-|b|),
        which linearly corrects for loss of bins at the edges.
        Default: False
    smooth : Quantity or None (optional)
        if smooth is a positive time, each bin in the raw cross-correlogram
        is averaged over a window (-smooth/2, +smooth/2) with the values in
        the neighbouring bins. If smooth <= w, no smoothing is performed.
        Default: None
    clip : bool (optional)
        whether to clip spikes from the same spike train falling in the
        same bin. If True, such spikes are considered as a single spike;
        otherwise they are considered as different spikes.
        Default: False.
    normed : bool (optional)
        whether to normalize the central value (corresponding to time lag
        0 s) to 1; the other values are rescaled accordingly.
        Default: False
    kernel : str or array (optional)
        kernel used for smoothing (see parameter smooth above). Can be:
        * list or array of floats defining the kernel weights
        * one of the following strings:
          * 'boxcar' : normalized boxcar window;
          * 'hamming': normalized hamming window;
          * 'hanning': normalized hanning window;
          * 'bartlett': normalized bartlett window;
        Default: 'boxcar'

    Returns
    -------
    AnalogSignal
        returns an analog signal, representing the CCH between the spike
        trains x and y at different time lags. AnalogSignal.times represents
        the left edges of the time bins. AnalogSignal.sampling_period
        represents the bin width used to compute the CCH.

    Example
    -------
    >>> import neo, quantities as pq
    >>> t1 = neo.SpikeTrain([1.2, 3.5, 8.7, 10.1] * pq.ms, t_stop=20*pq.ms)
    >>> t2 = neo.SpikeTrain([1.9, 5.2, 8.4] * pq.ms, t_stop=20*pq.ms)
    >>> CCH = cch(t1, t2, 3*pq.ms)
    >>> print CCH
    [ 0.  0.  0.  1.  2.  3.  3.  2.  1.  0.  0.] dimensionless
    >>> print CCH.times
    [-16.5 -13.5 -10.5  -7.5  -4.5  -1.5   1.5   4.5   7.5  10.5  13.5] ms
    """
    if isinstance(x, neo.core.SpikeTrain) and isinstance(y, neo.core.SpikeTrain):
        CCH, bins = ccht(x, y, w, window=lag, start=start, stop=stop,
            corrected=corrected, smooth=smooth, clip=clip, normed=normed,
            xaxis='time', kernel=kernel)

        if not isinstance(CCH, pq.Quantity):
            CCH = CCH * pq.dimensionless

        return neo.AnalogSignal(
            CCH, t_start=bins[0] - w / 2., sampling_period=w)

    else:
        CCH_exists = False
        for xx, yy in zip(x, y):
            if CCH_exists == False:
                CCH = cch(xx, yy, w, lag=lag, start=start, stop=stop,
                    corrected=corrected, smooth=smooth, clip=clip,
                    normed=normed, kernel=kernel)
                CCH_exists = True
            else:
                CCH += cch(xx, yy, w, lag=lag, start=start, stop=stop,
                    corrected=corrected, smooth=smooth, clip=clip,
                    normed=normed, kernel=kernel)
        CCH = CCH / float(len(x))

        return CCH


def cov(spiketrains, binsize, clip=True):
    '''
    Matrix of pairwise covariance coefficients for a list of spike trains.

    For each spike trains i,j in the list, the coavriance coefficient
    C[i, j] is given by the covariance between the vectors obtained by
    binning i and j at the desired bin size. Called b_i, b_j such vectors
    and m_i, m_j their respective averages:

                    C[i,j] = <b_i-m_i, b_j-m_j>

    where <.,.> is the scalar product of two vectors.
    If spiketrains is a list of n spike trains, a n x n matrix is returned.
    If clip is True, the spike trains are clipped before computing the
    covariance coefficients, so that the binned vectors b_i, b_j are binary.

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
        the square matrix of correlation coefficients. M[i,j] is the
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
    binned_sts = rep.Binned(
        spiketrains, binsize=binsize, t_start=t_start, t_stop=t_stop)

    # Create the binary matrix M of binned spike trains
    if clip is True:
        M = binned_sts.matrix_clipped()
    else:
        M = binned_sts.matrix_unclipped()

    # Return the matrix of correlation coefficients
    return np.cov(M)


def ccht2(x, y, binsize, corrected=False, smooth=0, normed=False,
          xaxis='time', **kwargs):
    """

    .. note::
        Same as ccht() [in turn deprecated].
        Should be faster, but is slower...deprecated. Use cch() instead

    .. See also::
        cch()

    """
    import np

    # Convert the inputs to arrays
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
    if 'clip' not in kwargs.keys():
        clip = True
    else:
        clip = kwargs['clip']
    #
    #**********************************************************************************************
    # setting the starting and stopping times for the cch; cutting the spike trains accordingly
    #**********************************************************************************************
    if 'start' not in kwargs.keys():
        start = min(0, min(x), min(y))
    else:
        start = kwargs['start']
    if 'stop' not in kwargs.keys():
        stop = max(max(x), max(y));
    else:
        stop = kwargs['stop']
    x_cut = x[np.all((start <= x, x <= stop), axis=0)]
    y_cut = y[np.all((start <= y, y <= stop), axis=0)]
    #
    #**********************************************************************************************
    # binning the (cut) trains and computing the bin difference
    #**********************************************************************************************
    if clip == True:
        x_filledbins = np.unique(np.array((x_cut - start) / binsize, int))
        y_filledbins = np.unique(np.array((y_cut - start) / binsize, int))
    else:
        x_filledbins = np.array((x_cut - start) / binsize, int)
        y_filledbins = np.array((y_cut - start) / binsize, int)
    bindiff = np.concatenate([i - y_filledbins for i in x_filledbins], axis=0)
    #
    #**********************************************************************************************
    # computing the number of bins and initializing counts and bin ids. Cutting the bin difference
    #**********************************************************************************************
    if 'window' in kwargs.keys():
        win = min(kwargs['window'], (stop - start) / 2.)
    else:
        win = (stop - start) / 2.
    Hlen = min(int((stop - start) / (2 * binsize)), int(np.ceil((win + smooth / 2.) / binsize)))
    Len = 2 * Hlen + 1
    Hbins = min(int(win / binsize), Hlen)
    counts = np.zeros(Len)
    bin_ids = np.arange(-Hlen, Hlen + 1)
    bindiff_cut = bindiff[np.all((bindiff >= -Hlen, bindiff <= Hlen), axis=0)]
    #
    #**********************************************************************************************
    # computing the counts
    #**********************************************************************************************
    for i in bindiff_cut:
        counts[Hlen + i] += 1
    #
    #**********************************************************************************************
    # correcting, smoothing and normalizing the counts, if requested
    #**********************************************************************************************
    if corrected == True:
        correction = float(Hlen + 1) / np.array(Hlen + 1 - abs(np.arange(-Hlen, Hlen + 1)), float)
        counts = counts * correction
    if smooth > binsize:
        if 'kernel' not in kwargs.keys():
            kerneltype = 'boxcar'
        else:
            kerneltype = kwargs['kernel']
        smooth_Nbin = min(int(smooth / binsize), Len)
        if kerneltype == 'hamming':
            win = np.hamming(smooth_Nbin)
        elif kerneltype == 'bartlett':
            win = np.bartlett(smooth_Nbin)
        elif kerneltype == 'hanning':
            win = np.hanning(smooth_Nbin)
        elif kerneltype == 'boxcar':
            win = np.ones(smooth_Nbin)
        kernel = win / sum(win)
        counts = np.convolve(counts, kernel, mode='same')
    if normed == True:
        counts = np.array(counts, float) / float(counts[Hlen])
    #
    #**********************************************************************************************
    # returning the CCH; the first output is the array of bin ids if xaxis == 'binid', and the array
    # of times (for the bin centers) if xaxis == 'time'.
    #**********************************************************************************************
    if xaxis == 'time':
        return bin_ids[Hlen - Hbins:Hlen + Hbins + 1] * binsize + start, counts[Hlen - Hbins:Hlen + Hbins + 1]
    else:
        return bin_ids[Hlen - Hbins:Hlen + Hbins + 1], counts[Hlen - Hbins:Hlen + Hbins + 1]
