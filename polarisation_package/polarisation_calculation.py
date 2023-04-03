#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Polarization Analysis for 3 component data.

References:

    Samson, J. C., and J. V. Olson. 1980. “Some Comments on the Descriptions of
    the Polarization States of Waves.” Geophysical Journal of the Royal
    Astronomical Society 61 (1): 115–29.
    doi:10.1111/j.1365-246X.1980.tb04308.x.

    Samson, J. C. 1983. “Pure States, Polarized Waves, and Principal Components
    in the Spectra.” Geophysical Journal of the Royal Astronomical Society
    72: 647–64.

:copyright:
    Martin van Driel (Martin@vanDriel.de), 2019
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    GPLv3
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import matplotlib
from matplotlib.colors import Normalize
from obspy.signal.util import next_pow_2
from obspy import UTCDateTime as utct
from obspy.signal.tf_misfit import cwt
from scipy import signal


def hanning_2d(std):
    win = np.einsum('...i,...j->...ij',
                    signal.hanning(std[0] + 2)[1:-1],
                    signal.hanning(std[1] + 2)[1:-1])
    win /= np.sum(win)
    return win


def compute_polarization(u1, u2, u3, ntsum=1, dsfact=1, nfsum=1, dsfacf=1):
    '''
    compute polarization ellipses parameters and degree of polarization using
    the eigenvalues of the smoothed covariance matrix

    u1, u2, u3: complex valued spectrograms, u1=vertical, shape (nf, nt)
    ntsum: number of samples for smoothing in time direction
    dsfact: downsample factor in time direction
    nfsum: number of samples for smoothing in frequency direction
    dsfact: downsample factor in frequency direction
    '''

    assert u2.shape == u1.shape
    assert u3.shape == u1.shape

    u = np.array([u1, u2, u3])
    u = np.moveaxis(u, 0, 2)

    # avoid eivenvector computation when not necessary
    if (not type(ntsum) == int) or nfsum > 1 or ntsum > 1:
        # compute spectral matrix, smooth and then estimate polarization vector
        # from eigenvectors as decribed in Samson (1983).

        # spectral matrix
        S = np.einsum('...i,...j->...ij', u.conj(), u)

        if type(ntsum) == int:
            w = hanning_2d([nfsum, ntsum])

            for j in range(0, S.shape[2]):
                for k in range(0, S.shape[3]):
                    S[:, :, j, k] = \
                        signal.convolve(S[..., j, k].real, w, mode='same') + \
                        signal.convolve(S[..., j, k].imag, w, mode='same') * 1j
        else:
            assert len(ntsum) == u1.shape[0]
            w_f = signal.hanning(nfsum + 2)[1:-1] * np.ones((nfsum, 1))
            w_f /= np.sum(w_f, axis=0)
            for j in range(0, S.shape[2]):
                for k in range(0, S.shape[3]):
                    S[:, :, j, k] = \
                        signal.convolve(S[..., j, k].real, w_f,
                                        mode='same') + \
                        signal.convolve(S[..., j, k].imag, w_f,
                                        mode='same') * 1j
                    for i in range(0, S.shape[0]):
                        w_t = signal.hanning(ntsum[i])
                        w_t /= np.sum(w_t)
                        S[i, :, j, k] = \
                            signal.convolve(S[i, :, j, k], w_t, mode='same')

        S = S[:, ::dsfact]
        S = S[::dsfacf]

        evalues, evectors = np.linalg.eig(S)
        # S is hermitian, so eigenvalues should be real
        evalues = np.abs(evalues)

        # Compute degree of polarization from eigenvalues of S, e. 18 in
        # Samson and Olson (1980).
        P = ((evalues[..., 2] - evalues[..., 0]) ** 2 +
             (evalues[..., 2] - evalues[..., 1]) ** 2 +
             (evalues[..., 1] - evalues[..., 0]) ** 2) / \
            ((3 - 1) * np.sum(evalues, axis=2) ** 2)

        u2 = evectors[..., 0].copy()
        u2[...] = 0.
        ev_idx = np.argmax(evalues, axis=-1)
        for i in range(u.shape[-1]):
            mask = ev_idx == i
            u2[mask, :] = \
                evectors[mask, :, i] * evalues[mask, np.newaxis, i] ** 0.5
    else:
        if dsfacf > 1 or dsfact > 1:
            raise ValueError('downsampling without smoothing makes no sense')

        u2 = u
        P = np.ones_like(u[..., 0])

    # compute minor and major axis of the polarization vector, eq 5 in
    # Samson and Olson (1980).
    gamma = np.arctan2(2 * np.einsum('...i,...i', u2.real, u2.imag),
                       np.einsum('...i,...i', u2.real, u2.real) -
                       np.einsum('...i,...i', u2.imag, u2.imag))
    phi = -0.5 * gamma

    r1 = (np.exp(1j * phi)[:, :, np.newaxis] * u2).real
    r2 = (np.exp(1j * phi)[:, :, np.newaxis] * u2).imag

    # choose positive direction for axes to get rid of ambiguity
    # this maps the azimuth to [0, 180] and leaves inclination free [-90, 90]
    mask = r1[..., -1] < 0
    r1 *= -1 * mask[:, :, np.newaxis] + (1 - mask[:, :, np.newaxis])

    mask = r2[..., -1] < 0
    r2 *= -1 * mask[:, :, np.newaxis] + (1 - mask[:, :, np.newaxis])

    # azimuth of the axes (assuming index -1 and -2 are the horizontal traces)
    azi1 = np.arctan2(r1[..., -1], r1[..., -2])
    azi2 = np.arctan2(r2[..., -1], r2[..., -2])

    elli = (r2 ** 2).sum(axis=-1) ** 0.5 / (r1 ** 2).sum(axis=-1) ** 0.5

    # inclination of the axes
    rr = (r1[..., 1] ** 2 + r1[..., 2] ** 2) ** 0.5
    inc1 = np.arctan2(r1[..., 0], rr)
    azi1[inc1 > 0.] = azi1[inc1 > 0.] + np.pi
    inc1 = abs(inc1)

    rr = (r2[..., 1] ** 2 + r2[..., 2] ** 2) ** 0.5
    inc2 = np.arctan2(r2[..., 0], rr)
    return azi1, azi2, elli, inc1, inc2, r1, r2, P


def imshow_alpha(ax, x, y, val, alpha, vmin, vmax, cmap):
    import matplotlib.dates as mdates
    x_lims = mdates.date2num([x[0], x[-1]])
    
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    c = norm(val)
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(c)
    colors[..., -1] = alpha
    ax.imshow(colors, extent=(x_lims[0], x_lims[-1], y[-1], y[0]),
              aspect='auto')

    cm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cm.set_array([])
    return cm


def pcolormesh_alpha(ax, x, y, val, alpha, vmin, vmax, cmap, bounds=None):
    
    if bounds is None:
        qm = ax.pcolormesh(x, y, val[:-1,:-1], vmin=vmin, vmax=vmax, cmap=cmap,
                           linewidth=0., rasterized=True) #, shading='flat' is implicitly used, which drops last column+row of val. Raises deprecation warning, so doing it manually
    
        colors = qm.cmap(qm.norm(qm.get_array()))
        norm = qm.norm
    else: #for custom colorscale
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        qm = ax.pcolormesh(x, y, val[:-1,:-1], cmap=cmap, norm=norm,
                           linewidth=0., rasterized=True)
        colors = qm.cmap(qm.norm(qm.get_array()))

    # if val.shape = (len(x), len(y)), then pcolormesh neglects one column +
    # row in val, hence need to adapt alpha
    colors[:, -1] = alpha[:len(y)-1, :len(x)-1].ravel()
    qm.set_color(colors)

    # create scalar mappable for colorbar
    cm = plt.cm.ScalarMappable(cmap=qm.cmap, norm=norm)
    cm.set_array([])

    # if the mappable array is not none, the colors are recomputed on draw from
    # the mappable
    qm._A = None
    
    

    return cm


def _check_traces(st_Z, st_N, st_E, tstart, tend):
    t0 = np.infty
    t1 = -np.infty
    for tr_Z, tr_N, tr_E in zip(st_Z, st_N, st_E):
        try:
            assert tr_N.stats.npts == tr_Z.stats.npts
            assert tr_E.stats.npts == tr_Z.stats.npts

            assert tr_N.stats.delta == tr_Z.stats.delta
            assert tr_E.stats.delta == tr_Z.stats.delta

            assert (tr_N.stats.starttime - tr_Z.stats.starttime) < \
                tr_N.stats.delta * 0.1
            assert (tr_E.stats.starttime - tr_Z.stats.starttime) < \
                tr_N.stats.delta * 0.1

            assert (tr_N.stats.endtime - tr_Z.stats.endtime) < \
                tr_N.stats.delta * 0.1
            assert (tr_E.stats.endtime - tr_Z.stats.endtime) < \
                tr_N.stats.delta * 0.1
        except AssertionError:
            print(tr_Z)
            print(tr_N)
            print(tr_E)
            raise

        t1 = max(t1, float(tr_Z.stats.endtime))
        t0 = min(t0, float(tr_Z.stats.starttime))

    if tstart is None:
        tstart = t0
    else:
        tstart = float(utct(tstart))

    if tend is None:
        tend = t1
    else:
        tend = float(utct(tend))

    dt = tr_N.stats.delta

    return tstart, tend, dt


def _calc_dop_windows(dop_specwidth, dop_winlen, dt, fmax, fmin, kind, nf,
                      nfft, overlap, winlen_sec):
    # Calculate width of smoothing windows for degree of polarization analysis
    if kind == 'spec':
        ntsum = int(dop_winlen / (winlen_sec * (1 - overlap)))
        df = 1. / (nfft * dt)
        nfsum = int(dop_specwidth / df)
        dsfact = max(1, ntsum // 2)
        dsfacf = max(1, nfsum // 2)
    else:
        periods = 1. / np.logspace(np.log10(fmin), np.log10(fmax), nf)
        ntsum = np.array(dop_winlen * periods / dt, dtype=int)
        df = (fmax / fmin) ** (1. / nf)
        nfsum = int(np.log(np.sqrt(dop_specwidth)) / np.log(df))
        if nfsum < 1:
            print(f'frequency step for DOP too small: {dop_specwidth:4.2f}' +
                  f'vs {df:4.2f}, setting to non-interpolation')
            nfsum = 1
        dsfacf = max(1, nfsum // 4)
        dsfact = max(1, int(dop_winlen))

    if type(ntsum) == int and ntsum < 1:
        raise ValueError('time window for DOP analysis is too short')
    if nfsum < 0.5:
        raise ValueError('spectral width DOP analysis is too small % 4.1f' %
                         nfsum)

    return nfsum, ntsum, dsfacf, dsfact


def _compute_spec(tr_Z, tr_N, tr_E, kind, fmin, fmax, winlen, nfft,
                  overlap=0.5, w0=10, nf=100):

    if kind == 'cwt':
        npts = tr_Z.stats.npts
        dt = tr_Z.stats.delta

        u1 = cwt(tr_Z.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)
        u2 = cwt(tr_N.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)
        u3 = cwt(tr_E.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)

        t = np.linspace(0, dt * npts, npts)
        f = np.logspace(np.log10(fmin),
                        np.log10(fmax),
                        nf)
    elif kind == 'spec':
        # parameters chosen to resemble matplotlib.mlab.specgram defaults
        kwargs = {'nperseg': winlen,
                  'fs': tr_Z.stats.sampling_rate,
                  'nfft': nfft,
                  'noverlap': int(winlen * overlap),
                  'mode': 'complex',
                  'scaling': 'density',
                  'window': 'hanning',
                  'detrend': False}

        f, t, u1 = spectrogram(tr_Z.data, **kwargs)
        f, t, u2 = spectrogram(tr_N.data, **kwargs)
        f, t, u3 = spectrogram(tr_E.data, **kwargs)

        # normalization for mode='complex' differs from 'psd'
        u1 *= 2 ** 0.5
        u2 *= 2 ** 0.5
        u3 *= 2 ** 0.5

    else:
        raise ValueError(
            'unknown TF method: %s (allowed: spec, cwt)' % kind)

    return f, t, u1, u2, u3


def _dop_elli_to_alpha(P, elli, use_alpha = True, use_alpha2 = False):
    if use_alpha:
        # map DOP to alpha such that alpha = 0 for DOP < 0.4 and 1 for
        # DOP > 0.6 and linear stretching in between
        alpha = P * 5. - 2.
        alpha[alpha > 1.] = 1.
        alpha[alpha < 0.] = 0.

        if use_alpha2:
            alpha2 = elli * 5 - 1.
            alpha2[alpha2 > 1.] = 1.
            alpha2[alpha2 < 0.] = 0.
            alpha2 = np.minimum(alpha, alpha2)
        else:
            alpha2 = alpha
    else:
        alpha = np.ones_like(P)
        alpha2 = np.ones_like(P)

    return alpha, alpha2



def detick(tr, detick_nfsamp, fill_val=None, freq_tick=1.0):
    # simplistic deticking by muting detick_nfsamp freqeuency samples around
    # 1Hz
    tr_out = tr.copy()
    Fs = tr.stats.sampling_rate
    NFFT = next_pow_2(tr.stats.npts)
    tr.detrend()
    df = np.fft.rfft(tr.data, n=NFFT)
    idx_1Hz = np.argmin(np.abs(np.fft.rfftfreq(NFFT) * Fs - freq_tick))
    if fill_val is None:
        fill_val = (df[idx_1Hz - detick_nfsamp - 1] + \
                    df[idx_1Hz + detick_nfsamp + 1]) / 2.
    df[idx_1Hz - detick_nfsamp:idx_1Hz + detick_nfsamp] /= \
        df[idx_1Hz - detick_nfsamp:idx_1Hz + detick_nfsamp] / fill_val
    tr_out.data = np.fft.irfft(df)[:tr.stats.npts]
    return tr_out

