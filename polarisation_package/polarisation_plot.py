#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 2021

@author: Géraldine Zenhäusern

:copyright:
    Géraldine Zenhäusern (geraldine.zenhaeusern@erdw.ethz.ch), 2022
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    GPLv3
"""
from os import makedirs
from os.path import join as pjoin, exists as pexists

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import polarisation_calculation as polarization
import seaborn as sns
from matplotlib.colorbar import make_axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from obspy import Stream
from obspy import UTCDateTime as utct
from obspy.signal.util import next_pow_2
from scipy import stats




def plot_polarization_event_noise(st, 
                              t_pick_P, t_pick_S,
                              timing_P, timing_S, timing_noise,
                              phase_P, phase_S,
                              delta_P = '', delta_S = '',
                              rotation = 'ZNE', BAZ=None,
                              BAZ_fixed=None, inc_fixed=None,
                              kind='cwt', fmin=0.1, fmax=10.,
                              winlen_sec=20., overlap=0.5,
                              tstart=None, tend=None, vmin=-180,
                              vmax=-140, log=True, fname='Polarisation_plot',
                              path='.',
                              dop_winlen=10, dop_specwidth=1.1,
                              nf=100, w0=8,
                              alpha_inc = None, alpha_elli = None, alpha_azi = None,
                              f_band_density = (0.3, 1.),
                              zoom = False,
                              differentiate=False, detick_1Hz=False):
    """
    

    Parameters
    ----------
    waveforms_VBB : obspy stream
        Input stream data rotated to ZNE. Data should either be in velocity; or displacement with differentiate = True.
    t_pick_P : List
        [start, end] seconds around P-arrival for polarisation calculation. If before pick, use e.g. '-5' for 5sec before arrival
    t_pick_S : List
        Same as for t_pick_P.
    timing_P : UTCDatetime or string
        Timing of first wave of interest (often P, but can of course be set arbitrarily). The first polarisation window is anchored on this and the back azimuth is estimated from it.
    timing_S : UTCDatetime or string
        Timing of second window anchor. No numerical baz value is estimated from it, but the same analysis is performed as for the first window.
    timing_noise : UTCDatetime or string
        Pre-event noise window anchor to compare the event polarisation to.
    phase_P : string
        Name of first signal window (for phase/plot labeling).
    phase_S : string
        Name of second signal window (for phase/plot labeling).
    delta_P: string, optional
        Picking uncertainty in seconds (will be used as float). Marks uncertainty width on Plot as a horizontal line to compare polarisation of signal to. 
        The default is '' and no uncertainty will be ploptted.
    delta_S : string, optional
        Same as previous for the second phase. The default is ''.
    rotation : string, optional
        Specify if traces should be rotated to 'RT' or 'LQT'. If these are given, the value of 'BAZ' is used. The default is 'ZNE'.
    BAZ : float, optional
        True back azimuth of the event. Will be marked on plots where back azimuth is shown. The default is None.
    BAZ_fixed : Int/float, optional
        Mainly for Mars: Add manual back azimuth of P-vector which will show in stereoplots to compare with data. The default is None.
    inc_fixed : int/float, optional
        Mainly for Mars: Add manual inclnation of P-vector which will show in stereoplots to compare with data. The default is None.
    kind : string, optional
        'spec' or 'cwt' for time-freq domain calculation. Spectrogram or continuous wavelet transform. The default is 'cwt'.
    fmin : float, optional
        Minimum frequency on f-axis. The default is 0.1.
    fmax : float, optional
        Maximum frequency on f-axis. The default is 10..
    winlen_sec : int, optional
        Window length for degree of polarisation (dop) windows. The default is 20.
    overlap : float, optional
        Overlap for windows of spectrograms (only relevant if kind = 'spec'). The default is 0.5.
    tstart : string, optional
        Starttime of event. The default is None.
    tend : string, optional
        Endtime of event. The default is None.
    vmin : int, optional
        Minimum signal amplitude in dB. The default is -180.
    vmax : int, optional
        Maximum signal amplitude in dB. The default is -140.
    log : Bool, optional
        Axis of frequency axis on log scale or not. The default is True.
    fname : string, optional
        Name for file when saving. The default is 'Polarisation_plot'.
    path : string, optional
        Point towards folder where plot will be saved. The default is '.'.
    dop_winlen : int, optional
        window length for degree of polarisation analysis. The default is 10.
    dop_specwidth : float, optional
        spectral width  for degree of polarisation analysis. The default is 1.1.
    nf : TYPE, optional
        DESCRIPTION. The default is 100.
    w0 : int, optional
        parameter for cwt, tradeoff between time and frequency resolution. Only applicable if kind='cwt'. The default is 8.
    alpha_inc : float, optional
        Factor describing how strong filtering based on inclination is. Check function 'polarisation_filtering'. The default is None.
    alpha_elli : TYPE, optional
        Factor describing how strong filtering based on ellipticity is. Check function 'polarisation_filtering'. The default is None.
    alpha_azi : TYPE, optional
        Factor describing how strong filtering based on azimuth is. Check function 'polarisation_filtering'. The default is None.
    f_band_density : tuple/list, optional
        Frequency band where back azimuth is estimated. The default is (0.3, 1.).
    zoom : Bool, optional
        Set to True if time-frequency window should zoom in on the two signal windows. The default is False.
    differentiate : Bool, optional
        Set to True if waveforms should be differentiated before the polarisation analysis. The default is False.
    detick_1Hz : Bool, optional
        Only applicable for Mars InSight data. Set to True if 1 Hz tick noise should be removed. The default is False.

    Returns
    -------
    None.

    """

    print('Processing waveforms...')
    
    #------------------------ Set parameters, pre-process waveforms -----------------------------------

    name_timewindows = [f'Signal {phase_P}', f'Signal {phase_S}', 'Noise', f'{phase_P}', f'{phase_S}'] #the last two are for the legend labeling

    #Process waveforms incl. rotation, trimming etc.
    st_Copy, components = waveform_processing(st, rotation, BAZ, differentiate, 
                                              timing_P, timing_S, timing_noise,
                                              tstart, tend)


    st_Z = Stream(traces=[st_Copy.select(component=components[0])[0]])
    st_N = Stream(traces=[st_Copy.select(component=components[1])[0]])
    st_E = Stream(traces=[st_Copy.select(component=components[2])[0]])
    
    #P window
    tstart_signal_P = utct(timing_P) + t_pick_P[0]
    tend_signal_P = utct(timing_S) - 5 if (utct(timing_P) + t_pick_P[1]) > (utct(timing_S) - 1) else  utct(timing_P) + t_pick_P[1] #Avoid going into S-window
    #S window
    tstart_signal_S = utct(timing_S) + t_pick_S[0]
    tend_signal_S = utct(timing_S) + t_pick_S[1]

    #Noise window
    tstart_noise = utct(timing_noise[0])
    tend_noise = utct(timing_noise[-1])
    
    #Set how the spectrogram windows are cut at time axis
    zoom_timewindow = [utct(utct(timing_P) - 120), utct(utct(timing_S) + 120)]
    normal_timewindow = [utct(tstart_noise - 30), utct(utct(timing_S) + 60)]
    

    tstart, tend, dt = polarization._check_traces(st_Z, st_N, st_E, tstart, tend)

    #----------------------- Plot preparation ---------------------------------------
    signal_P_row = 2
    signal_S_row = 3
    noise_row = 1

    fig, gs00 = create_major_plot_layout() # gs00 is the main gridspec: 3 rows of subplots
    axes0, axes1, gridspec_kw, nrows, box_legend, box_compass_colormap = create_subplot_layout(gs00)
    rect, color_windows = rectangles_for_time_windows(fmin, fmax, 
                                                      tstart_signal_P, tend_signal_P, 
                                                      tstart_signal_S, tend_signal_S, 
                                                      tstart_noise, tend_noise, 
                                                      nrows)


    winlen = int(winlen_sec / dt)
    nfft = next_pow_2(winlen) * 2

    # variables for statistics
    nbins = 30
    nts = 0

    # Calculate width of smoothing windows for degree of polarization analysis
    nfsum, ntsum, dsfacf, dsfact = polarization._calc_dop_windows(
        dop_specwidth, dop_winlen, dt, fmax, fmin,
        kind, nf, nfft, overlap, winlen_sec)

    if kind == 'spec':
        binned_data_signal_P = np.zeros((nrows, nfft // (2 * dsfacf) + 1, nbins))
        binned_data_signal_S = np.zeros_like(binned_data_signal_P)
        binned_data_noise = np.zeros_like(binned_data_signal_P)

    else:
        binned_data_signal_P = np.zeros((nrows, nf // dsfacf, nbins))
        binned_data_signal_S = np.zeros_like(binned_data_signal_P)
        binned_data_noise = np.zeros_like(binned_data_signal_P)


    #For KDE curve
    kde_list = [[[] for j in range(3)] for _ in range(nrows)]
    kde_dataframe_P = [[] for _ in range(nrows)]
    kde_dataframe_S = [[] for _ in range(nrows)]
    kde_noiseframe = [[] for _ in range(nrows)]
    kde_weights = [[[] for j in range(3)] for i in range(nrows)]

    #custom colormap for azimuth
    color_list = ['blue', 'cornflowerblue', 'goldenrod', 'gold', 'yellow', 'darkgreen', 'green', 'mediumseagreen', 'darkred', 'firebrick', 'tomato', 'midnightblue', 'blue']
    custom_cmap =  LinearSegmentedColormap.from_list('', color_list) #interpolated colormap - or use with bounds
    bounds = [0, 15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 360]
    
    print('Polarisation analysis...')

    #----------------------------------- Start of analysis ----------------------------------
    for tr_Z, tr_N, tr_E in zip(st_Z, st_N, st_E):
        if tr_Z.stats.npts < winlen * 4:
            continue

        #-------------------- Do polarisation calculation ----------------------------
        #Compute spectrogram
        if detick_1Hz:
            tr_Z_detick = polarization.detick(tr_Z, 10)
            tr_N_detick = polarization.detick(tr_N, 10)
            tr_E_detick = polarization.detick(tr_E, 10)
            f, t, u1, u2, u3 = polarization._compute_spec(tr_Z_detick, tr_N_detick, tr_E_detick, kind, fmin, fmax,
                                         winlen, nfft, overlap, nf=nf, w0=w0)
        else:
            f, t, u1, u2, u3 = polarization._compute_spec(tr_Z, tr_N, tr_E, kind, fmin, fmax,
                                             winlen, nfft, overlap, nf=nf, w0=w0)

        #Polarisation calculation
        azi1, azi2, elli, inc1, inc2, r1, r2, P = polarization.compute_polarization(
            u1, u2, u3, ntsum=ntsum, nfsum=nfsum, dsfacf=dsfacf, dsfact=dsfact)

        f = f[::dsfacf]
        t = t[::dsfact]
        t += float(tr_Z.stats.starttime)
        nts += len(t)

        bol_density_f_mask, bol_signal_P_mask, bol_signal_S_mask, bol_noise_mask, twodmask_P, twodmask_S, twodmask_noise = boolean_masks_f_t(f, t, 
                                                                                                                                             tstart_signal_P, tend_signal_P,
                                                                                                                                             tstart_signal_S, tend_signal_S, 
                                                                                                                                             tstart_noise, tend_noise, 
                                                                                                                                             f_band_density)
        if '-' in phase_S: #No second pick - histograms are empty
            bol_signal_S_mask[:] = False
            for i in range(3):
                twodmask_S[i][:] = False

        #Scalogram and alpha/masking of signals
        r1_sum, alpha, alpha2 = polarisation_filtering(r1, inc1, azi1, azi2, elli,
                                                       alpha_inc, alpha_azi, alpha_elli,
                                                       P)
        scalogram= 10 * np.log10(r1_sum)



        #Prepare x axis array (datetime)
        t_datetime = np.zeros_like(t,dtype=object)
        for i, time in enumerate(t):
             t_datetime[i] = utct(time).datetime

        # List with data, metadata, and alpha filter
        iterables = [
            (scalogram, vmin, vmax, np.ones_like(alpha),
             'amplitude\n[dB]', np.arange(vmin, vmax+1, 20), 'plasma', None),
            (np.rad2deg(azi1), 0, 360, alpha,
             'major azimuth\n[degree]', np.arange(0, 361, 90), custom_cmap, bounds), #was 45 deg steps, tab20b
            (np.rad2deg(abs(inc1)), -0, 90, alpha,
             'major inclination\n[degree]', np.arange(0, 91, 20), 'gnuplot', None)]

        # #--------------- Other options:---------------------
        #     #Minor axis azimuth
        #     (np.rad2deg(azi2), 0, 180, alpha2,
        #           'minor azimuth\n[degree]', np.arange(0, 181, 30), custom_cmap, bounds)
        #     #Minor axis inclination
        #     (np.rad2deg(inc2), -90, 90, alpha2,
        #           'minor inclination\n[degree]', np.arange(-90, 91, 30), 'gnuplot', None)
        #     #Ellipticity
        #     (elli, 0, 1, alpha,
        #          'ellipticity\n',  np.arange(0, 1.1, 0.2), 'gnuplot', None)
        
        
        # ------------------plot scalogram, ellipticity, major axis azimuth and inclination------------------------
        # Calculate histogram data
        for irow, [data, rmin, rmax, a, xlabel, xticks, cmap, boundaries] in \
                enumerate(iterables):

            ax = axes0[irow, 0]

            #plot data in time-frequency subplots
            if log and kind == 'cwt':
                # imshow can't do the log sampling in frequency
                cm = polarization.pcolormesh_alpha(ax, t_datetime, f, data,
                                                   alpha=a, cmap=cmap,
                                                   vmin=rmin, vmax=rmax, bounds=boundaries)

            else:
                cm = polarization.imshow_alpha(ax, t_datetime, f, data, alpha=a, cmap=cmap,
                                  vmin=rmin, vmax=rmax)

            #add colorbar on the left
            if tr_Z == st_Z[0]:
                cax, kw = make_axes(ax, location='left', fraction=0.07,
                                    pad=0.13)
                cbar = plt.colorbar(cm, cax=cax, ticks=xticks, **kw)
                cbar.ax.tick_params(labelsize=12) 

            #Get the f-t windows of the data (P, S, noise) for the KDE calculation later
            for i, mask in enumerate((twodmask_P[0], twodmask_S[0], twodmask_noise[0])):
                kde_list[irow][i] = data[mask]
                kde_weights[irow][i] = alpha[mask]
                
            #Calculate the histograms for the middle of (b)
            for i in range(len(f)):
                binned_data_signal_P[irow, i, :] += np.histogram(data[i,bol_signal_P_mask], bins=nbins,
                                                        range=(rmin, rmax),
                                                        weights=alpha[i,bol_signal_P_mask], density=True)[0]
                binned_data_signal_S[irow, i, :] += np.histogram(data[i,bol_signal_S_mask], bins=nbins,
                                                        range=(rmin, rmax),
                                                        weights=alpha[i,bol_signal_S_mask], density=True)[0]
                binned_data_noise[irow, i, :] += np.histogram(data[i,bol_noise_mask], bins=nbins,
                                                        range=(rmin, rmax),
                                                        weights=alpha[i,bol_noise_mask], density=True)[0]

    
    print('Generating plot...')
    
    #---------------------------- Axis parameters, turn on/off labels, prepare data -----------------------------
    #set how many major and minor ticks for the time axis - concise date version
    loc_major = mdates.AutoDateLocator(tz=None, minticks=4, maxticks=7)
    loc_minor = mdates.AutoDateLocator(tz=None, minticks=4, maxticks=15)
    formatter = mdates.ConciseDateFormatter(loc_major)

    #Time-frequency plots and histogram plots
    for ax in axes0:
        if zoom:
            ax[0].set_xlim(zoom_timewindow[0].datetime, zoom_timewindow[1].datetime)
        else:
            ax[0].set_xlim(normal_timewindow[0].datetime, normal_timewindow[1].datetime)
        ax[0].xaxis.set_major_formatter(formatter)
        ax[0].xaxis.set_major_locator(loc_major)
        ax[0].xaxis.set_minor_locator(loc_minor)

        for a in ax[:]:
            a.set_ylim(fmin, fmax)
            a.set_ylabel("frequency [Hz]", fontsize=12)
        if log:
            ax[0].set_yscale('log')
        ax[0].yaxis.set_ticks_position('both')
        ax[1].yaxis.set_ticks_position('both')
        ax[2].yaxis.set_ticks_position('both')
        # set tick position twice, otherwise labels appear right :/
        ax[signal_S_row].yaxis.set_ticks_position('right')
        ax[signal_S_row].yaxis.set_label_position('right')
        ax[signal_S_row].yaxis.set_ticks_position('both')

    for ax in axes1: #density
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position('both')

    for ax in axes0[0:-1, :].flatten(): #remove x axis labels for the upper plots
        ax.set_xlabel('')

    for ax in axes0[0:-1, 0]: #make it so that the spectrogram plots have linked x axes (i.e. time)
        ax.get_shared_x_axes().join(ax, axes0[-1, 0])

    for ax in axes0[:, 1]: #remove y label for histograms
        ax.set_ylabel('')

    for ax in axes0[:, 2]: #remove y label for histograms
        ax.set_ylabel('')
    for ax in axes0[:, 3]: #set y axis label, but rotate it so it's clear it applies to the left
        ax.set_ylabel('frequency [Hz]', rotation=-90, labelpad=15, fontsize=12)

    #time-frequncy plots
    for i,ax in enumerate(axes0[:, 0]):
        ax.grid(b=True, which='both', axis='x') #turn on the grid for the time ticks

        #Patches marking the time windows used in the analysis
        ax.add_patch(rect[i][0])
        ax.add_patch(rect[i][-1])

        #mark P/S arrival
        ax.axvline(x=utct(timing_P).datetime,ls='dashed',c='black')
        if not '-' in phase_S: #second pick available, so plot box and pick arrival times
            ax.axvline(x=utct(timing_S).datetime,ls='dashed',c='black')
            ax.add_patch(rect[i][1])
            
            
    #Turn of ticks of time-frequncy plots
    for ax in axes0[0:-1, 0]:
        ax.set_xticklabels('')
        
    #Set fontsize for the axes ticks
    for ax in axes0.flatten():
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.get_offset_text().set_size(12)
    for ax in axes1.flatten():
        ax.tick_params(axis="both", labelsize=12)

    #Make dictionary for P, S, and noise with data and their respective weights for the KDE plot
    for i in range(nrows):
        kde_dataframe_P[i] = {'P': kde_list[i][0],
                            'weights': kde_weights[i][0]}
        kde_dataframe_S[i] = {'S': kde_list[i][1],
                            'weights': kde_weights[i][1]}
        kde_noiseframe[i] = {'Noise': kde_list[i][2],
                             'weights': kde_weights[i][2]}


    #-------------------- Set titles, label the P and S timings, mark the boxes with labels -----------------------------------------------
    axes0[0, signal_P_row].set_title(f'{name_timewindows[0]}\n{t_pick_P[1]-t_pick_P[0]}s', fontsize=14)
    axes0[0, noise_row].set_title(f'{name_timewindows[2]}\n{tend_noise-tstart_noise:.0f}s', fontsize=14)
    axes1[0].set_title(f'Density\n{f_band_density[0]}-{f_band_density[1]} Hz', fontsize=14)
    
    axes0[0, 0].text(utct(tstart_signal_P-35).datetime, fmax+0.28*fmax, f'{name_timewindows[0]}', c=color_windows[0], fontsize=12)
    if not zoom or (zoom and (utct(tstart_noise).datetime >= utct(utct(timing_P) - 120).datetime and \
                              utct(tstart_noise).datetime < utct(utct(timing_S) + 120).datetime)):
        axes0[0, 0].text(utct(tstart_noise).datetime, fmax+0.28*fmax, f'{name_timewindows[2]}', c=color_windows[2], fontsize=12)
    axes0[0, 0].text(utct(timing_P).datetime, fmin-0.5*fmin, phase_P, c='black', fontsize=12)
    
    if not '-' in phase_S: #second pick available
        axes0[0, signal_S_row].set_title(f'{name_timewindows[1]}\n{t_pick_S[1]-t_pick_S[0]}s', fontsize=14)
        axes0[0, 0].text(utct(tstart_signal_S-35).datetime, fmax+0.28*fmax, f'{name_timewindows[1]}', c=color_windows[1], fontsize=12)
        axes0[0, 0].text(utct(timing_S).datetime, fmin-0.5*fmin, phase_S, c='black', fontsize=12)
        
    #Mark the picking uncertainty
    if not len(delta_P) == 0: #for some reason sometimes there is no uncertainty in the catalog
        axes0[0, 0].annotate(text='', 
                             xytext=((utct(timing_P)-float(delta_P)).datetime,0.08), 
                             xy=((utct(timing_P)+float(delta_P)).datetime,0.08),
                             arrowprops=dict(arrowstyle='|-|', mutation_scale=2.), xycoords='data', textcoords = 'data', annotation_clip=False)
    if not '-' in phase_S and not len(delta_S) == 0: 
        axes0[0, 0].annotate(text='', 
                             xytext=((utct(timing_S)-float(delta_S)).datetime,0.08), 
                             xy=((utct(timing_S)+float(delta_S)).datetime,0.08),
                             arrowprops=dict(arrowstyle='|-|', mutation_scale=2.), xycoords='data', textcoords = 'data', annotation_clip=False)


    #-----------Make histogram and KDE plots in b) --------------------
    for irow, [data, rmin, rmax, a, xlabel, xticks, cmap, boundaries] in \
            enumerate(iterables):

        #hist plot: signal P
        ax = axes0[irow, signal_P_row]
        cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                           f, binned_data_signal_P[irow] *(rmax-rmin),
                           cmap='hot_r', #pqlx,
                           vmin=0., vmax=10,
                           shading='auto')
        ax.axhspan(f_band_density[0], f_band_density[-1], color=color_windows[3], alpha=0.2) #mark f-band used in density plot
        ax.set_ylim(fmin, fmax)
        ax.set_xticks(xticks)
        #Color the outside lines of the plot
        for spine in ax.spines.values():
            spine.set_edgecolor(color_windows[0])
            spine.set_linewidth(2)

        #hist plot: signal S
        ax = axes0[irow, signal_S_row]
        cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                           f, binned_data_signal_S[irow] *(rmax-rmin),
                           cmap='hot_r', #pqlx,
                           vmin=0., vmax=10,
                           shading='auto')
        ax.axhspan(f_band_density[0], f_band_density[-1], color=color_windows[3], alpha=0.2) #mark f-band used in density plot
        ax.set_ylim(fmin, fmax)
        ax.set_xticks(xticks)
        #Color the outside lines of the plot
        for spine in ax.spines.values():
            spine.set_edgecolor(color_windows[1])
            spine.set_linewidth(2)

        #hist plot: noise
        ax = axes0[irow, noise_row]
        cm = ax.pcolormesh(np.linspace(rmin, rmax, nbins),
                           f, binned_data_noise[irow] *(rmax-rmin),
                           cmap='hot_r', #pqlx,
                           vmin=0., vmax=10,
                           shading='auto')
        ax.axhspan(f_band_density[0], f_band_density[-1], color=color_windows[3], alpha=0.2) #mark f-band used in density plot

        ax.set_ylim(fmin, fmax)
        ax.set_xticks(xticks)
        #Color the outside lines of the plot as a visual help
        for spine in ax.spines.values():
            spine.set_edgecolor(color_windows[2])
            spine.set_linewidth(2)

        #Set the ticks for the frequency axis manually if a log axis is used
        if log:
            for i in range(0, 4):
                axes0[irow, i].set_yscale('log')
                axes0[irow, i].set_yticks((0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0))
                axes0[irow, i].set_yticklabels(("0.1", "0.2", "0.5", "1", "2", "5", "10"))
                axes0[irow, i].yaxis.set_minor_formatter(NullFormatter()) #removes minor ticks between the major ticks which are set above
                axes0[irow, i].set_ylim(fmin, fmax)

        #Make the boxes on the leftmost side which say what each row shows: amplitude, azimuth etc
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax = axes0[irow, 0]
        ax.text(x=-0.43, y=0.5, transform=ax.transAxes, s=xlabel,
                ma='center', va='center', bbox=props, rotation=90, size=12)


        #density curves over some frequency band
        ax = axes1[irow]

        sns.kdeplot(data=kde_dataframe_P[irow], x='P', common_norm=False, ax=ax, clip = (rmin, rmax), 
                    color=color_windows[0], legend=False, weights = 'weights', bw_adjust=.6)
        sns.kdeplot(data=kde_dataframe_S[irow], x='S', common_norm=False, ax=ax, clip = (rmin, rmax), 
                    color=color_windows[1], legend=False, weights = 'weights', bw_adjust=.6)
        sns.kdeplot(data=kde_noiseframe[irow], x='Noise', common_norm=False, ax=ax, clip = (rmin, rmax), 
                    color=color_windows[2], fill=True, legend=False, weights = 'weights', bw_adjust=.6)

        #Turn off y-axis ticks completely since they don't tell anything in this context
        ax.set_xticks(xticks)
        ax.set_xlim(rmin,rmax)
        ax.set_xlabel('')
        ax.set_yticklabels('')
        ax.set_yticks([])
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_edgecolor(color_windows[3])
            spine.set_linewidth(2)


    #----------------- General plot tidying --------------------------------
    #Get BAZ from max density of P curve, mark in density column
    max_x, error = calculate_kde_maxima(kde_dataframe_P)

    if BAZ_fixed and inc_fixed: #if we use manual P vector
        BAZ_P = np.deg2rad(BAZ_fixed)
        inc_P = np.deg2rad(inc_fixed)
        error = [BAZ_fixed - 20, BAZ_fixed + 20]
        manualPvector = True
    else:
        BAZ_P = np.deg2rad(max_x[0])
        inc_P = np.deg2rad(max_x[1]) #needed later for polar plots
        manualPvector = False

    # Mark KDE peak, mark possible fixed baz
    ax = axes1[1]
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=max_x[0],c='r') #mark the polarisation BAZ from the maximum of the curve
    ax.scatter(max_x[0], ymax, color = 'r', marker = 'D', edgecolors = 'k', linewidths = 0.4, zorder = 100) #mark maximum with a diamond
    for i in range(len(color_list)): #add a mock version of the colorbar on top of the KDE
        ax.axvspan(bounds[i], bounds[i+1], ymin = 0.95, color = color_list[i],zorder=1)
        
    if BAZ_fixed: #if we use a manual P vector, mark that BAZ as well
        ax.axvline(x=BAZ_fixed,c='indigo') #mark the polarisation BAZ for manual values
        ax.scatter(BAZ_fixed, ymax, color = 'indigo', s = 80 ,marker = '*', edgecolors = 'k', linewidths = 0.4, zorder = 100)

    #Set grid lines in histograms, mark BAZ from catalog in there
    if BAZ is not None and ('ZNE' in rotation): #plot BAZ if it exists and if traces have NOT been rotated
        for ax in axes0[1, 1:]:
            ax.axvline(x=BAZ,ls='dashed',c='darkgrey')

        ax = axes1[1]
        ax.axvline(x=BAZ,ls='dashed',c='darkgrey') #mark catalog baz
        ax.scatter(BAZ, ymax, color = 'darkgrey', marker = 'v', edgecolors = 'k', linewidths = 0.4, zorder = 99)

    for ax in axes0[1:, 1:].flatten():
        ax.grid(b=True, which='both', axis='x', linewidth=0.2, color='grey')
    #Turn off y-axis ticks for left and middle histograms
    for ax in axes0[:, 1:-1].flatten():
        ax.set_yticklabels('')

    #Legend for density column
    colors = color_windows[:-1]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = [f'{name_timewindows[-2]}', f'{name_timewindows[-1]}', f'{name_timewindows[2]}']
    axes1[0].legend(lines, labels, loc='lower right', bbox_to_anchor=box_legend, fontsize=12, handlelength=0.8, ncol=3)

    #Plot compass rose to visualise azimuth colors and mark BAZ_mqs
    compass_rose(fig, gridspec_kw, box_compass_colormap, rotation, BAZ_P, BAZ, bounds, color_list, color_windows)
    

    axes0[0,0].text(-0.45, 2.8, '(a)', fontsize=23, transform=axes0[0,0].transAxes)
    axes0[0,0].text(-0.45, 1.2, '(b)', fontsize=23, transform=axes0[0,0].transAxes)
    axes0[0,0].text(-0.45, -2.8, '(c)', fontsize=23, transform=axes0[0,0].transAxes)
    axes0[0,0].text(2.3, -2.8, '(d)', fontsize=23, transform=axes0[0,0].transAxes)

    ## ---------------- add rest of subplots and save figure----------------

    savename = fname
    if zoom:
        savename += '_zoom'

    path_full = pjoin('Plots')
        
    if not pexists(path_full):
        makedirs(path_full)    

    #find out which version is plotted: zoom or normal. Save so that it can be marked in waveforms
    if zoom:
        specgram_timewindow = zoom_timewindow
    else:
        specgram_timewindow = normal_timewindow
        
    # Add subplot a): waveforms
    plot_waveforms(st_Copy, timing_P, timing_S, tend, specgram_timewindow,
                   f_band_density[0], f_band_density[1], gs00,
                   fname, np.rad2deg(BAZ_P), error, BAZ, phase_P, phase_S)
    
    #introduce a new subgrid at the bottom of the plot, so last row of gs00
    gs0 = gs00[-1].subgridspec(1, 2, wspace=0.3, hspace=None, height_ratios=[1], width_ratios=[4, 1])
    
    # Add subplot b): stereoplots with inclination-baz information
    plot_3D_polar_phase_analysis(BAZ_P, inc_P, BAZ, f_band_density, 
                                 iterables, alpha, 
                                 twodmask_P, twodmask_S, twodmask_noise, 
                                 nbins, props, name_timewindows,
                                 gs0)
    
    # Add subplot d): stereoplots with back azimuth estimate from the S-wave
    plot_baz_from_p_and_s(BAZ_P, inc_P, 
                          iterables, alpha, twodmask_P, twodmask_S, 
                          gs0, nxbins=45, nybins=30, manualPvector=manualPvector) #nxbins=45, nybins=30
        
    
    # Save the plot
    fig.savefig(pjoin(path_full, f'{savename}_joined.png'), dpi=200)
    plt.close('all')



def plot_3D_polar_phase_analysis(BAZ_P, inc_P, BAZ, f_band_density, 
                                 iterables, alpha, 
                                 twodmask_P, twodmask_S, twodmask_noise, 
                                 nbins, props, name_timewindows, gsxx):
    """
    Function to plot part (c) of the plot: polar projections showing inclination vs azimuth for two frequency bands. So two plots for each time window.

    Parameters
    ----------
    BAZ_P : float
        Back azimuth from polarisation in RAD.
    inc_P : float
        inclination of the P vector in RAD.
    BAZ : float
        MQS back azimuth in DEGREES, None if not available.
    f_band_density : list or tuple
        [min, max] of frequency band used for the analysis.
    iterables : list
        data from polarisation analysis: amplitude, azimuth, inclination.
    alpha : array
        used to mask non-polarised signals.
    twodmask_P : list
        2-D mask (f-t) for [all, low, high] frequency to calculate histograms.
    twodmask_S : list
        2-D mask (f-t) for [all, low, high] frequency to calculate histograms.
    twodmask_noise : list
        2-D mask (f-t) for [all, low, high] frequency to calculate histograms.
    nbins : int
        how many bins for histogram.
    props : TYPE
        make label boxes consistent with those of main figure (b).
    name_timewindows : list
        How to label the three time windows analysed.
    gsxx : TYPE
        Subgrid from the overall figure.

    Returns
    -------
    None.

    """
    gs21 = gsxx[0].subgridspec(2, 3, wspace=0.35, hspace=0.3, height_ratios=[1,1], width_ratios=[1,1,1])
    axes22 = gs21.subplots(subplot_kw={'projection': 'polar'})

    colormap = 'gist_heat_r'

    BAZ_Inc_P = [[] for i in range(2)]
    BAZ_Inc_S = [[] for i in range(2)]
    BAZ_Inc_noise = [[] for i in range(2)]

    f_middle = f_band_density[0] + (f_band_density[1]-f_band_density[0])/2

    [data, rmin, rmax, a, xlabel, xticks, cmap, boundaries] = iterables[1] #azimuth
    inc_data = iterables[-1][0] #inclination

    #Calculate the 2-D histograms for lower and higher frequency band
    for i in range(2):
        BAZ_Inc_P[i] = np.histogram2d(data[twodmask_P[i+1]], inc_data[twodmask_P[i+1]], 
                                      bins=nbins, range=((rmin, rmax),(0,90)), 
                                      weights=alpha[twodmask_P[i+1]], 
                                      density=True)[0]
        BAZ_Inc_S[i] = np.histogram2d(data[twodmask_S[i+1]], inc_data[twodmask_S[i+1]], 
                                      bins=nbins, range=((rmin, rmax),(0,90)), 
                                      weights=alpha[twodmask_S[i+1]], 
                                      density=True)[0]
        BAZ_Inc_noise[i] = np.histogram2d(data[twodmask_noise[i+1]], inc_data[twodmask_noise[i+1]], 
                                          bins=nbins, range=((rmin, rmax),(0,90)), 
                                          weights=alpha[twodmask_noise[i+1]], 
                                          density=True)[0]


    #Plot all histograms: they need to be transposed so inclination is on the radial axis
    P_hists = (BAZ_Inc_P[0].T, BAZ_Inc_P[1].T)
    S_hists = (BAZ_Inc_S[0].T, BAZ_Inc_S[1].T)
    Noise_hist = (BAZ_Inc_noise[0].T, BAZ_Inc_noise[1].T)
    y_lim = (np.linspace(0, 90, nbins), np.linspace(0, 90, nbins))
    axes_list = (axes22[0,:], axes22[1,:])
    for i, (P, S, N, ylim, ax) in enumerate(zip(P_hists, S_hists, Noise_hist, y_lim, axes_list)):
        ax[1].pcolormesh(np.radians(np.linspace(rmin, rmax, nbins)),
                                ylim, P,
                                cmap=colormap,
                                shading='auto')
        ax[2].pcolormesh(np.radians(np.linspace(rmin, rmax, nbins)),
                                ylim, S,
                                cmap=colormap,
                                shading='auto')
        ax[0].pcolormesh(np.radians(np.linspace(rmin, rmax, nbins)),
                                ylim, N,
                                cmap=colormap,
                                shading='auto')

    #Tell readers what happens in (c): azimuth versus inclination plots. Is plotted on the left
    axes22[0,0].text(x=-0.3, y=-0.15, transform=axes22[0,0].transAxes, s='major azimuth\nvs inclination',
                ma='center', va='center', bbox=props, rotation=90, size=14)
    
    #Tell readers which frequency band: plotted on the right
    axes22[0,2].text(x=1.3, y=0.5, transform=axes22[0,2].transAxes, s=f'{f_band_density[0]}-{f_middle:.2f} Hz',
                ma='center', va='center', bbox=props, rotation=270, size=14)
    axes22[1,2].text(x=1.3, y=0.5, transform=axes22[1,2].transAxes, s=f'{f_middle:.2f}-{f_band_density[1]} Hz',
                ma='center', va='center', bbox=props, rotation=270, size=14)

    for flat_ax in axes22[:,:].flatten(): #Some general plotting houskeeping
        flat_ax.set_theta_zero_location("N")
        flat_ax.set_theta_direction('clockwise')
        flat_ax.invert_yaxis() #so vertical is in the middle - like stereoplot
        flat_ax.grid(True)
        flat_ax.tick_params(axis='both', labelsize=12)
        
        if BAZ is not None: #Mark catalog back azimuth if available
                align_h = 'right' if BAZ > 180. else 'left'
                align_v = 'top' if 90. < BAZ < 270. else 'bottom'
                flat_ax.axvline(x=np.radians(BAZ), color='grey')
                flat_ax.text(np.radians(BAZ), -5, 'BAZ\nMQS', c='grey', fontsize=14, 
                             path_effects=[PathEffects.withStroke(linewidth=0.2, foreground="black")], 
                             horizontalalignment=align_h, verticalalignment = align_v)

    #Plot the orthogonal plane to the P wave
    BAZ_S, inc_S = vector_to_orthogonal_plane(BAZ_P, inc_P)
    for ax in axes22[:,1:].flatten():
        ax.scatter(BAZ_P,np.rad2deg(inc_P), color='C0', zorder=100) #P-vector: point
        ax.plot(BAZ_S, np.rad2deg(inc_S), color= 'C0', zorder=101) #Orthogonal plane: line

    #Set title for each column
    for ax, sub_title in zip((axes22[0,:].flatten()),
                             (name_timewindows[2], name_timewindows[0], name_timewindows[1])):
        ax.set_title(sub_title, fontsize=14)
    
    
    
    
def plot_baz_from_p_and_s(baz_KDE, inc_KDE,
                          iterables, alpha, twodmask_P, twodmask_S, 
                          gsxx, nxbins=72, nybins=30,
                          manualPvector = False):
    
    #Subplot part d) of the main plot
    
    #Define some colors for plotting
    colors = ['C0', 'Firebrick', 'C9'] #Color for P vector from KDE, S, and combined vector
    colormap = 'gist_heat_r'
    
    #Further divide into two subplots
    gs = gsxx[-1].subgridspec(2, 1, wspace=0.3, hspace=0.3, height_ratios=[1,1], width_ratios=[1])
    axes = gs.subplots(subplot_kw={'projection': 'polar'})
    
    #get data
    [data, rmin, rmax, a, xlabel, xticks, cmap, boundaries] = iterables[1] #azimuth
    inc_data = iterables[-1][0] #inclination


    #inclination vs BAZ histogram data for the P and S-wave window - full frequency band used for KDE
    BAZ_Inc_S = np.histogram2d(data[twodmask_S[0]], inc_data[twodmask_S[0]], 
                                  bins=[nxbins, nybins], range=((rmin, rmax),(0,90)), 
                                  weights=alpha[twodmask_S[0]])[0]
    BAZ_Inc_P = np.histogram2d(data[twodmask_P[0]], inc_data[twodmask_P[0]], 
                                      bins=[nxbins, nybins], range=((rmin, rmax),(0,90)), 
                                      weights=alpha[twodmask_P[0]])[0]


    # ! BAZ_Inc needs to be transposed! Then inclination is on radial/y; baz is on phi/x
    BAZ_Inc_P = BAZ_Inc_P.T
    BAZ_Inc_P += 1.0 #Water level so that P+S Plot is not completely dominated by only P
    BAZ_Inc_S = BAZ_Inc_S.T
    
    #prepare a list with same dimensions as BAZ_Inc_S.
    #Since each [baz,inc] space needs to hold a vector [x,y,z], this must be a list and not a np.array
    s_vector_array = [[0 for i in range(BAZ_Inc_S.shape[1])] for j in range(BAZ_Inc_S.shape[0])]  
    baz_step = int(360/nxbins) #get number of steps
    inc_step = int(90/nybins)
    
    #go though inclination and azimuth space - at each point, calculate the vector in [x,y,z]. 
    #Its length is given by the number of histogram counts in the same bin
    #The more signals with this azimuth, inclination are present in the S timewindow, the longer the resulting vector -> will increase cross-product further down
    for i,baz in enumerate(range(0, 360,baz_step)):
        for j,inc in enumerate(range(0, 90,inc_step)):
            count = BAZ_Inc_S[j,i]
            s_vector_array[j][i] = azi_inc_to_xyz_vector(np.deg2rad(baz), np.deg2rad(inc), r=count)
     
    #Calculate S-wave match
    #For all possible P waves (inclination and baz), calculate cross product with the whole S window:
    baz_likelihood = np.zeros([int(90/inc_step), int(360/baz_step)])
    for i,baz_P in enumerate(range(0, 360,baz_step)):
        for j,inc_P in enumerate(range(0, 90,inc_step)):
            #Calculate the P-vector for that point
            uP = azi_inc_to_xyz_vector(np.deg2rad(baz_P), np.deg2rad(inc_P))
            
            for sublist in s_vector_array: #flatten does not exist
                for s_vector in sublist:
                    cross_product = np.cross(uP, s_vector)
                    #since the length of the cross-product result is affected by the length of the two in-going vectors, 
                    #this will increase the likelihood at this point if s_vector is large 
                    #uP has length=1
                    baz_likelihood[j,i] += np.sqrt(cross_product[0]**2+
                                                cross_product[1]**2+
                                                cross_product[2]**2)
           
    
    baz_range = np.radians(np.arange(rmin, rmax+1, baz_step))
    inc_range = np.arange(0,91,inc_step)
    half_baz_step = np.deg2rad(0.5*baz_step) #move the markers into the middle of the rectangles, since they could be anywhere in there
    
    #Get location where P wave probability is maximum from the S wave
    maxIndex = np.where(baz_likelihood == np.amax(baz_likelihood))
    maxInc = maxIndex[0][0]
    maxbaz = maxIndex[1][0]
    
    #Get location where P wave probability is max from both P and S window
    baz_from_P_S = baz_likelihood*BAZ_Inc_P
    
    #get the index of the inclination (first axis in matrix later on) where the inclination is >= 50 degrees
    #this should limit where the combined P-vector can be placed
    P_lim_index = np.min(np.where(inc_range >= 50.)) 
    maxIndexCombined = np.where(baz_from_P_S == np.amax(baz_from_P_S[P_lim_index:,:]))
    
    #P wave probability from S-wave plot
    axes[0].pcolormesh(np.radians(np.arange(rmin, rmax+1, baz_step)),
                                np.arange(0,91,inc_step), baz_likelihood,
                                cmap=colormap,
                                shading='auto')

    #P wave probability from P+S plot
    axes[1].pcolormesh(np.radians(np.arange(rmin, rmax+1, baz_step)),
                                np.arange(0,91,inc_step), baz_from_P_S,
                                cmap=colormap,
                                shading='auto')
    
    if True in twodmask_S[0]: #if second pick is available, some part of the bool mask must be true
        axes[0].scatter(baz_range[maxbaz]+half_baz_step, inc_range[maxInc]+0.5*inc_step, color=colors[1], zorder=101) #P-vector from S-wave
        axes[0].scatter(baz_KDE, np.rad2deg(inc_KDE), color=colors[0], zorder=100) #P-vector from P window KDE
    
        axes[1].scatter(baz_KDE, np.rad2deg(inc_KDE), color=colors[0], zorder=100) #P-vector from P window KDE
        axes[1].scatter(baz_range[maxbaz]+half_baz_step, inc_range[maxInc]+0.5*inc_step, color=colors[1], zorder=101) #P-vector from S-wave
        axes[1].scatter(baz_range[maxIndexCombined[1][0]]+half_baz_step, inc_range[maxIndexCombined[0][0]]+0.5*inc_step, color=colors[2], zorder=102) #P-vector from combination
    
    #Legend
    lines = [Line2D([0], [0], color=c, marker='o') for c in colors]
    if manualPvector:
        P_label = 'Manual P'
    else:
        P_label = 'P from Signal P'
    labels = [P_label, 'P from Signal S', 'P from P+S']
    axes[0].legend(lines, labels, loc='center left', bbox_to_anchor=(-0.8, -0.2), fontsize=12, handlelength=0.8)
    
    axes[0].set_title('P wave from S window', fontsize=14)
    axes[1].set_title('P wave from P+S', fontsize=14)
    for ax in axes:
        ax.set_rgrids((80, 60, 40, 20, 0), labels=('80', '60', '40', '20', '0'))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction('clockwise')
        ax.invert_yaxis()
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=12)
        

def plot_waveforms(st, timing_P, timing_S, tend, specgram_timelim,
                   fmin, fmax, gsxx,
                   name, baz_preferred, error, BAZ, phase_P, phase_S):
    """
    Plot vertical waveforms, mark P/S arrivals and show which part of the waveforms are depicted in the polarisation analysis

    """

    #filter the data in the same f-band as KDE anylsis is done
    st.filter('bandpass',freqmin=fmin, freqmax=fmax, corners=6)
    st.trim(starttime=utct(specgram_timelim[0]),
             endtime=utct(tend))
    
    #Time axis in seconds since phase pick (P or S)
    t_offset = float(st[0].stats.starttime - utct(timing_P))
    xvec_env = st[0].times() + t_offset
    S_P = utct(timing_S)-utct(timing_P)
    
    #Get timing where spectrogram axis is shown
    specgram_xmin = utct(specgram_timelim[0])-utct(timing_P)
    specgram_xmax = utct(specgram_timelim[1])-utct(timing_P)
    
    #Prepare plot
    gs = gsxx[0].subgridspec(1, 2, wspace=0.1, hspace=None, height_ratios=[1], width_ratios=[2.3, 1])
    axes = gs.subplots()
    
    axes[0].plot(xvec_env, st.select(component='Z')[0].data, "k-") #now uses vertical data
    axes[0].set_xlim(xvec_env[0], xvec_env[-1])
    
    
    ymin, ymax = axes[0].get_ylim()
    xmin, xmax = axes[0].get_xlim()
    #mark P/S arrival
    axes[0].axvline(x=0.,ls='dashed',c='C0')
    axes[0].text(0., ymax+0.08*ymax, phase_P, c='C0', fontsize=14)
    if not '-' in phase_S: #second pick available
        axes[0].axvline(x=S_P,ls='dashed',c='C0')
        axes[0].text(S_P, ymax+0.08*ymax, phase_S, c='C0', fontsize=14)
        
    axes[0].scatter([specgram_xmin,specgram_xmax],[ymin,ymin], s=80, marker=6, c='indigo') #mark the time axis of the spectrogram. marker 6 = upward triangle with tip at ymin level
    axes[0].hlines(y=ymin, xmin=specgram_xmin, xmax=specgram_xmax, color='indigo')

    axes[0].set_xlabel('Time after P [s]', fontsize=14)
    axes[0].set_ylabel('Velocity [m/s]', fontsize=14)

    axes[0].tick_params(axis='both', labelsize=12)
    
    axes[-1].set_visible(False)
    
    deg_sign = u'\N{DEGREE SIGN}' #unicode degree sign as string
    if BAZ:
        BAZ = f'{BAZ:.0f}{deg_sign}'
    else:
        BAZ = '-'
        
    #Text box for labeling the whole plot - gives the true/input baz, calculated baz, and uncertainties
    axes[0].text(x=1.29, y=1.0, transform=axes[0].transAxes, 
                  s=f'{name}\nTrue BAZ: {BAZ}\nPreferred BAZ: {baz_preferred:.0f}{deg_sign}\nUncertainty: {error[0]:.0f}-{error[1]:.0f}{deg_sign}',
                ma='left', va='top', bbox=dict(facecolor='white', alpha=0.5), size=14)
    
    

def vector_to_orthogonal_plane(BAZ_P, inc_P):
    #Vector fun: calculates orthogonal plane (S-wave lies there somewhere) when given a vector (defined from BAZ and inclination; P-wave)
    #get P coordinates from kde curve maxima
    BAZ_S = []
    inc_S = []

    #Define uP vector in cartesian coordinates from BAZ and inclination (inclination from polarisation is NOT the spherical coordinate inclination)
    gamma = np.linspace(0,2*np.pi, num=300)
    uP = azi_inc_to_xyz_vector(BAZ_P, inc_P)

    #get two orthogonal vectors uS1, uS2
    uS1 = np.random.randn(3)  # take a random vector
    uS1 -= uS1.dot(uP) * uP / np.linalg.norm(uP)**2       # make it orthogonal to uP
    uS1 /= np.linalg.norm(uS1)  # normalize it
    uS2 = np.cross(uP, uS1)      # cross product with uP to get second vector

    for i in gamma: #loop from 0 to 2pi
        uS = np.sin(i)*uS1 + np.cos(i)*uS2 #general vector uS from linear combination of uS1 and uS2
        r = np.sqrt(uS[0]**2+uS[1]**2+uS[2]**2)
        BAZ_S.append(np.arctan2(uS[1],uS[0]))
        inclination = np.pi/2-np.arccos(uS[2]/r) #inclination again defined as for polarisation analysis: 90° is vertical
        if inclination < 0: #'upper' part of sphere, ignore
            inc_S.append(np.nan)
        elif inclination <= np.pi/2:
            inc_S.append(inclination)
        else: #is landing on the other side, re-map to 0-90°
            inc_S.append(np.pi-inclination)
            
    return BAZ_S, inc_S



def azi_inc_to_xyz_vector(azi, inc, r=1):
    #Angles in RAD!
    #r allows for weighting based on counts, defaults to one
    y = np.sin(np.pi/2-inc)*np.sin(azi)*r
    x = np.sin(np.pi/2-inc)*np.cos(azi)*r
    z = np.cos(np.pi/2-inc)*r
    vector = np.array([x, y, z])
    
    return vector


def calculate_kde_maxima(kde_dataframe_P):
    #Manually does what the seaborn KDE does in the main code
    #covariance_factor was hand-tuned so that the two KDE curves were the same. 
    #Is NOT the same as the bw_adjust factor in seaborn, there is no direct access to that
    max_x = [[],[]]
    for j, (i, xlim) in enumerate(zip((1,2), (360,90))):
        kernel = stats.gaussian_kde(kde_dataframe_P[i]['P'], weights = kde_dataframe_P[i]['weights'])
        kernel.covariance_factor = lambda : .17 #old:  lambda : .20
        kernel._compute_covariance()
        xs = np.linspace(-50,xlim+50,1000) #extend to positive and negative spaces so that the error can be wrapped around
        ys = kernel(xs)
        index = np.argmax(ys)
        max_x[j] = xs[index]
        
        #get the error of the BAZ from the full width of the half maximum
        if j==0:
            #find the FWHM
            error = fwhm_error_from_kde(xs, ys, index)
            
    return max_x, error


    
def fwhm_error_from_kde(xs, ys, index):
    #get the error of the BAZ from the full width of the half maximum (FWHM)
    #find the FWHM
    max_y = max(ys)
    indexes_ymax = [x for x in range(len(ys)) if ys[x] > max_y/2.0]
    
    #Get correct FWHM in case there are several peaks above the halfway mark
    index_local = indexes_ymax.index(index) #get index of the maximum index within the list
    for k in range(index_local, len(indexes_ymax)-1): #forwards through list
        if indexes_ymax[k+1] > indexes_ymax[k]+1:
            index_high = indexes_ymax[k]
            break
        elif k == len(indexes_ymax)-2: #if there is only one peak
            index_high = indexes_ymax[-1]
    for k in range(index_local, 0, -1): #backwards
        if indexes_ymax[k-1] < indexes_ymax[k]-1:
            index_low = indexes_ymax[k]
            break
        elif k == 1: #if there is only one peak
            index_low = indexes_ymax[0]

    left_error = xs[index_low]
    right_error = xs[index_high]
    
    #wrap the errors around 0: however, it does not calculate the KDE for wrapped data
    if left_error<0.:
        left_error = 360.+left_error #negative value, so 360-6, e.g
    if right_error>360.:
        right_error = right_error-360.

    error = [left_error, right_error]
        
    return error

def polarisation_filtering(r1, inc1, azi1, azi2, elli,
                           alpha_inc, alpha_azi, alpha_elli,
                           P):
    # Apply filtering based on degree of polarisation (dop) and possibly ellipticity/inclination/azimuth
    # alpha_azi only really makes sense with ZRT/LQT data
    if alpha_inc is not None:
        if alpha_inc > 0.: #When looking for S
            func_inc= np.cos
            func_azi= np.sin
        else: #When looking for P
            alpha_inc= -alpha_inc
            func_inc= np.sin
            func_azi= np.cos
    else:
        #look at azimuth without inclination, let's just set it like this.
        #So cosinus prefers P waves, set to sinus to prefer S waves (perpendicular to BAZ)
        func_azi= np.cos 

    r1_sum = (r1** 2).sum(axis=-1)
    if alpha_inc is not None:
        r1_sum *= func_inc(inc1)**(2*alpha_inc)
    if alpha_azi is not None:
        r1_sum *= abs(func_azi(azi1))**(2*alpha_azi)
    if alpha_elli is not None:
        r1_sum *= (1. - elli)**(2*alpha_elli)

    alpha, alpha2= polarization._dop_elli_to_alpha(P, elli)

    if alpha_inc is not None:
        alpha*= func_inc(inc1)**alpha_inc
    if alpha_azi is not None:
        alpha*= abs(func_azi(azi1))**alpha_azi
    if alpha_elli is not None:
        alpha*= (1. - elli)**alpha_elli
        
    return r1_sum, alpha, alpha2
    
def boolean_masks_f_t(f, t, 
                      tstart_signal_P, tend_signal_P,
                      tstart_signal_S, tend_signal_S, 
                      tstart_noise, tend_noise, 
                      f_band_density):
    #Prepare a 2D bool mask for the KDE analysis - in time and requency
    #Prepare 1D bool masks for time and for frequency
    
    
    #Prep bool mask for timing of the P, S, and noise window
    bol_signal_P_mask= np.array((t > tstart_signal_P, t< tend_signal_P)).all(axis=0)
    bol_signal_S_mask= np.array((t > tstart_signal_S, t< tend_signal_S)).all(axis=0)
    bol_noise_mask= np.array((t > tstart_noise, t< tend_noise)).all(axis=0)

    #get indexes where f lies in the defined f-band for density subplot
    twodmask_P = [[] for i in range(3)]
    twodmask_S = [[] for i in range(3)]
    twodmask_noise = [[] for i in range(3)]

    f_middle = f_band_density[0] + (f_band_density[1]-f_band_density[0])/2

    for i, (f_low, f_high) in enumerate(zip((f_band_density[0], f_band_density[0], f_middle),
                                            (f_band_density[1], f_middle, f_band_density[1]))):
        # Prepare three different masks: full frequency limit + cutting the frequency band in half for more in-depth analysis
        # Whole f-band, lower f-band, higher f-band
        bol_density_f_mask = np.array((f >= f_low, f < f_high)).all(axis=0)
        twodmask_P[i] = bol_density_f_mask[:, None] & bol_signal_P_mask[None, :]
        twodmask_S[i] = bol_density_f_mask[:, None] & bol_signal_S_mask[None, :]
        twodmask_noise[i] = bol_density_f_mask[:, None] & bol_noise_mask[None, :]
        
    return bol_density_f_mask, bol_signal_P_mask, bol_signal_S_mask, bol_noise_mask, twodmask_P, twodmask_S, twodmask_noise
    
def waveform_processing(waveforms_VBB, rotation, BAZ, differentiate, 
                           timing_P, timing_S, timing_noise,
                           tstart, tend):
    #Handles waveforms: make copy of stream data
    #rotate if specified, differentiate if specified
    #trim the waveforms
    #hands back stream with data, and components
    #Trims waveform based on tstart, tend
    st_Copy = waveforms_VBB.copy()  
    
    #Rotate the waveforms into different coordinate system: ZRT or LQT
    if 'ZNE' not in rotation:
        if 'RT' in rotation:
            st_Copy.rotate('NE->RT', back_azimuth=BAZ)
            components = ['Z', 'R', 'T']
            #need to use -R, otherwise it aligns with 180° instead of 0°
            tr_R_data = st_Copy[1].data
            tr_R_data *= -1

        elif 'LQT' in rotation:
            st_Copy.rotate('ZNE->LQT', back_azimuth=BAZ, inclination = 40.0)
            components = ['L', 'Q', 'T']
        else:
            raise Exception("Sorry, please pick valid rotation system: ZNE, RT, LQT")
    else:
        components = ['Z', 'N', 'E']


    #differentiate waveforms
    if differentiate:
        st_Copy.differentiate()

    # #trim the waveforms in length
    #Note on trimming: when data is de-ticked later, the trimming affects the results - traces do not line up perfectly
    #Important to use consistent trimming for all operations
    trim_time = [900., 900.]
    st_Copy.trim(starttime=utct(tstart) - trim_time[0], #og: -50, +850
                      endtime=utct(tend) + trim_time[1])

    return st_Copy, components


def compass_rose(fig, gridspec_kw, box_compass_colormap, rotation, BAZ_pol, BAZ_mqs, bounds, 
                 color_list, color_windows):
    """
    BAZ_pol is in radians, BAZ_mqs is in degrees
    MQS = Marsquake Service; but it just uses the input BAZ data
    """

    #Compass rose-type plot to see in which direction azimuth colormap lies with respect to NESW
    rose_axes = fig.add_axes([gridspec_kw['left']+box_compass_colormap[0],
                              gridspec_kw['top']-box_compass_colormap[1],
                              box_compass_colormap[2], box_compass_colormap[2]], polar=True) # Left, Bottom, Width, Height
    if 'ZNE' not in rotation: #rotate the colormap so that 0° is in direction of the BAZ
        theta = [x+BAZ_mqs for x in bounds]
        theta = np.array(theta)
        theta[theta > 360] = theta[theta > 360] - 360 #remap values over 360°
    else:
        theta = bounds
    radii = [1]*len(theta)
    #Width of pie segments: first and last entry separately since blue is both at beginning and end of the color list= half the width each
    width = [30]*(len(theta)-2)
    width.insert(0, 15)
    width.insert((len(width)), 15)

    rose_axes.bar(np.radians(theta), radii, width=np.radians(width), color=color_list, align='edge')

    rose_axes.set_theta_zero_location("N")
    rose_axes.set_theta_direction('clockwise')
    rose_axes.set_xticks(np.radians(range(0, 360, 90)))
    rose_axes.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=14)
    rose_axes.set_yticklabels('')
    rose_axes.tick_params(pad=-2.0)
    rose_axes.yaxis.grid(False)
    
    #Draw calculated polarisation baz in blue
    rose_axes.annotate('', xytext=(0.0, 0.0), xy=(BAZ_pol,1.4),
                            arrowprops=dict(facecolor=color_windows[0], edgecolor='black', 
                                            linewidth = 0.5, width=0.9, headwidth=6., headlength=6.),
                            xycoords='data', textcoords = 'data', annotation_clip=False)
    align_h = 'right' if np.rad2deg(BAZ_pol) > 180. else 'left'
    align_v = 'top' if 90. < np.rad2deg(BAZ_pol) < 270. else 'bottom'
    rose_axes.text(BAZ_pol, 1.4, 'BAZ\nPol', c=color_windows[0], fontsize=14, 
                   path_effects=[PathEffects.withStroke(linewidth=0.2, foreground="black")], 
                   horizontalalignment=align_h, verticalalignment = align_v)
    
    #If there is a catalog back azimuth, plot that as a grey arrow
    if BAZ_mqs is not None:
        # rose_axes.axvline(x=np.radians(BAZ), color='black')
        rose_axes.annotate('', xytext=(0.0, 0.0), xy=(np.radians(BAZ_mqs),1.4),
                            arrowprops=dict(facecolor=color_windows[2], edgecolor='black', 
                                            linewidth = 0.5, width=0.9, headwidth=6., headlength=6.),
                            xycoords='data', textcoords = 'data', annotation_clip=False)

        align_h = 'right' if BAZ_mqs > 180. else 'left'
        align_v = 'top' if 90. < BAZ_mqs < 270. else 'bottom'

        rose_axes.text(np.radians(BAZ_mqs), 1.4, 'BAZ\ntrue', c=color_windows[2], fontsize=14, 
                       path_effects=[PathEffects.withStroke(linewidth=0.2, foreground="black")], 
                       horizontalalignment=align_h, verticalalignment = align_v)
    rose_axes.set_ylim([0, 1])

def rectangles_for_time_windows(fmin, fmax, 
                                tstart_signal_P, tend_signal_P, 
                                tstart_signal_S, tend_signal_S, 
                                tstart_noise, tend_noise, 
                                nrows):
    #Produces rectangles - can not be used in multiple subplots, so each row needs its own rectangle
    #Mark the time window in the freq-time plot used for further analysis
    color_windows = ['C0', 'Firebrick', 'grey', 'Peru'] #signal P, S, noise, density-color
    
    rect = [[None for i in range(3)] for j in range(nrows)] #prepare rectangles to mark the time windows
    for j in range(nrows):
        rect[j][0] = patches.Rectangle((utct(tstart_signal_P).datetime,fmin+0.03*fmin),
                                       utct(tend_signal_P).datetime-utct(tstart_signal_P).datetime,
                                       fmax-fmin-0.03*fmax, linewidth=2,
                                       edgecolor=color_windows[0], fill = False) #signal
        rect[j][1] = patches.Rectangle((utct(tstart_signal_S).datetime,fmin+0.03*fmin),
                                       utct(tend_signal_S).datetime-utct(tstart_signal_S).datetime,
                                       fmax-fmin-0.03*fmax, linewidth=2,
                                       edgecolor=color_windows[1], fill = False) #signal
        rect[j][2] = patches.Rectangle((utct(tstart_noise).datetime,fmin+0.03*fmin),
                                       utct(tend_noise).datetime-utct(tstart_noise).datetime,
                                       fmax-fmin-0.03*fmax, linewidth=2,
                                       edgecolor=color_windows[2], fill = False) #noise
    return rect, color_windows


def create_major_plot_layout():
    #This defines the overall figure (incl size)
    
    fig = plt.figure(figsize=(14, 17))
    gs00 = gridspec.GridSpec(3, 1, figure=fig,
                            left=0.06, bottom=0.02, right=0.97, top=0.98,
                            wspace=None, hspace=0.28,
                            height_ratios=[1,3,2.8], width_ratios=[1])
    
    return fig, gs00

    
def define_plot_layout():
    # Create figure layout for middle/main part

    gridspec_kw = dict(width_ratios=[10, 2, 2, 2, 2],  # specgram, hist2d, hist2d
                       height_ratios=[1, 1, 1],
                       top=0.895,
                       bottom=0.055,
                       left=0.055,
                       right=0.985,
                       hspace=0.18,
                       wspace=0.1)
    box_legend = (1.08, 1.3)
    box_compass_colormap = [0.64, 0.02, 0.09] #offset from GRIDSPEC bounds left (+ is right), top (+ is down), width/height
    nrows = 3
    figsize_y = 9
        
    return gridspec_kw, nrows, figsize_y, box_legend, box_compass_colormap


def create_subplot_layout(gsxx):
    #Subplot Layout for middle part of the plot (b)
    #Time-frequency windows, histograms, density curves
    
    #get the layout parameters/boundaries
    gridspec_kw, nrows, figsize_y, box_legend, box_compass_colormap = define_plot_layout()
    
    gs0 = gsxx[1].subgridspec(1, 2, wspace=0.12, hspace=None, height_ratios=[1], width_ratios=[6, 1])

    #'Left' subplots
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows, 4, subplot_spec=gs0[0], wspace=gridspec_kw['wspace'], hspace=gridspec_kw['hspace'], height_ratios=gridspec_kw['height_ratios'], width_ratios=[3,1,1,1])
    axes0 = gs00.subplots()

    #'right' subplots - density curves
    # the following syntax does the same as the GridSpecFromSubplotSpec call above:
    gs01 = gs0[-1].subgridspec(nrows, 1, wspace=gridspec_kw['wspace'], hspace=gridspec_kw['hspace'], height_ratios=gridspec_kw['height_ratios'], width_ratios=[1])
    axes1 = gs01.subplots()
    
    return axes0, axes1, gridspec_kw, nrows, box_legend, box_compass_colormap



def calculate_baz_only(st, 
                    t_pick_P, t_pick_S,
                    timing_P, timing_S, timing_noise,
                    rotation = 'ZNE', BAZ = None,
                    kind='cwt', fmin=0.1, fmax=10.,
                    winlen_sec=20., overlap=0.5,
                    tstart=None, tend=None,
                    dop_winlen=10, dop_specwidth=1.1,
                    nf=100, w0=8,
                    alpha_inc = None, alpha_elli = None, alpha_azi = None,
                    f_band_density = (0.3, 1.),
                    differentiate=False, detick_1Hz=False):
    
    #Does exactly the same as plot_polarization_event_noise, but only calculates the back azimuth and prints it, no plotting

    def calculate_baz_kde(kde_dataframe_P):
        #Manually does what the seaborn KDE does in the main code
        #covariance_factor was hand-tuned so that the two KDE curves were exactly the same. Is NOT the same as the bw_adjust factor in seaborn, there is no direct access to that
        kernel = stats.gaussian_kde(kde_dataframe_P['P'], weights = kde_dataframe_P['weights'])
        kernel.covariance_factor = lambda : .17 #old:  lambda : .20
        kernel._compute_covariance()
        xs = np.linspace(-50,360+50,1000) #extend to positive and negative spaces so that the error can be wrapped around
        ys = kernel(xs)
        index = np.argmax(ys)
        max_x = xs[index]
        
        #get the error of the BAZ from the full width of the half maximum
        #find the FWHM
        error = fwhm_error_from_kde(xs, ys, index)
                
        return max_x, error

    st_Copy, components = waveform_processing(st, rotation, BAZ, differentiate, 
                                              timing_P, timing_S, timing_noise,
                                              tstart, tend)


    st_Z = Stream(traces=[st_Copy.select(component=components[0])[0]])
    st_N = Stream(traces=[st_Copy.select(component=components[1])[0]])
    st_E = Stream(traces=[st_Copy.select(component=components[2])[0]])
    
    #P window
    tstart_signal_P = utct(timing_P) + t_pick_P[0]
    tend_signal_P = utct(timing_S) - 5 if (utct(timing_P) + t_pick_P[1]) > (utct(timing_S) - 1) else  utct(timing_P) + t_pick_P[1] #Avoid going into S-window
    #S window
    tstart_signal_S = utct(timing_S) + t_pick_S[0]
    tend_signal_S = utct(timing_S) + t_pick_S[1]

    #Noise window
    tstart_noise = utct(timing_noise[0])
    tend_noise = utct(timing_noise[-1])

    tstart, tend, dt = polarization._check_traces(st_Z, st_N, st_E, tstart, tend)


    winlen = int(winlen_sec / dt)
    nfft = next_pow_2(winlen) * 2

    # variables for statistics
    nts = 0

    # Calculate width of smoothing windows for degree of polarization analysis
    nfsum, ntsum, dsfacf, dsfact = polarization._calc_dop_windows(
        dop_specwidth, dop_winlen, dt, fmax, fmin,
        kind, nf, nfft, overlap, winlen_sec)


    for tr_Z, tr_N, tr_E in zip(st_Z, st_N, st_E):
        if tr_Z.stats.npts < winlen * 4:
            continue

        if detick_1Hz:
            tr_Z_detick = polarization.detick(tr_Z, 10)
            tr_N_detick = polarization.detick(tr_N, 10)
            tr_E_detick = polarization.detick(tr_E, 10)
            f, t, u1, u2, u3 = polarization._compute_spec(tr_Z_detick, tr_N_detick, tr_E_detick, kind, fmin, fmax,
                                         winlen, nfft, overlap, nf=nf, w0=w0)
        else:
            f, t, u1, u2, u3 = polarization._compute_spec(tr_Z, tr_N, tr_E, kind, fmin, fmax,
                                             winlen, nfft, overlap, nf=nf, w0=w0)

        azi1, azi2, elli, inc1, inc2, r1, r2, P = polarization.compute_polarization(
            u1, u2, u3, ntsum=ntsum, nfsum=nfsum, dsfacf=dsfacf, dsfact=dsfact)

        f = f[::dsfacf]
        t = t[::dsfact]
        t += float(tr_Z.stats.starttime)
        nts += len(t)

        bol_density_f_mask, bol_signal_P_mask, bol_signal_S_mask, bol_noise_mask, twodmask_P, twodmask_S, twodmask_noise = boolean_masks_f_t(f, t, 
                                                                                                                                             tstart_signal_P, tend_signal_P,
                                                                                                                                             tstart_signal_S, tend_signal_S, 
                                                                                                                                             tstart_noise, tend_noise, 
                                                                                                                                             f_band_density)

        #Scalogram and alpha/masking of signals
        r1_sum, alpha, alpha2 = polarisation_filtering(r1, inc1, azi1, azi2, elli,
                                                       alpha_inc, alpha_azi, alpha_elli,
                                                       P)
        

        data = np.rad2deg(azi1)
        kde_list = data[twodmask_P[0]]
        kde_weights = alpha[twodmask_P[0]]
        

    kde_dataframe_P = {'P': kde_list,
                        'weights': kde_weights}

        #Get BAZ from max density of P curve, mark in density column
    baz, error = calculate_baz_kde(kde_dataframe_P)
    
    print(f'Back azimuth: {baz:.0f} deg; uncertainty: {error[0]:.0f}-{error[1]:.0f} deg')


