#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 7 2022

@author: Géraldine Zenhäusern
:copyright:
    Géraldine Zenhäusern (geraldine.zenhaeusern@erdw.ethz.ch), 2022
:license:
    GPLv3
"""
from obspy import read
from obspy import UTCDateTime as utct
import argparse
import polarisation_plot as ppl



def arguments():
    parser = argparse.ArgumentParser(
        description='Inputs to polarisation analysis.')
    
    parser.add_argument('file', action = 'store', help='mseed files for polarisation analysis. Direct input into Obspy Stream, so wildcards are allowed (Please put path in quote marks).')

    parser.add_argument('--name', type=str, dest='arg_event_name',
                        help='File name of plot.')

    parser.add_argument('-p', action='store_true', dest='arg_plot',
                        help='Set flag if plot should be made.')


    return parser.parse_args()


def polarisation(st, plot, name='Pol_plot'):
    
    t_pick_P = [-5, 10]
    t_pick_S = [-5, 10]
    
    #Event start/end
    tstart = utct('2000-01-01T00:00:00')
    tend = utct('2000-01-01T01:00:00')
    
    #Phase arrivals/anchors for windows
    timing_P = utct('2022-09-26T00:03:30')
    timing_S = utct('2022-09-26T00:03:50')
    timing_noise = [utct('2022-09-26T00:00:00'), utct('2022-09-26T00:01:00')]
    
    f_band = [0.3, 1.0] #Lower and upper bound of frequency band in Hz where back azimuth is estimated
    
    if plot:
        ppl.plot_polarization_event_noise(st, 
                                        t_pick_P, t_pick_S, #secs before/after pick which defines the polarisation window
                                        timing_P, timing_S, timing_noise,#P and S pick timing as strings
                                        'P', 'S', #Which phases/picks are used for the P and S windows - used for labeling
                                        BAZ=None,
                                        fmin=0.1, fmax=10.,
                                        tstart=tstart, tend=tend, vmin=-190,
                                        vmax=-135, fname=f'{name}',
                                        path = '.',
                                        alpha_inc = None, alpha_elli = 1.0, alpha_azi = None,
                                        f_band_density=f_band,
                                        zoom=True, differentiate = True)
    else:
        ppl.calculate_baz_only(st, 
                            t_pick_P, t_pick_S,
                            timing_P, timing_S, timing_noise,
                            rotation = 'ZNE', BAZ = None,
                            kind='cwt', fmin=0.1, fmax=10.,
                            winlen_sec=20., overlap=0.5,
                            tstart=tstart, tend=tend,
                            alpha_inc = None, alpha_elli = 1.0, alpha_azi = None,
                            f_band_density = f_band,
                            differentiate = True)



if __name__=='__main__':
    
    args = arguments()
    
    #Load ZNE rotated mseed files
    #IMPORTANT: The polarisation code was made for velocity data. Either input Displacement data and set differentiate = True; 
    #or use Velocity data directly.
    #The code works with any data, but the labels will be wrong otherwise (e.g. waveform y-axis label). 
    st = read(args.file)

    
    if args.arg_plot:
        polarisation(st, args.arg_plot, args.arg_event_name)
    else:
        polarisation(st, args.arg_plot)