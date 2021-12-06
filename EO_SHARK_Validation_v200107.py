# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:15:12 2019

@author: a001630
"""
import EO_SHARK_Validation_misc_v200107 as misc
import numpy as np
import matplotlib.pyplot as plt

def run_eo_validation():
    """
    """
    configpath='C:\\shd\python\\utveckling\\config_files\\eo_validation_config_tab.txt'
    
    # Initiate
    sharkpath,      \
    eo_path,        \
    arealimit,      \
    binstep,        \
    maxchl_xaxis,   \
    chl_param,      \
    grid_size,      \
    reader,         \
    sensor =        \
    misc.get_config(configpath)
    
    print '*****************'
    print '*****************'
    print 'Config'
    print 'sharkpath ', sharkpath
    print 'eo_path ' , eo_path
    print 'arealimit ',arealimit
    print 'binstep ', binstep
    print 'maxchl_xaxis ', maxchl_xaxis
    print 'chl_param ', chl_param
    print 'grid_size ', grid_size
    print 'reader ', reader
    print 'sensor ', sensor
    print '*****************'
    print '*****************'
    
    
    # Load shark chl data
    shark_head, \
    shark_chl=  \
    misc.load_shark_chl_surface_data(sharkpath, 
                                     arealimit)    
    
    # Get unique shark stations
    unique_shark_stations = \
    misc.get_shark_stations(shark_head, 
                            shark_chl)
    
    # Get eo chl data at shark stations and 
    # mean of all valid pixels values in array
    # Save the mean values and number of data in a NetCdf file 
    eo_head, \
    eo_chl=  \
    misc.get_eo_stn_chl(unique_shark_stations,
                        eo_path,
                        chl_param, 
                        grid_size,    
                        reader,
                        sensor)                        

    # Plot and save histogram of shark chl data
    misc.plot_hist_shark_chldata(shark_head,
                                 shark_chl,
                                 binstep, 
                                 maxchl_xaxis)

    # Plot and save histogram of eo chl data
    misc.plot_hist_eo_chldata(eo_head,
                              eo_chl,
                              binstep, 
                              maxchl_xaxis)

    
    # Plot number of station visits in shark data
    misc.plot_stationmap_shark_chldata(unique_shark_stations,
                                       shark_head, 
                                       shark_chl, 
                                       arealimit)

    # Plot number of station visits in eo data
    misc.plot_stationmap_eo_chldata(eo_head,
                                    eo_chl, 
                                    arealimit)


    # Save chl data
    misc.save_chl_data(shark_head,
                       shark_chl,
                       eo_head,
                       eo_chl)

    # Read the netcdf data and 
    # plot simple images of the mean field and the number of data
    misc.plot_chl_mean_maps(chl_param, 'may_sep')
    misc.plot_chl_mean_maps(chl_param, 'jun_aug')
    misc.plot_chl_mean_maps(chl_param, 'may')
    misc.plot_chl_mean_maps(chl_param, 'jun')
    misc.plot_chl_mean_maps(chl_param, 'jul')
    misc.plot_chl_mean_maps(chl_param, 'aug')
    misc.plot_chl_mean_maps(chl_param, 'sep')

    return 

#--------- Script start -------------------------------------------------------
if __name__ == '__main__':
    """
    Run script and plot figures with SHARK and EO chlorophyll data
    Histogram and map of stations
    """
    
    print '---***---'
    print 'Running'
    
    run_eo_validation()
    
    print 'Plot EO chlorophyll station map'

    
    print '---***---'
    print 'Done'
