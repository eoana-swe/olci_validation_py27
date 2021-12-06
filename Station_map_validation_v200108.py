# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 08:38:40 2019

@author: a001630
"""

import numpy as np
import matplotlib.pyplot as plt
import EO_SHARK_Validation_misc_v200107 as misc
from scipy.stats.mstats import gmean
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap
from datetime import datetime
from scipy import stats

import os
import sys
sys.path.append('C:\\shd\\python\\utveckling\\satpy')
os.environ['PPP_CONFIG_DIR'] = 'C:\\shd\python\\utveckling\\utveckling\\satpy\\satpy\\etc' 
configpath='C:\\shd\python\\utveckling\\config_files\\eo_validation_config_tab.txt'

def plot_hist(chl,
              step, 
              xax,
              titel_in):
    """    
    Plot histogram of chlorophylldata
    Select the bin step and xaxis. Default 1.0 and 100 mgChl/m3, respectively
    """   
        
    # Plot Histogram on chl
    fig=plt.figure() 
    axlabels=('Chlorophyll concentration ' + 
              '( $\mu$' + 'g Chl '+ 'L$^{-1}$)',
              'Frequency ') 
    bins=[i*1e-2 for i in range(0,int(np.ceil(max(chl)+10)*1e2),int(step*100)) ]
    N, bins, patches=plt.hist(chl, bins, color='gray', alpha=0.4, edgecolor='k')
    Titel=titel_in + str(sum(N)) 
    plt.title(Titel, Size=20,Weight=1000)
    plt.xlabel(axlabels[0], Size=15,Weight=1000)
    plt.xticks(Size=15,Weight=1000)
    plt.ylabel(axlabels[1], Size=15,Weight=1000)
    plt.yticks(Size=15,Weight=1000)

    # Add lines of statistical values
    mnchl=np.mean(chl)
    mdchl=np.median(chl)
    stdchl=np.std(chl)
    gmchl=gmean(chl)
    maxchl=max(chl)
    labels=('Mean chl, '+str(round(mnchl,1))+'; Std chl, '+str(round(stdchl,1)), 
            'Median  chl, '+str(round(mdchl,1)),
            'Geometric mean  chl, '+str(round(gmchl,1)),
            'Max  chl, '+str(round(maxchl,1)))     
    plt.axvline(mnchl,  color='k', linestyle='dashed', linewidth=2, label=labels[0])
    plt.axvline(mdchl,  color='b', linestyle='dashed', linewidth=2,  label=labels[1]) 
    plt.axvline(gmchl,  color='g', linestyle='dashed', linewidth=2, label=labels[2])
    plt.axvline(maxchl, color='r', linestyle='dashed', linewidth=2, label=labels[3])
    plt.legend(fontsize='large')    

    # Axis ranges
    xax=np.min([xax,np.ceil(max(chl)/10)*10])
    yax=np.ceil(max(N)/10)*10  
    plt.xlim(0, xax)
    plt.ylim(0, yax)  
    
    filename='.\\figures\\' + titel_in.split()[0] + '_histogram_common_stations.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  

    return 
    #-----End plot_hist--------------------------------------------------------

def plot_stationmap(lon,
                    lat,
                    nr,
                    chldata, 
                    arealimit,
                    titel_in):
    """    
    Plot map with stations of chlorophylldata
    Indicate the concentration of station visits by color and size of circle
    
    Output: 
    Plot 
    Imported list with unique stations shown in the plot
    """   
        
    fig=plt.figure() 
    # Plot restricted area map of the Baltic Sea
    lat1=arealimit[0]
    lat2=arealimit[1]
    lon1=arealimit[2]
    lon2=arealimit[3]
    m=Basemap(llcrnrlon=lon1, 
              llcrnrlat=lat1,
              urcrnrlon=lon2, 
              urcrnrlat=lat2, 
              resolution='h')

    m.fillcontinents(color='grey', alpha=0.3)
    m.fillcontinents(lake_color='white')
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawmeridians([i for i in range(lon1,lon2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)
    m.drawparallels([i for i in range(lat1,lat2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)


    # Create a number dependent size of the circles
    color_range=3
    color_factor=1
    if max(chldata) > 5:
        color_factor=2
    elif max(chldata) > 10:
        color_factor=3        
    elif max(chldata) > 15:
        color_factor=4        
    elif max(chldata) > 20:
        color_factor=5        
    sd=np.array(np.array(chldata)/float(color_range))    
    sd=np.where(sd>1, 1, sd)
    sd=np.where(sd<0.1, 0.1, sd)

    # Define own colormap and define a constant clim fitting the cmap
    Titel=titel_in    
    cbarlabel=('Geometric mean Chl concentration ' + 
              '( $\mu$' + 'g Chl '+ 'L$^{-1}$)')    
    limcax=[0, color_range*color_factor]
    nwcmp = ListedColormap(['white', 'black', 'grey', 'darkblue', 'brown',  
                            'khaki', 'green',  'yellow', 'orange', 'red']+ 
                           ['cyan']*5+ 
                           ['lime']*5+
                           ['violet']*5+ 
                           ['magenta']*5)            
    
    # Plot stations on the map
    m.scatter(lon, lat, marker="o", c=chldata, s=500*sd,  
              cmap=nwcmp, edgecolor='black', zorder=10)  
    plt.title(Titel,Size=25,Weight=1000)
    plt.clim(limcax)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)  
        
    filename='.\\figures\\' + titel_in.split()[0] + '_chl_map_common_stations.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  

    return 
    #-----End plot_stationmap--------------------------------------------------


def plot_bias_stationmap(lon,
                         lat,
                         nr,
                         chldata, 
                         arealimit,
                         titel_in):
    """    
    Plot map with stations of chlorophylldata
    Indicate the concentration of station visits by color and size of circle
    
    Output: 
    Plot 
    Imported list with unique stations shown in the plot
    """   
        
    fig=plt.figure() 
    # Plot restricted area map of the Baltic Sea
    lat1=arealimit[0]
    lat2=arealimit[1]
    lon1=arealimit[2]
    lon2=arealimit[3]
    m=Basemap(llcrnrlon=lon1, 
              llcrnrlat=lat1,
              urcrnrlon=lon2, 
              urcrnrlat=lat2, 
              resolution='h')

    m.fillcontinents(color='grey', alpha=0.3)
    m.fillcontinents(lake_color='white')
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawmeridians([i for i in range(lon1,lon2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)
    m.drawparallels([i for i in range(lat1,lat2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)


    # Create a number dependent size of the circles
    color_range=3
    color_factor=1

    sd=np.array(np.abs(np.array(chldata))/float(color_range))    
    sd=np.where(sd>1, 1, sd)
    sd=np.where(sd<0.1, 0.1, sd)

    # Define own colormap and define a constant clim fitting the cmap
    Titel=titel_in    
    cbarlabel=('Ratio geometric mean Chl (EO-Shark)/Shark')    
    limcax=[-color_range*color_factor, color_range*color_factor]       
                           
    # Plot stations on the map
    m.scatter(lon, lat, marker="o", c=chldata, s=500*sd,  
              cmap='seismic', edgecolor='black', zorder=10)  
    plt.title(Titel,Size=25,Weight=1000)
    plt.clim(limcax)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)  
        
    filename='.\\figures\\' + titel_in.split()[0] + '_chl_map_common_stations.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  

    return 
    #-----End plot_bias_stationmap---------------------------------------------


def plot_bias_stationmap_dates(lon,
                               lat,
                               nr,
                               chldata, 
                               arealimit,
                               titel_in):
    """    
    Plot map with stations of chlorophylldata
    Indicate the concentration of station visits by color and size of circle
    
    Output: 
    Plot 
    Imported list with unique stations shown in the plot
    """   
        
    fig=plt.figure() 
    # Plot restricted area map of the Baltic Sea
    lat1=arealimit[0]
    lat2=arealimit[1]
    lon1=arealimit[2]
    lon2=arealimit[3]
    m=Basemap(llcrnrlon=lon1, 
              llcrnrlat=lat1,
              urcrnrlon=lon2, 
              urcrnrlat=lat2, 
              resolution='h')

    m.fillcontinents(color='grey', alpha=0.3)
    m.fillcontinents(lake_color='white')
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawmeridians([i for i in range(lon1,lon2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)
    m.drawparallels([i for i in range(lat1,lat2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)


    # Create a number dependent size of the circles
    color_range=3
    color_factor=1

    sd=np.array(np.abs(np.array(chldata))/float(color_range))    
    sd=np.where(sd>1, 1, sd)
    sd=np.where(sd<0.1, 0.1, sd)

    # Define own colormap and define a constant clim fitting the cmap
    Titel=titel_in    
    cbarlabel=('Ratio geometric mean Chl (EO-Shark)/Shark')    
    limcax=[-color_range*color_factor, color_range*color_factor]       
                           
    # Plot stations on the map
    m.scatter(lon, lat, marker="o", c=chldata, s=500*sd,  
              cmap='seismic', edgecolor='black', zorder=10)  
    plt.title(Titel,Size=25,Weight=1000)
    plt.clim(limcax)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)  
        
    filename='.\\figures\\' + titel_in.split()[0] + '_chl_map_common_stations_and_dates.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  

    return 
    #-----End plot_bias_stationmap_dates---------------------------------------

def plot_corr_stationmap_dates(eo_chl_match,
                               shark_chl_match, 
                               titel_in): 
 
    x=shark_chl_match    
    y=eo_chl_match    
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    mn=np.min(x)
    mx=np.max(x)
    lx=len(x)
    x1=np.linspace(mn,mx,500)
    y1=gradient*x1+intercept

    Titel=titel_in + str(lx)

    axlabels=('Shark Chl '+'($\mu$' + 'g Chl '+ 'L$^{-1}$)',
          'EO Chl '+'($\mu$' + 'g Chl '+ 'L$^{-1}$)') 
          
    fig=plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(shark_chl_match, eo_chl_match,  marker="o", color="b")  
    plt.title(Titel,Size=25,Weight=1000)
    plt.xlabel(axlabels[0], Size=15,Weight=1000)
    plt.xticks(Size=15,Weight=1000)
    plt.ylabel(axlabels[1], Size=15,Weight=1000)
    plt.yticks(Size=15,Weight=1000)
    x_ax=np.ceil(max(shark_chl_match)/10)*10
    plt.xlim(0, x_ax)
    plt.ylim(0, x_ax)  
    ax.plot(x1,y1,'-r', label='Linear fit')   
    ax.text(0.85, 0.05,  
            'Slope=%1.2f\nIntercept=%1.2f\nR2=%1.2f\nP=%1.5f\nErr=%1.2f\n'%
            (np.round(gradient,2), 
             np.round(intercept,2), 
             np.round(r_value**2,2), 
             np.round(p_value,5), 
             np.round(std_err,2)), 
             color='r', 
             transform=ax.transAxes,
             Size=15)

    a=shark_chl_match
    a=np.append(a,0)
    ax.plot(a,a,'-k', label='One to one line')
    plt.legend(fontsize='large')    

    filename='.\\figures\\' + titel_in.split()[0] + '_chl_correlation_common_stations_and_dates.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  

    return    
    #-----End plot_corr_stationmap_dates---------------------------------------


def pick_plot_bias_stationmap(lon,
                              lat,
                              eo_nr,
                              shark_nr,
                              chldata,
                              eo_chl,
                              shark_chl,
                              eo_stn_data,
                              eo_stn_year,
                              eo_stn_month,
                              eo_stn_day,                                  
                              shark_stn_data,
                              shark_stn_year,
                              shark_stn_month,
                              shark_stn_day,   
                              arealimit,
                              titel_in):
    """    
    Plot map with stations of chlorophylldata
    Indicate the concentration of station visits by color and size of circle
    
    Output: 
    Plot 
    Imported list with unique stations shown in the plot
    """   
        
    fig=plt.figure() 
    # Plot restricted area map of the Baltic Sea
    lat1=arealimit[0]
    lat2=arealimit[1]
    lon1=arealimit[2]
    lon2=arealimit[3]
    m=Basemap(llcrnrlon=lon1, 
              llcrnrlat=lat1,
              urcrnrlon=lon2, 
              urcrnrlat=lat2, 
              resolution='h')

    m.fillcontinents(color='grey', alpha=0.3)
    m.fillcontinents(lake_color='white')
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawmeridians([i for i in range(lon1,lon2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)
    m.drawparallels([i for i in range(lat1,lat2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)


    # Create a number dependent size of the circles
    color_range=2
    color_factor=1

    sd=np.array(np.abs(np.array(chldata))/float(color_range))    
    sd=np.where(sd>1, 1, sd)
    sd=np.where(sd<0.1, 0.1, sd)

    # Define own colormap and define a constant clim fitting the cmap
    Titel=titel_in    
    cbarlabel=('Ratio geometric mean Chl (EO-Shark)/Shark')    
    limcax=[-color_range*color_factor, color_range*color_factor]       
                           
    # Plot stations on the map
    # 5 points picker tolerance                           
    point= m.scatter(lon, lat, marker="o", c=chldata, s=400*sd,  
              cmap='seismic', edgecolor='black', zorder=10, picker=3)  
    plt.title(Titel,Size=25,Weight=1000)
    plt.clim(limcax)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)  
        
    print 'Plot and pick in figure'

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    axlabels=('Date','Chlorophyll concentration ' + 
          '( $\mu$' + 'g Chl '+ 'L$^{-1}$)') 
    def onpick(event):
    
        if event.artist!=point: return True
    
        N = len(event.ind)
        if not N: return True
        
        for subplotnum, dataind in enumerate(event.ind):   

            eo_stn_date=list()
            shark_stn_date=list()
            for j in range(len(eo_stn_year[dataind])):
                eo_stn_date.append(datetime(eo_stn_year[dataind][j],
                                            eo_stn_month[dataind][j],
                                            eo_stn_day[dataind][j]))                                

            for j in range(len(shark_stn_year[dataind])):
                shark_stn_date.append(datetime(shark_stn_year[dataind][j],
                                               shark_stn_month[dataind][j],
                                               shark_stn_day[dataind][j]))                                
                                     
            figi = plt.figure()            
            ax = figi.add_subplot(2,1,1)
            ax.text(0.05, 0.9,  
                    'Lon=%1.2f\nLat=%1.2f\nRatio=%1.2f\nEO_Chl=%1.1f\nEO_nr=%1.0f\nShark_Chl=%1.1f\nShark_nr=%1.0f'%
                    (lon[dataind], 
                     lat[dataind], 
                     chldata[dataind], 
                     eo_chl[dataind], 
                     eo_nr[dataind],
                     shark_chl[dataind],
                     shark_nr[dataind]),
                    transform=ax.transAxes, va='top')
            plt.axis('off')

            ax2 = figi.add_subplot(2,1,2)
            ax2.scatter(eo_stn_date,eo_stn_data[dataind], marker="o", color="b", label='EO')
            ax2.scatter(shark_stn_date,shark_stn_data[dataind], marker="^", color='r', label='Shark')
            plt.xlabel(axlabels[0], Size=15,Weight=1000)
            plt.xticks(Size=15,Weight=1000)
            plt.ylabel(axlabels[1], Size=15,Weight=1000)
            plt.yticks(Size=15,Weight=1000)
            plt.legend(fontsize='large')    

        figi.show()

        return True
    
    fig.canvas.mpl_connect('pick_event', onpick)
    
    plt.show()
    #plt.close(fig)  

    return
    #-----End pick_plot_bias_stationmap----------------------------------------

def pick_plot_bias_stationmap_dates(lon,
                                    lat,
                                    eo_nr,
                                    shark_nr,
                                    chldata,
                                    eo_chl,
                                    shark_chl,
                                    eo_chl_match,
                                    shark_chl_match,
                                    date_match,                                    
                                    arealimit,
                                    titel_in):
    """    
    Plot map with stations of chlorophylldata, matching dates
    Indicate the concentration of station visits by color and size of circle
    
    Output: 
    Plot 
    Imported list with unique stations shown in the plot
    """   
        
    fig=plt.figure() 
    # Plot restricted area map of the Baltic Sea
    lat1=arealimit[0]
    lat2=arealimit[1]
    lon1=arealimit[2]
    lon2=arealimit[3]
    m=Basemap(llcrnrlon=lon1, 
              llcrnrlat=lat1,
              urcrnrlon=lon2, 
              urcrnrlat=lat2, 
              resolution='h')

    m.fillcontinents(color='grey', alpha=0.3)
    m.fillcontinents(lake_color='white')
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawmeridians([i for i in range(lon1,lon2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)
    m.drawparallels([i for i in range(lat1,lat2,1)], 
                     labels=[1,0,0,1], Size=12, Weight=1000)


    # Create a number dependent size of the circles
    color_range=2
    color_factor=1

    sd=np.array(np.abs(np.array(chldata))/float(color_range))    
    sd=np.where(sd>1, 1, sd)
    sd=np.where(sd<0.1, 0.1, sd)

    # Define own colormap and define a constant clim fitting the cmap
    Titel=titel_in    
    cbarlabel=('Ratio geometric mean Chl (EO-Shark)/Shark')    
    limcax=[-color_range*color_factor, color_range*color_factor]       
                           
    # Plot stations on the map
    # 5 points picker tolerance                           
    point= m.scatter(lon, lat, marker="o", c=chldata, s=400*sd,  
              cmap='seismic', edgecolor='black', zorder=10, picker=3)  
    plt.title(Titel,Size=25,Weight=1000)
    plt.clim(limcax)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)  
        
    print 'Plot and pick in figure'

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    axlabels=('Date','Chlorophyll concentration ' + 
          '( $\mu$' + 'g Chl '+ 'L$^{-1}$)')     
    def onpick(event):
    
        if event.artist!=point: return True
    
        N = len(event.ind)
        if not N: return True
        
        for subplotnum, dataind in enumerate(event.ind):                      
                                     
            figi = plt.figure()            
            ax = figi.add_subplot(2,1,1)
            ax.text(0.05, 0.9,  
                    'Lon=%1.2f\nLat=%1.2f\nRatio=%1.2f\nEO_Chl=%1.1f\nEO_nr=%1.0f\nShark_Chl=%1.1f\nShark_nr=%1.0f'%
                    (lon[dataind], 
                     lat[dataind], 
                     chldata[dataind], 
                     eo_chl[dataind], 
                     eo_nr[dataind],
                     shark_chl[dataind],
                     shark_nr[dataind]),
                     transform=ax.transAxes, va='top')
            plt.axis('off')

            ax2 = figi.add_subplot(2,1,2)
            ax2.scatter(date_match[dataind],eo_chl_match[dataind], marker="o", color="b", label='EO')
            ax2.scatter(date_match[dataind],shark_chl_match[dataind], marker="^", color='r', label='Shark')
            plt.xlabel(axlabels[0], Size=15,Weight=1000)
            plt.xticks(Size=15,Weight=1000)
            plt.ylabel(axlabels[1], Size=15,Weight=1000)
            plt.yticks(Size=15,Weight=1000)
            plt.legend(fontsize='large')    
            
        figi.show()
        
        return True
    
    fig.canvas.mpl_connect('pick_event', onpick)
    
    plt.show()
    #plt.close(fig)  

    return 
    #-----End pick_plot_bias_stationmap----------------------------------------
    
#--------- Script start -------------------------------------------------------
# Initiation of values
sharkpath,      \
eo_path,        \
arealimit,      \
step,           \
xax,            \
chl_param,      \
grid_size,      \
reader,         \
sensor =        \
misc.get_config(configpath)

pick=True
plot_all=True
plot_maps=True

# Import data saved by EO_SHARK_Validation
filename='.\\stations_data\\json_chldata.json'  
json_data=misc.import_json(filename)

# Extract data to numpy arrays
eo_year=np.array(json_data['eo_year'])
eo_month=np.array(json_data['eo_month'])
eo_day=np.array(json_data['eo_day'])
eo_lon=np.array(json_data['eo_lon'])
eo_lat=np.array(json_data['eo_lat'])
eo_chl=np.power(10,np.array(json_data['eo_chl']))
shark_year=np.array(json_data['shark_year'])
shark_month=np.array(json_data['shark_month'])
shark_day=np.array(json_data['shark_day'])
shark_lon=np.array(json_data['shark_lon'])
shark_lat=np.array(json_data['shark_lat'])
shark_chl=np.array(json_data['shark_chl'])

# Find unique stations
exd=misc.ExtractData()
lon_eo,   \
lat_eo,   \
nr_eo = \
exd.get_unique_stations(eo_lon,eo_lat)

# Calculate geometric means at common stations
gmean_lon=list()
gmean_lat=list()
shark_gmean=list()
eo_gmean=list()
eo_shark_ratio=list()
shark_nr=list()
eo_nr=list()
eo_stn_data=list()
eo_stn_year=list()
eo_stn_month=list()
eo_stn_day=list()

shark_stn_data=list()
shark_stn_year=list()
shark_stn_month=list()
shark_stn_day=list()

for i in range(len(lon_eo)):
    # Get index for all data at the specific station
    # and calculate geometric mean values
    # Do not use EO-Chl value < 0.2, similar to uncertain values in SHARK
    id_eo=np.where( (eo_lon.round(2)==lon_eo[i]) &
                    (eo_lat.round(2)==lat_eo[i]) &
                    (eo_chl[i]>=0.2))
    id_shark=np.where( (shark_lon.round(2)==lon_eo[i].round(2)) &
                       (shark_lat.round(2)==lat_eo[i].round(2)) )

    if len(id_shark[0])==0:
        continue
    else:
        gmean_lon.append(gmean(eo_lon[id_eo].round(2)))
        gmean_lat.append(gmean(eo_lat[id_eo].round(2)))
        eo_gmean.append(gmean(eo_chl[id_eo]))
        eo_nr.append(len(id_eo[0]))
        eo_stn_data.append(eo_chl[id_eo[0]])
        eo_stn_year.append(eo_year[id_eo[0]])
        eo_stn_month.append(eo_month[id_eo[0]])
        eo_stn_day.append(eo_day[id_eo[0]])            
    
        shark_gmean.append(gmean(shark_chl[id_shark]))    
        shark_nr.append(len(id_shark[0]))
        shark_stn_data.append(shark_chl[id_shark[0]])
        shark_stn_year.append(shark_year[id_shark[0]])
        shark_stn_month.append(shark_month[id_shark[0]])
        shark_stn_day.append(shark_day[id_shark[0]])    
        
        # Calculate the normalized bias between eo and shark data
        eo_shark_ratio.append((gmean(eo_chl[id_eo])-gmean(shark_chl[id_shark]))/
                               gmean(shark_chl[id_shark]))

# Calculate geometric means at common stations
# At matching dates
eo_gmean_match=list()
shark_gmean_match=list()
dates_nr_match=list()
lon_match=list()
lat_match=list()
eo_chl_match=list()
shark_chl_match=list()
date_match=list()
eo_shark_ratio_match=list()

for i in range(len(lon_eo)):
    # Get index for all data at the specific station
    # at the same date
    # and calculate geometric mean values
    # Do not use EO-Chl value < 0.2, similar to uncertain values in SHARK
    id_eo=np.where( (eo_lon.round(2)==lon_eo[i]) &
                    (eo_lat.round(2)==lat_eo[i]) &
                    (eo_chl[i]>=0.2))
    
    for j in range(len(id_eo[0])):
        id_match=np.where( (shark_lon.round(2)==lon_eo[i].round(2)) &
                           (shark_lat.round(2)==lat_eo[i].round(2)) &        
                           (shark_year==eo_year[id_eo[0][j]])       &
                           (shark_month==eo_month[id_eo[0][j]])    &
                           (shark_day==eo_day[id_eo[0][j]])        )
        
        if len(id_match[0])==0:
            continue
        elif len(id_match[0])>0:
            if len(id_match[0])>1:
                sharkchl=gmean(shark_chl[np.array(id_match[0])])
            else:
                sharkchl=np.float64(shark_chl[np.array(id_match[0])])
                
            lon_match.append(lon_eo[i].round(2))
            lat_match.append(lat_eo[i].round(2))
            eo_chl_match.append(eo_chl[id_eo[0][j]])                      
            date_match.append(datetime(eo_year[id_eo[0][j]],eo_month[id_eo[0][j]],eo_day[id_eo[0][j]]))
            shark_chl_match.append(sharkchl)
            
# 
# Find unique stations with match up data
lon_eo_match,  \
lat_eo_match,  \
nr_eo_match = \
exd.get_unique_stations(lon_match,lat_match)
eo_chl_match=np.array(eo_chl_match)
shark_chl_match=np.array(shark_chl_match)
date_match=np.array(date_match)
eo_stn_chl_match=list()
shark_stn_chl_match=list()
date_stn_match=list()

for i in range(len(lon_eo_match)):
    id_dates=np.where( (lon_match==lon_eo_match[i]) &
                       (lat_match==lat_eo_match[i]) )
    eo_gmean_match.append(gmean(eo_chl_match[id_dates[0]]))
    shark_gmean_match.append(gmean(shark_chl_match[id_dates[0]]))
    dates_nr_match.append(len(id_dates[0]))                            
    eo_stn_chl_match.append(eo_chl_match[id_dates[0]])
    shark_stn_chl_match.append(shark_chl_match[id_dates[0]])
    date_stn_match.append(date_match[id_dates[0]])

    # Calculate the normalized bias between eo and shark data
    eo_shark_ratio_match.append((eo_gmean_match[i]-shark_gmean_match[i])/shark_gmean_match[i])

if plot_all:
    
    # Plot histograms for the common station data
    plot_hist(shark_chl,
              step, 
              xax,
              titel_in='Shark-Match-Stn chlorophyll-a frequency histogram\n Nr data = ')    
              
        
    plot_hist(eo_chl,
              step, 
              xax,
              titel_in='EO-Match-Stn chlorophyll-a frequency histogram\n Nr data = ')    
    
    plot_hist(shark_chl_match,
              step, 
              xax,
              titel_in='Shark-Match-Date chlorophyll-a frequency histogram\n Nr data = ')    
              
        
    plot_hist(eo_chl_match,
              step, 
              xax,
              titel_in='EO-Match-Date chlorophyll-a frequency histogram\n Nr data = ')    
    
    
    plot_stationmap(gmean_lon,
                    gmean_lat,
                    nr_eo,
                    eo_gmean, 
                    arealimit,
                    titel_in='EO surface chlorophyll observations')
    
    plot_stationmap(gmean_lon,
                    gmean_lat,
                    nr_eo,
                    shark_gmean, 
                    arealimit,
                    titel_in='Shark surface chlorophyll observations')          
                    
    plot_bias_stationmap(gmean_lon,
                         gmean_lat,
                         nr_eo,
                         eo_shark_ratio, 
                         arealimit,
                         titel_in='Bias EO to Shark surface chl observations')                          
    
                            
    plot_bias_stationmap_dates(lon_eo_match,
                               lat_eo_match,
                               dates_nr_match,
                               eo_shark_ratio_match, 
                               arealimit,
                               titel_in='Bias EO to Shark surface chl observations\n Same dates') 
    
    plot_corr_stationmap_dates(eo_chl_match,
                               shark_chl_match, 
                               titel_in='Correlation EO to Shark surface chl observations\n Nr data = ') 

if plot_maps:
    # Read the netcdf data and 
    # plot simple images of the mean field and the number of data
    misc.plot_chl_mean_maps(chl_param, 'may_sep')
    misc.plot_chl_mean_maps(chl_param, 'jun_aug')
    misc.plot_chl_mean_maps(chl_param, 'may')
    misc.plot_chl_mean_maps(chl_param, 'jun')
    misc.plot_chl_mean_maps(chl_param, 'jul')
    misc.plot_chl_mean_maps(chl_param, 'aug')
    misc.plot_chl_mean_maps(chl_param, 'sep')
    

if pick:
    pick_plot_bias_stationmap(gmean_lon,
                              gmean_lat,
                              eo_nr,
                              shark_nr,
                              eo_shark_ratio,
                              eo_gmean,
                              shark_gmean,
                              eo_stn_data,
                              eo_stn_year,
                              eo_stn_month,
                              eo_stn_day,                                  
                              shark_stn_data,
                              shark_stn_year,
                              shark_stn_month,
                              shark_stn_day,                                  
                              arealimit,
                              titel_in='Bias EO to Shark surface chl observations')    


    pick_plot_bias_stationmap_dates(lon_eo_match,
                                    lat_eo_match,
                                    dates_nr_match,
                                    dates_nr_match,
                                    eo_shark_ratio_match,
                                    eo_gmean_match,
                                    shark_gmean_match,     
                                    eo_stn_chl_match,
                                    shark_stn_chl_match,
                                    date_stn_match,
                                    arealimit,
                                    titel_in='Bias EO to Shark surface chl observations\n Same dates')
