# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:32:15 2019

@author: a001630
"""
import os
import sys
sys.path.append('C:\\shd\\python\\utveckling\\satpy')
os.environ['PPP_CONFIG_DIR'] = 'C:\\shd\python\\utveckling\\utveckling\\satpy\\satpy\\etc'

import numpy as np
import matplotlib.pyplot as plt
import json
import netCDF4


from pandas import read_csv as read_csv
from pandas import DatetimeIndex as DatetimeIndex
from numpy import isnan as isnan
from numpy import array as array
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap
from satpy import Scene, find_files_and_readers
from scipy.stats.mstats import gmean
from datetime import datetime

def get_config(configpath):
    """
    Configurate the setup for EO validation at SHARK stations
    """

    # Read initialization configuration set-up 
    data = read_csv(configpath, sep='\t')

    # Set path to SHARK data
    sharkpath=data['Value'][0]

    # Set path to EO Sentinel 3 OLCI data
    eo_path=data['Value'][1]
    
    # Set area limitation values
    arealimit=[int(data['Value'][2]),
               int(data['Value'][3]),
               int(data['Value'][4]),
               int(data['Value'][5])]
                              
    # Bin step for Chl histogram
    binstep=float(data['Value'][6])
    
    # Set maximum value for x-axis in Chl Histogram
    maxchl_xaxis=float(data['Value'][7])

    # Set EO chl parameter
    chl_param=data['Value'][8]
    
    # Set BAWS grid size
    grid_size=data['Value'][9]
    
    # Set EO reader
    reader=data['Value'][10]
    
    # Set EO sensor
    sensor=data['Value'][11]
    
    return  sharkpath,      \
            eo_path,        \
            arealimit,      \
            binstep,        \
            maxchl_xaxis,   \
            chl_param,      \
            grid_size,      \
            reader,         \
            sensor
            
    #-----End extract_chldata--------------------------------------------------   

class ExtractData:
    """
    """
    def __init__(self):
        pass
    
    def extract_chldata(self, data, head, key):
        """
        """  
        from numpy import array as array

        try :      
            head=head.split(',')
            idr = [i for i, elem in enumerate(head) if key in elem]
            return array([row[idr[0]] for row in data])
        except Exception as e:
            print('Error! Code: {c}, Error occurred in extract_chldata, {m}'.
                    format(c = type(e).__name__, m = str(e)))
            return None
    #-----End extract_chldata--------------------------------------------------               

    def get_unique_stations(self, 
                            lon,
                            lat):
        """    
        Extract the unique numbers of stations with chl observations
        Use round off decimals 0.01 degree, about 1100 m
        """   
        # Decimal degrees 0.01 ~ 1100m
        a=np.array([[round(lon[i],2),
                     round(lat[i],2)] 
                     for i in range(len(lon))]) 
        a, nr =np.unique(a, axis=0, return_counts=True)
        lon_unique=[a[i][0] for i in range(len(a))]
        lat_unique=[a[i][1] for i in range(len(a))]    
        return lon_unique, lat_unique, nr
    #-----End get_unique_stations----------------------------------------------    
#-----End ExtractData----------------------------------------------------------    
            
class BooleanBase:
    """
    Collection of tools for boolean restrictions of pandas dataframe data sets 
    """
    def __init__(self):
        pass
                
    def bool_one(self, 
                 df, 
                 key, 
                 value, 
                 operator):
        """
        Use the key to select data column to be compared with the value
        Operator should be simple bool operator 
        Available now are ('<', '>', '>=', '<=', '==', '!=')
        
        Output: 
            A series of booleans for each element of df
        """               
        try :
            if operator.__eq__('>') :
                new_bool = df[key] > value
            elif operator.__eq__('<') :
                new_bool = df[key] < value        
            elif operator.__eq__('>=') :
                new_bool = df[key] >= value        
            elif operator.__eq__('<=') :
                new_bool = df[key] <= value        
            elif operator.__eq__('==') :
                new_bool = df[key] == value        
            elif operator.__eq__('!=') :
                new_bool = df[key] != value
            else: 
                print 'Operator do not exist.'
                return None           
            return new_bool

        except Exception as e:
            print('Error! Code: {c}, Error occurred in bool_one, {m}'.
                    format(c = type(e).__name__, m = str(e)))
            return None
    #-----End bool_combo-------------------------------------------------------    
    
    def bool_combo(self, 
                   df, 
                   key, 
                   value, 
                   operator, 
                   boolean):
        """
        Use the key to select data column to be compared with the value
        Operator should be simple bool operator 
        Available now are ('<', '>', '>=', '<=', '==', '!=')
        Combine the new boolean with the imported boolean using
        the boolean operator '&'

        Dependencies: 
            bolbase.bool_one (instance of BooleanBase())

        Output: 
            A series of booleans for each element of df
        """         
        try :
            new_bool= self.bool_one(df, key, value, 
                                    operator)
            return boolean & new_bool
            
        except Exception as e:
            print('Error! Code: {c}, Error occurred in bool_combo, {m}'.
                    format(c = type(e).__name__, m = str(e)))
            return None
    #-----End bool_combo-------------------------------------------------------

    def bool_combo2(self, 
                    df, 
                    keys, 
                    values, 
                    operators):
        """
        Combine several boolean tests to obtain one final series of 
        booleans for each element of df.         
        Use the keys to select data column to be compared with the values
        Operators should be simple boolean operators 
        Available now are ('<', '>', '>=', '<=', '==', '!=')
        Combines the booleans using the boolean operator '&'
        
        Example for SHARK data:
        #Restrict investigated area, max observation depth
        bolbase=BooleanBase()
        keys=       ['LATIT_DD', 'LATIT_DD', 'LONGI_DD', 'LONGI_DD', 'DEPH']
        operators=  ['>=', '<=', '>=', '<=','<=']
        depthlimit=[1.0]
        limitvalues= arealimit+depthlimit
        boolean=bolbase.bool_combo2(data, keys, limitvalues, operators)  
        data = data.loc[boolean, :]

        Dependencies: 
            bolbase.bool_combo, bolbase.bool_one (instance of BooleanBase())

        Output: 
            A series of booleans for each element of df        
        """                     
        try :           
            i=-1
            for key in keys:
                i=i+1
                if i == 0:
                    boolean=self.bool_one(df, key, values[i], 
                                          operators[i])
                else:
                    boolean=self.bool_combo(df, key, values[i], 
                                            operators[i], boolean)
            return boolean
        except:
            print 'Error occurred in bool_combo2'
            return None
    #-----End bool_combo2------------------------------------------------------            
#-----End Class BooleanBase----------------------------------------------------   
            
def get_zeros_array(array_shape=(1400, 1400)):
    return np.zeros(array_shape)
    #-----End get_zeros_array--------------------------------------------------

class ArrayOperations(object):
    """

    """
    def __init__(self, array_shape=None):

        if array_shape is not None:
            self.zeros = get_zeros_array(array_shape=array_shape)
        else:
            self.zeros = get_zeros_array()  # using default array_size

        self.nans = self.zeros*np.nan
            
        # valid_pixel_array: nr of all valid pixel
        self.valid_pixel_array = np.zeros(self.zeros.shape)

        # agg_data: Sum of annual blooms per pixel
        self.agg_data = np.zeros(self.zeros.shape)
    #-----End __init__---------------------------------------------------------

    def append_array(self, array, count=False):
        """
    
        :param array:
        :return:
        """
        data = np.where(~np.isnan(array), array, self.nans)
        data = np.where((np.isnan(data) & ~np.isnan(self.agg_data)), self.zeros, data)
        self.agg_data = self.agg_data + data
    
        if count:
            valid_data_pixels = np.where(~np.isnan(array), 1, 0)
            self.valid_pixel_array = self.valid_pixel_array + valid_data_pixels
    #-----End append_array-----------------------------------------------------

    def get_mean_array(self):
        """

        :return:
        """
        mean_array = self.agg_data / self.valid_pixel_array
        return mean_array, self.valid_pixel_array
    #-----End get_mean_array---------------------------------------------------
#-----End Class ArrayOperations------------------------------------------------   

            
def load_shark_chl_surface_data(path, 
                                arealimit):
    """    
    Load SHARK web data and extract surface chlorophyll values
    Restrict surface data to depth 0-1 meter
    Restrict investigated area with area[lat1,lat2,lon1,lon2]
    Remove nan values and chl data with quality flags '?','S','B','<'
    
    Input: 
        Data from SHARK web, dot and tab formatted 'utf-8'
        Chl-a bottle data in physical-chemical columns (with no duplictes)
        
    """   
    print 'Load SHARK chlorophyll data'
    
    bolbase=BooleanBase()   
    depthlimit=[1.0]
    
    data = read_csv(path, sep='\t', 
                    parse_dates=[11], 
                    encoding='utf-8',
                    low_memory=False) 

    #Restrict investigated area and max observation depth
    keys=       ['LATIT_DD', 'LATIT_DD', 'LONGI_DD', 'LONGI_DD', 'DEPH']
    operators=  ['>=', '<=', '>=', '<=','<=']
    limitvalues= arealimit+depthlimit
    boolean=bolbase.bool_combo2(data, keys, limitvalues, operators)  
    data = data.loc[boolean, :]

    #Remove NaN values
    boolean=~isnan(data['CPHL'])
    data = data.loc[boolean, :]

    #Remove bad and suspected data
    boolean = data['Q_CPHL'].isin([u'?',u'S',u'B',u'<'])
    data = data.loc[~boolean, :]
    
    # Use SHARK short names, save data in a list
    dep=array(data['DEPH'])
    date=DatetimeIndex(data['SDATE']).normalize()
    lat=array(data['LATIT_DD'])
    lon=array(data['LONGI_DD'])
    chl=array(data['CPHL'])
    wdis=array(data['WATER_DISTRICT'])
    wtyp=array(data['WATER_TYPE_AREA'])
    wcode=array(data['VISS_EU_ID'])      
    
    header='Year, Month, Day, Lat, Lon, Chl, Wdis, Wtyp, Wcode'
    chldata=[[date.year[i], 
              date.month[i], 
              date.day[i], 
              lat[i], 
              lon[i], 
              chl[i], 
              wdis[i], 
              wtyp[i], 
              wcode[i]] 
              for i in range(len(dep)) ]
              
    if not chldata:
        print 'No SHARK data found'
        
    print 'SHARK Chlorophyll data loaded'
    return  header, \
            chldata

    #-----End load_shark_chl_surface_data---------------------------------------

def get_shark_stations(head, 
                       chldata):
    """        
    Output: 
     A dictionary with unique SHARK stations
     Number of visit at the unique stations
     {'lon':lon, 'lat':lat, 'nr':nr}
    """   

    print 'Extract unique SHARK stations'
    exd=ExtractData()
    
    # Extract lat and lon from chldata
    lat=exd.extract_chldata(chldata,head,'Lat')    
    lon=exd.extract_chldata(chldata,head,'Lon')    

    # Extract the unique numbers of stations with chl observations
    lon, lat, nr = exd.get_unique_stations(lon,lat)
    
    return {'lon':lon, 'lat':lat, 'nr':nr}
    #-----End get_shark_stations-----------------------------------------------


def eo_scene(eo_path, 
             chl_param, 
             grid_size, 
             reader,
             sensor,
             start_time,
             end_time):        
    """
    Load scene
    Input example:
        eo_path= path to data e.g. 'C:\\shd\\python\\utveckling\\testfiler\\' 
        chl_param= 'chl_nn'
        grid_size='1000' or '300' 
        start_time=datetime(2018, 7, 30, 8, 01)
        end_time=datetime(2018, 7, 30, 11, 59)        
        reader='olci_l2'
        sensor='olci'
        
    Output: Scene 
    """        

    print 'Load EO scene'

    # Load Sentinel 3 OLCI data - Time in UTC
    filenames = find_files_and_readers(
         start_time,
         end_time,
         eo_path,
         reader,
         sensor)
     
    baws_area = 'baws'+grid_size+'_sweref99tm'
    scn = Scene(filenames=filenames)
    scn.load([chl_param, 'mask', 'latitude', 'longitude'])
    boundary=100
    scn = scn.resample(baws_area, radius_of_influence=int(grid_size)+boundary)  

    return scn
    #-----eo_scene-------------------------------------------------------------

def generate_filepaths(directory, pattern='', 
                       not_pattern='DUMMY_PATTERN', 
                       pattern_list=[], 
                       endswith='',
                       only_from_dir=True):
    """
    wd = 'X:\\Shd\\Python\\python-course-Anders-Hoglunds\\EO_Kari\\'
    generator_files = generate_filepaths(wd, endswith='.py', only_from_dir=False)

    for fid in generator_files:
        print(fid)
        
        :param directory:
        :param pattern:
        :param not_pattern:
        :param pattern_list:
        :param endswith:
        :param only_from_dir:
        :return:
    """
    for path, subdir, fids in os.walk(directory):
        if only_from_dir:
            if path != directory:
                continue
        for f in fids:
            if pattern in f and not_pattern not in f and f.endswith(endswith):
                if any(pattern_list):
                    for pat in pattern_list:
                        if pat in f:
                            yield os.path.abspath(os.path.join(path, f))
                else:
                    yield os.path.abspath(os.path.join(path, f))
    #-----End generate_filepaths-----------------------------------------------



def get_idx_pos(lat, lon, mlat, mlon):
    """
    Get indexes of closest grid point to (lon,lat) 
    in a master 2D field (mlon, mlat)
    """

    pos = abs(mlat - lat) + abs(mlon - lon)    
    idx=np.where(pos == pos.min())   
    return idx
    #-----End get_idx_pos------------------------------------------------------
        
def get_eo_stn_index(unique_stations,
                  eo_path, 
                  chl_param, 
                  grid_size, 
                  reader,
                  sensor):
    """
    Find EO grid index for a list of unique shark stations
        
    Output: A list of indexes idx
    """       
    print 'Get EO stations index'

    # Get one example with Sentinel 3 OLCI data 
    generator_files = generate_filepaths(eo_path, 
                                         pattern=chl_param, 
                                         only_from_dir=False) 

    for fname in generator_files:                
        yy=int(fname.split('_')[-12][0:4])
        mm=int(fname.split('_')[-12][4:6])
        dd=int(fname.split('_')[-12][6:8])
        scn=eo_scene(eo_path, 
                     chl_param, 
                     grid_size, 
                     reader,
                     sensor,
                     start_time=datetime(yy, mm, dd, 8, 01),
                     end_time  =datetime(yy, mm, dd, 13, 59))    
        break

    area_spec = scn.datasets[chl_param].area
    lons, lats = area_spec.get_lonlats()   

    print 'Find index of unique stations in BAWS grid'
    eo_stnid=[get_idx_pos(unique_stations['lat'][i], 
                          unique_stations['lon'][i], 
                          lats, 
                          lons) 
                          for i in range(len(unique_stations['lon']))] 

    return eo_stnid, lons, lats
    #-----get_eo_stn_index-----------------------------------------------------

def get_eo_stn_chl(unique_stations,
                   eo_path,
                   chl_param, 
                   grid_size,    
                   reader,
                   sensor):
    """
    Get EO chl data at stations from all files in eo_path
        
    Output: chlheader, chldata
    Note: Unit of chl is Log10 transformed
    
    In addition:    
    Save data maps of the mean values and number of data in a NetCdf file 
    Use the satpy method by overwriting the chl_param and mask    
    """

    # Get lats, lons and, shark stations index in Sentinel 3 OLCI data 
    eo_stnid, \
    lons,     \
    lats=     \
    get_eo_stn_index(unique_stations, 
                     eo_path, 
                     chl_param, 
                     grid_size, 
                     reader,
                     sensor)

    # Generate list of files with Sentinel 3 OLCI data 
    generator_files = generate_filepaths(eo_path, 
                                         pattern=chl_param, 
                                         only_from_dir=False)    
    print 'Get EO chla data'
    # Set header of EO-Chl data
    chlheader='Year, Month, Day, Lat, Lon, Chl'  

    # Initiate the list of EO-Chl data
    eo_chl=list()
    date_dict = {}
    init_aop=True
    for fname in generator_files: 
        date = fname.split('_')[-12][0:8]
        if date in date_dict:
            # Iterate only once every day
            continue
        date_dict[date] = True
        yy=int(date[0:4])
        mm=int(date[4:6])
        dd=int(date[6:8])

        scn=eo_scene(eo_path, 
                     chl_param, 
                     grid_size, 
                     reader,
                     sensor,
                     start_time=datetime(yy, mm, dd, 8, 01),
                     end_time  =datetime(yy, mm, dd, 13, 59))    

        scn.datasets[chl_param] = \
        scn.datasets[chl_param].where( \
        scn.datasets['mask'] != True, np.nan)

        print 'Year, Month, Day ', yy, mm, dd
        scn_val=scn.datasets[chl_param].values      

        # Save data for the mean of EO (log10) data)
        if init_aop:
            init_aop=False
            a_op = ArrayOperations(array_shape=scn_val.shape)
            b_op = ArrayOperations(array_shape=scn_val.shape)
            c_op = ArrayOperations(array_shape=scn_val.shape)
            d_op = ArrayOperations(array_shape=scn_val.shape)
            e_op = ArrayOperations(array_shape=scn_val.shape)
            f_op = ArrayOperations(array_shape=scn_val.shape)
            g_op = ArrayOperations(array_shape=scn_val.shape)
        a_op.append_array(scn_val, count=True)
        if (mm>=6) & (mm<=8):
            b_op.append_array(scn_val, count=True)
        if mm==5:
            c_op.append_array(scn_val, count=True)
        if mm==6:
            d_op.append_array(scn_val, count=True)
        if mm==7:
            e_op.append_array(scn_val, count=True)
        if mm==8:
            f_op.append_array(scn_val, count=True)
        if mm==9:
            g_op.append_array(scn_val, count=True)
        
        [eo_chl.append([yy, 
                        mm, 
                        dd, 
                        float(lats[eo_stnid[i]]), 
                        float(lons[eo_stnid[i]]), 
                        float(scn_val[eo_stnid[i]]) ] )
                        for i in range(len(eo_stnid)) 
                        if ~np.isnan(scn_val[eo_stnid[i]]) ]                                         
    a_mean, nr_a_data = a_op.get_mean_array()    
    b_mean, nr_b_data = b_op.get_mean_array()    
    c_mean, nr_c_data = c_op.get_mean_array()    
    d_mean, nr_d_data = d_op.get_mean_array()    
    e_mean, nr_e_data = e_op.get_mean_array()    
    f_mean, nr_f_data = f_op.get_mean_array()    
    g_mean, nr_g_data = g_op.get_mean_array()    
    
    save_nc(scn,chl_param,a_mean,nr_a_data,'may_sep')
    save_nc(scn,chl_param,b_mean,nr_b_data,'jun_aug')
    save_nc(scn,chl_param,c_mean,nr_c_data,'may')
    save_nc(scn,chl_param,d_mean,nr_d_data,'jun')
    save_nc(scn,chl_param,e_mean,nr_e_data,'jul')
    save_nc(scn,chl_param,f_mean,nr_f_data,'aug')
    save_nc(scn,chl_param,g_mean,nr_g_data,'sep')
    
    return chlheader, eo_chl
    #-----get_eo_stn_chl-------------------------------------------------------   

def save_nc(scn,chl_param,data,nr,period):
    """    
    Use the satpy method by overwriting the chl_param and mask    
    Save data to nc file
    """    
    scn.datasets[chl_param].data = data
    scn.datasets['mask'].data = nr
    save_map_data_in_nc_file(scn, chl_param, period)    

    return
    #-----End save_nc----------------------------------------------------------   

def plot_hist_shark_chldata(head,
                            chldata,
                            step, 
                            xax):
    """    
    Plot histogram of chlorophylldata
    Select the bin step and xaxis. Default 1.0 and 100 mgChl/m3, respectively
    """   
    
    print 'Plot shark chlorophyll histogram'
    exd=ExtractData()
    
    # Extract chl from chldata
    chl=exd.extract_chldata(chldata,head,'Chl')
    
    # Plot Histogram on chl
    fig=plt.figure() 
    axlabels=('Chlorophyll concentration ' + 
              '( $\mu$' + 'g Chl '+ 'L$^{-1}$)',
              'Frequency '+ '($\mu$' + 'g Chl '+ 'L$^{-1}$) ') 
    bins=[i*1e-2 for i in range(0,int(np.ceil(max(chl)+10)*1e2),int(step*100)) ]
    N, bins, patches=plt.hist(chl, bins, color='gray', alpha=0.4, edgecolor='k')
    Titel='Shark chlorophyll-a frequency histogram N = ' + str(sum(N)) 
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
    
    filename='.\\figures\\shark_histogram.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  

    return 
    #-----End plot_hist_shark_chldata------------------------------------------

def plot_hist_eo_chldata(head,
                         chldata,
                         step, 
                         xax):
    """    
    Plot histogram of satellite chlorophylldata transfomed to
    Chl concentration values by np.power(10,eo_chl).
    Select the bin step and xaxis. Default 1.0 and 100 mgChl/m3, respectively
    """   
    
    print 'Plot EO chlorophyll histogram'
    exd=ExtractData()
    
    # Extract chl from chldata
    eo_chl=exd.extract_chldata(chldata,head,'Chl')
        
    # Calculate the Chl concentration from log10 transformed eo data
    chl=np.power(10,eo_chl)

    fig=plt.figure() 
    axlabels=('EO chlorophyll concentration '+  
              '( $\mu$' + 'g Chl '+ 'L$^{-1}$)',
              'Frequency '+ '($\mu$' + 'g Chl '+ 'L$^{-1}$)') 
    bins=[i*1e-2 for i in range(0,int(np.ceil(max(chl)+10)*1e2),int(step*100)) ]
    N, bins, patches=plt.hist(chl, bins, color='gray', alpha=0.4, edgecolor='k')
    Titel='EO chlorophyll-a frequency histogram N = ' + str(sum(N)) 
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
    
    
    filename='.\\figures\\eo_histogram.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  
    
    return 
    #-----End plot_hist_eo_chldata---------------------------------------------
    
    
def plot_stationmap_shark_chldata(unique_stations,
                                  head, 
                                  chldata, 
                                  arealimit):
    """    
    Plot map with stations of chlorophylldata
    Indicate the number of station visits by color and size of circle
    
    Output: 
    Plot 
    Imported list with unique stations shown in the plot
    """   

    print 'Plot SHARK chlorophyll station map'
    
    # Extract the unique numbers of stations with chl observations
    lon=unique_stations['lon']
    lat=unique_stations['lat']
    nr =unique_stations['nr']
        
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
    color_range=30
    color_factor=1
    if max(nr) > 50:
        color_factor=2
    elif max(nr) > 100:
        color_factor=3        
    elif max(nr) > 150:
        color_factor=4        
    elif max(nr) > 200:
        color_factor=5        
    sd=np.array(nr/float(color_range))    
    sd=np.where(sd>1, 1, sd)

    # Define own colormap and define a constant clim fitting the cmap
    Titel='Shark stations with surface chlorophyll observations'    
    cbarlabel='Number of sampling visits'
    limcax=[0, color_range*color_factor]
    nwcmp = ListedColormap(['white', 'black', 'grey', 'darkblue', 'brown',  
                            'khaki', 'green',  'yellow', 'orange', 'red']+ 
                           ['cyan']*5+ 
                           ['lime']*5+
                           ['violet']*5+ 
                           ['magenta']*5)            
    
    # Plot stations on the map
    m.scatter(lon, lat, marker="o", c=nr, s=500*sd,  
              cmap=nwcmp, edgecolor='black', zorder=10)  
    plt.title(Titel,Size=25,Weight=1000)
    plt.clim(limcax)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)  
        
    filename='.\\figures\\shark_stations.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  

    return 
    #-----End plot_stationmap_shark_chldata------------------------------------

def plot_stationmap_eo_chldata(head, 
                               chldata, 
                               area):

    """    
    Plot map with stations of EO chlorophyll data
    Indicate the number of observations by color an size of circle
    
    Output: 
    Plot
    A list with unique stations shown in the plot
    """   

    print 'Plot EO chlorophyll station map'
    exd=ExtractData()
    
    # Extract lat and lon from chldata
    lat=exd.extract_chldata(chldata,head,'Lat')    
    lon=exd.extract_chldata(chldata,head,'Lon')    

    # Extract the unique numbers of stations with chl observations
    lon, lat, nr = exd.get_unique_stations(lon,lat)
        
    fig=plt.figure() 
    # Plot restricted area map of the Baltic Sea
    lat1=area[0]
    lat2=area[1]
    lon1=area[2]
    lon2=area[3]
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
    color_range=30
    color_factor=1
    if max(nr) > 50:
        color_factor=2
    elif max(nr) > 100:
        color_factor=3        
    elif max(nr) > 150:
        color_factor=4        
    elif max(nr) > 200:
        color_factor=5        
    sd=np.array(nr/float(color_range))    
    sd=np.where(sd>1, 1, sd)
        

    # Define own colormap and define a constant clim fitting the cmap
    Titel='EO-shark stations with surface chlorophyll observations'    
    cbarlabel='Number of sampling visits'
    limcax=[0, color_range*color_factor]
    nwcmp = ListedColormap(['white', 'black', 'grey', 'darkblue', 'brown',  
                            'khaki', 'green',  'yellow', 'orange', 'red']+ 
                           ['cyan']*5+ 
                           ['lime']*5+
                           ['violet']*5+ 
                           ['magenta']*5)            
    
    # Plot stations on the map
    m.scatter(lon, lat, marker="o", c=nr, s=500*sd, 
              cmap=nwcmp, edgecolor='black', zorder=10)  
    plt.title(Titel,Size=20,Weight=1000)
    plt.clim(limcax)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)    
        
    filename='.\\figures\\eo_stations.png'    
    print 'Plot and save figure: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close(fig)  

    return
    #-----End plot_stationmap_eo_chldata---------------------------------------

def export_json(data_dict={}, out_source='', indent=4):
    """
    :param data_dict:
    :param out_source:
    :param indent:
    :return:
    """
    with open(out_source, "w") as outfile:
        json.dump(data_dict, outfile, indent=indent)

    return
    #-----End export_json------------------------------------------------------

def import_json(filename):
    with open(filename, 'r') as f:
        data = f.read()
        json_data = json.loads(data)
    return json_data
    #-----End import_json------------------------------------------------------

def save_chl_data(shark_head,
                  shark_chl,
                  eo_head,
                  eo_chl):    

    exd=ExtractData()

    print 'Save chlorophyll data in json file'
    # Save chldata in json
    json_data={
    'shark_year'    :map(int,exd.extract_chldata(shark_chl,shark_head,'Year')),    
    'shark_month'   :map(int,exd.extract_chldata(shark_chl,shark_head,'Month')),    
    'shark_day'     :map(int,exd.extract_chldata(shark_chl,shark_head,'Day')),    
    'shark_lat'     :map(float,exd.extract_chldata(shark_chl,shark_head,'Lat')),    
    'shark_lon'     :map(float,exd.extract_chldata(shark_chl,shark_head,'Lon')),  
    'shark_chl'     :map(float,exd.extract_chldata(shark_chl,shark_head,'Chl')),  
    'eo_year'       :map(int,exd.extract_chldata(eo_chl,shark_head,'Year')),    
    'eo_month'      :map(int,exd.extract_chldata(eo_chl,shark_head,'Month')),    
    'eo_day'        :map(int,exd.extract_chldata(eo_chl,shark_head,'Day')),    
    'eo_lat'        :map(float,exd.extract_chldata(eo_chl,shark_head,'Lat')),    
    'eo_lon'        :map(float,exd.extract_chldata(eo_chl,shark_head,'Lon')),  
    'eo_chl'        :map(float,exd.extract_chldata(eo_chl,shark_head,'Chl'))}  

    filename='.\\stations_data\\json_chldata.json'  
    export_json(data_dict=json_data, out_source=filename)

    return
    #-----End save_chl_data----------------------------------------------------

def save_map_data_in_nc_file(scn, chl_param, period):
    """    
    Save the mean values and number of data in a NetCdf file 
    Use the satpy method by overwriting the chl_param and mask
    chl_param=mean chl data (note: it is log10 values)
    mask= number of data used for the calculation of means
    """       
    save_path='.\\map_data\\'+period+'_eo_map_mean_data.nc'    
    print 'Save mean map data in nc-file: '+ save_path

    scn.save_datasets(datasets=['mask', chl_param], 
                      filename=save_path, 
                      writer='cf')
    return                          
    #-----save_map_data_in_nc_file---------------------------------------------   


def plot_chl_mean_maps(chl_param, period):
    """    
    Plot the mean values and number of data in a NetCdf file 
    Used the satpy method by overwriting the chl_param and mask
    chl_param=mean chl data (note: it is log10 values)
    mask= number of data used for the calculation of means
    """           
    print 'Plot chlorophyll mean maps from NetCdf file'
    
    data_path='.\\map_data\\'+period+'_eo_map_mean_data.nc'
    nc=netCDF4.Dataset(data_path)
    lat=np.array(nc['latitude'])
    lon=np.array(nc['longitude'])
    chl=np.array(nc[chl_param])
    nr=np.array(nc['mask'])


    # 
    Titel='EO-mean values of chlorophyll'    
    cbarlabel='( $\mu$' + 'g Chl '+ 'L$^{-1}$)'
    limcax=[0, 5]

    plt.figure(1)
    plt.imshow(np.power(10,chl)) 

    plt.title(Titel,Size=20,Weight=1000)
    plt.clim(limcax)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)    
    plt.show()

    filename='.\\figures\\'+period+'_eo_map_mean_data.png'    
    print 'Plot and save mean values: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close()  

    # 
    Titel='EO-number of values for chl-mean'    
    cbarlabel='(Nr)'

    plt.figure(1)
    plt.imshow(nr) 

    plt.title(Titel,Size=20,Weight=1000)
    cbar=plt.colorbar()   
    cbar.set_label(cbarlabel, Size=20)    
    plt.show()

    filename='.\\figures\\'+period+'_eo_map_nr_data.png'    
    print 'Plot and save number of data: '+ filename

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.savefig(filename, dpi=600)
    plt.close()  

    return
    #-----End save_chl_mean_maps-----------------------------------------------
        