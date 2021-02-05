import numpy as np
import xarray as xr 
import logging


def read_obs(input_file, oi_grid, oi_param, simu_start_date, coarsening):
    """ 
    Read available observations (path given in input_file).

    Parameters
    ----------
    input_file : array of strings 
        Paths of all available observations
    oi_grid : 
        Grid used in the optimal interpolation.
    oi_param : 
        Dataset used to store and pass OI paramaters.
    simu_start_date : string
        Reference run initial date (e.g., '2012-10-01T00:00:00')
    coarsening : Dictionary
        Time to discard in the first and last few observation time steps (e.g., {'time': 5})

    Returns
    -------  
    ds_obs : Dataset
        Dataset of all available observations. 

    """
    
    logging.info('     Reading observations...')
    
    def preprocess(ds):
        return ds.coarsen(coarsening, boundary="trim").mean()
     
    ds_obs = xr.open_mfdataset(input_file, combine='nested', concat_dim='time', parallel=True, preprocess=preprocess) #.sortby('time')
    #ds_obs = ds_obs.coarsen(coarsening, boundary="trim").mean().sortby('time')
    ds_obs = ds_obs.sortby('time')
    
    lon_min = oi_grid.lon.min().values
    lon_max = oi_grid.lon.max().values
    lat_min = oi_grid.lat.min().values
    lat_max = oi_grid.lat.max().values
    time_min = oi_grid.time.min().values
    time_max = oi_grid.time.max().values
    
    ds_obs = ds_obs.sel(time=slice(time_min - np.timedelta64(int(2*oi_param.Lt.values), 'D'), 
                                   time_max + np.timedelta64(int(2*oi_param.Lt.values), 'D')), drop=True)
    
    # correct lon if domain is between [-180:180]
    if lon_min < 0:
        ds_obs['lon'] = xr.where(ds_obs['lon'] >= 180., ds_obs['lon']-360., ds_obs['lon'])
        
    ds_obs = ds_obs.where((ds_obs['lon'] >= lon_min - oi_param.Lx.values) & 
                          (ds_obs['lon'] <= lon_max + oi_param.Lx.values) &
                          (ds_obs['lat'] >= lat_min - oi_param.Ly.values) &
                          (ds_obs['lat'] <= lat_max + oi_param.Ly.values) , drop=True)
    
    vtime = (ds_obs['time'].values - np.datetime64(simu_start_date)) / np.timedelta64(1, 'D')
    ds_obs = ds_obs.assign_coords({'time': vtime})
    
    return ds_obs



def oi_param(Lx, Ly, Lt, noise):
    """
    Create dataset of OI parameters

    Parameters
    ----------
    Lx,Ly : scalar 
        OI spatial correlation on the x- and y-axis respectively

    Lt: scalar 
        OI temporal correlation 

    Returns
    -------
    ds_oi_param : Dataset
        Dataset used to store and pass OI paramaters. 

    """

    logging.info('     Set OI params...')

    ds_oi_param = xr.Dataset({'Lx' : Lx,
                              'Ly' : Ly,
                              'Lt' : Lt,
                              'noise' : noise})

    return ds_oi_param



def oi_grid(glon,glat,gtime,simu_start_date):
    """ 
    Set up OI grid.

    oi_grid uses glon, glat, gtime, simu_start_date to create an OI grid returned as ds_oi_grid.

    Parameters
    ----------

    Returns
    -------
    ds_oi_grid : Dataset
        Grid used in the optimal interpolation. 

    """

    logging.info('     Set OI grid...')

    nx = len(glon)
    ny = len(glat)
    nt = len(gtime)

    # define & initialize ssh array
    gssh = np.empty((nt, ny, nx))
    nobs = np.empty(nt)

    # Make 2D grid
    glon2, glat2 = np.meshgrid(glon,glat)
    fglon = glon2.flatten()
    fglat = glat2.flatten()

    ng = len(fglat) # number of grid points
    vtime = (gtime - np.datetime64(simu_start_date)) / np.timedelta64(1, 'D')


    ds_oi_grid = xr.Dataset({'gssh' : (('time', 'lat', 'lon'), gssh), 
                             'glon2' : (('lat', 'lon'), glon2),
                             'glat2' : (('lat', 'lon'), glat2),
                             'fglon' : (('ng'), fglon),
                             'fglat' : (('ng'), fglat),
                             'nobs' : (('time'), nobs)},
                              coords={'gtime': (vtime).astype(np.float),
                                      'time': gtime,
                                      'lat': glat, 
                                      'lon': glon,
                                      'ng': np.arange(ng)})

    return ds_oi_grid





def oi_core(it, ds_oi_grid, ds_oi_param, ds_obs,obsskiptime,obsskipac):
    """ 
    Performs OI inner algorithm.

    oi_core uses the pre-generated Datasets ds_oi_grid, ds_oi_param and ds_obs along with obsskiptime and
    obsskipac (for SWOT) to compute the ssh and update its value in ds_oi_grid. 

    Parameters
    ----------
    ds_oi_grid : Dataset
        Grid used in the optimal interpolation.
    ds_oi_param : Dataset
        Dataset used to store and pass OI paramaters.
    ds_obs : Dataset
        Dataset of all available observations. 

    Returns
    -------  
    """
    # Check if obs is swot-like (with 'nC', i.e., with across track) or nadir (no 'nC')
    try:
        getattr(ds_obs, 'nC')         
    except AttributeError:
        nC = 1
    else: 
        nC = int(ds_obs.nC.size) # To reduce the computing time one point out of two is used across track


    # indices in the right time window
    ind1 = np.where((np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[it]) < 2.*ds_oi_param.Lt.values))[0][::obsskiptime]

    if nC == 1: 
        ind0 = ~np.isnan(ds_obs.ssh_model.values[ind1]) 
        nobs = np.size(np.squeeze(ds_obs.lon.values[ind1][ind0]))
    else: 
        ind0 = ~np.isnan(ds_obs.ssh_model.values[::obsskipac,ind1])
        nobs = np.size(np.squeeze(ds_obs.lon.values[::obsskipac,ind1][ind0]))


    # print processing evolution
    print('Processing time-step : ', it, '/', len(ds_oi_grid.gtime.values) - 1, '      nobs = ', nobs, end="\r")

    # initialize OI matrices
    BHt = np.empty((len(ds_oi_grid.ng), nobs))
    HBHt = np.empty((nobs, nobs)) 

    if nC == 1: 
        obs_lon = np.squeeze(ds_obs.lon.values[ind1][ind0])
        obs_lat = np.squeeze(ds_obs.lat.values[ind1][ind0])
        obs_time = np.squeeze(ds_obs.time.values[ind1][ind0])
    else:  
        obs_lon = np.squeeze(ds_obs.lon.values[::obsskipac,ind1][ind0]) 
        obs_lat = np.squeeze(ds_obs.lat.values[::obsskipac,ind1][ind0]) 
        obs_time =  np.squeeze(np.multiply(np.ones_like(ds_obs.lat.values[::2,ind1]),ds_obs.time.values[ind1])[ind0])


    fglon = ds_oi_grid.fglon.values
    fglat = ds_oi_grid.fglat.values
    ftime = ds_oi_grid.gtime.values[it]


    # loop over observations in the time window
    for iobs in range(nobs): 

        # compute OI matrices
        BHt[:,iobs] = np.exp(-((ftime - obs_time[iobs])/ds_oi_param.Lt.values)**2 - 
                                ((fglon - obs_lon[iobs])/ds_oi_param.Lx.values)**2 - 
                                ((fglat - obs_lat[iobs])/ds_oi_param.Ly.values)**2)

        HBHt[:,iobs] = np.exp(-((obs_time - obs_time[iobs])/ds_oi_param.Lt.values)**2 -
                                 ((obs_lon - obs_lon[iobs])/ds_oi_param.Lx.values)**2 -
                                 ((obs_lat - obs_lat[iobs])/ds_oi_param.Ly.values)**2)

    del obs_lon, obs_lat, obs_time

    # observation error covariance matrix (diagonal)
    R = np.diag(np.full((nobs), ds_oi_param.noise.values**2))


    Coo = HBHt + R 
    Coo = np.ma.masked_invalid(Coo) 

    Mi = np.linalg.inv(Coo)  

    if nC == 1: 
        sol = np.dot(np.dot(BHt, Mi), ds_obs.ssh_model.values[ind1][ind0])
    else: 
        sol = np.dot(np.dot(BHt, Mi), ds_obs.ssh_model.values[::obsskipac,ind1][ind0])

    ds_oi_grid.gssh[it, :, :] = sol.reshape(ds_oi_grid.lat.size, ds_oi_grid.lon.size) 
    ds_oi_grid.nobs[it] = nobs
    