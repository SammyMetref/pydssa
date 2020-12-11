import numpy as np
import xarray as xr 
import logging

class ssh_mapping:
    """
    Mapping of 2d SSH fields using nadir or SWOT observations
     
    Requirement of initialization:
    ------------------------------ 
    obs_in: array of observation paths
    lonlat_minmax: domain [ min longitude, max longitude, min latitude, max latitude] 
    time_minmax: domain [min,max] time
    dx: zonal grid spatial step (in degree)
    dy: meridional grid spatial step (in degree)
    dt: temporal grid step 
    simu_start_date: Nature run initial date
    
    Available functions:
    --------------------
    # OI functions #
        - run_oi: run optimal interpolation (OI) mapping
        - oi_grid: set up OI grid
        - oi_param: create dataset of OI parameters
        - oi_core: performs OI inner algorithm
    """

    def __init__(self,obs_in=None,output=None,lonlat_minmax=None,time_minmax=None,dx=1.,dy=1.,dt=1.,simu_start_date=None): 
  
        # Observations
        self.obs_in = obs_in
        self.output_oi = output
        
        # Input grid 
        self.lon_min = lonlat_minmax[0]                               # domain min longitude
        self.lon_max = lonlat_minmax[1]                               # domain max longitude
        self.lat_min = lonlat_minmax[2]                               # domain min latitude
        self.lat_max = lonlat_minmax[3]                               # domain max latitude
        self.time_min = time_minmax[0]                                # domain min time
        self.time_max = time_minmax[1]                                # domain max time
        self.dx = dx                                                  # zonal grid spatial step (in degree)
        self.dy = dy                                                  # meridional grid spatial step (in degree)
        self.dt = dt                                                  # temporal grid step
 
        self.simu_start_date = simu_start_date                        # Nature run initial date

        self.glon = np.arange(self.lon_min, self.lon_max + self.dx, self.dx)           # output OI longitude grid
        self.glat = np.arange(self.lat_min, self.lat_max + self.dy, self.dy)           # output OI latitude grid
        self.gtime = np.arange(self.time_min, self.time_max + self.dt, self.dt)        # output OI time grid
 

    #####
    # OI functions

    def run_oi(self,Lx=1.,Ly=1.,Lt=7.,noise=0.05):
        """ 
        Run optimal interpolation (OI) mapping algorithm. 
        
        Parameters
        ----------
        
        Returns
        -------
            
        Notes
        -----
        The OI algorithm is a very commonly used mapping algorithm (e.g. for the DUACS-AVISO products) based on 
        prior assumptions on temporal and spatial correlations (see for instance Le Traon and Dibarboure., 1999).
        
        References
        ----------
        Le Traon, P. Y., & Dibarboure, G. (1999). Mesoscale Mapping Capabilities of Multiple-Satellite Altimeter 
        Missions, Journal of Atmospheric and Oceanic Technology, 16(9), 1208-1223. Retrieved Dec 8, 2020, from 
        https://journals.ametsoc.org/view/journals/atot/16/9/1520-0426_1999_016_1208_mmcoms_2_0_co_2.xml
        
        """
            
        # set OI param & grid
        ds_oi1_param = self.oi_param(Lx, Ly, Lt, noise)
        ds_oi1_grid = self.oi_grid(self.glon, self.glat, self.gtime, self.simu_start_date)
        
        # Read input obs + discard a bit...
        coarsening = {'time': 5}
        ds_oi1_obs = read_obs(self.obs_in, ds_oi1_grid, ds_oi1_param, self.simu_start_date, coarsening)
        
        # Run OI
        for it in range(len(self.gtime)):
            self.oi_core(it, ds_oi1_grid, ds_oi1_param, ds_oi1_obs)
                    
        # Write output                                                                  
        ds_oi1_grid.to_netcdf(self.output_oi)


    def oi_grid(self,glon, glat, gtime, simu_start_date):
        """ 
        Set up OI grid
        
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
        glon2, glat2 = np.meshgrid(glon, glat)
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


    def oi_param(self, Lx, Ly, Lt, noise):
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


    def oi_core(self,it, ds_oi_grid, ds_oi_param, ds_obs):
        """ 
        Performs OI inner algorithm
        
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

        ind1 = np.where((np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[it]) < 2.*ds_oi_param.Lt.values))[0]
        nobs = len(ind1)
        print('Processing time-step : ', it, '/', len(ds_oi_grid.gtime.values) - 1, '      nobs = ', nobs, end="\r")

        BHt = np.empty((len(ds_oi_grid.ng), nobs))
        HBHt = np.empty((nobs, nobs))

        obs_lon = ds_obs.lon.values[ind1]
        obs_lat = ds_obs.lat.values[ind1]
        obs_time = ds_obs.time.values[ind1]

        fglon = ds_oi_grid.fglon.values
        fglat = ds_oi_grid.fglat.values
        ftime = ds_oi_grid.gtime.values[it]

        for iobs in range(nobs):
            # print(iobs)

            BHt[:,iobs] = np.exp(-((ftime - obs_time[iobs])/ds_oi_param.Lt.values)**2 - 
                                    ((fglon - obs_lon[iobs])/ds_oi_param.Lx.values)**2 - 
                                    ((fglat - obs_lat[iobs])/ds_oi_param.Ly.values)**2)

            HBHt[:,iobs] = np.exp(-((obs_time - obs_time[iobs])/ds_oi_param.Lt.values)**2 -
                                     ((obs_lon - obs_lon[iobs])/ds_oi_param.Lx.values)**2 -
                                     ((obs_lat - obs_lat[iobs])/ds_oi_param.Ly.values)**2)

        del obs_lon, obs_lat, obs_time

        R = np.diag(np.full((nobs), ds_oi_param.noise.values**2))

        Coo = HBHt + R
        Mi = np.linalg.inv(Coo)

        sol = np.dot(np.dot(BHt, Mi), ds_obs.ssh_model.values[ind1])

        ds_oi_grid.gssh[it, :, :] = sol.reshape(ds_oi_grid.lat.size, ds_oi_grid.lon.size)
        ds_oi_grid.nobs[it] = nobs
    
     

#####
# Other useful functions

def read_obs(input_file, oi_grid, oi_param, simu_start_date, coarsening):
    """ 
    Read available observations (path given in input_file)

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
