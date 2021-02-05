import numpy as np
import xarray as xr 
import logging
import sys 

class ssh_mapping:
    """
    Mapping of 2d SSH fields using nadir or SWOT observations
     
    Requirement of initialization:
    ------------------------------ 
    obs_in: array of strings
        Observation paths
    lonlat_minmax: array_like
        Domain borders [ min longitude, max longitude, min latitude, max latitude] 
    time_minmax: array_like
        Domain [min,max] time
    dx: array_like (2D)
        Zonal grid spatial step (in degree)
    dy: array_like (2D)
        Meridional grid spatial step (in degree)
    dt: Numpy.timedelta64
        Temporal step 
    simu_start_date: string
        Nature run initial date (e.g., '2012-10-01T00:00:00')
    obsskiptime : int
        Jump index in observation grid point in time, for a less accurate but speedier run.
    obsskipac : int
        Jump index in observation grid point across track (SWOT), for a less accurate but speedier run.
    
    Available functions:
    --------------------
    OI functions 
        - run_oi: run optimal interpolation (OI) mapping
        - oi_grid: set up OI grid
        - oi_param: create dataset of OI parameters
        - oi_core: performs OI inner algorithm
    """

    def __init__(self,config_file = None, obs_in=None,output=None,lonlat_minmax=None,time_minmax=None,dx=1.,dy=1.,dt=1.,simu_start_date=None,obsskiptime=2,obsskipac=2): 
        
        if config_file is not None:
            # Config file (if needed)
            self.config_file = config_file
          
        else: 

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

            self.simu_start_date = simu_start_date                        # nature run initial date

            self.glon = np.arange(self.lon_min, self.lon_max + self.dx, self.dx)           # output OI longitude grid
            self.glat = np.arange(self.lat_min, self.lat_max + self.dy, self.dy)           # output OI latitude grid
            self.gtime = np.arange(self.time_min, self.time_max + self.dt, self.dt)        # output OI time grid

            self.obsskiptime = obsskiptime                                # jump observation grid points in time
            self.obsskipac = obsskipac                                    # jump observation grid points across track (SWOT)

    #####
    # OI run

    def run_oi(self,Lx=1.,Ly=1.,Lt=7.,noise=0.05):
        """ 
        Run optimal interpolation (OI) mapping algorithm. 
        
        When finished, run_oi uses Dataset.to_netcdf function to create a netcdf file stored in self.output_oi.
          
        Parameters
        ----------
        Lx,Ly : scalar 
            OI spatial correlation on the x- and y-axis respectively
         
        Lt: scalar 
            OI temporal correlation 
        
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
        
        sys.path.insert(1, '../pydssa/func_ssh_mapping/func_ssh_mapping_oi') 
        import func_ssh_mapping_oi as oi 
            
        # set OI param & grid
        ds_oi1_param = oi.oi_param(Lx, Ly, Lt, noise)
        ds_oi1_grid = oi.oi_grid(self.glon, self.glat, self.gtime, self.simu_start_date)
        
        # Read input obs + discard a bit...
        coarsening = {'time': 3}
        ds_oi1_obs = oi.read_obs(self.obs_in, ds_oi1_grid, ds_oi1_param, self.simu_start_date, coarsening)
        
        # Run OI
        for it in range(len(self.gtime)):
            oi.oi_core(it, ds_oi1_grid, ds_oi1_param, ds_oi1_obs,self.obsskiptime,self.obsskipac)
                    
        # Write output                                                                  
        ds_oi1_grid.to_netcdf(self.output_oi)
        

    #####
    # BFN-QG run

    def run_bfnqg(self):
        """ 
        Run back-and-forth nudgning with quasigeostrophic model (BFN-QG) mapping algorithm. 
        
        When finished, run_bfnqg creates netcdf files stored in the directory defined in config file.
          
        Parameters
        ----------
        Lx,Ly : scalar 
            OI spatial correlation on the x- and y-axis respectively
         
        Lt: scalar 
            OI temporal correlation 
        
        Returns
        -------
            
        Notes
        -----
        The BFN-QG algorithm is a new mapping algorithm based on the back-and-forth nudgning data assimilation scheme and
        the quasigeostrophic model (Le Guillou et al., 2020). The present codes are derived from the MASSH package
        (https://github.com/leguillf/MASSH).
        
        References
        ----------
        Le Guillou, F., Metref, S., Cosme, E., Le Sommer, J., Ubelmann, C., Verron, J., & Ballarotta, M. (2020). Mapping
        altimetry in the forthcoming SWOT era by back-and-forth nudging a one-layer quasi-geostrophic model, Journal of
        Atmospheric and Oceanic Technology, . Retrieved Jan 26, 2021, from
        https://journals.ametsoc.org/view/journals/atot/aop/JTECH-D-20-0104.1/JTECH-D-20-0104.1.xml
        
        """
        
        sys.path.insert(1, '../pydssa/func_ssh_mapping/func_ssh_mapping_bfnqg') 
        
        # Configuration file
        import exp as exp
        config = exp.exp(self.config_file)
        
        # State
        import state 
        State = state.State(config)
        
        # Model
        import mod
        Model = mod.Model(config,State)
        
        # Observations 
        import obs 
        dict_obs = obs.obs(config,State)
        
        # Assimilation 
        import ana 
        ana.ana(config,State,Model,dict_obs=dict_obs)
         

 