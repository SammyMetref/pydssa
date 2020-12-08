import numpy as np

class vertical_reconstruction:
    """
    Reconstruction of ocean 3d fields using surface maps
    
    Requirement of initialization:
    ssh:   sea surface height
    ssd:   sea surface density 
    sst:   sea surface temperature 
    lon:   longitude  
    lat:   latitudes  
    z:     vertical levels
    N0:    effective buoyancy  
    C:     SQG ratio between SST and SSH
    sppad: proportional spectral padding coefficient
    corrN: vertical correlation of N0 
    """

    def __init__(self,ssh=None,ssd=None,sst=None,lon=None,lat=None,z=None,N0=None,C=1.,sppad=1,corrN=None,periodsmooth=30):
        """
        vertical reconstruction attributes
        """ 

        # Input surface maps
        self.ssh = ssh
        self.ssd = ssd
        self.sst = sst
        self.lon = lon
        self.lat = lat
        self.z = z #np.linspace(0,-1000,100)

        # Problem parameters
        self.N0 = N0 #80*self.f0                                     # Predefined
        self.C = C #12                                         # Predefined
        self.g = 9.81
        self.sp_pad = sppad
        self.corrN = corrN
        self.periodsmooth = periodsmooth

    def init_esqg(self):
        """
        initialize arrays for esqg reconstruction
        """
        
        if self.ssh is None: 
            raise ValueError('SSH value is needed for eSQG reconstruction')
         
        
        # Problem size in physical space
        self.nx = np.shape(self.lon)[0]
        self.ny = np.shape(self.lon)[1]
        self.nz = np.shape(self.z)[0] 
        
        # Set Coriolis parameter f
        self.f0 = 2*2*np.pi/86164*np.sin(self.lat*np.pi/180)   
        
        # Compute dx and dy in meters
        tmp, self.dx = np.gradient(self.lon*111320*np.cos(self.lat[:,:]* np.pi / 180))
        self.dy, tmp = np.gradient(self.lat*110540)
        
        # Smooth ssh value at border to mimic periodicity
        ssh_smoothed = smooth_border_domain(self.ssh,self.periodsmooth)
        
        # Compute surface anomaly maps
        self.ssha =ssh_smoothed - ssh_smoothed.mean()                  # Filter a priori ? nd.gaussian_filter( ,0.5)  
        
        # Initialize arrays
        self.psi = np.zeros([self.nx,self.ny,self.nz])
        self.relvort = np.zeros([self.nx,self.ny,self.nz])
        self.relvort_norm = np.zeros([self.nx,self.ny,self.nz])
        self.buoy = np.zeros([self.nx,self.ny,self.nz])
        self.vertvel = np.zeros([self.nx,self.ny,self.nz])
        self.corr_surf = np.zeros([self.nz]) 
        
        
    def init_esqg_spectral_space(self):
        """
        initialize spectral space and arrays for esqg reconstruction
        """
        
        import scipy.fftpack as fft
        
        # Compute Fourier transform 
        self.ssha_hat = fft.fft2(self.ssha,[self.sp_pad*self.nx,self.sp_pad*self.ny]) 

        # Problem size in Fourier space
        self.skx = np.shape(self.ssha_hat)[0]
        self.sky = np.shape(self.ssha_hat)[1]


        self.f0_hat = np.ones([self.skx,self.sky])*self.f0.mean()
        self.f0_hat[:self.nx,:self.ny] = self.f0
        self.N0_hat = np.ones([self.skx,self.sky])*self.N0.mean()
        self.N0_hat[:self.nx,:self.ny] = self.N0

        # Initialize N on the vertical 
        if self.corrN is None: 
            self.N = np.dstack([self.N0_hat]*self.nz)
        else:
            self.N = np.zeros([self.skx,self.sky,self.nz]) 
            for iz in range(self.nz): 
                self.N[:,:,iz] = self.N0_hat*self.corrN[iz]

        # Compute kh 
        kx = 2*np.pi*fft.fftfreq(self.skx)
        ky = 2*np.pi*fft.fftfreq(self.sky)  
        self.k_h = np.zeros([self.skx,self.sky])
        for ik in np.arange(self.skx):
            for il in np.arange(self.sky): 
                        self.k_h[ik,il] = np.sqrt((kx[ik]/self.dx[ik%self.nx,0] ) ** 2 + (ky[il]/self.dy[0,il%self.ny] ) ** 2)   
        
        
        # Initialize spectral arrays 
        self.psi_hat = np.zeros([self.skx,self.sky,self.nz],dtype=complex)
        self.relvort_hat = np.zeros([self.skx,self.sky,self.nz],dtype=complex)
        self.buoy_hat = np.zeros([self.skx,self.sky,self.nz],dtype=complex) 
        self.vertvel_hat = np.zeros([self.skx,self.sky,self.nz],dtype=complex) 
         
            
    def run_esqg(self): 
        """
        run esqg reconstruction
        """

        import scipy.fftpack as fft

        # Initialize esqg 
        self.init_esqg()
        
        # Initialize spectral space
        self.init_esqg_spectral_space()
        
        
        # Loop on the vertical
        for iz in range(self.nz): 
            # Compute the streamfunction in Fourier space
            corr_surf0 = np.exp(self.N[:,:,iz]/self.f0_hat*self.k_h*self.z[iz])
            self.corr_surf[iz] =  np.mean(corr_surf0)
            self.psi_hat[:,:,iz] = self.g/self.f0_hat*self.ssha_hat[:,:]*corr_surf0

            # Compute the relative vorticity in Fourier space
            self.relvort_hat[:,:,iz] = - self.k_h**2 * self.psi_hat[:,:,iz] 

            # Compute the buoyancy in Fourier space
            self.buoy_hat[:,:,iz] = self.N[:,:,iz]/self.C * self.k_h * self.psi_hat[:,:,iz]
            
            # Compute the vertical velocity in Fourier space
            jacob_tmp_hat = np.zeros([self.skx,self.sky],dtype=complex)
            if iz == 0:
                psi_s = np.real(fft.ifft2(self.psi_hat[:,:,iz])[:self.nx,:self.ny]) 
                buoy_s = np.real(fft.ifft2(self.buoy_hat[:,:,iz])[:self.nx,:self.ny])  
                psi_s = smooth_border_domain(psi_s,self.periodsmooth)
                buoy_s = smooth_border_domain(buoy_s,self.periodsmooth)
                jacob_s = jacobian(psi_s,buoy_s,self.dx,self.dy)  
                jacob_s = smooth_border_domain(jacob_s,self.periodsmooth)
                jacob_s_hat = fft.fft2(jacob_s,[self.skx,self.sky]) 
            else : 
                psi_tmp = np.real(fft.ifft2(self.psi_hat[:,:,iz])[:self.nx,:self.ny]) 
                buoy_tmp = np.real(fft.ifft2(self.buoy_hat[:,:,iz])[:self.nx,:self.ny]) 
                psi_tmp = smooth_border_domain(psi_tmp,self.periodsmooth)
                buoy_tmp = smooth_border_domain(buoy_tmp,self.periodsmooth)
                jacob_tmp = jacobian(psi_tmp,buoy_tmp,self.dx,self.dy) 
                jacob_tmp = smooth_border_domain(jacob_tmp,self.periodsmooth)
                jacob_tmp_hat = fft.fft2(jacob_tmp,[self.skx,self.sky]) 
            self.vertvel_hat[:,:,iz] = - ( self.C/self.N[:,:,iz])**2*( -jacob_s_hat*np.exp(self.N[:,:,iz]/self.f0_hat*self.k_h*self.z[iz]) + jacob_tmp_hat )

            # Back to the physical space 
            self.psi[:,:,iz] = np.real(fft.ifft2(self.psi_hat[:,:,iz])[:self.nx,:self.ny])
            self.relvort[:,:,iz] = np.real(fft.ifft2(self.relvort_hat[:,:,iz])[:self.nx,:self.ny])
            self.relvort_norm[:,:,iz] = self.relvort[:,:,iz]/self.f0
            self.buoy[:,:,iz] = np.real(fft.ifft2(self.buoy_hat[:,:,iz])[:self.nx,:self.ny])
            self.vertvel[:,:,iz] = np.real(fft.ifft2(self.vertvel_hat[:,:,iz])[:self.nx,:self.ny])*1e3
            
            
    def plot_reconstruction_outputs(self,crop1=0,crop2=-1,crop3=0,crop4=-1): 
        """ 
        plot_reconstruction_outputs

    DESCRIPTION 
        Plot surface variables (ssha, ssda, ssta) and 3D variables (psi, relvort_norm, buoy, vertvel) if they exist.

        Returns:  None
        """
        
        import matplotlib.pyplot as plt
        
        # Plot Surface variables [SSH, SSD, SST]
        print("Plotting surface variable")
        
        var_surf = ['ssha', 'ssda', 'ssta']
        var_3d_long = ['SSH anomaly', 'SSD anomaly', 'SST anomaly']
        
        ivar = 0
        for varname in var_surf: 
            try:
                getattr(self, varname)         
            except AttributeError:
                print(var_3d_long[ivar]+' is not available for plotting')
            else: 
                var = getattr(self, varname)
                plt.figure(figsize=(3,3)) 
                plt.contourf(self.lon[crop1:crop2,crop3:crop4],self.lat[crop1:crop2,crop3:crop4],var[crop1:crop2,crop3:crop4],cmap=plt.cm.get_cmap('bwr'))
                plt.colorbar()
                plt.ylabel(var_3d_long[ivar],fontsize=20)
                plt.title('Horizontal map')
            ivar +=1
            
        # Plot 3D variable: Streamfunction, Relative vorticity, Buoyancy and Vertical velocity
        print("Plotting 3D variables")
        
        var_3d = ['psi', 'relvort_norm', 'buoy','vertvel']
        var_3d_long = ['Stream function', 'Relative vorticity', 'Buoyancy', 'Vertical velocity (10$^{-3}$)']
        
        ivar = 0
        for varname in var_3d: 
            try:
                getattr(self, varname)         
            except AttributeError:
                print(var_3d_long[ivar]+' is not available for plotting')
            else: 
                var = getattr(self, varname)
                levs = np.arange(-np.max(np.abs(var[crop1:crop2,crop3:crop4,:])),np.max(np.abs(var[crop1:crop2,crop3:crop4,:])), 2*np.max(np.abs(var[crop1:crop2,crop3:crop4,:]))/100)
                plt.figure(figsize=(35,6))
                plt.subplot(151)
                plt.contourf(self.lon[crop1:crop2,crop3:crop4],self.lat[crop1:crop2,crop3:crop4],var[crop1:crop2,crop3:crop4,0],cmap=plt.cm.get_cmap('bwr'),levels=levs)
                plt.colorbar()
                plt.ylabel(var_3d_long[ivar],fontsize=20)
                plt.title('Horizontal map at 0 m')
                plt.subplot(152)
                plt.contourf(self.lon[crop1:crop2,crop3:crop4],self.lat[crop1:crop2,crop3:crop4],var[crop1:crop2,crop3:crop4,30],cmap=plt.cm.get_cmap('bwr'),levels=levs)
                plt.colorbar()
                plt.title('Horizontal map at '+str("%.1f" %self.z[30])+' m')
                plt.subplot(153)
                plt.contourf(self.lon[crop1:crop2,crop3:crop4],self.lat[crop1:crop2,crop3:crop4],var[crop1:crop2,crop3:crop4,60],cmap=plt.cm.get_cmap('bwr'),levels=levs)
                plt.colorbar()
                plt.title('Horizontal map at '+str("%.1f" %self.z[60])+' m')
                plt.subplot(154)
                plt.contourf(self.lon[int(self.nx/2),crop3:crop4],self.z[:],np.transpose(var[int(self.nx/2),crop3:crop4,:]),cmap=plt.cm.get_cmap('bwr'),levels=levs)
                plt.colorbar()
                plt.title('Zonal cross section at '+str("%.1f" % self.lat[int(self.nx/2),1])+'°N')
                plt.subplot(155)
                plt.contourf(self.lat[crop1:crop2,int(self.ny/2)],self.z[:],np.transpose(var[crop1:crop2,int(self.ny/2),:]),cmap=plt.cm.get_cmap('bwr'),levels=levs)
                plt.colorbar()
                plt.title('Meridional cross section at '+str("%.1f" % np.abs(self.lon[1,int(self.ny/2)]))+'°W')
            ivar +=1


def jacobian(aa,bb,dx,dy):
    """ 
    
    NAME 
        jacobian

    DESCRIPTION 
        Computer the jacobian J(aa,bb) of two matrices aa and bb in function of x and y

        Returns:  Jacobian array
        
    """
    aa_x = np.gradient(aa)[1]/dx
    aa_y = np.gradient(aa)[0]/dy 
    bb_x = np.gradient(bb)[1]/dx 
    bb_y = np.gradient(bb)[0]/dy
    
    return aa_x*bb_y - bb_x*aa_y


def smooth_border_domain(ssh,bordersize=30):
    """
    NAME 
        smooth_border_domain

    DESCRIPTION 
        Smooth SSH to periodic at border of the domain
        
        Args: 
            ssh : array of ssh values to smooth
            bordersize : Distance from border above where smoothing is performed

        Returns:  ssh field with smooth borders 
        
    """
    plot4verif = False
    if plot4verif:
        import matplotlib.pyplot as plt 
        plt.figure(figsize=(15,6))
        plt.subplot(121)
        plt.imshow(ssh)
        plt.colorbar()
        
    
    ssh_dim1 = np.shape(ssh)[0]
    ssh_dim2 = np.shape(ssh)[1]
    ssh_smoothed = ssh
    smoother = ana_gaspari_cohn(np.arange(bordersize),2*bordersize) #np.arange(bordersize)[::-1]/(bordersize-1)#
    
    for i in range(ssh_dim1):
        ssh_smoothed[i,:bordersize] = ssh_smoothed[i,:bordersize]*smoother[::-1] + ssh_smoothed[i,ssh_dim2-bordersize:]*(1-smoother[::-1]) 
        ssh_smoothed[i,ssh_dim2-bordersize:] = ssh_smoothed[i,ssh_dim2-bordersize:]*smoother + ssh_smoothed[i,:bordersize]*(1-smoother) 
    for j in range(ssh_dim2):
        ssh_smoothed[:bordersize,j] = ssh_smoothed[:bordersize,j]*smoother[::-1] + ssh_smoothed[ssh_dim1-bordersize:,j]*(1-smoother[::-1]) 
        ssh_smoothed[ssh_dim1-bordersize:,j] = ssh_smoothed[ssh_dim1-bordersize:,j]*smoother + ssh_smoothed[:bordersize,j]*(1-smoother) 
        
    if plot4verif: 
        plt.subplot(122)
        plt.imshow(ssh_smoothed)
        plt.colorbar()
        plt.show()
           
    return ssh   


def ana_gaspari_cohn(r,c):
    """
    NAME 
        ana_smoothing_gaspari_cohn

    DESCRIPTION 
        Gaspari-Cohn function. 
        
        Args: 
            r : array of value whose the Gaspari-Cohn function will be applied
            c : Distance above which the return values are zeros


        Returns:  smoothed values 
            
    """ 
    
    if type(r) is float:
        ra = np.array(r)
    else:
        ra = r
    if c<=0:
        return np.zeros_like(ra)
    else:
        ra = 2*np.abs(ra)/c
        gp = np.zeros_like(ra)
        i= np.where(ra<=1.)[0]
        gp[i]=-0.25*ra[i]**5+0.5*ra[i]**4+0.625*ra[i]**3-5./3.*ra[i]**2+1.
        i =np.where((ra>1.)*(ra<=2.))[0]
        gp[i] = 1./12.*ra[i]**5-0.5*ra[i]**4+0.625*ra[i]**3+5./3.*ra[i]**2-5.*ra[i]+4.-2./3./ra[i]
        if type(r) is float:
            gp = gp[0]
    return gp



