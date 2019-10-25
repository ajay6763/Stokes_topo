#!/usr/bin/python
from __future__ import division
import time
starttime = time.time()
#import Stokes2D is not being imported

from scipy.interpolate import RegularGridInterpolator ## interpolation to regularised grid 
import numpy as np
import scipy
import scipy.sparse            # Needed for some older Spyder versions
import scipy.sparse.linalg     # Needed for some older Spyder versions

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pylab import rcParams
from copy import copy, deepcopy
import matplotlib.gridspec as gridspec


# Subfunctions: 
def interpolate_2D(x,y,z,profile_len,reso,step):
    print('Model info \n')
    #reso=input('Enter resolution along the profile: \n')
    #reso=5
    #profile_len=input('Enter resolution the profile length: \n')
    #profile_len=625 
    f = RegularGridInterpolator((x, y), z) #interpolate.interp2d(y,x,z,kind='linear')
    #x_new = np.arange(min(x_),max(x_),step=step)
    #y_new = np.arange(min(y_),max(y_),step=step)
    #points = np.array([x_new,y_new])
    #z_new = f(points)
    return f
def idp(ix,iz,nz):
    # purpose: find index value for the p component in 2-D Stokes matrix
    fout = 3*((ix-1)*nz + iz) - 3
    return fout
    
def idvx(ix,iz,nz):
    # purpose: find index value for the vx component in 2-D Stokes matrix
    fout = 3*((ix-1)*nz + iz) - 2
    return fout
    
def idvz(ix,iz,nz):
    # purpose: find index value for the vz component in 2-D Stokes matrix
    fout = 3*((ix-1)*nz + iz) - 1
    return fout
    
def getpvxvz(sol,xx):
    # purpose: extract p, vx, and vz from sol vector from matrix inversion
    # sol contains [p1vx1vz1p2vx2vz2p3vx3vz3....]
    [nz,nx]=np.shape(xx)
    pp=sol[::3]          # Extract every 3rd position as p from sol
    vx=sol[1::3]         # ... and vx
    vz=sol[2::3]         # ... and vz
    p=np.reshape(pp,(nz,nx), order='F')     # shape solvp into nx-by-nz mesh
    vx=np.reshape(vx,(nz,nx), order='F')    # idem for vx
    vz=np.reshape(vz,(nz,nx), order='F')    # idem for vz
    p=p[1:,1:]           # remove first row and column: ghost points
    meanp=np.mean(p)     # subtract mean to make average p=0
    p=p-meanp               # 

    vx=vx[0:-1,0:]       # remove ghost points from vx
    vz=vz[0:,0:-1]       # remove ghost points from vz
    return [p, vx, vz]
    
def preppvxvzplot(pp,vx,vz,xx,islip):
    # Purpose: interpolate p, vx, and vz to the base points
    #          for plotting
    # Method:  p, vx, and vz are each defined at their own location 
    #          on the staggered grid, which makes plotting difficult
    #          vx on staggered grid is vertically between base points
    #            expand vx array with top and bottom row, note that this
    #            done differently for free-slip and no-slip.
    #            Then interpolate vertically to midpoints, which are the 
    #            base points 
    #          vz on staggered grid is horizontally between base points
    #            done as for vx, but horizontally, not vertically
    #          p on staggered grid is diagonally between base points
    #            done as vx and vz, but both horizontally and vertically
    #            This implies both hor and vert interpolation
    #          p, vx, and vz now all have nx by nz points
    # Arguments:
    #     pp = raw pressure field
    #     vx is raw x-velocity field
    #     vz is raw z-velocity field
    #     islip is slip type on bnds: 1=free-slip, -1=no-slip
    
    # vertically interpolate vx to base points: 
    vxplot=np.zeros(np.shape(xx))
    vxplot[1:-1,:]=0.5*(vx[0:-1,:]+vx[1:,:])
    # and extrapolate vx from 1st/last row to top/bottom bnd.
    vxplot[0,:]=vx[0,:]
    vxplot[-1,:]=vx[-1,:]
    
    # horizontally interpolate vz to base points: 
    vzplot=np.zeros(np.shape(xx))
    vzplot[:,1:-1]=0.5*(vz[:,0:-1]+vz[:,1:])
    # and extrapolate vz from 1st/last column to lef/right bnd.
    vzplot[:,0]=vz[:,0]
    vzplot[:,-1]=vz[:,-1]
    
    # interpolate p to base points for plotting purposes:
    pplot=np.zeros(np.shape(xx))
    # bilinear interpolation of p-points to all internal points:
    pplot[1:-1,1:-1]=0.25*(pp[0:-1,0:-1]+pp[1:,0:-1]+pp[0:-1,1:]+pp[1:,1:])
    pplot[0,1:-1]=0.5*(pp[0,0:-1]+pp[0,1:])
    # Boundary points only have two nearest p-points:
    pplot[-1,1:-1]=0.5*(pp[-1,0:-1]+pp[-1,1:])
    pplot[1:-1,0]=0.5*(pp[0:-1,0]+pp[1:,0])
    pplot[1:-1,-1]=0.5*(pp[0:-1,-1]+pp[1:,-1])
    # Corner points only have one associated internal point:
    pplot[0,0]=pp[0,0]
    pplot[-1,0]=pp[-1,0]
    pplot[0,-1]=pp[0,-1]
    pplot[-1,-1]=pp[-1,-1]
    
    return [pplot,vxplot,vzplot]


# Main routine: 
def Stokes2Dfunc(Ra, T, xx, zz):
    islip = 1 # 1=free-slip -1=no-slip
    
    [nz,nx] = np.shape(xx)
    nxz=3*nx*nz  # total nr of unknowns (nx * nz * (vx+vz+p))
    dx   = xx[0,1]-xx[0,0]
    dz   = zz[1,0]-zz[0,0]   
    
    A    = scipy.sparse.lil_matrix((nxz,nxz))
    A.setdiag(1)
    rhs  = np.zeros(nxz)             # create rhs (buoyancy force) vector
    # Fill in info in matrix for Stokes_z for internal points & left/right bnd. points:
    # Note: 1) other points (top/bottom bnd.pnts and  unused points have vz=0, which is default
    #       2) Index counters: ix=1 to nx-1, and iz=1 to nz (unusual for Python)
    for iz in range (2,nz):          # iz=1 & iz=nz are default (i.e. vz=0) bc's: no calc needed
        for ix in range (1,nx):      # ix=nx is unused point ix=1 & nx-1 are boundary, 
                                     #     but vz still needs calculating
            # calculate indices of all relevant grid points for vz and p:
            # for vz
            vc = idvz(ix,iz,nz)      # calculate matrix index for central vz point:
            if (ix>1):
                vl = idvz(ix-1,iz,nz)# idem, for left vx point
            if (ix<nx-1):
                vr = idvz(ix+1,iz,nz)# idem, for right vz point
            vt = idvz(ix,iz+1,nz)    # idem, for top vz point
            vb = idvz(ix,iz-1,nz)    # idem, for bottom vz point
            # for p:
            pt = idp(ix+1,iz+1,nz)   # idem, for left p point
            pb = idp(ix+1,iz,nz)     # idem, for right p point
            
            # fill in matrix components:
            irow = idvz(ix,iz,nz)
            A[irow,vc] = -2/dx**2-2/dz**2 # valid for internal points only
            if (ix>1):
                A[irow,vl] = 1/dx**2
            else:
                # free-slip add correction to central point
                A[irow,vc] = A[irow,vc] + islip*1/dx**2
    
            if (ix<nx-1):
                A[irow,vr] = 1/dx**2
            else:
                # free-slip add correction to central point
                A[irow,vc] = A[irow,vc] + islip*1/dx**2
            A[irow,vt] = 1/dz**2
            A[irow,vb] = 1/dz**2
            A[irow,pb] = 1/dz
            A[irow,pt] = -1/dz
            
            # rhs: Ra*T'
            avT  = 0.5*(T[iz-1,ix-1]+T[iz-1,ix])
            rhs[irow] = avT*Ra[iz,ix]
            #print 'you are using advT'
            # rhs: Ra*drho'
            #avdrho  = 0.5*(drho[iz-1,ix-1]+drho[iz-1,ix])
            #rhs[irow] = avdrho*Ra
            #print 'You are using advrho'
    # Fill in info in matrix for Stokes_x for internal points & top/bottom bnd. points:
    # Note: other points (left/right bnd.pnts and unused points have vx=0, which is default
    for ix in range (2,nx):          # ix=1 & nx are default (i.e. vx=0) bc's: no calc, needed
        for iz in range (1,nz):      # iz=nz are unused points, iz=1&nz-1 are boundaries, 
                                     #     but vx still needs calculating there
            # calculate indices of all relevant grid points for vx and p:
            # for vx
            vc = idvx(ix,iz,nz)      # calculate matrix index for central vx point:
            vl = idvx(ix-1,iz,nz)    # idem, for left vx point
            vr = idvx(ix+1,iz,nz)    # idem, for right point
            if (iz<nz-1):
                vt = idvx(ix,iz+1,nz)# idem, for top vx point
            if (iz>1):
                vb = idvx(ix,iz-1,nz)# idem, for bottom vx point
            # for p:
            pl = idp(ix,iz+1,nz) # idem, for left p point
            pr = idp(ix+1,iz+1,nz)   # idem, for right p point
            
            # fill in matrix components:
            irow = idvx(ix,iz,nz)
            A[irow,vc] = -2/dx**2-2/dz**2 # valid for internal points only
            A[irow,vl] = 1/dx**2        
            A[irow,vr] = 1/dx**2         
            if (iz<nz-1):            # top bnd.point
                A[irow,vt] = 1/dz**2
            else:
                # free-slip add correction to central point
                A[irow,vc] = A[irow,vc] + islip*1/dz**2 
            if(iz>1):                # bottom bnd.point
                A[irow,vb] = 1/dz**2
            else:
                # free-slip add correction to central point
                A[irow,vc] = A[irow,vc] + islip*1/dz**2 # free-slip
            A[irow,pl] = 1/dx
            A[irow,pr] = -1/dx
            # all rhs components here are 0: is default
    
    # Fill in info in matrix for continuity eqn for all pressure points:
    for ix in range (2,nx+1):       # pressure point ix=1 is unused point
        for iz in range (2,nz+1):   # pressure point iz=1 is unused point
            irow=idp(ix,iz,nz)
            vxl=idvx(ix-1,iz-1,nz)
            vxr=idvx(ix,iz-1,nz)
            vzb=idvz(ix-1,iz-1,nz)
            vzt=idvz(ix-1,iz,nz)
            A[irow,vxl]=-1/dx
            A[irow,vxr]=1/dx
            A[irow,vzb]=-1/dz
            A[irow,vzt]=1/dz
            A[irow,irow]=0
    
    # fix p=0 at one point: lowerleft corner:
    irow=idp(2,2,nz)               
    A[irow,irow]=1
    rhs[irow]=0
    
    # Solve system:
    sol=scipy.sparse.linalg.spsolve(A,rhs)
    
    # extrac p, vx, and vz solutions from solution vector:
    [pp,vx,vz] = getpvxvz(sol,xx)
#    vrms=vrmscalc(vx,vz)
#    print('vrms=',vrms)
    
    # preparing p, vx, and vz for plotting:
    [pplot,vxplot,vzplot]=preppvxvzplot(pp,vx,vz,xx,islip)
       
    return [pplot,vxplot,vzplot]          


def twoDadvdiff (fin,vx,vz,dx,dz,dt):
# Performs 1 advection-diffusion timestep
#   Top b.c.: fixed T
#   Side bnds: symmetry
#   Advection scheme: simple upwind
#   Uniform grid and kappa assumed

    # Initialize a timestep df/dt vector:
    dfdt=np.zeros(np.shape(fin))

    # Calculate 2nd derivatives in x- & z-dir.:
    d2fdx2=np.diff(fin,n=2,axis=1)/dx**2
    d2fdz2=np.diff(fin,n=2,axis=0)/dz**2
        
    # Apply diffusion:
    dfdt[1:-1,1:-1] = d2fdx2[1:-1,:]+ d2fdz2[:,1:-1]
    #   Natural b.c.'s at side boundaries:
    dfdt[1:-1,0]    = dfdt[1:-1, 0]   + 2*(fin[1:-1, 1]-fin[1:-1, 0])/dx**2
    dfdt[1:-1,-1]   = dfdt[1:-1,-1]   + 2*(fin[1:-1,-2]-fin[1:-1,-1])/dx**2
    
    # Advection: upwind approach: 
    [nz,nx]=np.shape(fin)
    for i in range(1,nx-1):
        for j in range(0,nz):
            if vx[j,i]>=0:
                dfdtx=vx[j,i]*(fin[j,i-1]-fin[j,i])/dx
            else:
                dfdtx=vx[j,i]*(fin[j,i]-fin[j,i+1])/dx
            dfdt[j,i]=dfdt[j,i]+dfdtx
    for i in range(0,nx):
        for j in range(1,nz-1):
            if vz[j,i]>=0:
                dfdtz=vz[j,i]*(fin[j-1,i]-fin[j,i])/dz
            else:
                dfdtz=vz[j,i]*(fin[j,i]-fin[j+1,i])/dz
            dfdt[j,i]=dfdt[j,i]+dfdtz
    # Add dt * df/dt-vector to old solution:
    fout=fin+dt*dfdt
    return fout
###################################################################################################################################################
# Main code: 
###################################################################################################################################################
# Initialisation:

###################################################################################################################################################
# Dimensional variables:
kappa    = 1e-6                # thermal diffusivity
Tm       = 1550                # mantle temperature in degC
Tlab     = 1320
deltaT   = Tm-Tlab
g        = 9.81
alpha    = 3e-5                # K-1
hdim     = 400e3              # dimensional height of box: 1000 km
eta      = 1e22                # How can I use a none constant viscosity?
rho      = 3400.      

###################################################################################################################################################
# Mesh setup:
h        = 1.0                 # nondimensional box height
w        = 1.0                 # box of aspect ratio 1
dx       = 0.02                # discretization step in meters
dz       = 0.05
nx       = w/dx+1
nx       = int(nx)
nz       = h/dz+1
nz       = int(nz) 
niveles_z = nz      
niveles_x = nx 
x        = np.linspace(0,w,niveles_x) # array for the finite difference mesh
z        = np.linspace(0,h,niveles_z)
dx       = w/(nx-1)            # Adjust requested dx & dz to fit in equidistant grid space
dz       = h/(nz-1) 
[xx,zz]  = np.meshgrid(x,z)
print nx,'nx',nz,'nz'
nxz=3*nx*nz

        

###################################################################################################################################################
# initial density and temperature distribution from LiMod2D)
###################################################################################################################################################
data= np.loadtxt('post_processing_output.dat',usecols=(0,1,2,3,4,5,6))
Tm       = max(data[:,2])                # mantle temperature in degC
Tlab     = 1320
deltaT   = Tm-Tlab

reso = 5
profile_len=625
depth=400 
step=0.05
y_=-data[0:95,1]/400
t=np.arange(0,profile_len+reso,reso)
x_=t/profile_len
print len(x_)
print len(y_)
T=[]
D=[]
temp=0
for i in range(len(x_)):
  T_ = []
  D_ = []
  for j in range(len(y_)):
    T_.append(data[temp,2]/max(data[:,2]))
    D_.append(data[temp,6]/max(data[:,6]))
    temp=temp+1
  T.append(T_)
  D.append(D_)
### making X,Y grid
X,Y=np.meshgrid(y_,x_)
T_func= interpolate_2D(x_,y_,T,profile_len,5,step)
D_func= interpolate_2D(x_,y_,D,profile_len,5,step)
drho = 1.*np.ones(np.shape(xx))
Told     = 1.*np.ones(np.shape(xx))
m,n=np.shape(xx)
for i in range(m):
    for j in range(n):
        #print xx[i,j],zz[i,j]
        Told[i,j]=T_func((xx[i,j],zz[i,j]))
        drho[i,j]=D_func([xx[i,j],zz[i,j]])
drho[0:int(nx/3),:]=0.5        # Symmetric buoyancy for both odd and even 
drho[int(nx/3):int(nx/2),:]=0.5
###################################################################################################################################################
# Raleight number; At the moment calculated with constant density; Will adapt this part where it will be calculated using density distribution from 
# LitMod
###################################################################################################################################################
'''
Ra  = np.zeros(nxz)
Ra  = (alpha*drho*g*deltaT*(hdim**3))/(eta*kappa)
print Ra,'Ra'
'''
Raold  = np.zeros(np.shape(xx))
#Ra  = (alpha*3000*g*1300*(hdim**3))/(eta*kappa)
m,n=np.shape(xx)
for i in range(m):
    for j in range(n):
        #print xx[i,j],zz[i,j]
        Raold[i,j]= (alpha*drho[i,j]*g*Told[i,j]*(hdim**3))/(eta*kappa)#  T_func((xx[i,j],zz[i,j]))
        #drho[i,j]=D_func([xx[i,j],zz[i,j]])

#print Ra,'Ra'
###################################################################################################################################################
# Inialising topography array
###################################################################################################################################################
topo_ini = np.linspace(0,w,niveles_x)
topo_obs = np.loadtxt('topo.inp')
###################################################################################################################################################
# Time variables:
###################################################################################################################################################
dt_diff  = 0.2*(dx**2)          # timestep in seconds
print dt_diff, 'timestp in seconds'
nt       = 20                 # number of tsteps
secinmyr = 1e6*365*24*3600     # amount of seconds in 1 Myr
print dt_diff/3600/24/365      # time step in years
t        = 0                   # set initial time to zero
nplot    = 10                   # plotting interval: plot every nplot timesteps

###################################################################################################################################################
# Initial temperature distribution; This will also be taken from LiMod
###################################################################################################################################################
#Ttop     = Tlab                   # surface T
#Told     = 1.*np.ones(np.shape(xx)) # Initial temperature T=0.9 everywhere
#print np.shape(xx)
'''
Told[(zz==0)]=Ttop
Told[(zz>0.0e3/hdim) & (zz<100e3/hdim)]=1350./Tm
Told[(zz>100e3/hdim) & (zz<200e3/hdim)]=1360./Tm
Told[(zz>200e3/hdim) & (zz<300e3/hdim)]=1380./Tm
Told[(zz>300e3/hdim) & (zz<400e3/hdim)]=1400./Tm
Told[(zz>400e3/hdim) & (zz<500e3/hdim)]=1450./Tm
Told[(zz>490e3/hdim) & (zz<600e3/hdim)]=1500./Tm
Told[(zz>600e3/hdim) & (zz<700e3/hdim)]=1550./Tm
Told[(zz>700e3/hdim) & (zz<800e3/hdim)]=1600./Tm
Told[(zz>0.0e3/hdim) & (zz<300e3/hdim) & (xx>200e3/hdim) & (xx<800e3/hdim)]=1350./Tm
Told[(zz>100.0e3/hdim) & (zz<400e3/hdim) & (xx>400e3/hdim) & (xx<500e3/hdim)]=1450./Tm
'''
#for zz[:] > 0.5:
#    Told= Told + 0.01*np.random.random(np.shape(xx))  # Add random noise
#Told     = Told + np.random.binomial(2,0.25)
'''
plt.figure(2,figsize=(10,10))
plt.imshow(Told*Tm, 
                   extent=(0,w*1000.,0,h*500.),
                   clim=(Tlab,1.0*Tm),
                   interpolation='bilinear', 
                   cmap='jet')
plt.colorbar(orientation='horizontal', shrink=0.8)
plt.show()
'''
#plt.close(2)

cada=0.001 #10kyears
dt_1=cada/hdim**2*kappa*secinmyr
tt=5./hdim**2*kappa*secinmyr  #hasta 5My

nt=int(tt/dt_1)
print nt 
nplot=5
print nplot, 'nplot'
# timestepping
for it in range(1,nt):
    ################################################################################################################################################### 
   	# Stokes velocity
        ###################################################################################################################################################
        [pp,vx,vz] = Stokes2Dfunc(Raold, Told, xx, zz)
        #vx[:,0]=500000. #this imposed velocity is not being added to stokes velocity
        ###################################################################################################################################################
        # Calculate topography
        ###################################################################################################################################################        
        # Calculate topography
        #topo=-(2*vz[1,:]/dz-pp[0,:])*kappa*1e27/Ra[:,1]/4000/10/1000e3**2
        topo=-(2*vz[0,:]/dz-pp[0,:])*g*drho[0,:]/Raold[0,:]
        #topo =pp[0,:]*profile_len
        #avtopo=np.sum(topo)/np.size(topo)
        #topo = topo/max(topo)
        '''
        topo=-(2*vz[1,:]/dz-pp[0,:])*g*rho/Ra
        avtopo=np.sum(topo)/np.size(topo)
        topo = topo-avtopo
        topo2 = topo_ini+topo
        '''
        # Calculate next Courant timestep:
        vxmax    = (abs(vx)).max()
        vzmax    = (abs(vz)).max()
        dt_adv   = min(dx/vxmax,dz/vzmax)  # advection timestep
        dt       = 0.5*min(dt_diff, dt_adv)  # total timestep
	    
    
    ###################################################################################################################################################
	# Advect temeperature with velocities from Stokes 	
	###################################################################################################################################################
	# numerical solution
        Tnew = twoDadvdiff(Told,vx,vz,dx,dz,dt)
        Ranew = twoDadvdiff(Raold,vx,vz,dx,dz,dt)
        #update time
        t=t+dt
        ###################################################################################################################################################
        # Plotting part
        ###################################################################################################################################################
        if (it%nplot==0):
            tmyrs=round(t*hdim**2/kappa/secinmyr,2) # dimensional time in Myrs
            print ("Time :", tmyrs)
       	    fig1 = plt.figure(1)
            #fig1.set_size_inches(18.5, 10.5, forward=True)
            #fig1.clf()
            gs = gridspec.GridSpec(4, 1,height_ratios=[1,1,2,2])
            # Set up GridSpec
            #gs = gridspec.GridSpec(3, 1)
            ax1 = plt.subplot(gs[0]) # plt.subplot(gs[0:1, 2:])  # Temperature
            #ax2 =  plt.subplot2grid((4, 1), (1, 0)) #plt.subplot(gs[3:6, 6:])  # New density
            #gs = gridspec.GridSpec(3, 1,height_ratios=[1,1,3])
            #ax1 = plt.subplot(311) #(gs[0:3, 0:])  # Temperature
            ax1.plot(x*profile_len,vz[0,:])
            ax4 = plt.subplot(gs[1]) # plt.subplot(gs[0:1, 2:])  # Temperature
            ax4.plot(x*profile_len,pp[0,:])
            topo_ob_plt=topo_obs[:,1]/max(topo_obs[:,1])
            #ax1.plot(topo_obs[:,0],topo_obs[:,1]/1000)
            
            ax1.set_title('T after '+str(tmyrs)+' Myrs')
            ax1.set_ylabel('Topography (km)')
            #ax2 = plt.subplot(312) #(gs[0:3, 0:])  # Temperature
            #ax2.plot(x*hdim*1e-3,topo2)
            #plt.ylabel('Topography (km)')
            #ax3 = plt.subplot(313) #(gs[3:6, 0:])  # New density
            ax2= plt.subplot(gs[2])
            im2 = plt.imshow(Raold, 
                   extent=(0,h*profile_len,h*depth,0),
                   #clim=(Tlab,1.0*Tm),
                   interpolation='nearest', 
                   cmap='jet')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="2%", pad=0.2)
            cbar = plt.colorbar(im2, cax=cax)
            cbar.ax.set_title('Rayleigh Number',fontsize=11) 
            ax3 = plt.subplot(gs[3])  #plt.subplot(gs[1:, 0:1])  # Temp profile
            im1 = plt.imshow(Tnew*Tm, 
                   extent=(0,h*profile_len,h*depth,0),
                   #clim=(Tlab,1.0*Tm),
                   interpolation='nearest', 
                   cmap='jet')
            #ax3.invert_yaxis()
            #ax3.set_xticklabels(xtick_label,fontsize=12)
            #ax3.set_yticklabels(ytick_label,fontsize=12)
            CS= ax3.contour(Tnew*Tm, [200,600,800,1000,1200,1300,1400,1500], colors='white', origin='image', extent=(0,h*profile_len,h*depth,0))
            ax3.clabel(CS, inline=1, fontsize=10,fontcolor='white')

            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="2%", pad=0.2)
            cbar = plt.colorbar(im1, cax=cax,ticks=np.linspace(200,1550,6))
            cbar.ax.invert_yaxis()
            cbar.ax.set_title('$^{\circ}C$',fontsize=11)
            cbar.ax.tick_params(labelsize=12) 
            #ax3.axis('equal')
            ax3.set_ylabel('Depth (km)')
            ax3.set_xlabel('Distance (km)')
            #ax3.set_xticklabels(xtick_label,fontsize=12)
            #ax3.set_yticklabels(ytick_label,fontsize=12)
            #plt.colorbar(orientation='horizontal', shrink=0.8)
            #plt.quiver(xx*profile_len, (h-zz)*depth, vx, -vz, units='width')
            ax3.set_title('T after '+str(tmyrs)+' Myrs')
            #fig1.colorbar(neg, ax=ax3)
            #ax3.colorbar(orientation='horizontal', shrink=0.8) ## fix this colorbar thing
            Q = ax3.quiver(xx*profile_len, (h-zz)*depth, vx, -vz, units='width')
            #qk = ax3.quiverkey(Q, 0.2, 0.2, 2, r'$2 \frac{m}{s}$', labelpos='E',
            #       coordinates='figure')
            #ax3.quiver(xx*profile_len, (h-zz)*depth, vx, -vz, units='width')
            
            #
            #plt.tightlayout()
            fig1.set_tight_layout(True)
            plt.pause(0.0001)
            plt.draw()
        # prepare for next time step:
        Told = Tnew
        Raold = Ranew
