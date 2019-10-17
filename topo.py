#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import pylab as plt
from Stokes2D import Stokes2Dfunc 

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

# Main code: 
# Initialisation:
# Dimensional variables:
kappa    = 1e-6                # thermal diffusivity
Tm       = 1350                # mantle temperature in degC
Ra       = 1e4
hdim     = 1000e3              # dimensional height of box: 1000 km

# Mesh setup:
h        = 1.0                 # nondimensional box height
w        = 1.0                 # box of aspect ratio 1
dx       = 0.05                # discretization step in meters
dz       = 0.05
nx       = w/dx+1
nz       = h/dz+1         
dx       = w/(nx-1)            # Adjust requested dx & dz to fit in equidistant grid space
dz       = h/(nz-1) 
x        = np.linspace(0,w,nx) # array for the finite difference mesh
z        = np.linspace(0,h,nz)
[xx,zz]  = np.meshgrid(x,z)

# Time variables:
dt_diff  = 0.2*dx**2           # timestep in Myrs
nt       = 500                # number of tsteps
secinmyr = 1e6*365*24*3600     # amount of seconds in 1 Myr
t        = 0                   # set initial time to zero
nplot    = 5                   # plotting interval: plot every nplot timesteps

# Initial condition:

Ttop     = 0                   # surface T
Told=0.5*np.ones(np.shape(xx)) # Initial temperature T=0.5 everywhere
Told = Told + 0.1*np.random.random(np.shape(xx))  # Add random noise
Told[0,:]=0.                   # Set top and bottom T to 0 and 1 resp.
Told[-1,:]=1.0

nplot    = 20                 # Plot every nplot timesteps

# timestepping
for it in range(1,nt):
    # Stokes velocity
    [pp,vx,vz] = Stokes2Dfunc(1e4, Told, xx, zz)

    # Calculate topography
    topo=-(2*vz[1,:]/dz-pp[0,:])*kappa*1e27/Ra/4000/10/1000e3**2
    avtopo=np.sum(topo)/np.size(topo)
    topo = topo-avtopo
    
    # Calculate next Courant timestep:
    vxmax    = (abs(vx)).max()
    vzmax    = (abs(vz)).max()
    dt_adv   = min(dx/vxmax,dz/vzmax)  # advection timestep
    dt       = 0.5*min(dt_diff, dt_adv)  # total timestep
    
    # numerical solution
    Tnew = twoDadvdiff(Told,vx,vz,dx,dz,dt)

    #update time
    t=t+dt

    # plot solution:
    if (it%nplot==0):
        tmyrs=int(t*hdim**2/kappa/secinmyr)   # dimensional time in Myrs
        plt.figure(1)                         # T-v plot                       
        plt.clf()
        plt.imshow(Tnew, 
                   extent=(0,h,0,h),
                   clim=(0,1.0),
                   interpolation='bilinear', 
                   cmap='jet')
        plt.quiver(xx, h-zz, vx, -vz, units='width')
        plt.title('T after '+str(tmyrs)+' Myrs')
        plt.pause(0.00001)
        plt.figure(2)                        # Topography plot
        plt.clf()
        plt.plot(x*hdim*1e-3,topo)
        plt.xlabel('x(km)')
        plt.ylabel('topography(m)')
        plt.title('Topography')
        plt.pause(0.00001)
        
    # prepare for next time step:
    Told = Tnew