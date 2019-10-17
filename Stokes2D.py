#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#########################################################################
# This code consists of subroutines to solve 2D Stokes equation.        #
#=======================================================================#
# Author: Kittiphon Boonma, kboonma@ictja.csic.es                       #
# Created 09/04/2017                                                    #
#########################################################################
# Setup libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.sparse.linalg
####################################

#############################################################################
############ Session 4: Building a convection code ##########################
#############################################################################

# Subfunctions:
#===========================================================================#
def idp(ix,iz,nz):
    # purpose: find index value for the p component in 2-D Stokes matrix
    fout = 3*((ix-1)*nz + iz) - 3
    return fout
#===========================================================================#
def idvx(ix,iz,nz):
    # purpose: find index value for the vx component in 2-D Stokes matrix
    fout = 3*((ix-1)*nz + iz) - 2
    return fout
#===========================================================================#
def idvz(ix,iz,nz):
    # purpose: find index value for the vz component in 2-D Stokes matrix
    fout = 3*((ix-1)*nz + iz) - 1
    return fout
#===========================================================================#
def getpvxvz(sol,xx):
    # purpose: extract p, vx, and vz from sol vector from matrix inversion
    # sol contains [p1 vx1 vz1 p2 vx2 vz2 p3 vx3 vz3....]
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

#===========================================================================#
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
    
#    vxplot[:,:] = 0.5*vxplot[:,:]
#    vxplot[-5::,:] = 1*vxplot[-5::,:]
#    vzplot[:,:] = 0.005*vzplot
#    vzplot[-5::,:] = 0*vzplot[-5::,:]

    return [pplot,vxplot,vzplot]
#===========================================================================#

def Stokes2Dfunc(Ra, T, xx, zz):
    islip = 1  # 1=free-slip -1=no-slip

    [nz,nx] = np.shape(xx)
    nxz  = 3*nx*nz  # total nr of unknowns (nx * nz * (vx+vz+p))
    dx   = xx[0,1]-xx[0,0]
    dz   = zz[1,0]-zz[0,0]

    A    = scipy.sparse.lil_matrix((nxz,nxz)) # create and empty sparse matrix
    A.setdiag(np.ones(nxz))                 # set diagonal elements to 1
    rhs  = np.zeros(nxz)             # create rhs (buoyancy force) vector
    drho = np.zeros(np.shape(xx))    # create density distribution matrix
    drho[:,0:int(nx/2)]=-0.5         # Symmetric buoyancy for both odd and even
    drho[:,int(nx/2)+1:]=0.5         # number grids: if odd-> middle column rho=0
    Ra = 1e5                         # Rayleigh number

    # Fill in info in matrix for Stokes_z for internal points & left/right bnd. points:
    # Note: 1) other points (top/bottom bnd.pnts and ghstpnts have vz=0, which is default
    #       2) Index counters: ix=1 to nx-1, and iz=1 to nz (unusual for Python)
    for iz in range (2,nz):          # iz=1 & iz=nz are default (i.e. vz=0) bc's: no calc needed
        for ix in range (1,nx):      # ix=nx is ghostpoint ix=1 & nx-1 are boundary,
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

            # rhs: Ra*drho'
            # rhs: Ra*T'
            avT  = 0.5*(T[iz-1,ix-1]+T[iz-1,ix])
            rhs[irow] = -avT*Ra
            #avdrho  = 0.5*(drho[iz-1,ix-1]+drho[iz-1,ix])
            #rhs[irow] = avdrho*Ra

    # Fill in info in matrix for Stokes_x for internal points & top/bottom bnd. points:
    # Note: other points (left/right bnd.pnts and ghstpnts have vx=0, which is default
    for ix in range (2,nx):          # ix=1 & nx are default (i.e. vx=0) bc's: no calc, needed
        for iz in range (1,nz-3):      # iz=nz are ghostpoints, iz=1&nz-1 are boundaries,
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
    for ix in range (2,nx+1):       # pressure point ix=1 is a ghostpoint
        for iz in range (2,nz+1):   # pressure point iz=1 is a ghostpoint
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
    
    ################################# TEST ADDITION #######################

    
    ########################################################################

    # Solve system:
    sol=scipy.sparse.linalg.spsolve(A,rhs)
    # extrac p, vx, and vz solutions from solution vector:
    [pp,vx,vz] = getpvxvz(sol,xx)


    
#    ########################################################################    
#    zz = zz[::-1]
#    [mz,mx] = np.shape(xx)
#    for i in range(mx):
##            T_old[np.int(dmoho/dz):np.int(dlab/dz),i] = np.linspace(Tmoho,Tlab,(slab_thickness)/dz)
##            T_old[np.int(dlab/dz):np.int(h/dz)+1,i] = np.linspace(Tlab,Tbottom,((h-dlab)/dz)+1)
#      for j in range(mz): 
#          #Background temp. (asth) gradient and vel. for the whole box
##              if (xx[j,i]>=0 and xx[j,i]<=w \
##                and zz[j,i]>=0 and zz[j,i]<=h): 
##                  T_old[j,i] = 273
#        if (xx[j,i]>=0 and xx[j,i]<=w \
#            and zz[j,i]>=0 and zz[j,i]<=h): 
#              #T_old[j,i] = Tbottom-dT_asth*(h-zz[j,i])
#              #T_old[j,i] = Tlab+dT_asth*(dlab+zz[j,i])
#              vx[j,i] = 0*xh[j,i]
#              vz[j,i] = 0*zh[j,i]
#          ############### Temperature setup #########################
#          #Lithospheric mantle layer
##              if (xx[j,i]>=0 and xx[j,i]<=w \
##                  and zz[j,i]>=dmoho and zz[j,i]<=dlab):
##                  T_old[j,i] = Tlab-dT_lith*(dlab-zz[j,i])
#              #T_old[j,i] = Tmoho+dT_lith*(dmoho+zz[j,i])
#              
#          ############## Initial velocity setup ##################            
#  #              # Right side - Lithospheric slab
#  #              if (zz[j,i] >= dmoho and zz[j,i] <= dlab \
#  #                  and xx[j,i] >= 470000+(zz[j,i]-dmoho)): 
#  #                  vx[j,i] = 0*xh[j,i]
#  #                  vz[j,i] = 0*zh[j,i]
#  #                  
#          # Left side - Lithospheric slab
#        if (zz[j,i] >= dmoho and zz[j,i] <= dlab \
#              and xx[j,i] <= hinge_ax-(zz[j,i]-dmoho)/gradient1):
#              vx[j,i] = 0*xh[j,i]+pres_vel
#              vz[j,i] = 0*zh[j,i]
#          # Triangular block (joint)
#        if (zz[j,i] >= dmoho and zz[j,i] <= dlab \
#              and xx[j,i] >= hinge_ax-(zz[j,i]-dmoho)/gradient1 \
#              and xx[j,i] <= hinge_ax+(zz[j,i]-dmoho)/gradient2):
#              vx[j,i] = 0*xh[j,i]+pres_vx
#              vz[j,i] = 0*zh[j,i]-pres_vz                   
#          # Hanging lithospheric slab    
#        if (zz[j,i] >= dlab and zz[j,i] <= dlab \
#              and xx[j,i] <= hinge_bx+wid+(zz[j,i]-dlab)/gradient2 \
#              and xx[j,i] >= hinge_bx+(zz[j,i]-dlab)/gradient2):
#              vx[j,i] = 0*xh[j,i]+pres_vx
#              vz[j,i] = 0*zh[j,i]-pres_vz
#  
#          ###########################################################
#          
#    zz = zz[::-1]
#    return [vx,vz,xx,zz]
          ########################################################################

    vxmax=vx.max()
    vzmax=vz.max()
    print('vxmax= %e' %vxmax)
    print('vzmax= %e' %vzmax)
    print('=========================')

    # preparing p, vx, and vz for plotting:
    [pplot,vxplot,vzplot]=preppvxvzplot(pp,vx,vz,xx,islip)
    plt.figure(1)
    #plt.clf()
    plt.subplot(2, 1, 2)
    plt.imshow(pplot,
               extent=(0,1,0,1),
               #clim=(0,Tm),
               interpolation='bilinear',
               cmap='jet')
    plt.gca().invert_yaxis()
    plt.xlabel('Distance in x-direction [km] $\longrightarrow$')
    plt.ylabel('$\longleftarrow$ Depth [km]')
    plt.quiver(xx, zz[::-1], vxplot, vzplot, units='width')
    plt.title('Stokes flow')
    plt.pause(0.0001)
    #plt.draw()

    return [pplot,vxplot,vzplot]