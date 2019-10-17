from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def interpolate_2D(x,y,z,profile_len,reso,step):
	f = RegularGridInterpolator((x, y), z) #interpolate.interp2d(y,x,z,kind='linear')
	return f


data= np.loadtxt('post_processing_output.dat',usecols=(0,1,2,3,4,5,6))
reso = 5
profile_len=625 
step=0.01
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
    T_.append(data[temp,2])
    D_.append(data[temp,6])
    temp=temp+1
  T.append(T_)
  D.append(D_)
T_func= interpolate_2D(x_,y_,T,profile_len,5,step)
D_func= interpolate_2D(x_,y_,D,profile_len,5,step)

###################################################################################################################################################
# Initialisation:
###################################################################################################################################################
# Mesh setup:
###################################################################################################################################################
h        = 1.0                 # nondimensional box height
w        = 1.0                 # box of aspect ratio 1
dx       = step                # discretization step in meters
dz       = step
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

# Dimensional variables:
kappa    = 1e-6                # thermal diffusivity
Tm       = 1650                # mantle temperature in degC
Tlab     = 1320
deltaT   = Tm-Tlab
g        = 9.81
alpha    = 3e-5                # K-1
hdim     = 1000e3              # dimensional height of box: 1000 km
eta      = 1e22                # How can I use a none constant viscosity?
rho      = 3400.              


###################################################################################################################################################
# initial density distribution (can imposrted from LiMod)
###################################################################################################################################################
             # create rhs (buoyancy force) vector
drho = 1.*np.ones(np.shape(xx))
Told     = 1.*np.ones(np.shape(xx))
m,n=np.shape(xx)
for i in range(m-1):
	for j in range(n-1):
		#print xx[i,j],zz[i,j]
		Told[i,j]=T_func((xx[i,j],zz[i,j]))
		drho[i,j]=D_func([xx[i,j],zz[i,j]])

### making X,Y grid
fig=plt.figure()
ax1= plt.subplot(221)
X,Y=np.meshgrid(y_,x_)
ax1.contourf(Y,X,T,40,cmap='RdGy_r')
plt.gca().invert_yaxis()


ax2= plt.subplot(222)
ax2.contourf(Told,40,cmap='RdGy_r')
plt.gca().invert_yaxis()
#plt.colorbar()

ax3= plt.subplot(223)
X,Y=np.meshgrid(y_,x_)
ax3.contourf(Y,X,D,40,cmap='RdGy_r')
plt.gca().invert_yaxis()

ax4= plt.subplot(224)
plt.contourf(drho,40,cmap='RdGy_r')
plt.gca().invert_yaxis()
#plt.colorbar()
#plt.plot(melt_mantle)
plt.show()


'''
drho = np.zeros(np.shape(xx))    # create density distribution matrix
drho = 3300.*np.ones(np.shape(xx))
drho[0:int(nx/3),:]=3300        # Symmetric buoyancy for both odd and even 
drho[int(nx/3):int(nx/2),:]=3300 

###################################################################################################################################################
# Raleight number; At the moment calculated with constant density; Will adapt this part where it will be calculated using density distribution from 
# LitMod
###################################################################################################################################################
Ra  = np.zeros(nxz)
Ra  = (alpha*rho*g*Tm*(hdim**3))/(eta*kappa)
print Ra,'Ra'

'''



