import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def interpolate_2D(x,y,z,profile_len,reso,step):
	print('Model info \n')
	#reso=input('Enter resolution along the profile: \n')
	#reso=5
	#profile_len=input('Enter resolution the profile length: \n')
	#profile_len=625 
	f = RegularGridInterpolator((x, y), z) #interpolate.interp2d(y,x,z,kind='linear')
	x_new = np.arange(min(x_),max(x_),step=step)
	y_new = np.arange(min(y_),max(y_),step=step)
	points = np.array([x_new,y_new])
	z_new = f(points)
	return x_new,y_new,z_new


data= np.loadtxt('post_processing_output.dat',usecols=(0,1,2,3,4,5,6))
reso = 5
profile_len=625 
y_=-data[0:95,1]
x_=range(0,profile_len,reso)
print len(x_)
print len(y_)
Z=[]
temp=0
for i in range(len(x_)):
  tmp = []
  for j in range(len(y_)):
    tmp.append(data[temp,2])
    temp=temp+1
  Z.append(tmp)

### making X,Y grid
X,Y=np.meshgrid(y_,x_)
plt.contourf(Y,X,Z,40,cmap='RdGy_r')

x_new,y_new,Z_inter= interpolate_2D(x_,y_,Z,625,5,10)
plt.contourf(Z_inter,40,cmap='RdGy_r')


plt.gca().invert_yaxis()
plt.colorbar()
#plt.plot(melt_mantle)
plt.show()