
# load libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

data = np.loadtxt("timing.dat")

plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color='red',ecolor="gray",capsize=2,elinewidth=1,label="1xGPU Titan Xp")
plt.errorbar(data[:,0],data[:,3],yerr=data[:,4],color='blue',ecolor="gray",capsize=2,elinewidth=1,label="1xCPU Intel i7")
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel('Number of Atoms', size=12)
plt.ylabel("IS-SPA Force Compute Time (ms)", size=12)
plt.legend(loc=2)
plt.savefig('isspa_force_gpu_timing_natoms.png',dpi=300)
plt.close()


error = np.sqrt( (data[:,2]/data[:,1])**2 + (data[:,4]/data[:,3])**2 )
plt.errorbar(data[:,0],data[:,3]/data[:,1],yerr=error,color='red',ecolor="gray",capsize=2,elinewidth=1,label="1xGPU Titan Xp")
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel('Number of Atoms', size=12)
plt.ylabel("GPU Speed-up Factor", size=12)
plt.savefig('isspa_force_gpu_enhancement.png',dpi=300)
plt.close()

