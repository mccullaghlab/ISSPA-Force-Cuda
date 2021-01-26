import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm

stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def uCDD(x):
        f = -332./2.35/x
        return f

def uLJ(x):
        f=0.152*(7./x)**6*((7./x)**6-2.)
        return f


values = (3.5,5.0,7.0,10.0,12.0,15.0)
#values = (3.5,5.0)
j = 3
colors = ["k","r","b","g","c","m"]
# Plot PMF values from isspa code
for i,val in enumerate(values):
        data = np.loadtxt("pmf.%s.dat" %(val))
        data[:,j] += uCDD(data[-1,1])
        plt.plot(data[:,1], data[:,j], c = colors[i], label = "$R_{z}$ = %s $\AA$" %(val))

# Plot CDD + LJ PMF
CDD_data = np.zeros((len(data),2),dtype=float)
for i in range(len(data)):
        x = data[i,1]
        CDD_data[i,0] = x
        CDD_data[i,1] = uCDD(x) + uLJ(x)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "CDD + LJ Dimer", linestyle='--')


# Plot Dimer PMF plus CDD from 3rd atom
DCDD_data = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p0.5_m1.0/pmf.dat")
for  i in range(len(DCDD_data)):
        DCDD_data[i,1] = (-DCDD_data[i,1] + DCDD_data[i,2])/2.0
plt.plot(DCDD_data[:,0], DCDD_data[:,1], c = 'g', label = "Dimer + CDD$_{3}$", linestyle='--')

        
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{z}$ $(\AA)$', size=12)
plt.ylabel(r'$u_{pmf}$ $(kcal \cdot mol^{-1})$', size=12)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),  shadow=True, ncol=3, fontsize = 'medium')
plt.xticks(np.arange(0, 16, 1.0))
plt.xlim((0,15))
plt.ylim((-40.0, 5.0))
plt.savefig('PMP.zz.pmf.pdf')
plt.savefig('PMP.zz.pmf.png')
plt.close()

