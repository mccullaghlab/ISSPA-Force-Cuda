import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def fCDD(q1,q2,r):
    diel = 2.3473
    qConv = 18.2223**2
    f = qConv*q1*q2*(1-(1./diel))*(2./3.)/(r**2)
    return f

def fC(q1,q2,r):
    qConv = 18.2223**2
    f = qConv*q1*q2/(r**2)
    return f


q1 = 1.0
q2 = -1.0

rcut = 12.0
rcut2 = 25.0
# Plot CDD + LJ force
x = np.arange(2.0,100.1,0.1)
nBins = len(x)
CDD_data = np.zeros((nBins,3),dtype=float)
for i in range(nBins):
    z12 = x[i]
    CDD_data[i,0] = z12
    CDD_data[i,1] += fC(q1,q2,z12)
    CDD_data[i,2] += fC(q1,q2,z12)
    if z12 > 2*rcut:
        CDD_data[i,1] += fCDD(q1,q2,z12)
    if z12 > 2*rcut2:
        CDD_data[i,2] += fCDD(q1,q2,z12)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'r', label = "(r_{cut} > 12.0)$", linestyle='--')            
plt.plot(CDD_data[:,0], CDD_data[:,2], c = 'r', label = "(r_{cut} > 25.0)$", linestyle='--')            
    

plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f \right >$', size=12)
plt.legend(loc='upper right',  shadow=True, ncol=1, fontsize = 'medium')
#plt.xticks(np.arange(0, 26, 1.0))
plt.xlim((12,100))
#plt.ylim((0.0,6.0))
plt.savefig('p1.0_m1.0.cforces.pdf')
plt.savefig('p1.0_m1.0.cforces.png')
plt.close()

