import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def fCDD_z(q1,q2,p1,p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    z = p2[2] - p1[2]
    r = np.sqrt(x*x+y*y+z*z)
    diel = 2.3473
    qConv = 18.2223**2
    f = qConv/diel*q1*q2*z/(r**3)
    return f

def fLJ_z(p1,p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    z = p2[2] - p1[2]
    r = np.sqrt(x*x+y*y+z*z)
    sig = 7.0/np.power(2.0,1./6.)
    f = 24*0.152*z*(sig**-2)*(sig/r)**(8)*(2*(sig/r)**6-1)
    return f

q1 = 0.5
q2 = -0.5
p1 = (0,0,0)

# Plot the dimer forces
tf = np.loadtxt("../LJ2.forces.combined.dat")
nBins = len(tf)
f = np.zeros((nBins,2),dtype=float)
for i in range(nBins):
    f[i,0] = tf[i,0]
    f[i,1] += tf[i,6]
plt.plot(f[:,0], f[:,1], c = 'k', label = "$f^{12}$")

# Plot CDD + LJ force
x = np.arange(2.0,25.1,0.1)
nBins = len(x)
CDD_data = np.zeros((nBins,2),dtype=float)
for i in range(nBins):
    z12 = x[i]
    CDD_data[i,0] = z12
    p2 = (0,0,z12)
    CDD_data[i,1] += fLJ_z(p1,p2)
    CDD_data[i,1] += fCDD_z(q1,q2,p1,p2)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'r', label = "$f_{CDD} + f_{LJ}$", linestyle='--')            
    
#plt.title(r"$R^{\parallel}_{13}$ = %s $\AA$" %(val))
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f \right >$', size=12)
plt.legend(loc='lower right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 26, 1.0))
#plt.xlim((0,15))
#plt.ylim((-5.0,2.5))
plt.xlim((1,25))
plt.ylim((-3,2.0))
plt.savefig('p0.5_m0.5.forces.pdf')
plt.savefig('p0.5_m0.5.forces.png')
plt.close()

# Plot the dimer forces
tf = np.loadtxt("../LJ2.c_forces.combined.dat")
nBins = len(tf)
f = np.zeros((nBins,3),dtype=float)
for i in range(nBins):
    f[i,0] = tf[i,0]
    f[i,1] += tf[i,3]
    f[i,2] += tf[i,6]
plt.plot(f[:,0], -f[:,1], c = 'k', label = "+0.5 ion - IS-SPA")
plt.plot(f[:,0], f[:,2], c = 'r', label = "-0.5 ion - IS-SPA")

plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f \right >$', size=12)
plt.legend(loc='upper right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 26, 1.0))
plt.xlim((0,25))
plt.ylim((0.0,1.0))
plt.savefig('p0.5_m0.5.cforces.pdf')
plt.savefig('p0.5_m0.5.cforces.png')
plt.close()

