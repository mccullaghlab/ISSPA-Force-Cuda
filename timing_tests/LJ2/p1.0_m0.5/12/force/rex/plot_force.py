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

def fC(x):
    eps=2.35
    rc=24.95
    f = 332.*(1.-1./eps)*(x*(8.*rc-3.*x)/24./rc**4-np.log(1.+x/rc)/8./x**2+rc/8./x/(x+rc)**2-x**2/32./rc**4+3./16./rc**2)
    return f

q1 = 1.0
q2 = -0.5
p1 = (0,0,0)

#plot '../LJ2.qp0.50.qm1.00.IS-SPA.dn0.21.dat' u 1:($4+0.5*fC($1)) w l lw 3 dt 1 lc rgb 'blue' t '$+0.5$ ion - IS-SPA',\
#   '' u 1:($5+fC($1)) w l lw 3 dt 1 lc rgb 'red'  t '$-1.0$ ion - IS-SPA',\
#   332.*0.5*(1.-1./2.35)/x**2 lc rgb "black" lw 3 dt 1 t 'CDD - $\epsilon = 2.35$'
dat = np.loadtxt("LJ2.qp1.00.qm0.50.IS-SPA.dn0.21.dat")
data = np.zeros((len(dat),3),dtype=float)
for i in range(len(dat)):
    data[i,0] = dat[i,0]
    data[i,1] =  dat[i,3]-q1*q2*fC(dat[i,0])
    data[i,2] =  dat[i,4]-q1*q2*fC(dat[i,0])
    
# Plot the forces from rex
plt.plot(data[:,0], data[:,1], c = 'b', label = "+1.0 ion - IS-SPA")
plt.plot(data[:,0], data[:,2], c = 'r', label = "-0.5 ion - IS-SPA")


# Plot the dimer forces
tf = np.loadtxt("../../LJ2.c_forces.combined.dat")
nBins = len(tf)
f = np.zeros((nBins,3),dtype=float)
for i in range(nBins):
    f[i,0] = tf[i,0]
    f[i,1] += tf[i,3]
    f[i,2] += tf[i,6]
plt.plot(f[:,0], -f[:,1], c = 'k', label = "+1.0 ion - cuda")
plt.plot(f[:,0], f[:,2], c = 'g', label = "-0.5 ion - cuda")

plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f \right >$', size=12)
plt.legend(loc='upper right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 26, 1.0))
plt.xlim((0,25))
plt.ylim((0.0,2.0))
plt.savefig('p1.0_m0.5.cforces.pdf')
plt.savefig('p1.0_m0.5.cforces.png')
plt.close()

