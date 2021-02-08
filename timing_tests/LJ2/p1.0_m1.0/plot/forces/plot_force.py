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

def fCDD_solv_z(q1,q2,p1,p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    z = p2[2] - p1[2]
    r = np.sqrt(x*x+y*y+z*z)
    diel = 2.3473
    qConv = 18.2223**2
    f = 332.*q1*q2*(1.-1./2.35)*z/(r**3)
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
q2 = -1.0
p1 =(0,0,0)

dat = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p1.0_m1.0/12/force/rex/LJ2.qp1.00.qm1.00.IS-SPA.dn0.21.dat")
data = np.zeros((len(dat),3),dtype=float)
for i in range(len(dat)):
    data[i,0] = dat[i,0]
    data[i,1] =  dat[i,3]-q1*q2*fC(dat[i,0])
    data[i,2] =  dat[i,4]-q1*q2*fC(dat[i,0])

# Plot the forces from rex                                                                                                                                                          
#plt.plot(data[:,0], data[:,1], c = 'b', label = "+1.0 ion - IS-SPA")
#plt.plot(data[:,0], data[:,2], c = 'r', label = "-1.0 ion - IS-SPA")


#vals = (12,15,17,20,25)
vals = (12,17,25)
colors = ("c","m","g",'b','r')
# Plot the dimer forces
for j,val in enumerate(vals):
    tf = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p1.0_m1.0/%s/LJ2.c_forces.combined.dat" %(val))
    plt.errorbar(tf[:,0], -tf[:,1], yerr=tf[:,2], c = colors[j], linestyle='--', linewidth = 1.0)
    plt.errorbar(tf[:,0], tf[:,3], yerr=tf[:,4], c = colors[j], label="$r_{cut}$ = %s" %(val), linewidth = 1.0)

# Plot CDD
x = np.arange(2.0,50.1,0.1)
nBins = len(x)
CDD_data = np.zeros((nBins,2),dtype=float)
for i in range(nBins):
    z12 = x[i]
    CDD_data[i,0] = z12
    p2 = (0,0,z12)
    CDD_data[i,1] -= fCDD_solv_z(q1,q2,p1,p2)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "$f_{CDD}$")            
    
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f \right >$', size=12)
plt.legend(loc='upper right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 51, 5.0))
plt.xlim((0,50))
plt.ylim((0.0,0.50))
plt.savefig('p1.0_m1.0.cforces.pdf')
plt.savefig('p1.0_m1.0.cforces.png')
plt.close()


# Plot the dimer forces
for j,val in enumerate(vals):
    tf = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p1.0_m1.0/%s/LJ2.forces.combined.dat" %(val))
    plt.errorbar(tf[:,0], tf[:,3], yerr=tf[:,4], c = colors[j], label="$r_{cut}$ = %s" %(val), linewidth = 1.0)

# Plot CDD + LJ force
x = np.arange(2.0,50.1,0.1)
nBins = len(x)
CDD_data = np.zeros((nBins,2),dtype=float)
for i in range(nBins):
    z12 = x[i]
    CDD_data[i,0] = z12
    p2 = (0,0,z12)
    CDD_data[i,1] += fLJ_z(p1,p2)
    CDD_data[i,1] += fCDD_z(q1,q2,p1,p2)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "$f_{CDD} + f_{LJ}$")            
    
#plt.title(r"$R^{\parallel}_{13}$ = %s $\AA$" %(val))
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f \right >$', size=12)
plt.legend(loc='lower right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 51, 5.0))
plt.xlim((0,50))
plt.ylim((-7.00,2.0))
plt.savefig('p1.0_m1.0.tforces.pdf')
plt.savefig('p1.0_m1.0.tforces.png')
plt.close()

