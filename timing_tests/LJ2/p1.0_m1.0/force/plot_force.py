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
name="p1.0_m1.0"
# Create CDD array                                                                                                     
x = np.arange(2.0,60.1,0.1)
nBins = len(x)
CDD_data = np.zeros((nBins,2),dtype=float)
for i in range(nBins):
    z12 = x[i]
    CDD_data[i,0] = z12
    p2 = (0,0,z12)
    CDD_data[i,1] -= fCDD_solv_z(q1,q2,p1,p2)

###########################
####  Plot Diff Force  ####
###########################
# Plot the dimer differences for enow+e0now forces                                                                                                                           
enow_25 = np.loadtxt("../25/LJ2.enow.combined.dat")
e0now_25 = np.loadtxt("../25/LJ2.e0now.combined.dat")
vals = (12,15,17,20)
colors = ("c","m","g",'b','r')
for j,val in enumerate(vals):
    enow = np.loadtxt("../%s/LJ2.enow.combined.dat" %(val))
    e0now = np.loadtxt("../%s/LJ2.e0now.combined.dat" %(val))
    plt.plot(enow[:,0], -(enow[:,1]+e0now[:,1])+(enow_25[:,1]+e0now_25[:,1]), c = colors[j], linewidth = 1.0, label="%s - 25" %(val))
    plt.plot(enow[:,0], enow[:,3]+e0now[:,3]-enow_25[:,3]-e0now_25[:,3], c = colors[j], linewidth = 1.0, linestyle='--')
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f_{no \ pair} \right >$', size=12)
plt.legend(loc='lower right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 61, 5.0))
plt.xlim((0,60))
#plt.ylim((-0.15,0.02))
plt.savefig('%s.diff.pdf' %(name))
plt.savefig('%s.diff.png' %(name))
plt.close()

vals = (12,25)
colors = ("c","m","g",'b','r')
###########################
####  Plot pair Force  ####
###########################
# Plot the dimer forces
for j,val in enumerate(vals):
    tf = np.loadtxt("../%s/LJ2.pair.combined.dat" %(val))
    plt.errorbar(tf[:,0], -tf[:,1], yerr=tf[:,2], c = colors[j], linestyle='--', linewidth = 1.0)
    plt.errorbar(tf[:,0], tf[:,3], yerr=tf[:,4], c = colors[j], label="$r_{cut}$ = %s" %(val), linewidth = 1.0)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "$f_{CDD}$")    
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f_{dir} \right >$', size=12)
plt.legend(loc='upper right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 61, 5.0))
plt.xlim((0,60))
plt.ylim((0.0,0.5))
plt.savefig('%s.pair.pdf' %(name))
plt.savefig('%s.pair.png' %(name))
plt.close()


###########################
###  Plot e0now Force  ####
###########################
# Plot the dimer forces
for j,val in enumerate(vals):
    tf = np.loadtxt("..//%s/LJ2.e0now.combined.dat" %(val))
    plt.errorbar(tf[:,0], -tf[:,1], yerr=tf[:,2], c = colors[j], linestyle='--', linewidth = 1.0)
    plt.errorbar(tf[:,0], tf[:,3], yerr=tf[:,4], c = colors[j], label="$r_{cut}$ = %s" %(val), linewidth = 1.0)
#plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "$f_{CDD}$")    
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f_{e0now} \right >$', size=12)
plt.legend(loc='lower right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 61, 5.0))
plt.xlim((0,60))
#plt.ylim((0.0,1.30))
plt.savefig('%s.e0now.pdf' %(name))
plt.savefig('%s.e0now.png' %(name))
plt.close()

###########################
####  Plot enow Force  ####
###########################
# Plot the dimer forces
for j,val in enumerate(vals):
    tf = np.loadtxt("..//%s/LJ2.enow.combined.dat" %(val))
    plt.errorbar(tf[:,0], -tf[:,1], yerr=tf[:,2], c = colors[j], linestyle='--', linewidth = 1.0)
    plt.errorbar(tf[:,0], tf[:,3], yerr=tf[:,4], c = colors[j], label="$r_{cut}$ = %s" %(val), linewidth = 1.0)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "$f_{CDD}$")    
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f_{enow} \right >$', size=12)
plt.legend(loc='upper right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 61, 5.0))
plt.xlim((0,60))
plt.ylim((0.0,5.50))
plt.savefig('%s.enow.pdf' %(name))
plt.savefig('%s.enow.png' %(name))
plt.close()

    
###########################
## Plot Coulombic Force  ##
###########################
# Plot the dimer forces
for j,val in enumerate(vals):
    tf = np.loadtxt("..//%s/LJ2.c_forces.combined.dat" %(val))
    plt.errorbar(tf[:,0], -tf[:,1], yerr=tf[:,2], c = colors[j], linestyle='--', linewidth = 1.0)
    plt.errorbar(tf[:,0], tf[:,3], yerr=tf[:,4], c = colors[j], label="$r_{cut}$ = %s" %(val), linewidth = 1.0)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "$f_{CDD}$")    
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f_{coul} \right >$', size=12)
plt.legend(loc='upper right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 61, 5.0))
plt.xlim((0,60))
plt.ylim((0.0,5.50))
plt.savefig('%s.cforces.pdf' %(name))
plt.savefig('%s.cforces.png' %(name))
plt.close()

###########################
## Plot Coulombic Force  ##
###########################
# Plot CDD + LJ force                                                                                                                                                        
x = np.arange(2.0,60.1,0.1)
nBins = len(x)
CDD_data = np.zeros((nBins,2),dtype=float)
for i in range(nBins):
    z12 = x[i]
    CDD_data[i,0] = z12
    p2 = (0,0,z12)
    CDD_data[i,1] += fLJ_z(p1,p2)
    CDD_data[i,1] += fCDD_z(q1,q2,p1,p2)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "$f_{CDD} + f_{LJ}$")
vals = (12,25)
colors = ("c","m","g",'b','r')
# Plot the dimer forces
for j,val in enumerate(vals):
    tf = np.loadtxt("../%s/LJ2.forces.combined.dat" %(val))
    plt.errorbar(tf[:,0], tf[:,3], yerr=tf[:,4], c = colors[j], label="$r_{cut}$ = %s" %(val), linewidth = 1.0)
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
plt.ylabel(r'$\left < f_{tot} \right >$', size=12)
plt.legend(loc='lower right',  shadow=True, ncol=1, fontsize = 'medium')
plt.xticks(np.arange(0, 61, 5.0))
plt.xlim((0,60))
plt.ylim((-6.5,1.0))
plt.savefig('%s.tforces.pdf' %(name))
plt.savefig('%s.tforces.png' %(name))
plt.close()

