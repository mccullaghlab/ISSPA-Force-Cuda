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
q2 = -1.0
q3 = 0.5
values = (3.5,5.0,7.0,10.0,12.0,15.0)
nVals = len(values)
cutoff = 12.0
cutoff = 25.0
#cutoff = 49.9
step = 2.0

# Create trimer force array
tf = np.loadtxt("../LJ3.forces.combined.dat",usecols=(1,4))
f_tri = tf.reshape((nVals,int(len(tf)/nVals),2))

# Create trimer force array with 2-3 isspa forces removed
tf = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ3_23/p1.0_m1.0_n0.0/parallel/LJ3.forces.combined.dat",usecols=(1,7))
f12 = tf.reshape((nVals,int(len(tf)/nVals),2))

# Plot PMF values from isspa code                                                                                                             
for j,val in enumerate(values):
    p1 = (0,0,0)
    p3 = (0,0,-val)

    # Plot the trimer forces
    plt.plot(f_tri[j,:,0], f_tri[j,:,1], c = 'k', label = "$f^{all}$")

    # Plot CDD + LJ force
    x = np.arange(2.0,50.0,0.1)
    nBins = len(x)
    CDD_data = np.zeros((nBins,2),dtype=float)
    for i in range(nBins):
        z12 = x[i]
        CDD_data[i,0] = z12
        p2 = (0,0,z12)
        #print(cdd_force(z12,val)+lj_force(z12)+lj_force(z12+val),fLJ_z(p1,p2) + fLJ_z(p3,p2) + fCDD_z(q1,q2,p1,p2) + fCDD_z(q2,q3,p3,p2))
        CDD_data[i,1] += fLJ_z(p1,p2) + fLJ_z(p3,p2)
        CDD_data[i,1] += fCDD_z(q1,q2,p1,p2) + fCDD_z(q2,q3,p3,p2)
    plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'r', label = "$f_{CDD} + f_{LJ}$", linestyle='--')
        
    # Plot the dimer forces
    tf = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p0.5_m1.0/LJ2.forces.combined.dat")
    nBins = len(tf)
    f = np.zeros((nBins,2),dtype=float)
    for i in range(nBins):
        f[i,0] = tf[i,0]
        f[i,1] += tf[i,3]
        z12 = f[i,0]
        p2 = (0,0,z12)    
        f[i,1] +=  fCDD_z(q2,q3,p3,p2) + fLJ_z(p3,p2)
    #plt.plot(f[:,0], f[:,1], c = 'b', label = "$f^{12}$ + $f_{CDD}^{23}$ + $f_{LJ}^{23}$", linestyle='--')
    plt.plot(f[:,0], f[:,1], c = 'b', label = "$f^{12}$ + $f_{CDD}^{23}$ + $f_{LJ}^{23}$")
        

    ## Plot the trimer forces minues 2-3 isspa forces
    #nBins = len(f12[j])
    #for i in range(nBins):
    #    z12 = f12[j,i,0]
    #    p2 = (0,0,z12)    
    #    f12[j,i,1] +=  fCDD_z(q2,q3,p3,p2) + fLJ_z(p3,p2)
    #plt.plot(f12[j,:,0], f12[j,:,1], c = 'g', label = "$f^{all}$ - $f_{IS-SPA}^{23}$", linestyle='--')

    
    
    
    plt.title(r"$R^{\parallel}_{13}$ = %s $\AA$" %(val))
    plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
    plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
    plt.ylabel(r'$\left < f \right >$', size=12)
    plt.legend(loc='lower right',  shadow=True, ncol=1, fontsize = 'medium')
    plt.xticks(np.arange(0, cutoff+1, step))
    plt.xlim((0,cutoff))
    plt.ylim((-7.0,2.0))
    plt.savefig('p0.5_m1.0_p0.5.%s.zz.forces.pdf' %(val))
    plt.savefig('p0.5_m1.0_p0.5.%s.zz.forces.png' %(val))
    plt.close()

