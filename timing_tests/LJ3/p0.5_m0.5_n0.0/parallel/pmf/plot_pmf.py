import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm

stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def uCDD(q1,q2,z):
        f = q1*q2*332./2.35/z
        return f

def uCDD3z(q2,q3,z12,z13):
        z23 = z12 + z13
        f = q2*q3*332./2.35/z23
        return f

def uCDD3y(q2,q3,z12,z13):
        r23 = np.sqrt(z12*z12 + z13*z13)
        f = q2*q3*332./2.35/r23
        return f

def uLJ(z):
        f=0.152*(7./z)**6*((7./z)**6-2.)
        return f

def uLJ3z(z12,z13):
        z23 = z12 + z13
        f=0.152*(7./z23)**6*((7./z23)**6-2.)
        return f

def uLJ3y(z12,z13):
        r23 = np.sqrt(z12*z12 + z13*z13)
        f=0.152*(7./r23)**6*((7./r23)**6-2.)
        return f


q1=0.5
q2=-0.5
q3=0.0
cutoff = 15.0
cutoff = 25.0
#cutoff = 49.9
step = 2.0
values = (3.5,5.0,7.0,10.0,12.0,15.0)
# Plot PMF values from isspa code
for i,val in enumerate(values):
        z13 = val

        # Plot isspa trimer
        data = np.loadtxt("pmf.%s.dat" %(val))
        for i in range(len(data)):
                if data[i,1] == cutoff:
                        index = i
        # shift the value at 15 to be equal to CDD+LJ
        z12 = data[index,1]
        data[:,3] += -data[index,3] + uCDD(q1,q2,z12) + uCDD3z(q2,q3,z12,z13) + uLJ(z12) + uLJ3z(z12,z13)
        plt.plot(data[:,1], data[:,3], "k", label = "$u_{pmf}$")

        # Plot CDD + LJ PMF
        CDD_data = np.zeros((len(data),2),dtype=float)
        for i in range(len(data)):
                z12 = data[i,1]
                CDD_data[i,0] = z12
                CDD_data[i,1] = uCDD(q1,q2,z12) + uCDD3z(q2,q3,z12,z13) + uLJ(z12) + uLJ3z(z12,z13)
        plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'r', label = "$u_{CDD} + u_{LJ}$", linestyle='--')

        # Plot Dimer PMF plus CDD from 3rd atom
        DCDD_data = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p0.5_m0.5/25/pmf.dat")
        for i in range(len(DCDD_data)):
                if DCDD_data[i,0] == cutoff:
                        index = i
        # Add in CDD + LJ from atom 3 to atom 1
        for  i in range(len(DCDD_data)):
                z12 = DCDD_data[i,0]
                DCDD_data[i,2] =  DCDD_data[i,2] + uCDD3z(q2,q3,z12,z13) + uLJ3z(z12,z13)                
        z12 = DCDD_data[index,0]
        DCDD_data[:,2] += -DCDD_data[index,2] + uCDD(q1,q2,z12) + uCDD3z(q2,q3,z12,z13) + uLJ(z12) + uLJ3z(z12,z13)                
        plt.plot(DCDD_data[:,0], DCDD_data[:,2], c = 'b', label = "$u_{pmf}^{12}$ + $u_{CDD}^{23}$ + $u_{LJ}^{23}$", linestyle='--')

        ## Plot trimer PMF removing isspa interactions between 2-3
        #data = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ3_23/p1.0_m1.0_n0.0/parallel/pmf/pmf.%s.dat" %(val))
        ## Add in CDD + LJ from atom 3 to atom 1
        #for i in range(len(data)):
        #        z12 = data[i,1]
        #        data[i,3] = data[i,3] + uCDD3z(q1,q2,z12,z13) + uLJ3z(z12,z13)
        ## shift the value at 15 to be equal to CDD+LJ
        #data[:,3] += -data[-1,3] + uCDD(q1,q2,z12) + uCDD3z(q2,q3,z12,z13) + uLJ(z12) + uLJ3z(z12,z13)
        #
        #plt.plot(data[:,1], data[:,3], "g", label = "$u_{pmf}$ - $u_{IS-SPA}^{23}$",linestyle='--')


        plt.title(r"$R^{\parallel}_{13}$ = %s $\AA$" %(val))
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
        plt.ylabel(r'$u_{pmf}$ $(kcal \cdot mol^{-1})$', size=12)
        plt.legend(loc='lower right',  shadow=True, ncol=1, fontsize = 'medium')
        plt.xticks(np.arange(0, cutoff+1, step))
        plt.xlim((0,cutoff))
        plt.ylim((-10.0, 5.0))
        plt.savefig('p0.5_m0.5_p0.5.%s.zz.pmf.pdf' %(val))
        plt.savefig('p0.5_m0.5_p0.5.%s.zz.pmf.png' %(val))
        plt.close()

