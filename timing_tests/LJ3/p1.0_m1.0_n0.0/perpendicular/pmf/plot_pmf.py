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


q1=1.0
q2=-1.0
q3=0.0
values = (3.5,5.0,7.0,10.0,12.0,15.0)
j = 3
# Plot PMF values from isspa code
for i,val in enumerate(values):
        z13 = val

        # Plot isspa trimer
        data = np.loadtxt("pmf.%s.dat" %(val))
        z12 = data[-1,1]
        data[:,j] += uCDD(q1,q2,z12) + uCDD3y(q2,q3,z12,z13) + uLJ(z12) + uLJ3y(z12,z13)
        plt.plot(data[:,1], data[:,j], "k", label = "$u_{pmf}$")

        # Plot CDD + LJ PMF
        CDD_data = np.zeros((len(data),2),dtype=float)
        for i in range(len(data)):
                z12 = data[i,1]
                CDD_data[i,0] = z12
                CDD_data[i,1] = uCDD(q1,q2,z12) + uCDD3y(q2,q3,z12,z13) + uLJ(z12) + uLJ3y(z12,z13)
        plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'r', label = "$u_{CDD} + u_{LJ}$", linestyle='--')

        # Plot Dimer PMF plus CDD from 3rd atom
        DCDD_data = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p1.0_m1.0/12/pmf.dat")
        for i in range(len(DCDD_data)):
                if DCDD_data[i,0] == 15.0:
                        index = i
        for  i in range(len(DCDD_data)):
                z12 = DCDD_data[i,0]
                DCDD_data[i,1] =  DCDD_data[i,2] - DCDD_data[index,2] + uCDD(q1,q2,z12) + uCDD3y(q2,q3,z12,z13) + uLJ(z12) + uLJ3y(z12,z13)                
                #DCDD_data[i,1] =  (-DCDD_data[i,1]+DCDD_data[i,2])/2.0 + uCDD3y(q2,q3,z12,z13) + uLJ3y(z12,z13)
        plt.plot(DCDD_data[:,0], DCDD_data[:,1], c = 'b', label = "u$_{pmf}^{12}$ + u$_{CDD}^{23}$ + u$_{LJ}^{23}$", linestyle='--')

        # Plot trimer PMF removing isspa interactions between 2-3
        data = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ3_23/p1.0_m1.0_n0.0/perpendicular/pmf/pmf.%s.dat" %(val))
        z12 = data[-1,1]
        data[:,j] += uCDD(q1,q2,z12) + uCDD3y(q2,q3,z12,z13) + uLJ(z12) + uLJ3y(z12,z13)
        plt.plot(data[:,1], data[:,j], "g", label = "$u_{pmf}$ - $u_{IS-SPA}^{23}$",linestyle='--')


        plt.title(r"$R^{\perp}_{13}$ = %s $\AA$" %(val))
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'$R_{12}$ $(\AA)$', size=12)
        plt.ylabel(r'$u_{pmf}$ $(kcal \cdot mol^{-1})$', size=12)
        plt.legend(loc='lower right',  shadow=True, ncol=1, fontsize = 'medium')
        plt.xticks(np.arange(0, 16, 1.0))
        plt.xlim((0,15))
        plt.ylim((-40.0, 5.0))
        plt.savefig('p1.0_m1.0_n0.0.%s.yz.pmf.pdf' %(val))
        plt.savefig('p1.0_m1.0_n0.0.%s.yz.pmf.png' %(val))
        plt.close()

