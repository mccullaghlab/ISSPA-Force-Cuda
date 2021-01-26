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
values = (3.5,5.0,7.0,10.0,12.0,15.0)
#values = (3.5,5.0)
j = 3
colors = ["k","r","b","g","c","m"]
# Plot PMF values from isspa code
for i,val in enumerate(values):
        data = np.loadtxt("pmf.%s.dat" %(val))
        z12 = data[-1,1]
        data[:,j] += uCDD(q1,q2,z12)
        plt.plot(data[:,1], data[:,j], c = colors[i], label = "$R_{z}$ = %s $\AA$" %(val))

# Plot CDD + LJ PMF
CDD_data = np.zeros((len(data),2),dtype=float)
for i in range(len(data)):
        z12 = data[i,1]
        CDD_data[i,0] = z12
        z13 = val
        CDD_data[i,1] = uCDD(q1,q2,z12) + uCDD3z(q2,q3,z12,z13) + uLJ(z12) + uLJ3z(z12,z13)
plt.plot(CDD_data[:,0], CDD_data[:,1], c = 'k', label = "CDD + LJ Dimer", linestyle='--')


# Plot Dimer PMF plus CDD from 3rd atom
DCDD_data = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p0.5_m0.5/pmf.dat")
for  i in range(len(DCDD_data)):
        DCDD_data[i,1] = (-DCDD_data[i,1] + DCDD_data[i,2])/2.0 + uCDD3z(q2,q3,z12,z13)
plt.plot(DCDD_data[:,0], DCDD_data[:,1], c = 'g', label = "Dimer + CDD$_{3}$", linestyle='--')

        
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{z}$ $(\AA)$', size=12)
plt.ylabel(r'$u_{pmf}$ $(kcal \cdot mol^{-1})$', size=12)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),  shadow=True, ncol=3, fontsize = 'medium')
plt.xticks(np.arange(0, 16, 1.0))
plt.xlim((0,15))
plt.ylim((-15.0, 5.0))
plt.savefig('PM0.5N.zz.pmf.pdf')
plt.savefig('PM0.5N.zz.pmf.png')
plt.close()

