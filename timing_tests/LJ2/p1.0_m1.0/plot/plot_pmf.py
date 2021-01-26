import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()



def plot_4d(xdata1, ydata1, label1, xdata2, ydata2, label2, xdata3, ydata3, label3, xdata4, ydata4, label4, x_axis, y_axis, system):
        plt.plot(xdata1, ydata1, 'k',label = label1)
        plt.plot(xdata2, ydata2, 'r',label = label2)
        plt.plot(xdata3, ydata3, 'b',label = label3)
        plt.plot(xdata4, ydata4, 'g',label = label4)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=4, fontsize = 'medium')
        plt.xticks(np.arange(0, 16, 1.0))
        plt.xlim((0,15))
        plt.ylim((-35.0, 5.0))
        plt.savefig('%s.png' %(system))
        plt.close()

def plot_3d(xdata1, ydata1, label1, xdata2, ydata2, label2, xdata3, ydata3, label3,  x_axis, y_axis, system):
        plt.plot(xdata1, ydata1, 'k',label = label1)
        plt.plot(xdata2, ydata2, 'r',label = label2)
        plt.plot(xdata3, ydata3, 'b',label = label3)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=3, fontsize = 'medium')
        plt.xticks(np.arange(0, 16, 1.0))
        plt.xlim((0,15))
        plt.ylim((-35.0, 5.0))
        plt.savefig('%s.png' %(system))
        plt.close()

def uC(x,rc,eps):
        uC = -332.*(1.-1./eps)*(x**2/6./rc**3-5.*x**3/96./rc**4+3.*x/16./rc**2+1./8./(x+rc)+np.log(1.+x/rc)/8./x-13./12.)
        return uC
def uLJ(x):
        uLJ=0.152*(7./x)**6*((7./x)**6-2.)
        return uLJ

data10 = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p1.00_M1.00/10/pmf.dat")
data15 = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p1.00_M1.00/15/pmf.dat")
data25 = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/p1.00_M1.00/25/pmf.dat")

cpu_dat = np.loadtxt("LJ2.q1.00.IS-SPA.dn0.21.dat")
exp_dat = np.loadtxt("decomposed_z1.0_r7.0.out")

data = np.zeros((len(cpu_dat),3),dtype=float)
data2 = np.zeros((len(exp_dat),2),dtype=float)
data3 = np.zeros((len(exp_dat),2),dtype=float)
eps=2.35
rc=24.95

for i in range(len(exp_dat)):
        data3[i,0] = exp_dat[i,0]
        data3[i,1] = uLJ(data3[i,0])-332./2.35/data3[i,0]
for i in range(len(exp_dat)):
        data2[i,0] = exp_dat[i,0]
        data2[i,1] = exp_dat[i,5]+exp_dat[i,6]+uLJ(exp_dat[i,0])-332./exp_dat[i,0]+12.05
for i in range(len(cpu_dat)):
        data[i,0] = cpu_dat[i,0]
        data[i,1] = ((cpu_dat[i,5]+cpu_dat[i,7])+uC(cpu_dat[i,0],rc,eps)+uLJ(cpu_dat[i,0])-332./cpu_dat[i,0]-196.03)
        data[i,2] = ((cpu_dat[i,6]+cpu_dat[i,8])+uC(cpu_dat[i,0],rc,eps)+uLJ(cpu_dat[i,0])-332./cpu_dat[i,0]-196.85)
        
data10[:,2] = (data10[:,2]-data10[:,1])*0.5
data15[:,2] = (data15[:,2]-data15[:,1])*0.5
data25[:,2] = (data25[:,2]-data25[:,1])*0.5
        
plot_3d(data10[:,0], data10[:,2], "$R_{max}$=10 $\AA$", data15[:,0], data15[:,2], "$R_{max}$=15 $\AA$", data25[:,0], data25[:,2], "$R_{max}$=25 $\AA$", "Distance ($\AA$)", "$u_{pmf}$", "LJ2.pmf")
plot_3d(data25[:,0], data25[:,2], "GPU", data2[:,0], data2[:,1], "Explicit", data3[:,0], data3[:,1], "CDD + LJ", "Distance ($\AA$)", "$u_{pmf}$", "LJ2.comp.pmf")
plot_4d(data25[:,0], data25[:,2], "GPU", data[:,0], data[:,1],"CPU", data2[:,0], data2[:,1], "Explicit", data3[:,0], data3[:,1], "CDD + LJ", "Distance ($\AA$)", "$u_{pmf}$", "LJ2.25.pmf")
