import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()


def plot_4d(xdata1, ydata1, xdata2, ydata2, xdata3, ydata3, xdata4, ydata4, x_axis, y_axis, system):
        plt.plot(xdata1, ydata1, 'k',label ="LJ+ gpu")
        plt.plot(xdata2, ydata2, 'r',label ="LJ- gpu")
        plt.plot(xdata3, ydata3, 'g',label ="LJ+ cpu")
        plt.plot(xdata4, ydata4, 'b',label ="LJ- cpu")
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=2, fontsize = 'medium')                                                         
        #plt.xlim((0,200))
        #plt.ylim((1.5, 3.0))
        plt.savefig('%s.png' %(system))
        plt.close()

def plot_4rat(xdata1, ydata1, xdata2, ydata2, xdata3, ydata3, xdata4, ydata4, x_axis, y_axis, system):
        ydat1 = np.zeros(len(xdata1),dtype=float)
        ydat2 = np.zeros(len(xdata1),dtype=float)
        
        for i in range(len(xdata1)):
                ydat1[i] = ydata1[i]/ydata3[i]
                ydat2[i] = ydata2[i]/ydata4[i]        
        plt.plot(xdata1, ydat1, 'k',label ="LJ+ ratio (gpu/cpu)")
        plt.plot(xdata2, ydat2, 'r',label ="LJ - ratio (gpu/cpu)")
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=2, fontsize = 'medium')                                                         
        #plt.xlim((0,200))
        #plt.ylim((1.5, 3.0))
        plt.savefig('%s.ratio.png' %(system))
        plt.close()

def plot_4diff(xdata1, ydata1, xdata2, ydata2, xdata3, ydata3, xdata4, ydata4, x_axis, y_axis, system):
        ydat1 = np.zeros(len(xdata1),dtype=float)
        ydat2 = np.zeros(len(xdata1),dtype=float)
        for i in range(len(xdata1)):
                ydat1[i] = ydata1[i]-ydata3[i]
                ydat2[i] = ydata2[i]-ydata4[i]        
        plt.plot(xdata1, ydat1, 'k',label ="LJ+ diff (gpu-cpu)")
        plt.plot(xdata2, ydat2, 'r',label ="LJ- diff (gpu-cpu)")
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=2, fontsize = 'medium')                                                         
        #plt.xlim((0,200))
        #plt.ylim((1.5, 3.0))
        plt.savefig('%s.diff.png' %(system))
        plt.close()


ljf = np.loadtxt("LJ2.lj_forces.combined.dat")
cf = np.loadtxt("LJ2.c_forces.combined.dat")
cpu_dat = np.loadtxt("/home/ryan/Desktop/isspa/isspa2_v3_forces/timing_tests/LJ2/rex_force/LJ2.q1.00.IS-SPA.dn0.21.dat")

nBins = len(ljf)
ljt = np.zeros((3,nBins),dtype=float)
ct = np.zeros((3,nBins),dtype=float)


for i in range(nBins):
    ljt[0,i] = ljf[i,0]
    #ljt[1,i] = np.sqrt(ljf[i,1]*ljf[i,1]+ljf[i,2]*ljf[i,2]+ljf[i,3]*ljf[i,3])
    #ljt[2,i] = np.sqrt(ljf[i,4]*ljf[i,4]+ljf[i,5]*ljf[i,5]+ljf[i,6]*ljf[i,6])
    ljt[1,i] = -ljf[i,3]
    ljt[2,i] = ljf[i,6]

    
    ct[0,i] = cf[i,0]
    #ct[1,i] = np.sqrt(cf[i,1]*cf[i,1]+cf[i,2]*cf[i,2]+cf[i,3]*cf[i,3])
    #ct[2,i] = np.sqrt(cf[i,4]*cf[i,4]+cf[i,5]*cf[i,5]+cf[i,6]*cf[i,6])
    ct[1,i] = -cf[i,3]
    ct[2,i] = cf[i,6]

rcut = 25
eps=2.3473
q1 = 1.0
q2 = -1.0
for i in range(len(cpu_dat)):
        A = -q1*q2*(1-1/eps)
        r = cpu_dat[i,0]
        if cpu_dat[i,0] < 2*rcut:
                f1 = r*(8*rcut-3*r)/(24*rcut*rcut*rcut*rcut)  
                f2 = -np.log(1+r/rcut)/(8*r*r)
                f3 = rcut/(8*r*(r+rcut)**2)
                f4 = -r*r/(32*rcut*rcut*rcut*rcut)
                f5 = 3/(16*rcut*rcut)
                f = A*(f1+f2+f3+f4+f5)
        #else:
        #        f1 = 2/(3*r*r)
        #        f2 = -np.log((r+rcut)/(r-rcut))/(8*r*r)
        #        f3 = rcut*(r*r+rcut*rcut)/(4*r*(r*r-rcut*rcut)**2)
        #        f = A*(f1+f2+f3)
        cpu_dat[i,3] += f
        cpu_dat[i,4] += f
        
    
plot_4d(ljt[0,:], ljt[1,:], ljt[0,:], ljt[2,:], cpu_dat[:,0], cpu_dat[:,1], cpu_dat[:,0], cpu_dat[:,2], "Distance $(\AA)$", "LJ Force", "LJ_force")
plot_4d(ct[0,:], ct[1,:], ct[0,:], ct[2,:], cpu_dat[:,0], cpu_dat[:,3], cpu_dat[:,0], cpu_dat[:,4], "Distance $(\AA)$", "C Force", "C_force")
#plot_4rat(ct[0,:], ct[1,:], ct[0,:], ct[2,:], cpu_dat[:,0], cpu_dat[:,3], cpu_dat[:,0], cpu_dat[:,4], "Distance $(\AA)$", "C Force", "C_force")
#plot_4diff(ct[0,:], ct[1,:], ct[0,:], ct[2,:], cpu_dat[:,0], cpu_dat[:,3], cpu_dat[:,0], cpu_dat[:,4], "Distance $(\AA)$", "C Force", "C_force")
