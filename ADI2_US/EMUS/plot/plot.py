import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def plot_4(xdata1, ydata1, yerr1, label1, xdata2, ydata2, yerr2, label2, xdata3, ydata3, yerr3, label3, xdata4, ydata4, yerr4, label4, x_axis, y_axis, system):
        plt.errorbar(xdata1, ydata1, yerr1, color='k',label=label1,errorevery=3,elinewidth=1.5)
        plt.errorbar(xdata2, ydata2, yerr2, color='r',label=label2,errorevery=3,elinewidth=1.5)
        plt.errorbar(xdata3, ydata3, yerr3, color='b',label=label3,errorevery=3,elinewidth=1.5)
        plt.errorbar(xdata4, ydata4, yerr4, color='g',label=label4,errorevery=3,elinewidth=1.5)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=2, fontsize = 'medium')
        plt.xlim((0,16))
        #plt.ylim((1.5, 3.0))
        plt.savefig('%s.PMFs.png' %(system))
        plt.close()

def plot_3(xdata1, ydata1, yerr1, label1, xdata2, ydata2, yerr2, label2, xdata3, ydata3, yerr3, label3, x_axis, y_axis, system):
        plt.errorbar(xdata1, ydata1, yerr1, color='k',label=label1,errorevery=3,elinewidth=1.5)
        plt.errorbar(xdata2, ydata2, yerr2, color='r',label=label2,errorevery=3,elinewidth=1.5)
        plt.errorbar(xdata3, ydata3, yerr3, color='b',label=label3,errorevery=3,elinewidth=1.5)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3, fontsize = 'medium')
        plt.xlim((0,16))
        plt.ylim((-5.5, 2.0))
        plt.savefig('%s.png' %(system))
        plt.savefig('%s.pdf' %(system))
        plt.close()

data = np.loadtxt("ADI2.exp.IS-SPA.dn0.21.pmf.bs.dat")
for i in range(len(data)):
        if data[i,0] < 3.00:
                index = i
exp_data = []
for i in range(index,len(data)):
        temp = []
        temp.append(data[i,0])
        temp.append(data[i,9])
        temp.append(data[i,10])
        exp_data.append(temp)
exp_data = np.array(exp_data)

data = np.loadtxt("ADI2.IS-SPA.dn0.21.pmf.bs.dat")
for i in range(len(data)):
        if data[i,0] < 3.00:
                index = i
pap_data = []
for i in range(index,len(data)):
        temp = []
        temp.append(data[i,0])
        temp.append(data[i,9])
        temp.append(data[i,10])
        pap_data.append(temp)
pap_data = np.array(pap_data)


data = np.loadtxt("ADI2.vac.pmf.bs.dat")
for i in range(len(data)):
        if data[i,0] < 3.00:
                index = i
vac_data = []
for i in range(index,len(data)):
        temp = []
        temp.append(data[i,0])
        temp.append(data[i,5])
        temp.append(data[i,6])
        vac_data.append(temp)
vac_data = np.array(vac_data)

cpu_data = np.loadtxt("../CPU/vacuum/ADI2.shifted_pmf.dat")
cpu_data = np.loadtxt("../CPU/ADI2.shifted_pmf.dat")
gpu_data = np.loadtxt("../GPU/ADI2.shifted_pmf.dat")

i=2
#plot_4(exp_data[:,0], exp_data[:,1], exp_data[:,2], "Explicit",  pap_data[:,0], pap_data[:,1], pap_data[:,2], "IS-SPA Paper", vac_data[:,0], vac_data[:,1], vac_data[:,2], "Vacuum", cpu_data[:,0], cpu_data[:,1], cpu_data[:,2], "ISSPA CPU (new)", 'Distance ($\AA$)', '$u_{pmf}$ (kcal/mol)', "ADI2.cpu")
plot_3(exp_data[:,0], exp_data[:,1], exp_data[:,2], "explicit",  gpu_data[i:,0], gpu_data[i:,1], gpu_data[i:,2], "IS-SPA CUDA", cpu_data[i:,0], cpu_data[i:,1], cpu_data[i:,2], "IS-SPA", 'R ($\AA$)', '$u_{pmf}$ $(kcal \cdot mol^{-1}$)', "ADP_dimer_pmf")
