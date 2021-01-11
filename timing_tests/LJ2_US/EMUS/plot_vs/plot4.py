import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def plot_1d(xdata1, ydata1, yerr1, xdata2, ydata2, yerr2, xdata3, ydata3, yerr3, xdata4, ydata4, yerr4, x_axis, y_axis, system):
        plt.errorbar(xdata1, ydata1, yerr1, color='k',label="Explicit",errorevery=3,elinewidth=1.5)
        plt.errorbar(xdata2, ydata2, yerr2, color='r',label="IS-SPA (cpu)",errorevery=3,elinewidth=1.5)
        plt.errorbar(xdata3, ydata3, yerr3, color='b',label="IS-SPA (gpu)",errorevery=3,elinewidth=1.5)
        plt.errorbar(xdata4, ydata4, yerr4, color='g',label="IS-SPA (gpu) rep 1",errorevery=3,elinewidth=1.5)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=2, fontsize = 'medium')
        plt.xlim((0,16))
        #plt.ylim((1.5, 3.0))
        plt.savefig('%s.PMF.4.png' %(system))
        plt.close()


data1 = np.loadtxt("ADI2.exp.pmf.bs.dat")
data2 = np.loadtxt("ADI2.IS-SPA.dn0.21.pmf.bs.dat")
data3 = np.loadtxt("PDI.shifted_pmf.dat")
data4 = np.loadtxt("ADI2.shifted_pmf.rep1.dat")

index1 = 0
for i in range(len(data1)):
        if data1[i,0] < 3.00:
                index1 = i
index2 = 0
for i in range(len(data2)):
        if data2[i,0] < 3.00:
                index2 = i
i=9
plot_1d(data1[index1:,0], data1[index1:,i], data1[index1:,i+1], data2[index2:,0], data2[index2:,i], data2[index2:,i+1], data3[:,0], data3[:,1], data3[:,2], data4[:,0], data4[:,1], data4[:,2], 'Distance ($\AA$)', '$u_{pmf}$ (kcal/mol)', "ADI2")
