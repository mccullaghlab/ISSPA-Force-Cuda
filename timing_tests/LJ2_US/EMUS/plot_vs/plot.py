import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def plot_1d(xdata3, ydata3, yerr3, x_axis, y_axis, system):
        plt.errorbar(xdata3, ydata3, yerr3, color='b',label="IS-SPA (gpu)",errorevery=3,elinewidth=1.5)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=3, fontsize = 'medium')
        plt.xlim((0,25))
        plt.xticks(np.arange(0,25.0, 1.0))
        #plt.ylim((1.5, 3.0))
        plt.savefig('%s.PMF.png' %(system))
        plt.close()


data1 = np.loadtxt("LJ2.shifted_pmf.dat")

i=9
plot_1d(data1[:,0], data1[:,1], data1[:,2], 'Distance ($\AA$)', '$u_{pmf}$ (kcal/mol)', "LJ2")
