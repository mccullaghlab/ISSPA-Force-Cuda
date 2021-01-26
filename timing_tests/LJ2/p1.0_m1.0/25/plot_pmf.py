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

def plot_2d(xdata1, ydata1, label1, xdata2, ydata2, label2,  x_axis, y_axis, system):
        plt.plot(xdata1, ydata1, 'k',label = label1)
        plt.plot(xdata2, ydata2, 'r',label = label2)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=2, fontsize = 'medium')
        plt.xticks(np.arange(0, 16, 1.0))
        plt.xlim((0,15))
        plt.ylim((-35.0, 5.0))
        plt.savefig('%s.png' %(system))
        plt.close()


data = np.loadtxt("pmf.dat")

plot_2d(data[:,0], -data[:,1], "$q = +1$", data[:,0], data[:,2], "$q = -1$", "Distance ($\AA$)", "$u_{pmf}$", "LJ_PMF")
plot_2d(data[:,0], data[:,3], "$q = +1$", data[:,0], -data[:,4], "$q = -1$", "Distance ($\AA$)", "$u_{pmf}$", "LJ_ISSPA_PMF")

