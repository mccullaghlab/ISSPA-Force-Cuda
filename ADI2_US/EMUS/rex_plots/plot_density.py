import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def plot_3(data1, label1, data2, label2, data3, label3, x_axis, y_axis, system):
        hist1, edges1 = np.histogram(data1[:,1],density=True,bins=60)
        hist2, edges2 = np.histogram(data2[:,1],density=True,bins=60)
        #hist1 /= np.amax(hist1)
        #hist2 /= np.amax(hist2)
        plt.plot(edges1[1:],hist1,label=label1)
        plt.plot(edges2[1:],hist2,label=label2)
        plt.plot(data3[:,0],data3[:,1],label=label3)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3, fontsize = 'medium')
        plt.xlim((15,17))
        #plt.ylim((1.5, 3.0))                                                                                                                                                                                                     
        plt.savefig('%s.png' %(system))
        plt.close()

def calc_dist(k,T,r0,r):
        T *= 1.9872041E-3 #k_b
        p = np.exp(-k/2.0*(r-r0)**2/T)*r*r
        return p

data1 = np.loadtxt("ADI2.CPU.window.16.0.dat")
data2 = np.loadtxt("ADI2.GPU.window.16.0.dat")
r = np.arange(0,30,0.01)
data3 = []
for i in range(len(r)):
        data3.append((r[i],calc_dist(20.0,298.0,16.0,r[i])))
data3 = np.array(data3)
data3[:,1] /= 150
plot_3(data1, "CPU", data2, "GPU", data3, "Calculated", "Distance ($\AA$)", "Probability Density", "CV")


print(np.std(data1[:,1])*np.std(data1[:,1])*20/1.9872041E-3)
print(np.std(data2[:,1])*np.std(data2[:,1])*20/1.9872041E-3)
