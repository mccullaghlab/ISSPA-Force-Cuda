import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import MDAnalysis as mda
from matplotlib.ticker import NullFormatter
stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def plot_3(data1, label1, data2, label2, x_axis, y_axis):
        plt.plot(data2[:,0], data2[:,1], color='r',label=label2)
        plt.plot(data1[:,0], data1[:,1], color='k',label=label1)
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
        KE = 1.5*1.9872041E-3*298.0*44.0
        plt.axhline(y=KE, color='b', linestyle='-')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=2, fontsize = 'medium')
        plt.xlim((0,50.0))
        #plt.ylim((1.5, 3.0))                                                                                                                                                                                     
        plt.savefig('KE.png')
        plt.close()



skip = 250

dat1 = np.loadtxt("CPU_KE.dat")
data1 = []
avg = 0
pos = 0
count = 0
for i in range(len(dat1)):
        pos += dat1[i,0]*0.002
        avg += dat1[i,1]
        count += 1
        if count == skip:
                avg /= skip
                pos /= skip
                data1.append((pos,avg))
                avg = 0
                pos = 0
                count = 0
data1 = np.array(data1)

dat2 = np.loadtxt("GPU_KE.dat")
data2 = []
avg = 0
pos = 0
count = 0
for i in range(len(dat2)):
        pos += dat2[i,0]*0.002
        avg += dat2[i,1]
        count += 1
        if count == skip:
                avg /= skip
                pos /= skip
                data2.append((pos,avg))
                avg = 0
                pos = 0
                count = 0
data2 = np.array(data2)

print(np.average(dat1[:,1])/1.5/44./(1.9872041*10**(-3)))
print(np.average(dat2[:,1])/1.5/44./(1.9872041*10**(-3)))
print(np.average(dat1[:,1]))
print(np.average(dat2[:,1]))
plot_3(data1, "CPU", data2, "GPU", "Time (ps)", "Kinetic Energy (kcal/mol)")

