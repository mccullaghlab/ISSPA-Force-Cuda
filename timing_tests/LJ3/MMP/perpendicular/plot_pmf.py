import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm

stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()


def plot_hm(Z, ext, v_min, v_max, x_axis, y_axis, system):
        fig, ax = plt.subplots()
        im = ax.imshow(Z, interpolation='bilinear', cmap=cm.viridis, origin='lower', extent=ext, vmax=v_max, vmin=v_min)
        #im = ax.imshow(Z, interpolation='bilinear', cmap=cm.plasma, origin='lower', extent=ext)
        plt.colorbar(im)
        #plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xticks(np.arange(4.0, 15.0, 1.0))
        plt.yticks(np.arange(4.0, 15.0, 1.0))
        plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=12)
        plt.ylabel(r'%s' %(y_axis), size=12)
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

ext = [data[0,1],data[-1,1],data[0,0],data[-1,0]]
zbins = int((ext[1]-ext[0])/0.1)
ybins = int((ext[3]-ext[2])/0.1)
print(zbins,ybins)
Z1 = np.zeros((ybins+1,zbins+1),dtype=float)
Z2 = np.zeros((ybins+1,zbins+1),dtype=float)
Z3 = np.zeros((ybins+1,zbins+1),dtype=float)
Z4 = np.zeros((ybins+1,zbins+1),dtype=float)

for i in range(len(data)):
        y = round((data[i,0]-ext[2])/0.1)
        z = round((data[i,1]-ext[0])/0.1)
        
        Z1[y,z] = data[i,2]
        Z2[y,z] = data[i,3]
        Z3[y,z] = data[i,4]
        Z4[y,z] = data[i,5]

r = np.arange(ext[0],ext[1]+0.1,0.1)
plot_2d(r, Z1[0], "$R_{y}$=%.1f $(\AA)$" %(r[0]), r, Z1[ybins], '$R_{y}$=%.1f $(\AA)$' %(r[ybins]),  "Distance $(\AA)$", "$u_{pmf}$", "p_total_pmf")
plot_2d(r, Z2[0], "$R_{y}$=%.1f $(\AA)$" %(r[0]), r, Z2[ybins], '$R_{y}$=%.1f $(\AA)$' %(r[ybins]),  "Distance $(\AA)$", "$u_{pmf}$", "m_total_pmf")
plot_2d(r, Z3[0], "$R_{y}$=%.1f $(\AA)$" %(r[0]), r, Z3[ybins], '$R_{y}$=%.1f $(\AA)$' %(r[ybins]),  "Distance $(\AA)$", "$u_{pmf}$", "p_isspa_pmf")
plot_2d(r, Z4[0], "$R_{y}$=%.1f $(\AA)$" %(r[0]), r, Z4[ybins], '$R_{y}$=%.1f $(\AA)$' %(r[ybins]),  "Distance $(\AA)$", "$u_{pmf}$", "m_isspa_pmf")
        
plot_hm(Z1,ext,np.min(Z1),5,"$R_{z} (\AA)$","$R_{y} (\AA)$", "p_total_pmf.2d")
plot_hm(Z2,ext,np.min(Z2),5,"$R_{z} (\AA)$","$R_{y} (\AA)$", "m_total_pmf.2d")
plot_hm(Z3,ext,np.min(Z3),0,"$R_{z} (\AA)$","$R_{y} (\AA)$", "p_isspa_pmf.2d")
plot_hm(Z4,ext,np.min(Z4),0,"$R_{z} (\AA)$","$R_{y} (\AA)$", "m_isspa_pmf.2d")
