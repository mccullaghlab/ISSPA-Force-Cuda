import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.cm as cm

stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()

def CDD(x):
        f = -332./2.35/x
        return f

values = (3.5,5.0,7.0,10.0,12.0,15.0)
j = 3
colors = ["k","r","b","g","c","m"]
for i,val in enumerate(values):
        data = np.loadtxt("pmf.%s.dat" %(val))
        data[:,j] += CDD(data[-1,1])
        plt.plot(data[:,1], data[:,j], c = colors[i], label = "$R_{z}$ = %s $\AA$" %(val))
        
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel(r'$R_{z}$ $(\AA)$', size=12)
plt.ylabel(r'$u_{pmf}$ $(kcal \cdot mol^{-1})$', size=12)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),  shadow=True, ncol=3, fontsize = 'medium')
plt.xticks(np.arange(0, 16, 1.0))
plt.xlim((0,15))
plt.ylim((-40.0, 5.0))
plt.savefig('MMP.zz.pmf.png')
plt.close()

