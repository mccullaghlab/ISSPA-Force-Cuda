import numpy as np
import sys
import math

ft = np.loadtxt("LJ3.forces.combined.dat")
flj = np.loadtxt("LJ3.lj_forces.combined.dat")
fc = np.loadtxt("LJ3.c_forces.combined.dat")

dx=0.1
zBins = int((ft[-1,1]-ft[0,1])/dx)+1
yBins = 6
print(yBins)
print(zBins)


upt = np.zeros((yBins,zBins),dtype=float)
upi = np.zeros((yBins,zBins),dtype=float)
umt = np.zeros((yBins,zBins),dtype=float)
umi = np.zeros((yBins,zBins),dtype=float)

vals1 = (3.5,5.0,7.0,10.0,12.0,15.0) 



for y in range(yBins):
    vals = []
    for z in range(zBins-1):
        vals.append(ft[z,1])
        hdx = dx*0.5
        upt[y,z+1] = upt[y,z]-hdx*(ft[y*zBins+z,4]+ft[y*zBins+z+1,4])
        umt[y,z+1] = umt[y,z]-hdx*(ft[y*zBins+z,7]+ft[y*zBins+z+1,7])
        upi[y,z+1] = upi[y,z]-hdx*(flj[y*zBins+z,4]+flj[y*zBins+z+1,4]+fc[y*zBins+z,4]+fc[y*zBins+z+1,4])
        umi[y,z+1] = umi[y,z]-hdx*(flj[y*zBins+z,7]+flj[y*zBins+z+1,7]+fc[y*zBins+z,7]+fc[y*zBins+z+1,7])
    upt[y,:]-=upt[y,zBins-1]
    umt[y,:]-=umt[y,zBins-1]
    upi[y,:]-=upi[y,zBins-1]
    umi[y,:]-=umi[y,zBins-1]
    vals.append(ft[zBins-1,1])
    vals = np.array(vals)
    out = open("pmf/pmf.%s.dat" %(vals1[y]),'w')
    for z in range(zBins-1):
        out.write("  %5.1f  %5.1f  %10.5f  %10.5f  %10.5f  %10.5f\n" %(vals1[y],vals[z],-upt[y,z],umt[y,z],upi[y,z],-umi[y,z]))
    out.close
