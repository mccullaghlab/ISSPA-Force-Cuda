import numpy as np
import sys

ft = np.loadtxt("LJ2.forces.combined.dat")
flj = np.loadtxt("LJ2.lj_forces.combined.dat")
fc = np.loadtxt("LJ2.c_forces.combined.dat")


nBins = len(ft)
print(nBins)

upt = np.zeros(nBins,dtype=float)
upi = np.zeros(nBins,dtype=float)
umt = np.zeros(nBins,dtype=float)
umi = np.zeros(nBins,dtype=float)

for i in range(nBins-1):
    if ft[i,0] > 2.99:
        hdx = 0.1*0.5
        upt[i+1] = upt[i]-hdx*(ft[i,3]+ft[i+1,3])
        umt[i+1] = umt[i]-hdx*(ft[i,6]+ft[i+1,6])
        upi[i+1] = upi[i]-hdx*(flj[i,3]+flj[i+1,3]+fc[i,3]+fc[i+1,3])
        umi[i+1] = umi[i]-hdx*(flj[i,6]+flj[i+1,6]+fc[i,6]+fc[i+1,6])

upt-=upt[nBins-1]
umt-=umt[nBins-1]
upi-=upi[nBins-1]
umi-=umi[nBins-1]

out = open("pmf.dat",'w')
for i in range(nBins):
    out.write("  %5.1f  %10.5f  %10.5f  %10.5f  %10.5f\n" %(ft[i,0],upt[i],umt[i],upi[i],umi[i]))
out.close
