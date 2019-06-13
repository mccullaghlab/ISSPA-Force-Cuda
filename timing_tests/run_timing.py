#!/usr/local/anaconda3/bin/python
import numpy as np
import sys
import os

nTimingRuns = 5
# job parameters
prmtop = "PDI1.prmtop"
inputCoord = "PDI1.rst"
nMC = 10
nSteps = 10000
deltaWrite = 1000
# writing timing.cfg
cfg = open("timing.cfg", "w")
cfg.write("prmtop = %s\n" % (prmtop))
cfg.write("inputCoord = %s\n" % (inputCoord))
cfg.write("nMC = %d\n" % (nMC))
cfg.write("nSteps = %d\n" % (nSteps))
cfg.write("deltaWrite = %d\n" % (deltaWrite))
cfg.close()

#
nsPerDay = np.empty(nTimingRuns,dtype=float)
#
for i in range(nTimingRuns):

    os.system('./total_force_cuda.x timing.cfg > timing.log')
    output = open("timing.log", 'r')
    for line in output:
        if "Average ns/day" in line:
            print(line)
            nsPerDay[i] = float(line.split('=')[1])
    output.close()


print("Average ns/day: ", np.mean(nsPerDay))
print("Stdev ns/day: ", np.std(nsPerDay))
