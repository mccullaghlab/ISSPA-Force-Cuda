import numpy as np
import sys
import os
import MDAnalysis as mda

top_file = "ADI2.prmtop"
traj1 = "fixed.xyz"
traj2 = "ADI2.run0.window.16.0.velocities.xyz"

u1 = mda.Universe(top_file,traj1)
u2 = mda.Universe(top_file,traj2)
for atom in u1.atoms:
        if atom.mass == 1.008:
                atom.mass = 12.01
for atom in u2.atoms:
        if atom.mass == 1.008:
                atom.mass = 12.01

KE1 = []
for ts in u1.trajectory:
        KE = 0
        for atom in u1.atoms:
                for j in range(3):
                        KE += 0.5*atom.mass*atom.position[j]*atom.position[j]
        KE1.append(KE)
KE1 = np.array(KE1)
out = open("CPU_KE.dat",'w')
for i in range(len(KE1)):
        out.write("  %10i  %12.8f\n" %(i,KE1[i]))
out.close()


KE2 = []
for ts in u2.trajectory:
        KE = 0
        for atom in u2.atoms:
                for j in range(3):
                        KE += 0.5*atom.mass*atom.position[j]*atom.position[j]
        KE2.append(KE)
KE2 = np.array(KE2)
out = open("GPU_KE.dat",'w')
for i in range(len(KE2)):
        out.write("  %10i  %12.8f\n" %(i,KE2[i]))
out.close()
