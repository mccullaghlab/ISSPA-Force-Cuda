import numpy as np
import sys
import MDAnalysis as mda
import glob

top_file = "LJ3.prmtop"

values1 = (3.5,5.0,7.0,10.0,12.0,15.0)
values2 = np.arange(3.5, 50.1, 0.1)
out = open("LJ3.forces.combined.dat",'w')
for val1 in values1:
    for val2 in values2:
        print("%.1f %.1f" %(val1,val2))
        u = mda.Universe(top_file,"%.1f/%.1f/LJ3.%.1f_%.1f.forces.xyz" %(val1,val2,val1,val2))
        nAtoms = len(u.atoms)
        nSteps = len(u.trajectory)
        forces = np.zeros((nSteps,nAtoms,3),dtype=float)
        for ts in u.trajectory:
            forces[ts.frame] = u.atoms.positions
        out.write("  %.1f  %.1f" %(val1,val2))
        for i in range(len(u.atoms)):
            out.write("  %12.6f  %12.6f" %(np.average(forces[:,i,2]),np.std(forces[:,i,2])/np.sqrt(nSteps)))
        out.write("\n")
out.close

out = open("LJ3.lj_forces.combined.dat",'w')
for val1 in values1:
    for val2 in values2:
        print("%.1f %.1f" %(val1,val2))
        u = mda.Universe(top_file,"%.1f/%.1f/isspa_lj_force.xyz" %(val1,val2))
        nAtoms = len(u.atoms)
        nSteps = len(u.trajectory)
        forces = np.zeros((nSteps,nAtoms,3),dtype=float)
        for ts in u.trajectory:
            forces[ts.frame] = u.atoms.positions
        out.write("  %.1f  %.1f" %(val1,val2))
        for i in range(len(u.atoms)):
            out.write("  %12.6f  %12.6f" %(np.average(forces[:,i,2]),np.std(forces[:,i,2])/np.sqrt(nSteps)))
        out.write("\n")
out.close

out = open("LJ3.c_forces.combined.dat",'w')
for val1 in values1:
    for val2 in values2:
        print("%.1f %.1f" %(val1,val2))
        u = mda.Universe(top_file,"%.1f/%.1f/isspa_C_force.xyz" %(val1,val2))
        nAtoms = len(u.atoms)
        nSteps = len(u.trajectory)
        forces = np.zeros((nSteps,nAtoms,3),dtype=float)
        for ts in u.trajectory:
            forces[ts.frame] = u.atoms.positions
        out.write("  %.1f  %.1f" %(val1,val2))
        for i in range(len(u.atoms)):
            out.write("  %12.6f  %12.6f" %(np.average(forces[:,i,2]),np.std(forces[:,i,2])/np.sqrt(nSteps)))
        out.write("\n")
out.close


