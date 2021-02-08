import numpy as np
import sys
import MDAnalysis as mda
import glob

top_file = "LJ2.prmtop"
files = glob.glob("4.*/LJ2.*.forces.xyz")


values = np.arange(0.1, 50.1, 0.1)

out = open("LJ2.forces.combined.dat",'w')
for val in values:
    print("%.1f" %(val))
    u = mda.Universe(top_file,"%.1f/LJ2.%.1f.forces.xyz" %(val,val))
    nAtoms = len(u.atoms)
    nSteps = len(u.trajectory)
    forces = np.zeros((nSteps,nAtoms,3),dtype=float)
    for ts in u.trajectory:
        forces[ts.frame] = u.atoms.positions
    out.write("  %.1f" %(val))
    for i in range(len(u.atoms)):
        out.write("  %12.6f  %12.6f" %(np.average(forces[:,i,2]),np.std(forces[:,i,2])/np.sqrt(nSteps)))
    out.write("\n")
out.close

out = open("LJ2.lj_forces.combined.dat",'w')
for val in values:
    print("%.1f" %(val))
    u = mda.Universe(top_file,"%.1f/isspa_lj_force.xyz" %(val))
    nAtoms = len(u.atoms)
    nSteps = len(u.trajectory)
    forces = np.zeros((nSteps,nAtoms,3),dtype=float)
    for ts in u.trajectory:
        forces[ts.frame] = u.atoms.positions
    out.write("  %.1f" %(val))
    for i in range(len(u.atoms)):
        out.write("  %12.6f  %12.6f" %(np.average(forces[:,i,2]),np.std(forces[:,i,2])/np.sqrt(nSteps)))
    out.write("\n")
out.close

out = open("LJ2.c_forces.combined.dat",'w')
for val in values:
    print("%.1f" %(val))
    u = mda.Universe(top_file,"%.1f/isspa_C_force.xyz" %(val))
    nAtoms = len(u.atoms)
    nSteps = len(u.trajectory)
    forces = np.zeros((nSteps,nAtoms,3),dtype=float)
    for ts in u.trajectory:
        forces[ts.frame] = u.atoms.positions
    out.write("  %.1f" %(val))
    for i in range(len(u.atoms)):
        out.write("  %12.6f  %12.6f" %(np.average(forces[:,i,2]),np.std(forces[:,i,2])/np.sqrt(nSteps)))
    out.write("\n")
out.close
