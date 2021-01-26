import numpy as np
import sys
import MDAnalysis as mda
import glob

top_file = "LJ2.prmtop"
files = glob.glob("4.*/LJ2.*.forces.xyz")


values = np.arange(0.1, 50.0, 0.1)

out = open("LJ2.forces.combined.dat",'w')
for val in values:
    if val < 10:
        print("%3.1f" %(val))
        u = mda.Universe(top_file,"%3.1f/LJ2.%3.1f.forces.xyz" %(val,val))
        nAtoms = len(u.atoms)
    else:
        print("%4.1f" %(val))
        u = mda.Universe(top_file,"%4.1f/LJ2.%4.1f.forces.xyz" %(val,val))
        nAtoms = len(u.atoms)
    out.write("  %4.1f" %(val))
    for atom in u.atoms:
        out.write("  %12.6f  %12.6f  %12.6f" %(atom.position[0],atom.position[1],atom.position[2]))
    out.write("\n")
out.close

out = open("LJ2.lj_forces.combined.dat",'w')
for val in values:
    if val < 10:
        print("%3.1f" %(val))
        u = mda.Universe(top_file,"%3.1f/isspa_lj_force.xyz" %(val))
        nAtoms = len(u.atoms)
    else:
        print("%4.1f" %(val))
        u = mda.Universe(top_file,"%4.1f/isspa_lj_force.xyz" %(val))
        nAtoms = len(u.atoms)
    out.write("  %4.1f" %(val))
    for atom in u.atoms:
        out.write("  %12.6f  %12.6f  %12.6f" %(atom.position[0],atom.position[1],atom.position[2]))
    out.write("\n")
out.close

out = open("LJ2.c_forces.combined.dat",'w')
for val in values:
    if val < 10:
        print("%3.1f" %(val))
        u = mda.Universe(top_file,"%3.1f/isspa_C_force.xyz" %(val))
        nAtoms = len(u.atoms)
    else:
        print("%4.1f" %(val))
        u = mda.Universe(top_file,"%4.1f/isspa_C_force.xyz" %(val))
        nAtoms = len(u.atoms)
    out.write("  %4.1f" %(val))
    for atom in u.atoms:
        out.write("  %12.6f  %12.6f  %12.6f" %(atom.position[0],atom.position[1],atom.position[2]))
    out.write("\n")
out.close

