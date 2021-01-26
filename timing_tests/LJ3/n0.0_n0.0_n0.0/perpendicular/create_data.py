import numpy as np
import sys
import MDAnalysis as mda
import glob

top_file = "LJ3.prmtop"

values1 = (3.5,5.0,7.0,10.0,12.0,15.0)
values2 = np.arange(3.5, 15.1, 0.1)
out = open("LJ3.forces.combined.dat",'w')
for val1 in values1:
    for val2 in values2:
        print("%.1f  %.1f" %(val1,val2))
        u = mda.Universe(top_file,"%.1f/%.1f/LJ3.%.1f_%.1f.forces.xyz" %(val1,val2,val1,val2))
        nAtoms = len(u.atoms)
        out.write("  %4.1f  %4.1f" %(val1,val2))
        for atom in u.atoms:
            out.write("  %12.6f  %12.6f  %12.6f" %(atom.position[0],atom.position[1],atom.position[2]))
        out.write("\n")
out.close

out = open("LJ3.lj_forces.combined.dat",'w')
for val1 in values1:
    for val2 in values2:
        print("%.1f  %.1f" %(val1,val2))
        u = mda.Universe(top_file,"%.1f/%.1f/isspa_lj_force.xyz" %(val1,val2))
        nAtoms = len(u.atoms)
        out.write("  %4.1f  %4.1f" %(val1,val2))
        for atom in u.atoms:
            out.write("  %12.6f  %12.6f  %12.6f" %(atom.position[0],atom.position[1],atom.position[2]))
        out.write("\n")
out.close

out = open("LJ3.c_forces.combined.dat",'w')
for val1 in values1:
    for val2 in values2:
        print("%.1f  %.1f" %(val1,val2))
        u = mda.Universe(top_file,"%.1f/%.1f/isspa_C_force.xyz" %(val1,val2))
        nAtoms = len(u.atoms)
        out.write("  %4.1f  %4.1f" %(val1,val2))
        for atom in u.atoms:
            out.write("  %12.6f  %12.6f  %12.6f" %(atom.position[0],atom.position[1],atom.position[2]))
        out.write("\n")
out.close


