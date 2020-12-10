import numpy as np
import sys
import MDAnalysis as mda

top_file="LJ2.prmtop"
traj_file="LJ2.positions.xyz"
frc_file=["Csolv_force.xyz","LJsolv_force.xyz","Csolv_CDD_force.xyz"]
u = mda.Universe(top_file,traj_file)

skip = 1

for name in frc_file:
    print(name)
    file = open(name,'r')
    lines = file.readlines()
    nAtoms = int(lines[0])
    nFrames = len(u.trajectory)
    atom = 0
    frame = -1
    frc = np.zeros((nFrames,nAtoms,3),dtype=float)
    for i,line in enumerate(lines):
        temp = line.split()
        if len(temp) == 1:
            atom = 0
            frame += 0.5
            continue
        else:
            frc[int(frame),atom,0] = float(temp[4])
            frc[int(frame),atom,1] = float(temp[5])
            frc[int(frame),atom,2] = float(temp[6])
            atom += 1

    avg = 0.0
    for ts in u.trajectory:    
        if ts.frame < skip:
            BA_dist = u.atoms[0].position -  u.atoms[1].position
            AB_dist = u.atoms[1].position -  u.atoms[0].position
            dist = np.linalg.norm(u.atoms[0].position-u.atoms[1].position)
            AB_norm = AB_dist/dist
            BA_norm = BA_dist/dist
            
            frc_norm = np.zeros(np.shape(frc[0]),dtype=float)
            dist = np.linalg.norm(frc[ts.frame,0])
            frc_norm[0] = frc[ts.frame,0]/dist
            dist = np.linalg.norm(frc[ts.frame,1])
            frc_norm[1] = frc[ts.frame,1]/dist
            
            align1 = np.dot(AB_norm,frc_norm[0])
            avg += align1
            align2 = np.dot(BA_norm,frc_norm[1])
            avg += align2
        
    avg /= nAtoms*skip    
    print(avg)
                
