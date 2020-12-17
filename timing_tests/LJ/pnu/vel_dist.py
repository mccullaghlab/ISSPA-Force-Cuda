import numpy as np
import sys
import MDAnalysis as mda
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import glob
stdev = np.std
sqrt = np.sqrt
nullfmt = NullFormatter()


#def plot_3(vx, vy, vz):
#    b = 20
#    xhist, xedges = np.histogram(np.abs(vx),bins=b)
#    yhist, yedges = np.histogram(np.abs(vy),bins=b)
#    zhist, zedges = np.histogram(np.abs(vz),bins=b)
#    
#    plt.plot(xedges[1:], xhist, 'k',label=r"$V_{x}$")
#    plt.plot(yedges[1:], yhist, 'r',label=r"$V_{y}$")
#    plt.plot(zedges[1:], zhist, 'b',label=r"$V_{z}$")
#    plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
#    plt.xlabel(r'Velocities', size=12)
#    plt.ylabel(r'Counts', size=12)
#    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=1, fontsize = 'medium')
#    plt.xlim((0,0.8))
#    #plt.ylim((1.5, 3.0))
#    plt.savefig('vel_dist.png')
#    plt.close()

def plot_3(vx, vy, vz):
    b = 20
    xhist, xedges = np.histogram(np.abs(vx),bins=b)
    yhist, yedges = np.histogram(np.abs(vy),bins=b)
    zhist, zedges = np.histogram(np.abs(vz),bins=b)
    
    plt.plot(xedges[1:], xhist, 'k',label=r"pnu = 0.01")
    plt.plot(yedges[1:], yhist, 'r',label=r"pnu = 0.05")
    plt.plot(zedges[1:], zhist, 'b',label=r"pnu = 0.10")
    plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
    plt.xlabel(r'Velocities', size=12)
    plt.ylabel(r'Counts', size=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=1, fontsize = 'medium')
    plt.xlim((0,0.8))
    #plt.ylim((1.5, 3.0))
    plt.savefig('vel_dist.png')
    plt.close()

def plot_1(v):
    b = 20
    hist, edges = np.histogram(np.abs(v),bins=b)

    plt.plot(edges[1:], hist, 'k')
    plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
    plt.xlabel(r'Velocities', size=12)
    plt.ylabel(r'Counts', size=12)
    plt.xlim((0,0.8))
    #plt.ylim((1.5, 3.0))
    plt.savefig('vel_dist.png')
    plt.close()

def plot_1s(v):
    b = 20
    half = int(len(v)/2.0)
    hist1, edges1 = np.histogram(np.abs(v[:half]),bins=b)
    hist2, edges2 = np.histogram(np.abs(v[half:]),bins=b)

    plt.plot(edges1[1:], hist1, 'k',label="0")
    plt.plot(edges2[1:], hist2, 'r',label="1")
    plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
    plt.xlabel(r'Velocities', size=12)
    plt.ylabel(r'Counts', size=12)
    plt.xlim((0,0.8))
    #plt.ylim((1.5, 3.0))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),  shadow=True, ncol=1, fontsize = 'medium')
    plt.savefig('vel_dist.png')
    plt.close()



top_file = "LJ2.prmtop"



vel_file = "LJ2.velocities.10.xyz"
u = mda.Universe(top_file,vel_file)
#vx = []
#vy = []
#vz = []
v = []
for ts in u.trajectory:
    for atom in u.atoms:
        #vx.append(atom.position[0])
        #vy.append(atom.position[1])
        #vz.append(atom.position[2])
        v.append(atom.position[0])
        v.append(atom.position[1])
        v.append(atom.position[2])

#vx = np.array(vx)
#vy = np.array(vy)
#vz = np.array(vz)
v1 = np.array(v)

vel_file = "LJ2.velocities.50.xyz"
u = mda.Universe(top_file,vel_file)
#vx = []
#vy = []
#vz = []
v = []
for ts in u.trajectory:
    for atom in u.atoms:
        #vx.append(atom.position[0])
        #vy.append(atom.position[1])
        #vz.append(atom.position[2])
        v.append(atom.position[0])
        v.append(atom.position[1])
        v.append(atom.position[2])

#vx = np.array(vx)
#vy = np.array(vy)
#vz = np.array(vz)
v2 = np.array(v)

vel_file = "LJ2.velocities.100.xyz"
u = mda.Universe(top_file,vel_file)
#vx = []
#vy = []
#vz = []
v = []
for ts in u.trajectory:
    for atom in u.atoms:
        #vx.append(atom.position[0])
        #vy.append(atom.position[1])
        #vz.append(atom.position[2])
        v.append(atom.position[0])
        v.append(atom.position[1])
        v.append(atom.position[2])

#vx = np.array(vx)
#vy = np.array(vy)
#vz = np.array(vz)
v3 = np.array(v)

plot_3(v1, v2, v3)
