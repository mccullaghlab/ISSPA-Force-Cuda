# -*- coding: utf-8 -*-
"""
Example script with basic usage of the EMUS package.  The script follows the quickstart guide closely, with slight adjustments (for simplicity we have moved all plotting commands to the bottom of the script).
"""
import numpy as np                  
from emus import emus, avar
from emus import usutils as uu
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

# Define Simulation Parameters
T = 300                             # Temperature in Kelvin
k_B = 1.9872041E-3                  # Boltzmann factor in kcal/mol
kT = k_B * T
meta_file = 'metadata.dat'         # Path to Meta File
dim = 1                             # 1 Dimensional CV space.
#period = 360                         # Dihedral Angles periodicity

# Load data
#psis, cv_trajs, neighbors = uu.data_from_meta(meta_file,dim,T=T,k_B=k_B,period=period)
psis, cv_trajs, neighbors = uu.data_from_meta(meta_file,dim,T=T,k_B=k_B)
#psis, cv_trajs, neighbors = uu.data_from_meta(meta_file,dim,T=T,k_B=k_B,nsig=6)
nbins = 200                          # Number of Histogram Bins.

# Calculate the partition function for each window
z, F = emus.calculate_zs(psis,neighbors=neighbors)

# Calculate error in each z value from the first iteration.
zerr, zcontribs, ztaus  = avar.calc_partition_functions(psis,z,F,iat_method='acor')
#zerr, zcontribs, ztaus  = avar.calc_partition_functions(psis,z,F,neighbors,iat_method='acor')

# Calculate the PMF from EMUS
#domain = ((-180.0,180.))            # Range of dihedral angle values
domain = ((3.0,16.0))            # Range of length values
pmf,edges = emus.calculate_pmf(cv_trajs,psis,domain,z,nbins=nbins,kT=kT,use_iter=False)   # Calculate the pmf

# Calculate z using the MBAR iteration.
z_iter_1, F_iter_1 = emus.calculate_zs(psis,n_iter=1)
z_iter_2, F_iter_2 = emus.calculate_zs(psis,n_iter=2)
z_iter_5, F_iter_5 = emus.calculate_zs(psis,n_iter=5)
z_iter_1k, F_iter_1k = emus.calculate_zs(psis,n_iter=1000)
#z_iter_1, F_iter_1 = emus.calculate_zs(psis,neighbors=neighbors,n_iter=1)
#z_iter_2, F_iter_2 = emus.calculate_zs(psis,neighbors=neighbors,n_iter=2)
#z_iter_5, F_iter_5 = emus.calculate_zs(psis,neighbors=neighbors,n_iter=5)
#z_iter_1k, F_iter_1k = emus.calculate_zs(psis,neighbors=neighbors,n_iter=1000)

# Calculate new PMF
iterpmf,edges = emus.calculate_pmf(cv_trajs,psis,domain,nbins=nbins,z=z_iter_1,kT=kT)
#iterpmf,edges = emus.calculate_pmf(cv_trajs,psis,domain,neighbors=neighbors,nbins=nbins,z=z_iter_1k,kT=kT)


# Get the asymptotic error of each histogram bin.
pmf_av_mns, pmf_avars = avar.calc_pmf(cv_trajs,psis,domain,z,F,nbins=nbins,kT=kT,iat_method=np.average(ztaus,axis=0))
#pmf_av_mns, pmf_avars = avar.calc_pmf(cv_trajs,psis,domain,z,F,neighbors=neighbors,nbins=nbins,kT=kT,iat_method=np.average(ztaus,axis=0))

### Data Output Section ###

# Plot the EMUS, Iterative EMUS pmfs.
pmf_centers = (edges[0][1:]+edges[0][:-1])/2.


out=open("EMUS_PMF.dat",'w')
out2=open("EMUS_C_vals.dat",'w')
for i in range(len(pmf_centers)):
    out.write("  %10.5f  %10.5f  %10.5f\n" %(pmf_centers[i], (pmf_av_mns[i]+1.2*np.log(pmf_centers[i])), np.sqrt(pmf_avars[i]))) 
    #out.write("  %10.5f  %10.5f  %10.5f\n" %(pmf_centers[i], (pmf_av_mns[i]+1.2*np.log(pmf_centers[i])), np.sqrt(pmf_avars[i]))) 
out.close
for i in range(len(z_iter_1)):
    out2.write("  %10.5f  %10.10f\n" %(-np.log(z_iter_1[i])*kT,kT*zerr[i]/z_iter_1[i]))
out2.close

plt.figure()
plt.errorbar(pmf_centers,pmf_av_mns,yerr=np.sqrt(pmf_avars),label='EMUS PMF w. AVAR')
plt.plot(pmf_centers,iterpmf,label='Iter EMUS PMF')
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel('$\AA$ Distance', size=16)
plt.ylabel('Unitless FE', size=16)
plt.legend()
plt.title('EMUS and Iterative EMUS potentials of Mean Force', size=16)
plt.savefig("EMUS_PMF.png")
plt.savefig("EMUS_PMF.eps")
plt.savefig("EMUS_PMF.pdf")
plt.close()

# Plot the relative normalization constants as fxn of max iteration. 
#plt.errorbar(np.arange(len(z)),-np.log(z),yerr=np.sqrt(zerr)/z,label="Iteration 0")
plt.plot(-np.log(z_iter_1),label="Iteration 1")
plt.plot(-np.log(z_iter_2),label="Iteration 2",linestyle='--')
plt.plot(-np.log(z_iter_5),label="Iteration 5",linestyle='--')
plt.plot(-np.log(z_iter_1k),label="Iteration 1k",linestyle='--')
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.xlabel('Window Index', size=16)
plt.ylabel('Unitless Free Energy', size=16)
plt.title('Window Free Energies and Iter No.', size=16)
plt.legend(loc='upper left')
plt.savefig("rel_norm_const.png")
plt.savefig("rel_norm_const.eps")
plt.savefig("rel_norm_const.pdf")
plt.close()


print("Asymptotic coefficient of variation for each partition function:")
print(np.sqrt(zerr)/z)

