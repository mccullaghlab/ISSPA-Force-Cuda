import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def plot_1d(xdata1, ydata1, yerr1, color1, x_axis, y_axis, system):
        plt.errorbar(xdata1, ydata1, yerr1, color=color1,label="EMUS",errorevery=3,elinewidth=1.5)
        plt.grid(b=True,which='major',axis='both',color='#808080',linestyle='--')
        plt.xlabel(r'%s' %(x_axis), size=16)
        plt.ylabel(r'%s' %(y_axis), size=16)
        plt.xlim((0,16))
        plt.ylim((-10, 4.0))
        plt.savefig('%s.PMF.png' %(system))
        plt.savefig('%s.PMF.pdf' %(system))
        plt.savefig('%s.PMF.eps' %(system))
        plt.close()
        
        
def parse_meta_file(meta_file):
        # READ META DATA FILE TO SUBSEQUENTLY ANALYZE THE CORRECT DATASET
        file_list = []
        data_list = []
        with open(meta_file,'r') as f:
                for line in f:
                        temp = line.split()
                        if temp[0] == '#':
                                continue
                        else:
                                file_list.append(temp[0])
                                data_list.append([float(temp[1]),float(temp[2])])                                
        data_list = np.array(data_list)
        nProds = len(file_list)

        return file_list, data_list, nProds

def parse_PMF_file(EMUS_file,C_val_file):
	global bin_centers, free_energy, fe_err, C_values
	bin_centers_l = []
	free_energy_l = []
	fe_err_l = []
	C_values_l = []	
	zerr_l = []
	with open(EMUS_file,'r') as f:
		for line in f:
			temp = line.split()
			if temp[1] != "inf":
				bin_centers_l.append(float(temp[0]))
				free_energy_l.append(float(temp[1]))
				fe_err_l.append(float(temp[2]))
	with open(C_val_file,'r') as f:
		for line in f:
			temp = line.split()
			C_values_l.append(float(temp[0]))
			zerr_l.append(temp[1])
			
	bin_centers = np.array(bin_centers_l)
	free_energy = np.array(free_energy_l)
	fe_err = np.array(fe_err_l)
	C_values = np.array(C_values_l)
	zerr = np.array(zerr_l)
	return bin_centers, free_energy, fe_err, C_values, zerr


# -------------------------------------------------
# READ IN DATA
EMUS_file="EMUS_PMF.dat"
C_val_file="EMUS_C_vals.dat"
meta_file ="metadata.dat"
system="ADI2"
k = 0.001987 # Kcal K^-1 mol^-1
T = 298.
kT = k*T
four_pi = 4*np.pi
count = 0 
zeros = np.zeros
boltz = 2*kT
nBins = 200

file_list, data_list, nProds = parse_meta_file(meta_file)
bin_centers, free_energy, fe_error, C_values, zerr = parse_PMF_file(EMUS_file,C_val_file)


# -------------------------------------------------
# CREATE PMF WITH ERROR BARS

# shift PMF so that at far distances it goes to zero
#shift = np.max(free_energy[len(free_energy)-3])
shift = np.max(free_energy[-1])
free_energy -= shift
print("Minimum Free Energy: %f" %(np.min(free_energy)))
out = open("%s.shifted_pmf.dat" %(system),'w')
for i in range(len(free_energy)):
        out.write("  %12.8e  %12.8e  %12.8e\n" %(bin_centers[i],free_energy[i],fe_error[i]))
out.close
# plot PMF with error bars
plot_1d(bin_centers, free_energy, fe_error, 'k', "Distance $\AA$", "Relative Free Energy (kcal mol$^{-1}$)", system)


# -------------------------------------------------
# CREATE UNSTITCHED PMF AND PROBABILITY DISTRIBUTIONS
# LOOP THROUGH ALL DATA FILES, COLLECT DATA, HISTOGRAM DATA INTO FREQ, PROB DENSITY, AND FREE ENERGY COUNTERS
#plt.figure(2)
#plt.errorbar(bin_centers, free_energy, fe_error, color='k',label="PMF",errorevery=5,elinewidth=2)

for k in range(nProds):
        i = nProds - k - 1
        print(i)
        # loading file into a numpy array
        with open('%s' %(file_list[i]),'r') as f:
                temp = np.loadtxt(f,dtype=np.float)
        # collecting data to be used for creating the histograms
        x_min = np.ndarray.min(temp[:,1])
        x_max = np.ndarray.max(temp[:,1])
        delta_x = (x_max - x_min)/nBins
        nValues = len(temp)
        prob_density_divisor = nValues*delta_x
        fe_divisor = prob_density_divisor*four_pi

        half_bins = zeros(nBins)
        for j in range(nBins):
                half_bins[j] = x_min + delta_x*(j+0.5)
        counts = zeros(nBins)           # binning data with no weighting to use later as a counter
        prob_density = zeros(nBins)     # binning data with prob density weighting to observe shape of distribution and overlap between windows
        fe_counts = zeros(nBins)        # binning data with boltzmann weighting to subsequently calc free energy within a window
        for j in range(nValues):
                exponent = data_list[i][1]*(temp[j][1] - data_list[i][0])**2/boltz
                index = int((temp[j][1]-x_min)/delta_x)
                if index == nBins:
                        counts[-1] += 1
                        prob_density[-1] += 1/prob_density_divisor
                        fe_counts[-1] += 1/(fe_divisor*temp[j][1]**2*np.exp(-exponent))
                else:
                        counts[index] += 1
                        prob_density[index] += 1/prob_density_divisor
                        fe_counts[index] += 1/(fe_divisor*temp[j][1]**2*np.exp(-exponent))

        # HISTOGRAM PROB DENSITY OF ALL DATA FILES ONTO THE SAME PLOT; ALLOWS FOR VISUALIZATION OF OVERLAP BETWEEN WINDOWS AND VARIATIONS IN PRODUCTION RUN DISTRIBUTIONS WITHIN EACH WINDOW                                                              
        plt.figure(1)
        plt.plot(half_bins[:],prob_density[:])
	#plt.errorbar(half_bins[:],prob_density[:], zerr[:] ,label="EMUS",errorevery=3,elinewidth=2)
        for j in range(nBins):
                fe_counts[j] = -kT*np.log(fe_counts[j])         # taking negative log of the boltzmann weighted fe counter;
                fe_counts[j] += C_values[i]     # subtracting out the constant used in WHAM to align the windows with each other; this will align the unstitched free energy surfaces, allowing for comparison of overlap of windows
        # PLOT THE FREE ENERGY SURFACE FOR EACH WINDOW ONTO THE SAME PLOT; ALLOWS FOR VISUALIZATION/COMPARISON OF FREE ENERGY SURFACES BEFORE BEING STITCHED TOGETHER BY WHAM; IF GAPS OR LARGE VALUE DISPARITIES BETWEEN WINDOWS ARE PRESENT, THIS INDICATES THAT WINDOWS ARE SAMPLING DIFFERENT DISTRIBUTIONS (AKA NOT ERGODIC)
        plt.figure(2)
        if k == 0:
                shift = fe_counts[i]
        plt.plot(half_bins[counts > 40],fe_counts[counts > 40]-shift)
plt.figure(1)
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.ylabel('Probability Density')
plt.xlabel(r'Distance ($\AA$)',size=14)
plt.savefig('%s.data_histogram.png' %(system),dpi=300)
plt.savefig('%s.data_histogram.eps' %(system),dpi=300)
plt.savefig('%s.data_histogram.pdf' %(system),dpi=300)
plt.close()

plt.figure(2)
plt.grid(b=True, which='major', axis='both', color='#808080', linestyle='--')
plt.ylabel(r'Relative Free Energy (kCal mol$^{-1}$)',size=14)
plt.xlabel(r'Distance ($\AA$)',size=14)
plt.savefig('%s.unstitched_fe.png' %(system),dpi=300)
plt.savefig('%s.unstitched_fe.pdf' %(system),dpi=300)
plt.savefig('%s.unstitched_fe.pdf' %(system),dpi=300)
plt.close()
