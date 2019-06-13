


nTypes = 21
f = open("isspa_density_params.dat",'w')
for i in range(nTypes):
    f.write("%15.8f %15.8f %15.8f\n" % (1.714, 4.118, 2.674))
f.close()

nTypes = 21
nRs = 160
f = open("isspa_forces.dat",'w')
for i in range(nRs):
    f.write("%15.8f" % ((i+0.5)*0.1))
    for j in range(nTypes):
        f.write("%15.8f" % (1.714))
    f.write("\n")
f.close()

