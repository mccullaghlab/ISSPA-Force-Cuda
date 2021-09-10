import numpy as np
import sys



names = ["LJsolu","Csolu","Csolv_pair","Csolv","LJsolv","Csolv_CDD"]
#names = ["LJsolu","Csolu","Csolv_pair","LJsolv","Csolv"]

for name in names:
    file = open("ISSPA_force_%s_10000.xyz" %(name),'r')
    lines = file.readlines()
    start=[]
    stop = []
    for num,line in enumerate(lines):
        if num == 0:
            nAtoms = int(line)
        temp = line.split()
        if len(temp) == 1:
            if len(lines[num+1].split()) != 1:
                start.append(num+1)
            else:
                if num != 0:
                    stop.append(num-1)
        if num == len(lines)-1:
            stop.append(num)
            
    avg = np.zeros((nAtoms,3),dtype=float)
    for i in range(len(start)):
        count = 0
        for j in range(start[i],stop[i]):
            temp = lines[j].split()
            avg[count,0] += float(temp[4])
            avg[count,1] += float(temp[5])
            avg[count,2] += float(temp[6])
            count += 1
    avg /= len(start)

    file2 = open("ADI2.multi.frc.%s.dat" %(name),'r')
    lines = file2.readlines()
    count = 0
    avg_ratio = 0
    for i,line in enumerate(lines):
        temp = line.split()
        avg_ratio += avg[count,0]/float(temp[3])
        avg_ratio += avg[count,1]/float(temp[4])
        avg_ratio += avg[count,2]/float(temp[5])
        count += 1
    avg_ratio /= count*3
    
    print name,avg_ratio
