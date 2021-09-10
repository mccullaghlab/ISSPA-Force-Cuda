import numpy as np
import sys

file = open("ADI2.run0.window.16.0.xyz")
lines = file.readlines()

out = open("fixed.xyz",'w')
for line in lines:
    temp = line.split()
    if int(temp[0]) == 1:
        out.write("44\n")
        out.write("44\n")
        out.write(line)
    else:
        out.write(line)

out.close()
