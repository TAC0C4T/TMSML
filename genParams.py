import random
from sys import argv
import csv
from os import mkdir, remove

mkdir("params")

for i in range(int(argv[1])):

    with open(f"params/params{i}.csv", 'w', newline='') as f:
        for i in range(int(argv[2])):
            writer = csv.writer(f)
            width = random.randrange(10, 300)
            ipi = random.randrange(100, 300)
            pulses = random.randrange(1, 20)
            writer.writerow(["Rectangular",width,1/(float(ipi)/1000),ipi,pulses,255,30,5,0,1000,2,"-47.79,74.76,58.94","-41.2,71.4,55.3","1.0,0.0,0.0","0.0,1.0,0.0",2.111602,453.99443,215])