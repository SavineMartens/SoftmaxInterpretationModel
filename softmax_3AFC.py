import numpy as np
import matplotlib.pyplot as plt
import glob

dir_to_loop = './IR/NH/*.npy'

# get IR
R_name = 
RT_max_name = 

R = np.load(R_name)
RT_max = np.load(RT_max_name)

# loop RT
files = glob.glob(dir_to_loop)
for file in files:
    RT = np.load(file)
    
    
# 