import numpy as np
import os
import json
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import matplotlib.pyplot as plt

txt_file = 'num_grasps_2.txt'

with open(txt_file, 'r') as f:
    lines = f.readlines()
    lines = [int(line.strip()) for line in lines if line.strip()]
    
print(lines[0])

# visualize
plt.figure(figsize=(10, 5))
plt.plot(lines, label='Number of Grasps', color='blue')
plt.xlabel('Index')
plt.ylabel('Number of Grasps')
plt.title('Number of Grasps Over Indices')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
