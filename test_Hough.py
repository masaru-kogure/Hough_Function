import matplotlib.pyplot as plt
from houghfunction_v3 import houghfunction as hf
import matplotlib
import numpy as np


hough, dhdx, hough_u, hough_v, h, x = hf.hough_function(nlat=94, N=62, s=4, sigma=2)
#hough, hough_u, hough_v, h, x = hf.hough_function(nlat=92, N=30)
houghn = hough * 0
houghn_u = hough_u * 0
houghn_v = hough_v * 0

for i in range(0, 6):
    hough[:,i] = hf.normalize(hough[:,i])
    hough_u[:,i] = hf.normalize(hough_u[:,i])
    hough_v[:,i] = hf.normalize(hough_v[:,i])
#hough_u = hf.normalize(hough_u)
gslat = np.arccos(x)/(2 * np.pi) * 360 -90
h = np.array([f"{num:.2f}" for num in h])
a = 12 * hough[:,5] 
print(np.sum(a * hough[:,5])/np.sum(hough[:,5] * hough[:,5]))
