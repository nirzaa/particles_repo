from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
xdata = np.array([ -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ydata = np.array([1.2, 4.2, 6.7, 8.3, 10.6, 11.7, 13.5, 14.5, 15.7, 16.1, 16.6, 16.0, 15.4, 14.4, 14.2, 12.7, 10.3, 8.6, 6.1, 3.9, 2.1])

xdata = np.array([169.65077989, 174.70147193, 179.75216397, 184.80285601,
       189.85354805, 194.9042401 , 199.95493214, 205.00562418,
       210.05631622, 215.10700826, 220.1577003 , 225.20839235,
       230.25908439, 235.30977643, 240.36046847, 245.41116051,
       250.46185255, 255.5125446 , 260.56323664, 265.61392868,
       270.66462072, 275.71531276, 280.7660048 , 285.81669685,
       290.86738889, 295.91808093, 300.96877297, 306.01946501,
       311.07015705, 316.1208491 , 321.17154114])[:-1]

ydata = np.array([ 2,  2,  7, 13, 23, 25, 31, 30, 17,  8,  6,  7,  1,  1,  0,  1,  0,
        0,  0,  1,  0,  1,  1,  0,  0,  0,  0,  1,  0,  1])


xdata = xdata[ydata > 1]
ydata = ydata[ydata > 1]

ymean = np.mean(ydata)
ystd = np.std(ydata) * 2

# Recast xdata and ydata into numpy arrays so we can use their handy features
ydata_original = np.asarray(ydata)
ydata = ydata_original[np.logical_and(ydata_original < ymean+ystd, ydata_original > ymean-ystd)]
xdata = np.asarray(xdata)
xdata = xdata[np.logical_and(ydata_original < ymean+ystd, ydata_original > ymean-ystd)]


plt.plot(xdata, ydata, 'o')
  
# Define the Gaussian function
def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y
parameters, covariance = curve_fit(Gauss, xdata, ydata)
  
fit_A = parameters[0]
fit_B = parameters[1]
  
fit_y = Gauss(xdata, fit_A, fit_B)
plt.plot(xdata, ydata, 'o', label='data')
plt.plot(xdata, fit_y, '-', label='fit')
plt.legend()
plt.savefig('shan_scripts/sand.jpg')