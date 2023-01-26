import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from numpy.fft import fft, ifft, fftshift, fftfreq 


path_to_csv = "/home/ngomariz/Escritorio/kusto-querys/dataframe.csv"
dframe = pd.read_csv(path_to_csv)

"""
    LAeq --> FFT --> F
    F --> sqF = |F|²
    |F|² --> iFFT --> Autocorrelation
"""

# # # Colums
times = dframe["timedelta"]
LAeq = dframe["LAeq"].to_numpy()

# # # FFT, iFFT
F = fft(LAeq, norm="ortho")
sqF = F * np.conjugate(F)
acorr = ifft(sqF)

# # # Plots
plt.plot(times, LAeq, '--', label=r'$L \equiv L_{Aeq}$')
plt.plot(times, acorr, label=r'$L \star L$')
plt.legend()
plt.grid()
plt.show()
