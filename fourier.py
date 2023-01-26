import pandas as pd 
import numpy as np
from numpy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from librosa import stft
from librosa.display import specshow

# # # Extract dataset
path_to_csv = "/home/ngomariz/Escritorio/kusto-querys/dataframe.csv"
dframe = pd.read_csv(path_to_csv)

# # # Fill NaN values
dframe.fillna(method='bfill', inplace=True)

# # # Create a timedelta column
dframe["timestamp"] = pd.to_datetime(dframe["timestamp"])
t0 = dframe["timestamp"][0]
dframe["timedelta"] = pd.to_timedelta(
    dframe["timestamp"] - pd.Series([t0] * len(dframe))
).dt.total_seconds()/3600
dframe.to_csv(path_to_csv)


# # # Plot LAeq time evolution
plt.plot(dframe["timedelta"], dframe["LAeq"])
plt.grid()
plt.show()

# # # Calculation of FFT
N = len(dframe)
laeq = dframe["LAeq"].to_numpy()
fourier = fft(laeq) / np.sqrt(len(dframe))
fourier = fourier[:N//2]

#### Frequency space in days/2^{-1}####
sr = 24 # Un sample cada hora --> 24 samples por día
freqs = fftfreq(N, d=1/sr)[:N//2]

plt.plot(freqs, np.abs(fourier))
plt.grid()
plt.show()
