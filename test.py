import numpy as np
import matplotlib.pyplot as plt
from redo import gauss_func, GaussianModel
import json


# file_path = "./src/prueba_streetCanyon.json"
# with open(file_path, "r") as file:
#     geojs = json.load(file)


# # # Configuration
RUNMODE = "plan"  # Generate the x-y view at z=0
STABILITY = {"condition": "constant", "value": 6.0}
WIND = {"condition": "constant", "speed": 2.0, "direction": 0.0}
STACKS = [(0.0, 0.0, 3.0)]  # Coordinates of the sources
flux = [0.76514]

xlims = (-25.0, 25.0)
ylims = xlims
zlims = (0.0, 10.0)
tlims = (0.0, 15.0)
deltas = {"dx": 0.1, "dy": 0.1, "dz": 0.5, "dt": 1.0}

# # # Calculation
solver = GaussianModel(xlims, ylims, zlims, tlims, runmode="plan", **deltas)
C = solver.solve(flux, STACKS, STABILITY, WIND)

print(C)

plt.pcolor(C[:, :, 0])
plt.show()
