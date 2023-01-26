###########################################################################
# GAUSSIAN PLUME MODEL FOR TEACHING PURPOSES                              #
# PAUL CONNOLLY (UNIVERSITY OF MANCHESTER, 2017)                          #
# THIS CODE IS PROVIDED `AS IS' WITH NO GUARANTEE OF ACCURACY             #
# IT IS USED TO DEMONSTRATE THE EFFECTS OF ATMOSPHERIC STABILITY,         #
# WINDSPEED AND DIRECTION AND MULTIPLE STACKS ON THE DISPERSION OF        #
# POLLUTANTS FROM POINT SOURCES                                           #
###########################################################################

import numpy as np
import sys
from scipy.special import erfcinv as erfcinv
import tqdm as tqdm
import time
import json
import pandas as pd
from gauss_func import gauss_func
import xarray as xr
from shapely.geometry import Point, Polygon

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})



def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_centers(x, y, coordinates = (2.175371592675371, 41.394719212134426)):
    
    centers = []
    coor = []
    for pos_y in y:
        for pos_x in x:
            
            centers.append([coordinates[0] + pos_x*9e-6, coordinates[1] + pos_y*9e-6])
            coor.append([pos_x,pos_y])
    return centers,coor

def get_cells(x,y,centers):
    
    cells =  []
    
    latitude = np.zeros((len(y) + 1, len(x) + 1))
    longitude = np.zeros((len(y) + 1, len(x) + 1))

    c = 0
    
    for i in range(len(y)):
        for j in range(len(x)):
            latitude[i,j] = centers[c][1] - 0.009/2
            longitude[i,j] = centers[c][0]  - 0.0115/2
            
            c = c+1

    i = len(y)

    for j in range(len(x)):
        latitude[i,j] = latitude[i-1,j] + (latitude[i-1,j] - latitude[i-2,j])
        longitude[i,j] = longitude[i-1,j] + (longitude[i-1,j] - longitude[i-2,j])

    j = len(x)

    for i in range(len(y)):
        latitude[i,j] = latitude[i,j-1] + (latitude[i,j-1] - latitude[i,j-2])
        longitude[i,j] = longitude[i,j-1] + (longitude[i,j-1] - longitude[i,j-2])

    i = len(y)
    j = len(x)
    
    latitude[i,j] = latitude[i,j-1] + (latitude[i,j-1] - latitude[i,j-2])
    longitude[i,j] = longitude[i,j-1] + (longitude[i,j-1] - longitude[i,j-2])
    
    for i in range(len(y)):
        for j in range(len(x)):
            cells.append([[
                    [
                      longitude[i,j] + 0,
                      latitude[i,j] + 0,
                    ],
                    [
                      longitude[i,j+1] + 0,
                      latitude[i,j+1] + 0,
                    ],
                    [
                      longitude[i+1,j+1] + 0,
                      latitude[i+1,j+1] + 0,
                    ],
                    [
                      longitude[i+1,j] + 0,
                      latitude[i+1,j] + 0,
                    ],
                    [
                      longitude[i,j] + 0,
                      latitude[i,j] + 0,
                    ]
                  ]])

    return cells

def get_flux(centers):
    with open('../BC01.json', 'r') as outfile:
        big = json.load(outfile)
    
    f = []
    
    for point in centers:
        p = Point(point)
        id = 0
        flag = True
        
        while id < len(big['features']) and flag:
            square = Polygon(big['features'][id]['geometry']['coordinates'][0])
            flag = not(p.within(square))
            id = id + 1
        
        id = id - 1
        
        data = xr.open_dataset(f'/home/hopu/Descargas/Catalunya/EMISIONES/BC01/EMI_2020/EMIS.BC01.01.NO2.s.nc' , mode = 'r')
        fluxes = data['NO2'][6,:,:,:,:,:]
        fluxes = fluxes.mean(dim = ['nlevel_emep', 'Time', 'type_day'])
        
        contador = 0
        i = 0
        
        
        noEncontrado = True
        
        while i < fluxes.shape[1] and noEncontrado:
            j = 0
            
            while j < fluxes.shape[0] and noEncontrado:
                
                
                if contador == id:
                    noEncontrado = False
                else:
                    contador = contador + 1
                    j = j + 1
            
            i = i + 1

        f.append((fluxes[i,j].data/fluxes.mean().data) + 1)
        
    return f

def get_factor(cells):
    with open('../barcelona-geoJSON.json', 'r') as outfile:
        big = json.load(outfile)
    
    factors = []
    
    for point in cells:
        p = Polygon(point[0])
        id = 0
        flag = True
        
        while id < len(big['features']) and flag:
            square = Polygon(big['features'][id]['geometry']['coordinates'][0])
            if p.overlaps(square) or p.within(square):
                flag = False
            
            id = id + 1
        
        if flag:
            factors.append(0.5)
        else:
            factors.append(big['features'][id]['properties']['multiplicador'])
            
    return factors

def to_geojson(x, y, coordinates = (2.175371592675371, 41.394719212134426)):
    
    result = {'type': "FeatureCollection",
              'features': []}
    
    centers, coor = get_centers(x, y, coordinates)

    cells = get_cells(x, y, centers)
    fluxes = get_flux(centers)
    factors = get_factor(cells)
    print(len(factors))
    for c in range(len(cells)):
        
        result['features'].append({"type": "Feature","id": c,"properties": {"id": c,'H': 2, "flux": float(fluxes[c]*factors[c]), "factor": float(factors[c]), "x": float(coor[c][0]), "y": float(coor[c][1]) },"geometry": {"coordinates": cells[c],"type": "Polygon"}})
    
    return result

def calibration(csv, station):
    pass
        
###########################################################################
# Do not change these variables                                           #
###########################################################################


# SECTION 0: Definitions (normally don't modify this section)
# view
PLAN_VIEW=1
HEIGHT_SLICE=2
SURFACE_TIME=3
NO_PLOT=4

# wind field
CONSTANT_WIND=1
FLUCTUATING_WIND=2
PREVAILING_WIND=3

# number of stacks
ONE_STACK=1
TWO_STACKS=2
THREE_STACKS=3

# stability of the atmosphere
CONSTANT_STABILITY=1
ANNUAL_CYCLE=2
stability_str=['Very unstable','Moderately unstable','Slightly unstable', \
    'Neutral','Moderately stable','Very stable']
# Aerosol properties
HUMIDIFY=2
DRY_AEROSOL=1

SODIUM_CHLORIDE=1
SULPHURIC_ACID=2
ORGANIC_ACID=3
AMMONIUM_NITRATE=4
nu=[2., 2.5, 1., 2.]
rho_s=[2160., 1840., 1500., 1725.]
Ms=[58.44e-3, 98e-3, 200e-3, 80e-3]
Mw=18e-3


dxy=100          # resolution of the model in both x and y directions
dz=10
x=np.mgrid[-2500:2500+dxy:dxy] # solve on a 5 km domain
y=x              # x-grid is same as y-grid

geojson = to_geojson(x, y, coordinates = (2.175371592675371, 41.394719212134426))

with open('prueba_streetCanyon.json', 'w') as outfile:
    json.dump(geojson, outfile)
 
###########################################################################



# SECTION 1: Configuration
# Variables can be changed by the user+++++++++++++++++++++++++++++++++++++
RH=0.90
aerosol_type=AMMONIUM_NITRATE

dry_size=60e-9
humidify=DRY_AEROSOL

stab1=6 # set from 1-6
stability_used=CONSTANT_STABILITY


output=PLAN_VIEW
x_slice=26 # position (1-50) to take the slice in the x-direction
y_slice=1  # position (1-50) to plot concentrations vs time

wind=CONSTANT_WIND

stack_x=[]
stack_y=[]

Q=[] # mass emitted per unit time
H=[] # stack height, m
factor=[]

for feature in geojson['features']:
    stack_x.append(feature['properties']['x'])
    stack_y.append(feature['properties']['y'])
    
    Q.append(feature['properties']['flux']) # mass emitted per unit time
    H.append(feature['properties']['H']) # stack height, m
    factor.append(feature['properties']['factor'])

stack_x=np.array(stack_x)
stack_y=np.array(stack_y)

Q=np.array(Q) # mass emitted per unit time
H=np.array(H) # stack height, m
factor=np.array(factor)

stacks=len(stack_x)

days=1          # run the model for 365 days
#--------------------------------------------------------------------------
times=np.mgrid[1:(days)*24+1:1]/24.

Dy=10.
Dz=10.

# SECTION 2: Act on the configuration information

# Decide which stability profile to use
if stability_used == CONSTANT_STABILITY:
   
   stability=stab1*np.ones((days*24,1))
   stability_str=stability_str[stab1-1]
elif stability_used == ANNUAL_CYCLE:

   stability=np.round(2.5*np.cos(times*2.*np.pi/(365.))+3.5)
   stability_str='Annual cycle'
else:
   sys.exit()


# decide what kind of run to do, plan view or y-z slice, or time series
if output == PLAN_VIEW or output == SURFACE_TIME or output == NO_PLOT:

   C1=np.zeros((len(x),len(y),days*24)) # array to store data, initialised to be zero

   [x,y]=np.meshgrid(x,y) # x and y defined at all positions on the grid
   z=np.zeros(np.shape(x))    # z is defined to be at ground level.
elif output == HEIGHT_SLICE:
   z=np.mgrid[0:500+dz:dz]       # z-grid

   C1=np.zeros((len(y),len(z),days*24)) # array to store data, initialised to be zero

   [y,z]=np.meshgrid(y,z) # y and z defined at all positions on the grid
   x=x[x_slice]*np.ones(np.shape(y))    # x is defined to be x at x_slice       
else:
   sys.exit()



# Set the wind based on input flags++++++++++++++++++++++++++++++++++++++++
wind_speed=5.*np.ones((days*24,1)) # m/s
if wind == CONSTANT_WIND:
   wind_dir=0.*np.ones((days*24,1))
   wind_dir_str='Constant wind'
elif wind == FLUCTUATING_WIND:
   wind_dir=360.*np.random.rand(days*24,1)
   wind_dir_str='Random wind'
elif wind == PREVAILING_WIND:
   wind_dir=-np.sqrt(2.)*erfcinv(2.*np.random.rand(24*days,1))*40. #norminv(rand(days.*24,1),0,40)
   # note at this point you can add on the prevailing wind direction, i.e.
   # wind_dir=wind_dir+200
   wind_dir[np.where(wind_dir>=360.)]= \
        np.mod(wind_dir[np.where(wind_dir>=360)],360)
   wind_dir_str='Prevailing wind'
else:
   sys.exit()
#--------------------------------------------------------------------------



# SECTION 3: Main loop
# For all times...
C1=np.zeros((len(x),len(y),len(wind_dir)))
for i in tqdm.tqdm(range(0,len(wind_dir))):  # Loop for 
   for j in range(0,stacks):
        C=np.ones((len(x),len(y)))
        C=gauss_func(Q[j],wind_speed[i],wind_dir[i],x,y,z,
            stack_x[j],stack_y[j],H[j],Dy,Dz,stability[i])
        C1[:,:,i]=C1[:,:,i]+C



# SECTION 4: Post process / output

# decide whether to humidify the aerosol and hence increase the mass
if humidify == DRY_AEROSOL:
   print('do not humidify')
elif humidify == HUMIDIFY:
   mass=np.pi/6.*rho_s[aerosol_type]*dry_size**3.
   moles=mass/Ms[aerosol_type]
        
   nw=RH*nu[aerosol_type]*moles/(1.-RH)
   mass2=nw*Mw+moles*Ms[aerosol_type]
   C1=C1*mass2/mass 
else:
   sys.exit()


# SECTION 5: Create Dataframe
C_mean = np.mean(C1,axis=2)*1e6
concentration = []

for c in C_mean:
    for cc in c:
        concentration.append(round(cc,2))

csv = pd.DataFrame()

csv['NOx'] = concentration
csv['NOx'] = csv['NOx']

csv = csv.iloc [0:len(geojson['features']),:]
csv['id'] = csv.index
csv['flux'] = Q
csv['factor'] = factor

reference = pd.read_csv('../Urbano/ES1438A.csv')
NOx = reference['NO2_reference'] + reference['NO_reference']
calibration_factor = csv['NOx'].mean()/NOx.mean()
print(calibration_factor)
csv['NOx_calibrated'] = csv['NOx']/calibration_factor

csv.to_csv('prueba.csv', index = False)

# output the plots
if output == PLAN_VIEW:
   plt.figure()
   plt.ion()
   
   plt.pcolor(x,y,np.mean(C1,axis=2)*1e6, cmap='jet')
   plt.title(stability_str + '\n' + wind_dir_str)
   plt.xlabel('x (metres)')
   plt.ylabel('y (metres)')
   cb1=plt.colorbar()
   cb1.set_label('ug/m3')
   plt.show()

elif output == HEIGHT_SLICE:
   plt.figure()
   plt.ion()
   
   plt.pcolor(y,z,np.mean(C1,axis=2)*1e6, cmap='jet')      
   plt.clim((0,1e2))
   plt.xlabel('y (metres)')
   plt.ylabel('z (metres)')
   plt.title(stability_str + '\n' + wind_dir_str)
   cb1=plt.colorbar()
   cb1.set_label('ug/m3')
   plt.show()

elif output == SURFACE_TIME:
   f,(ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
   ax1.plot(times,1e6*np.squeeze(C1[y_slice,x_slice,:]))
   try:
      ax1.plot(times,smooth(1e6*np.squeeze(C1[y_slice,x_slice,:]),24),'r')
      ax1.legend(('Hourly mean','Daily mean'))
   except:
      sys.exit()
      
   ax1.set_xlabel('time (days)')
   ax1.set_ylabel('Mass loading (ug/m3)')
   ax1.set_title(stability_str +'\n' + wind_dir_str)

   ax2.plot(times,stability)
   ax2.set_xlabel('time (days)')
   ax2.set_ylabel('Stability parameter')
   f.show()
   
elif output == NO_PLOT:
   print('don''t plot')
else:
   sys.exit()
   




