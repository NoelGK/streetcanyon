import numpy as np
import sys
import time
import json
from calc_sigmas import calc_sigmas



def gauss_func(x, y, z, x0, y0, z0, Q, wind_speed, wind_dir, stability="constant"):
    """
    Args:
        x : x
        y : y
        z : z
        x0 : x coordinate of the source. Defaults to None.
        y0 : y coordinate of the source. Defaults to None.
        Q : mass emitted per unit time
        wind_mag : wind absolute value
        wind_dir : wind direction

    Returns:
        _type_: _description_
    """
    # # # Make stack at x0, y0 be the center
    x = x - x0 
    y = y - y0
    d = np.sqrt(x**2 + y**2)  # distance from each point to the center
    
    wx, wy = wind_speed * np.sin( (wind_dir - 180)*np.pi/180 ), wind_speed * np.cos( (wind_dir - 180)*np.pi/180 )

    # # # Calculate dot product of coordinates and wind to obtain the angle at each point
    dot_product = wx*x + wy*y
    cosine = dot_product / (d * wind_speed)
    sine = np.sqrt(1 - cosine**2)

    # # # Calculate the vertical and horizontal components of wind.
    downwind = cosine * d
    crosswind = sine * d

    # # # Indices where downwind > 0
    indx = np.where(downwind > 0.0)

    # # # Shape of the function
    C = np.zeros((x.shape[0], y.shape[1]))
    sigma_y, sigma_z = calc_sigmas(stability, downwind)
    exps = np.exp( -(z[indx] - z0)**2 / (2 * sigma_y[indx]**2) ) + np.exp( -(z[indx] + z0)**2 / (2 * sigma_y[indx]**2) )

    C[indx] = Q / ( 2* np.pi * wind_speed * sigma_y[indx] * sigma_z[indx] ) \
        * np.exp( -crosswind[indx]**2 / (2 * sigma_y[indx]**2) ) \
        * exps
    
    return C


def create_space(xlims, ylims, zlims, **kwargs):  # Cambiar para que devuelve x, y, z con shape=(Nx, Ny)
    x = np.mgrid[xlims[0] : xlims[1] : kwargs["dx"]]
    y = np.mgrid[ylims[0] : ylims[1] : kwargs["dy"]]
    z = np.mgrid[zlims[0] : zlims[1] : kwargs["dz"]]
    return x, y, z, np.ones(x.shape), np.zeros(x.shape)


class GaussianModel:
    def __init__(self, xlims, ylims, zlims, tlims, runmode, **kwargs):
        """Initialize GaussianModel class

        Args:
            xlims (tuple): 
            ylims (tuple): 
            zlims (tuple): 
            runmode (str): 
        """
        self.runmode = runmode
        self.x, self.y, self.z = self.set_space(xlims, ylims, zlims, **kwargs)
        self.t = self.set_time(tlims, **kwargs)

    def set_space(self, xlims, ylims, zlims, **kwargs):
        """
            Formats the solution space depending on the kind of run:
            "plan" --> x, y ~ variables, z ~ ground level
            "hslice" --> y, z ~ variables, x ~ fixed value

        Args:
            lims: lower/upper limits of x, y and z
            kwargs: dx, dy, dz, x_value
        """
        if self.runmode == "plan":
            x, y, _, _, z = create_space(xlims, ylims, zlims, **kwargs)
            x, y = np.meshgrid(x, y)

        elif self.runmode == "hslice":
            _, y, z, x, _ = create_space(xlims, ylims, zlims, **kwargs)
            y, z = np.meshgrid(y, z)
            x *= kwargs["x_value"]

        return x, y, z

    def set_time(self, tlims, **kwargs):
        """Set the time values for the solution.

        Args:
            tlims (tuple): lower and upper limits of time domain
            kwargs = {"dt": _}

        Returns:
            np.array: [t0, t1, t2, ...]
        """
        return np.mgrid[tlims[0] : tlims[1] : kwargs["dt"]]

    def set_conditions(self, stability, wind):
        # # # Stability conditions
        if stability["condition"] == "constant":
            stab = stability["value"] * np.ones(len(self.t))
        else:
            print("Not supported stability conditions " + stability["condition"])
        
        if wind["condition"] == "constant":
            wind_speed = wind["speed"] * np.ones(len(self.t))
            wind_dir = wind["direction"] * np.ones(len(self.t))
        else:
            print("Not supported wind conditions " + wind['condition'])

        return stab, wind_speed, wind_dir

    def set_shape(self):
        if self.runmode == "plan":
            shape = (self.x.shape[1], self.y.shape[0], len(self.t))
        else:
            shape = (self.y.shape[1], self.z.shape[0], len(self.t))

        return shape

    def solve(self, Q, stacks, stability, wind):
        """Solves the concentrations for each time and point in the domain

        Args:
            Q (_type_): _description_
            stacks (_type_): _description_
            stability (_type_): _description_
            wind (_type_): _description_

        Returns:
            _type_: _description_
        """
        # # # Set time space
        shape = self.set_shape()
        
        # # # Set stability and wind conditions 
        stability, wind_speed, wind_dir = self.set_conditions(stability, wind)

        # # # Calculation of solution
        C = np.zeros(shape=shape)

        # Loop for each time index
        for (i, speed_i), dir_i, stab in zip(enumerate(wind_speed), wind_dir, stability):
            # Loop for each source
            for (x0, y0, z0), Qi in zip(stacks, Q):
                C[:, :, i] += gauss_func(
                    self.x, self.y, self.z, x0, y0, z0,
                    Qi, speed_i, dir_i
                )
        
        return C
