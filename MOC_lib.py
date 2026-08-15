#Author: Alex Kult
#Description: Create the geometry of a rocket nozzle using the method of characteristics
#Date: 5-14-2025

import numpy as np
from scipy.optimize import brentq

def prandtl_meyer(gamma, mach):
    if mach <= 1:
            raise ValueError("Mach number not greater than 1")
    term1 = np.sqrt((gamma + 1) / (gamma - 1))
    term2 = np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (mach**2 - 1)))
    term3 = np.arctan(np.sqrt(mach**2 - 1))
    nu = term1 * term2 - term3
    return nu #nu in radians

def invert_prandtl_meyer_angle(gamma, nu): #nu in radians
    def equation(mach):
        if mach <= 1:
            return np.inf
            raise ValueError("Mach number not greater than 1")
        term1 = np.sqrt((gamma + 1) / (gamma - 1))
        term2 = np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (mach**2 - 1)))
        term3 = np.arctan(np.sqrt(mach**2 - 1))
        return nu - (term1 * term2 - term3)

    # Search interval must bracket a root
    a, b = 1.0001, 50
    mach = brentq(equation, a, b)
    return mach

def mach_angle(mach):
    mu = np.arcsin(1/mach) #radians
    return mu

def invert_mach_angle(mu): #mu in radians
    mach = 1/np.sin(mu)
    return mach