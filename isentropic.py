# --- LIBRARIES ---
import math
from scipy.optimize import fsolve

# --- FUNCTIONS ---
def press_ratio(gamma, press, press_t):
    pres_ratio = press / press_t
    mach_sqrd = (2/(gamma - 1)) * ((pres_ratio)**((1-gamma)/gamma) - 1)
    mach = math.sqrt(mach_sqrd)
    return mach
    
def a_ratio(gamma, mach):
    area_ratio = ((gamma + 1)/2)**(-(gamma + 1)/(2*(gamma - 1))) * 1/mach * (1 + (gamma - 1)/2 * mach**2)**((gamma + 1)/(2*(gamma - 1)))
    return area_ratio

def temp_ratio(gamma, mach, temp_t):
    temp = temp_t*(1 + (gamma - 1)/2*mach**2)**(-1)
    return temp

def mach_find(area_ratio, gamma, regime='subsonic'):
    def equation(M):    
        term1 = (gamma + 1)/2
        term2 = (1 + (gamma - 1)/2 * M**2)
        exp = (gamma + 1) / (2 * (gamma - 1))
        
        area_ratio_calc = term1**(-exp) * term2**exp / M
        return area_ratio_calc - area_ratio

    if regime == 'subsonic':
        initial_guess = 0.1
    elif regime == 'supersonic':
        initial_guess = 2.0

    mach, = fsolve(equation, initial_guess)
    
    return mach