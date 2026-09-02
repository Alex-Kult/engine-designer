#Author: Alex Kult
#Description: Create the geometry of a rocket nozzle using the method of characteristics
#Date: 9-2-2026
#
# This file is still the PLANAR (2-D slab) method of characteristics -- see
# MOC_axisymmetric.py for the axisymmetric version, which is what an
# axisymmetric (round) rocket nozzle actually needs. Kept here, fixed, as a
# correct planar reference / fallback.

import MOC_lib as moc
import numpy as np

## Inputs
n = 100 #number of characteristic lines
theta_min = 0.000001 #kickoff angle
throat_mach = 1.00000001 #kickoff mach number
mesh_type = "average" #"leading" is calculated from angles of prior points, "average" is calculated from average of prior and current points
interpolation = True #Make a function from nozzle wall points

def moc_geometry(gamma, mach_e, throat_rad):

    ## Initialziing Arrays
    x_wall_lst = [0]
    y_wall_lst = [throat_rad]

    char_points = -np.ones((n,n+1,2))
    nu_map = -np.ones((n,n+1))
    mach_map = -np.ones((n,n+1))
    mu_map = -np.ones((n,n+1))
    theta_map = -np.ones((n,n+1))

    #Max expansion angle
    theta_max = 0.5*moc.prandtl_meyer(gamma, mach_e) #radians

    initial_theta_lst = np.linspace(theta_min, theta_max, n) #Interpolated thetas
    initial_slope_lst = initial_theta_lst - moc.mach_angle(throat_mach) #Slopes of characteristics from throat

    #First negative characteristic line
    nu_kick = moc.prandtl_meyer(gamma, throat_mach)
    k_minus_lst = 2*initial_theta_lst + nu_kick   # FIX #1 (was: initial_theta_lst + nu_kick)
    k_plus_lst = -k_minus_lst
    nu_1 = k_minus_lst[0]
    nu_map[0,0] = nu_1
    mach_1 = moc.invert_prandtl_meyer_angle(gamma, nu_1)
    mach_map[0,0] = mach_1
    mu_1 = moc.mach_angle(mach_1)
    mu_map[0,0] = mu_1

    #Finding xy coordinates of first point on centerline
    m_1 = np.tan(initial_slope_lst[0])
    x_1 = y_wall_lst[0]/(-m_1)
    char_points[0,0,0] = x_1
    char_points[0,0,1] = 0

    #First positive characteristic Line
    theta_map[:,0] = 0

    for i in range(n - 1):
        #Finding Invariants
        k_minus = k_minus_lst[i+1]
        k_plus = k_plus_lst[0]

        #Calculating point parameters
        nu = 0.5*(k_minus - k_plus)
        theta = 0.5*(k_minus + k_plus)
        mach = moc.invert_prandtl_meyer_angle(gamma, nu)
        mu = moc.mach_angle(mach)

        #Updating arrays
        nu_map[0,i+1] = nu
        theta_map[0,i+1] = theta
        mach_map[0,i+1] = mach
        mu_map[0,i+1] = mu

        #Calculate slope from reference points
        if mesh_type == "leading":
            m_1 = np.tan(initial_slope_lst[i+1])
            m_2 = np.tan(theta_map[0,i] + mu_map[0,i])
        elif mesh_type == "average":
            m_1 = np.tan(0.5*(initial_slope_lst[i+1] + theta - mu))
            m_2 = np.tan(0.5*(theta_map[0,i] + mu_map[0,i] + theta + mu))
        else:
            raise ValueError("Please select a valid mesh type")

        #Calculate new point location
        x = (y_wall_lst[0] - char_points[0,i,1] + m_2*char_points[0,i,0] - m_1*x_wall_lst[0]) / (m_2 - m_1)
        y = y_wall_lst[0] + m_1*(x - x_wall_lst[0])

        #Update coordinate array
        char_points[0,i+1,0] = x
        char_points[0,i+1,1] = y

    #Calculating parameters of first wall point
    wall_ang_1 = initial_theta_lst[-1]
    theta = wall_ang_1                                    # FIX #2 (was: theta_map[0,n-1], i.e. theta_max/2)
    nu = nu_map[0,n-1] + (theta - theta_map[0,n-1])        # keep K- continuation consistent
    mach = moc.invert_prandtl_meyer_angle(gamma, nu)
    mu = moc.mach_angle(mach)

    #Updating arrays
    nu_map[0,n] = nu
    theta_map[0,n] = theta
    mach_map[0,n] = mach
    mu_map[0,n] = mu

    #Calculating slope from reference points
    m_1 = np.tan(wall_ang_1) #ensures initial expansion angle is correct
    m_2 = np.tan(theta + mu)

    #Calculating wall point location
    x = (y_wall_lst[0] - char_points[0,-2,1] + m_2*char_points[0,-2,0] - m_1*x_wall_lst[0]) / (m_2 - m_1)
    y = y_wall_lst[0] + m_1*(x - x_wall_lst[0])

    #Updating coordinate array
    x_wall_lst.append(x)
    y_wall_lst.append(y)
    char_points[0,-1,0] = x
    char_points[0,-1,1] = y

    ## Calculating characteristics 2 through n
    for i in range(n-1):
        #Finding Invariants
        k_minus = k_minus_lst[i+1]
        k_plus = k_plus_lst[i+1]

        #Calculating centerline point parameters
        nu_1 = k_minus
        nu_map[i+1,0] = nu_1
        mach_1 = moc.invert_prandtl_meyer_angle(gamma, nu_1)
        mach_map[i+1,0] = mach_1
        mu_1 = moc.mach_angle(mach_1)
        mu_map[i+1,0] = mu_1

        #Finding xy coordinates of centerline point
        theta = theta_map[i,1]
        mu = mu_map[i,1]
        m_1 = np.tan(theta - mu)

        x_0 = char_points[i,1,0]
        y_0 = char_points[i,1,1]
        x_1 = y_0/(-m_1) + x_0

        #Updating coordinate array
        char_points[i+1,0,0] = x_1
        char_points[i+1,0,1] = 0

        #Calculating interior points on characteristic line
        for j in range(n - i - 2):
            #Updating left running invariant
            k_minus = k_minus_lst[i+j+2]

            #Calculating point parameters
            nu = 0.5*(k_minus - k_plus)
            theta = 0.5*(k_minus + k_plus)
            mach = moc.invert_prandtl_meyer_angle(gamma, nu)
            mu = moc.mach_angle(mach)

            #Updating arrays
            nu_map[i+1,j+1] = nu
            theta_map[i+1,j+1] = theta
            mach_map[i+1,j+1] = mach
            mu_map[i+1,j+1] = mu

            #Calculating slope from reference points
            if mesh_type == "leading":
                m_1 = np.tan(theta_map[i,j+2] - mu_map[i,j+2])
                m_2 = np.tan(theta_map[i+1,j] + mu_map[i+1,j])
            elif mesh_type == "average":
                m_1 = np.tan(0.5*(theta_map[i,j+2] - mu_map[i,j+2] + theta - mu))
                m_2 = np.tan(0.5*(theta_map[i+1,j] + mu_map[i+1,j] + theta + mu))
            else:
                raise ValueError("Please select a valid mesh type")

            #Calculating new point location
            x = (char_points[i,j+2,1] - char_points[i+1,j,1] + m_2*char_points[i+1,j,0] - m_1*char_points[i,j+2,0]) / (m_2 - m_1)
            y = char_points[i,j+2,1] + m_1*(x - char_points[i,j+2,0])

            #Updating coordinate array
            char_points[i+1,j+1,0] = x
            char_points[i+1,j+1,1] = y

        #Finding wall point of characteristic
        wall_ang = initial_theta_lst[-i-2]
        theta = theta_map[i+1,n-i-2]
        nu = nu_map[i+1,n-i-2]
        mach = moc.invert_prandtl_meyer_angle(gamma, nu)
        mu = moc.mach_angle(mach)

        #Updating arrays
        nu_map[i+1,n-i-1] = nu
        theta_map[i+1,n-i-1] = theta
        mach_map[i+1,n-i-1] = mach
        mu_map[i+1,n-i-1] = mu

        #Calculating slopes from reference points
        if mesh_type == "leading":
            m_1 = np.tan(wall_ang)
        elif mesh_type == "average":
            m_1 = np.tan(0.5*(wall_ang + theta))
        else:
            raise ValueError("Please select a valid mesh type")
        m_2 = np.tan(theta + mu)

        #Calculating new point location
        x = (char_points[i,-i-1,1] - char_points[i+1,-i-3,1] + m_2*char_points[i+1,-i-3,0] - m_1*char_points[i,-i-1,0]) / (m_2 - m_1)
        y = char_points[i,-i-1,1] + m_1*(x - char_points[i,-i-1,0])

        #Updating coordinate array
        x_wall_lst.append(x)
        y_wall_lst.append(y)
        char_points[i+1,-i-2,0] = x
        char_points[i+1,-i-2,1] = y

    ##Output
    area_ratio = char_points[n-1,1,1]**2/throat_rad**2
    Re = char_points[n-1,1,1]
    Le = char_points[n-1,1,0]

    return theta_max, area_ratio, Re, Le
